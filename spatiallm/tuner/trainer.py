import gc
import math
import os
from copy import copy
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, IterableDataset
from transformers import Seq2SeqTrainer, get_scheduler
from transformers.trainer import is_datasets_available, seed_worker
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing_extensions import override

if is_datasets_available():
    import datasets
else:
    datasets = None

from spatiallm.tuner.framework import logging
from spatiallm.tuner.framework.utils import is_transformers_version_greater_than
from spatiallm.tuner.framework.callbacks import (
    LogCallback,
    ReporterCallback,
    get_swanlab_callback,
)
from spatiallm.tuner.framework.loader import load_tokenizer, load_model
from spatiallm.tuner.hparams import get_train_args, read_args
from spatiallm.tuner.hparams.training_args import recommended_eval_steps
from spatiallm.tuner.data import (
    IGNORE_INDEX,
    get_dataset,
    get_template_and_fix_tokenizer,
    register_spatiallm_templates,
    SFTDataCollatorWith4DAttentionMask,
)


if TYPE_CHECKING:
    from transformers import (
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    from spatiallm.tuner.hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )

logger = logging.get_logger(__name__)


class CompleteAccumulationBatchSampler(BatchSampler):
    """Drop the final incomplete gradient-accumulation window each epoch.

    The wrapped sampler still controls ordering and receives ``set_epoch`` from
    the distributed DataLoader, so a new shuffle is sampled for every epoch.
    Only a prefix containing an integer number of complete accumulation windows
    is exposed to the Trainer.
    """

    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool,
        gradient_accumulation_steps: int,
    ) -> None:
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        super().__init__(sampler, batch_size, drop_last)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)

    def __len__(self) -> int:
        complete_windows = super().__len__() // self.gradient_accumulation_steps
        return complete_windows * self.gradient_accumulation_steps

    def __iter__(self):
        complete_micro_batches = len(self)
        for batch_index, batch in enumerate(super().__iter__()):
            if batch_index >= complete_micro_batches:
                break
            yield batch

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


def build_eval_collator(data_collator, data_args):
    """Keep training augmentation unchanged while giving validation its own plugin."""
    if all(getattr(data_args, key) is None for key in (
        "eval_do_augmentation", "eval_geometric_augmentation", "eval_point_sampling"
    )):
        return None
    eval_collator = copy(data_collator)
    eval_collator.template = copy(data_collator.template)
    eval_collator.template.mm_plugin = data_collator.template.mm_plugin.copy_for_evaluation(
        do_augmentation=data_args.eval_do_augmentation,
        geometric_augmentation=data_args.eval_geometric_augmentation,
        point_sampling=data_args.eval_point_sampling,
    )
    logger.info_rank0(
        "Independent validation collator: "
        f"do_augmentation={eval_collator.template.mm_plugin.do_augmentation}, "
        f"geometric_augmentation={eval_collator.template.mm_plugin.geometric_augmentation}, "
        f"point_sampling={data_args.eval_point_sampling or 'train'}"
    )
    return eval_collator


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        gen_kwargs: Optional[dict[str, Any]] = None,
        eval_data_collator=None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.eval_data_collator = eval_data_collator

        self.finetuning_args = finetuning_args
        self._lm_loss_sums: dict[str, Optional[torch.Tensor]] = {
            "train": None,
            "eval": None,
        }
        self._lm_loss_counts = {"train": 0, "eval": 0}
        self._lm_train_optimizer_steps: set[int] = set()
        self._bbox_metric_sums: dict[str, dict[str, torch.Tensor]] = {
            "train": {},
            "eval": {},
        }
        self._bbox_metric_counts = {"train": 0, "eval": 0}
        self._bbox_point_alignment_metric_sums: dict[
            str, dict[str, torch.Tensor]
        ] = {
            "train": {},
            "eval": {},
        }
        self._bbox_point_alignment_metric_counts = {"train": 0, "eval": 0}
        self._oom_split_batches_since_log = 0
        self._oom_split_batches_total = 0
        self._oom_split_samples_since_log = 0
        self._oom_split_loss_divisor = 1
        self._num_training_steps: Optional[int] = None
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

    @override
    def get_train_dataloader(self):
        """Build a loader whose length is divisible by gradient accumulation.

        Transformers' default loop consumes a short final accumulation window
        when the number of micro-batches is not divisible by GAS. That changes
        the effective loss scale once per epoch. We keep the sampler's normal
        epoch-wise shuffle and expose only complete windows instead.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if datasets is not None and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, IterableDataset):
            sampler = self._get_train_sampler()
            if self.args.drop_incomplete_accumulation_window:
                batch_sampler = CompleteAccumulationBatchSampler(
                    sampler=sampler,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                )
            else:
                batch_sampler = BatchSampler(
                    sampler,
                    self._train_batch_size,
                    self.args.dataloader_drop_last,
                )
            dataloader_params["batch_sampler"] = batch_sampler
            dataloader_params.pop("batch_size")
            dataloader_params["worker_init_fn"] = seed_worker
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        else:
            # Iterable datasets have no sampler whose ordering we can control;
            # retain the Trainer's normal handling for that unsupported case.
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        train_dataloader = self.accelerator.prepare(
            DataLoader(train_dataset, **dataloader_params)
        )
        if (
            self.args.auto_eval_steps
            and self.args.do_train
            and self.args.do_eval
            and self.args.eval_strategy == "steps"
            and self.args.max_steps <= 0
        ):
            steps_per_epoch = max(
                1,
                len(train_dataloader) // self.args.gradient_accumulation_steps,
            )
            estimated_steps = math.ceil(
                float(self.args.num_train_epochs) * steps_per_epoch
            )
            eval_steps = recommended_eval_steps(estimated_steps)
            if eval_steps is not None:
                self.args.eval_steps = eval_steps
                logger.info_rank0(
                    "Auto eval cadence: "
                    f"estimated_steps={estimated_steps}, eval_steps={eval_steps}"
                )
        return train_dataloader

    @override
    def get_eval_dataloader(self, eval_dataset=None):
        original = self.data_collator
        try:
            if self.eval_data_collator is not None:
                self.data_collator = self.eval_data_collator
            return super().get_eval_dataloader(eval_dataset)
        finally:
            # DataLoader captures its collator; restore before training resumes.
            self.data_collator = original

    @override
    def get_test_dataloader(self, test_dataset):
        original = self.data_collator
        try:
            if self.eval_data_collator is not None:
                self.data_collator = self.eval_data_collator
            return super().get_test_dataloader(test_dataset)
        finally:
            self.data_collator = original

    def save_final_checkpoint(self) -> str:
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        if os.path.isdir(checkpoint_dir):
            logger.info_rank0(
                f"Final-step checkpoint already exists: {checkpoint_dir}"
            )
            return checkpoint_dir

        logger.info_rank0(f"Saving final-step checkpoint: {checkpoint_dir}")
        self._save_checkpoint(self.model, trial=None, metrics=None)
        return checkpoint_dir

    @override
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        self._num_training_steps = int(num_training_steps)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None or not self.args.reset_scheduler_on_resume:
            return
        checkpoint_name = os.path.basename(os.path.normpath(str(checkpoint)))
        suffix = checkpoint_name.removeprefix("checkpoint-")
        checkpoint_step = int(suffix) if suffix.isdigit() else None
        total_steps = self._num_training_steps
        if (
            checkpoint_step is None
            or total_steps is None
            or total_steps <= checkpoint_step
        ):
            return
        remaining_steps = total_steps - checkpoint_step
        if self.args.warmup_steps > 0:
            warmup_steps = min(int(self.args.warmup_steps), remaining_steps)
        else:
            warmup_steps = int(remaining_steps * float(self.args.warmup_ratio))
        for group in self.optimizer.param_groups:
            group["lr"] = self.args.learning_rate
            group["initial_lr"] = self.args.learning_rate
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=remaining_steps,
        )
        logger.info_rank0(
            "Reset scheduler after resume: "
            f"checkpoint_step={checkpoint_step}, "
            f"remaining_steps={remaining_steps}, warmup_steps={warmup_steps}"
        )

    @staticmethod
    def _unwrap_model(model: "torch.nn.Module") -> "torch.nn.Module":
        while hasattr(model, "module"):
            model = model.module
        return model

    def _bbox_loss_scale(self, model: "torch.nn.Module") -> float:
        core_model = self._unwrap_model(model)
        if not getattr(core_model, "bbox_regression_aux", False):
            return 1.0
        point_config = getattr(core_model.config, "point_config", {})
        warmup_ratio = float(
            point_config.get("bbox_regression_loss_warmup_ratio", 0.05)
        )
        if warmup_ratio <= 0:
            return 1.0
        max_steps = int(getattr(self.state, "max_steps", 0))
        if max_steps <= 0:
            return 1.0
        warmup_steps = max(1, math.ceil(max_steps * warmup_ratio))
        return min(1.0, float(self.state.global_step + 1) / warmup_steps)

    def _bbox_point_alignment_loss_scale(
        self,
        model: "torch.nn.Module",
    ) -> float:
        core_model = self._unwrap_model(model)
        if not getattr(core_model, "bbox_point_alignment_aux", False):
            return 1.0
        point_config = getattr(core_model.config, "point_config", {})
        warmup_ratio = float(
            point_config.get("bbox_point_alignment_loss_warmup_ratio", 0.05)
        )
        if warmup_ratio <= 0:
            return 1.0
        max_steps = int(getattr(self.state, "max_steps", 0))
        if max_steps <= 0:
            return 1.0
        warmup_steps = max(1, math.ceil(max_steps * warmup_ratio))
        return min(1.0, float(self.state.global_step + 1) / warmup_steps)

    @override
    def compute_loss(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        core_model = self._unwrap_model(model)
        if getattr(core_model, "bbox_regression_aux", False):
            inputs = dict(inputs)
            inputs["bbox_regression_loss_scale"] = self._bbox_loss_scale(model)
            inputs["bbox_regression_accumulation_divisor"] = (
                float(
                    self.args.gradient_accumulation_steps
                    * self._oom_split_loss_divisor
                )
                if model.training and num_items_in_batch is not None
                else 1.0
            )
        if getattr(core_model, "bbox_point_alignment_aux", False):
            inputs = dict(inputs)
            inputs["bbox_point_alignment_loss_scale"] = (
                self._bbox_point_alignment_loss_scale(model)
            )
            inputs["bbox_point_alignment_accumulation_divisor"] = (
                float(
                    self.args.gradient_accumulation_steps
                    * self._oom_split_loss_divisor
                )
                if model.training and num_items_in_batch is not None
                else 1.0
            )

        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        metrics = getattr(core_model, "_last_bbox_regression_metrics", {})
        if metrics:
            split = "train" if model.training else "eval"
            sums = self._bbox_metric_sums[split]
            for key, value in metrics.items():
                detached = value.detach()
                sums[key] = sums.get(key, detached.new_zeros(())) + detached
            self._bbox_metric_counts[split] += 1
        alignment_metrics = getattr(
            core_model,
            "_last_bbox_point_alignment_metrics",
            {},
        )
        if alignment_metrics:
            split = "train" if model.training else "eval"
            sums = self._bbox_point_alignment_metric_sums[split]
            for key, value in alignment_metrics.items():
                detached = value.detach()
                sums[key] = sums.get(key, detached.new_zeros(())) + detached
            self._bbox_point_alignment_metric_counts[split] += 1
        lm_loss = getattr(core_model, "_last_lm_loss", None)
        if lm_loss is not None:
            split = "train" if model.training else "eval"
            detached_lm_loss = lm_loss.detach()
            current_sum = self._lm_loss_sums[split]
            self._lm_loss_sums[split] = (
                detached_lm_loss.clone()
                if current_sum is None
                else current_sum + detached_lm_loss
            )
            if split == "train":
                # With Transformers 4.46, every micro-batch CE is normalized
                # by the token count of its full accumulation window. Sum the
                # contributions and average by optimizer step, matching
                # Trainer's train/loss semantics.
                self._lm_train_optimizer_steps.add(int(self.state.global_step))
            else:
                self._lm_loss_counts[split] += 1
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _split_prepared_batch(
        inputs: dict[str, Union["torch.Tensor", Any]],
    ) -> list[dict[str, Union["torch.Tensor", Any]]]:
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim == 0:
            raise ValueError(
                "OOM batch splitting requires a batched input_ids tensor."
            )
        batch_size = int(input_ids.shape[0])
        if batch_size <= 1:
            return [inputs]

        point_clouds = inputs.get("point_clouds")
        point_offsets = inputs.get("point_cloud_offsets")
        packed_point_ranges = None
        if torch.is_tensor(point_offsets):
            if point_offsets.ndim != 1 or point_offsets.numel() != batch_size:
                raise ValueError(
                    "Packed point_cloud_offsets must contain one cumulative "
                    "offset per local sample."
                )
            ends = [int(value) for value in point_offsets.detach().cpu().tolist()]
            starts = [0, *ends[:-1]]
            packed_point_ranges = list(zip(starts, ends))

        split_batches = []
        for batch_index in range(batch_size):
            sample = {}
            for key, value in inputs.items():
                if key == "point_clouds" and packed_point_ranges is not None:
                    if not torch.is_tensor(point_clouds):
                        raise ValueError(
                            "Packed point_clouds must be represented by a tensor."
                        )
                    start, end = packed_point_ranges[batch_index]
                    sample[key] = point_clouds[start:end]
                elif key == "point_cloud_offsets" and packed_point_ranges is not None:
                    start, end = packed_point_ranges[batch_index]
                    sample[key] = point_offsets.new_tensor([end - start])
                elif (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == batch_size
                ):
                    sample[key] = value[batch_index : batch_index + 1]
                elif isinstance(value, list) and len(value) == batch_size:
                    sample[key] = value[batch_index : batch_index + 1]
                elif isinstance(value, tuple) and len(value) == batch_size:
                    sample[key] = value[batch_index : batch_index + 1]
                else:
                    sample[key] = value
            split_batches.append(sample)
        return split_batches

    def _backward_training_loss(self, loss: "torch.Tensor") -> "torch.Tensor":
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.use_apex:
            raise RuntimeError(
                "oom_split_batch_on_cuda_oom does not support Apex training."
            )
        loss = loss * self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        num_items_in_batch=None,
    ) -> "torch.Tensor":
        if not self.args.oom_split_batch_on_cuda_oom:
            return super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
        if self.args.world_size != 1:
            raise RuntimeError(
                "oom_split_batch_on_cuda_oom currently supports single-process "
                "training only; rank-local OOM recovery is unsafe under DDP."
            )

        input_ids = inputs.get("input_ids")
        if (
            not torch.is_tensor(input_ids)
            or input_ids.ndim == 0
            or input_ids.shape[0] <= 1
        ):
            return super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        prepared_inputs = self._prepare_inputs(inputs)

        loss = None
        oom_message = None
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(
                    model,
                    prepared_inputs,
                    num_items_in_batch=num_items_in_batch,
                )
        except torch.OutOfMemoryError as error:
            # Only the forward/loss phase is retried. An OOM raised by
            # backward propagates normally so partially written gradients are
            # never reused.
            oom_message = str(error).splitlines()[0]

        if oom_message is None:
            del prepared_inputs
            return self._backward_training_loss(loss)

        batch_size = int(input_ids.shape[0])
        loss = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.warning_rank0(
            "CUDA OOM during forward/loss at optimizer_step="
            f"{self.state.global_step}; retrying this local batch as "
            f"{batch_size} sequential bs=1 samples. Original error: "
            f"{oom_message}"
        )

        split_batches = self._split_prepared_batch(prepared_inputs)
        total_loss = torch.zeros((), device=self.args.device)
        self._oom_split_loss_divisor = batch_size
        try:
            for sample in split_batches:
                with self.compute_loss_context_manager():
                    sample_loss = self.compute_loss(
                        model,
                        sample,
                        num_items_in_batch=num_items_in_batch,
                    )
                total_loss = total_loss + self._backward_training_loss(sample_loss)
                del sample_loss
                torch.cuda.empty_cache()
        finally:
            self._oom_split_loss_divisor = 1
            del prepared_inputs

        self._oom_split_batches_since_log += 1
        self._oom_split_batches_total += 1
        self._oom_split_samples_since_log += batch_size
        return total_loss

    @override
    def log(self, logs: dict[str, float]) -> None:
        split = "eval" if any(key.startswith("eval") for key in logs) else "train"
        if split == "train" and self.args.oom_split_batch_on_cuda_oom:
            logs["oom_recovery/split_batches"] = float(
                self._oom_split_batches_since_log
            )
            logs["oom_recovery/split_samples"] = float(
                self._oom_split_samples_since_log
            )
            logs["oom_recovery/total_split_batches"] = float(
                self._oom_split_batches_total
            )
            self._oom_split_batches_since_log = 0
            self._oom_split_samples_since_log = 0
        lm_loss_sum = self._lm_loss_sums[split]
        lm_loss_count = (
            len(self._lm_train_optimizer_steps)
            if split == "train"
            else self._lm_loss_counts[split]
        )
        if lm_loss_sum is not None and lm_loss_count > 0:
            reduced_lm_loss = lm_loss_sum.clone()
            count_tensor = reduced_lm_loss.new_tensor(float(lm_loss_count))
            if dist.is_initialized():
                dist.all_reduce(reduced_lm_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            key = "eval_lm_loss" if split == "eval" else "lm_loss"
            logs[key] = float((reduced_lm_loss / count_tensor).item())
            self._lm_loss_sums[split] = None
            self._lm_loss_counts[split] = 0
            if split == "train":
                self._lm_train_optimizer_steps.clear()
        count = self._bbox_metric_counts[split]
        if count > 0:
            prefix = "eval_bbox" if split == "eval" else "bbox"
            count_tensor = next(
                iter(self._bbox_metric_sums[split].values())
            ).new_tensor(float(count))
            if dist.is_initialized():
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            for key, value in self._bbox_metric_sums[split].items():
                reduced = value.clone()
                if dist.is_initialized():
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                logs[f"{prefix}/{key}"] = float((reduced / count_tensor).item())
            self._bbox_metric_sums[split] = {}
            self._bbox_metric_counts[split] = 0
        alignment_count = self._bbox_point_alignment_metric_counts[split]
        if alignment_count > 0:
            prefix = (
                "eval_bbox_point_alignment"
                if split == "eval"
                else "bbox_point_alignment"
            )
            count_tensor = next(
                iter(self._bbox_point_alignment_metric_sums[split].values())
            ).new_tensor(float(alignment_count))
            if dist.is_initialized():
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            for key, value in self._bbox_point_alignment_metric_sums[
                split
            ].items():
                reduced = value.clone()
                if dist.is_initialized():
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                logs[f"{prefix}/{key}"] = float(
                    (reduced / count_tensor).item()
                )
            self._bbox_point_alignment_metric_sums[split] = {}
            self._bbox_point_alignment_metric_counts[split] = 0
        super().log(logs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = (
                self.processing_class.pad_token_id
            )
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    register_spatiallm_templates(
        cutoff_len=data_args.cutoff_len,
        num_bins=data_args.num_bins,
        world_size=data_args.world_size,
        do_augmentation=data_args.do_augmentation,
        random_rotation=data_args.random_rotation,
        geometric_augmentation=data_args.geometric_augmentation,
        point_token_bbox_mask=data_args.point_token_bbox_mask,
        point_token_bbox_expand_ratio=data_args.point_token_bbox_expand_ratio,
        point_cloud_batch_encoding=data_args.point_cloud_batch_encoding,
        point_token_scorer_gt_mask=data_args.point_token_scorer_gt_mask,
        bbox_regression_aux=data_args.bbox_regression_aux,
        bbox_point_alignment_aux=data_args.bbox_point_alignment_aux,
        bbox_filter_inverse_rotation=data_args.bbox_filter_inverse_rotation,
    )

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(model_args, data_args, training_args)
    model = load_model(
        tokenizer, data_args, model_args, finetuning_args, training_args.do_train
    )

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=(
            8 if training_args.do_train else None
        ),  # for shift short attention
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id
    ] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        eval_data_collator=build_eval_collator(data_collator, data_args),
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        if training_args.save_final_checkpoint:
            trainer.save_final_checkpoint()
        else:
            trainer.save_model()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )

    callbacks: list[Any] = []
    callbacks.append(LogCallback())
    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    callbacks.append(
        ReporterCallback(model_args, data_args, finetuning_args, generating_args)
    )  # add to last

    run_sft(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
    )

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    _training_function(config={"args": args})


if __name__ == "__main__":
    run_exp()
