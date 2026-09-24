#!/usr/bin/env python3
"""Train stage-2 SpatialLM from offline scorer-filtered point tokens."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

import spatiallm  # noqa: F401 - registers custom SpatialLM AutoClasses
from spatiallm.tuner.data.collator import _encode_messages_example
from spatiallm.tuner.data.template import (
    IGNORE_INDEX,
    get_template_and_fix_tokenizer,
    register_spatiallm_templates,
)
from spatiallm.tuner.hparams.data_args import DataArguments
from spatiallm.tuner.hparams.finetuning_args import FinetuningArguments
from spatiallm.tuner.framework.utils import count_parameters
from spatiallm.tuner.trainer import CustomSeq2SeqTrainer


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "0923_train_filtered_stage2.yaml"
)


def torch_dtype_from_name(name: str) -> torch.dtype | str | None:
    if name in {None, "auto"}:
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def checkpoint_step(path: Path) -> int | None:
    if not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.removeprefix("checkpoint-")
    if not suffix.isdigit():
        return None
    return int(suffix)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        step = checkpoint_step(child)
        if step is not None:
            checkpoints.append((step, child))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_resume_checkpoint(config: dict[str, Any]) -> str | None:
    explicit = config.get("resume_from_checkpoint")
    output_dir = Path(config["output_dir"])

    if explicit not in (None, "", "auto"):
        if isinstance(explicit, bool):
            if not explicit:
                return None
            latest = find_latest_checkpoint(output_dir)
            if latest is None:
                print(f"No checkpoint found in {output_dir}; starting from scratch.")
                return None
            print(f"Resuming from latest checkpoint: {latest}")
            return str(latest)
        checkpoint = Path(explicit)
        print(f"Resuming from configured checkpoint: {checkpoint}")
        return str(checkpoint)

    if config.get("overwrite_output_dir", False):
        return None

    if not config.get("auto_resume_from_latest_checkpoint", True):
        return None

    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        if output_dir.exists():
            print(f"Output directory exists but no checkpoint was found: {output_dir}")
        return None

    print(f"Auto-resuming from latest checkpoint: {latest}")
    return str(latest)


def configure_wandb_env(config: dict[str, Any]) -> None:
    report_to = config.get("report_to", "none")
    if isinstance(report_to, str):
        reports = {report_to.lower()}
    else:
        reports = {str(item).lower() for item in report_to}
    if "wandb" not in reports:
        return

    wandb_project = config.get("wandb_project")
    if wandb_project:
        os.environ["WANDB_PROJECT"] = str(wandb_project)

    wandb_entity = config.get("wandb_entity")
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = str(wandb_entity)

    wandb_run_name = config.get("wandb_run_name") or config.get("run_name")
    if wandb_run_name:
        os.environ["WANDB_NAME"] = str(wandb_run_name)


def resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_scorer_checkpoint(value: str | Path) -> Path:
    path = resolved_path(value)
    if path.is_file():
        return path
    if (path / "scorer.pt").is_file():
        return path / "scorer.pt"
    checkpoints: list[tuple[int, Path]] = []
    for candidate in path.glob("checkpoint-*"):
        scorer_path = candidate / "scorer.pt"
        suffix = candidate.name.removeprefix("checkpoint-")
        if candidate.is_dir() and suffix.isdigit() and scorer_path.is_file():
            checkpoints.append((int(suffix), scorer_path))
    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]
    raise FileNotFoundError(f"No scorer.pt found under {path}")


def require_same_path(
    actual: str | Path | None,
    expected: str | Path,
    description: str,
) -> None:
    if actual is None:
        raise ValueError(f"Missing lineage field: {description}")
    actual_path = resolved_path(actual)
    expected_path = resolved_path(expected)
    if actual_path != expected_path:
        raise ValueError(
            f"{description} mismatch: actual={actual_path}, "
            f"expected={expected_path}"
        )


def validate_scorer_filtered_lineage(
    config: dict[str, Any],
    datasets: dict[str, "FilteredPointTokenDataset"],
) -> dict[str, Any]:
    if not config.get("validate_scorer_filtered_lineage", False):
        return {}

    model_path = resolved_path(config["model_name_or_path"])
    required_model_files = [
        model_path / "config.json",
        model_path / "model.safetensors",
    ]
    missing_model_files = [
        str(path) for path in required_model_files if not path.is_file()
    ]
    if missing_model_files:
        raise FileNotFoundError(
            "BBoxMask initialization checkpoint is incomplete:\n"
            + "\n".join(missing_model_files)
        )

    scorer_path_value = config.get("scorer_path_for_lineage")
    if not scorer_path_value:
        raise ValueError(
            "validate_scorer_filtered_lineage requires "
            "scorer_path_for_lineage."
        )
    scorer_path = resolve_scorer_checkpoint(scorer_path_value)

    scorer_checkpoint = torch.load(
        scorer_path,
        map_location="cpu",
        weights_only=False,
    )
    scorer_args = scorer_checkpoint.get("args") or {}
    scorer_config = scorer_checkpoint.get("config") or {}
    require_same_path(
        scorer_args.get("projector_model_path"),
        model_path,
        "scorer projector_model_path",
    )

    split_reports = {}
    for split_name, dataset in datasets.items():
        metadata = dataset.metadata
        if metadata.get("feature_type") != "projected_point_tokens":
            raise ValueError(
                f"{split_name} filtered cache feature_type must be "
                f"projected_point_tokens, got {metadata.get('feature_type')!r}."
            )
        for auxiliary_key in (
            "bbox_regression_aux",
            "bbox_point_alignment_aux",
        ):
            if config.get(auxiliary_key, False) and not metadata.get(
                auxiliary_key,
                False,
            ):
                raise ValueError(
                    f"{split_name} filtered cache does not contain required "
                    f"{auxiliary_key} message metadata."
                )
        require_same_path(
            metadata.get("projector_model_path"),
            model_path,
            f"{split_name} filtered-cache projector_model_path",
        )
        require_same_path(
            metadata.get("scorer_path"),
            scorer_path,
            f"{split_name} filtered-cache scorer_path",
        )

        source_cache_dir_value = metadata.get("source_cache_dir")
        if not source_cache_dir_value:
            raise ValueError(
                f"{split_name} filtered cache has no source_cache_dir."
            )
        source_cache_dir = resolved_path(source_cache_dir_value)
        source_index_path = source_cache_dir / "index.json"
        if not source_index_path.is_file():
            raise FileNotFoundError(source_index_path)
        with source_index_path.open("r", encoding="utf-8") as handle:
            source_index = json.load(handle)
        source_metadata = source_index.get("metadata") or {}
        if source_metadata.get("feature_type") != "point_encoder_context":
            raise ValueError(
                f"{split_name} scorer source feature_type must be "
                f"point_encoder_context, got "
                f"{source_metadata.get('feature_type')!r}."
            )
        if source_metadata.get("projected") is not False:
            raise ValueError(
                f"{split_name} scorer source cache must contain unprojected "
                "point-encoder context."
            )
        require_same_path(
            source_metadata.get("model_path"),
            model_path,
            f"{split_name} source-cache point encoder model_path",
        )

        expected_scorer_cache = scorer_args.get(
            "train_cache_dir"
            if split_name == "train"
            else "eval_cache_dir"
        )
        require_same_path(
            expected_scorer_cache,
            source_cache_dir,
            f"scorer {split_name} cache_dir",
        )
        source_feature_dim = int(source_metadata.get("feature_dim", -1))
        scorer_encoder_dim = int(
            scorer_config.get("encoder_feature_dim", -2)
        )
        filtered_token_dim = int(metadata.get("point_token_dim", -1))
        scorer_token_dim = int(scorer_config.get("point_token_dim", -2))
        if source_feature_dim != scorer_encoder_dim:
            raise ValueError(
                f"{split_name} encoder feature dim mismatch: "
                f"cache={source_feature_dim}, scorer={scorer_encoder_dim}"
            )
        if filtered_token_dim != scorer_token_dim:
            raise ValueError(
                f"{split_name} projected token dim mismatch: "
                f"cache={filtered_token_dim}, scorer={scorer_token_dim}"
            )
        split_reports[split_name] = {
            "source_cache_dir": str(source_cache_dir),
            "source_model_path": str(model_path),
            "source_feature_type": source_metadata["feature_type"],
            "source_feature_dim": source_feature_dim,
            "projected_point_token_dim": filtered_token_dim,
            "selection": dataset.selection_summary,
        }

    report = {
        "bbox_mask_model_path": str(model_path),
        "scorer_path": str(scorer_path),
        "scorer_step": int(scorer_checkpoint.get("step", -1)),
        "scorer_projector_model_path": str(
            resolved_path(scorer_args["projector_model_path"])
        ),
        "splits": split_reports,
    }
    print(
        "Validated scorer-filtered lineage:\n"
        + json.dumps(report, indent=2, ensure_ascii=False),
        flush=True,
    )
    del scorer_checkpoint
    return report


class WandbRuntimeConfigCallback(TrainerCallback):
    def __init__(self, runtime_config: dict[str, Any]):
        self.runtime_config = runtime_config

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        try:
            import wandb
        except ImportError:
            return control
        if wandb.run is not None:
            wandb.config.update(
                self.runtime_config,
                allow_val_change=True,
            )
        return control


class FilteredPointTokenDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        max_samples: int | None = None,
        max_point_tokens: int | None = None,
        raw_point_token_threshold_exclusive: int | None = None,
        shard_cache_size: int = 1,
        require_bbox_regression_aux: bool = False,
        require_bbox_point_alignment_aux: bool = False,
    ):
        self.cache_dir = cache_dir
        index_path = cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open("r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        self.metadata = self.index["metadata"]
        all_samples = self.index["samples"]
        self.total_index_samples = len(all_samples)
        self.raw_point_token_threshold_exclusive = (
            raw_point_token_threshold_exclusive
        )
        if raw_point_token_threshold_exclusive is not None:
            if raw_point_token_threshold_exclusive < 0:
                raise ValueError(
                    "raw_point_token_threshold_exclusive must be non-negative."
                )
            missing_raw_count = [
                sample
                for sample in all_samples
                if "raw_token_count" not in sample
            ]
            if missing_raw_count:
                raise ValueError(
                    f"{index_path} has {len(missing_raw_count)} samples without "
                    "raw_token_count; cannot apply the requested region-length "
                    "filter."
                )
            self.samples = [
                sample
                for sample in all_samples
                if int(sample["raw_token_count"])
                > raw_point_token_threshold_exclusive
            ]
        else:
            self.samples = list(all_samples)
        self.selected_index_samples = len(self.samples)
        if not self.samples:
            raise ValueError(
                f"No samples remain after filtering {index_path} with "
                "raw_token_count > "
                f"{raw_point_token_threshold_exclusive}."
            )
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.max_point_tokens = max_point_tokens
        self.shard_cache_size = max(1, shard_cache_size)
        self.require_bbox_regression_aux = bool(require_bbox_regression_aux)
        self.require_bbox_point_alignment_aux = bool(
            require_bbox_point_alignment_aux
        )
        self._shard_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_shard(self, shard_rel: str) -> dict[str, Any]:
        if shard_rel in self._shard_cache:
            shard = self._shard_cache.pop(shard_rel)
            self._shard_cache[shard_rel] = shard
            return shard
        shard = torch.load(self.cache_dir / shard_rel, map_location="cpu")
        self._shard_cache[shard_rel] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    @staticmethod
    def _center_crop_tokens_and_grid(
        point_tokens: torch.Tensor,
        point_grid_coords: torch.Tensor,
        max_point_tokens: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point_tokens.ndim != 2:
            raise ValueError(
                f"point_tokens must be [T, D], got {point_tokens.shape}."
            )
        if (
            point_grid_coords.ndim != 2
            or point_grid_coords.shape[-1] != 3
            or point_grid_coords.shape[0] != point_tokens.shape[0]
        ):
            raise ValueError(
                "grid_coord must be [T, 3] and aligned with point_tokens, "
                f"got tokens={point_tokens.shape}, grid={point_grid_coords.shape}."
            )
        if max_point_tokens is None or point_tokens.shape[0] <= max_point_tokens:
            return point_tokens, point_grid_coords
        excess = point_tokens.shape[0] - max_point_tokens
        trim_start = excess // 2
        trim_end = trim_start + max_point_tokens
        return point_tokens[trim_start:trim_end], point_grid_coords[
            trim_start:trim_end
        ]

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.samples[index]
        shard = self._load_shard(entry["shard"])
        item = shard["items"][entry["item_index"]]
        if "grid_coord" not in item:
            raise KeyError(
                f"Filtered cache item has no grid_coord: "
                f"{entry['shard']}:{entry['item_index']}"
            )
        point_tokens, point_grid_coords = self._center_crop_tokens_and_grid(
            item["point_tokens"],
            item["grid_coord"],
            self.max_point_tokens,
        )
        if "messages" not in item:
            raise KeyError(
                f"Filtered cache item has no processed messages: "
                f"{entry['shard']}:{entry['item_index']}"
            )
        messages = item["messages"]
        assistant_messages = [
            message
            for message in messages
            if message.get("role") == "assistant"
        ]
        if self.require_bbox_regression_aux and not any(
            "_bbox_regression" in message for message in assistant_messages
        ):
            raise KeyError(
                "Filtered cache assistant message is missing required "
                f"_bbox_regression metadata: {entry['shard']}:"
                f"{entry['item_index']}"
            )
        if self.require_bbox_point_alignment_aux and not any(
            "_bbox_point_alignment" in message
            for message in assistant_messages
        ):
            raise KeyError(
                "Filtered cache assistant message is missing required "
                f"_bbox_point_alignment metadata: {entry['shard']}:"
                f"{entry['item_index']}"
            )
        return {
            "point_tokens": point_tokens,
            "point_grid_coords": point_grid_coords,
            "messages": messages,
            "scene_id": item.get("scene_id", ""),
            "point_cloud": item.get("point_cloud", ""),
        }

    @property
    def point_token_dim(self) -> int:
        dim = self.metadata.get("point_token_dim")
        if dim is not None:
            return int(dim)
        return int(self[0]["point_tokens"].shape[-1])

    @property
    def selection_summary(self) -> dict[str, int | None]:
        return {
            "total_index_samples": self.total_index_samples,
            "selected_index_samples": self.selected_index_samples,
            "loaded_samples": len(self.samples),
            "raw_point_token_threshold_exclusive": (
                self.raw_point_token_threshold_exclusive
            ),
        }


class ShardLocalBatchSampler(BatchSampler):
    """Build batches from one shard at a time to avoid random 1GB shard loads."""

    def __init__(
        self,
        dataset: FilteredPointTokenDataset,
        batch_size: int,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        self.shard_to_indices: dict[str, list[int]] = {}
        for index, sample in enumerate(dataset.samples):
            self.shard_to_indices.setdefault(str(sample["shard"]), []).append(index)
        self.shards = list(self.shard_to_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        shard_order = list(range(len(self.shards)))
        if self.shuffle_shards:
            shard_order = torch.randperm(len(self.shards), generator=generator).tolist()

        for shard_idx in shard_order:
            indices = list(self.shard_to_indices[self.shards[shard_idx]])
            if self.shuffle_within_shard:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.shard_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


class FilteredPointTokenCollator:
    def __init__(
        self,
        tokenizer,
        template,
        cutoff_len: int,
        compute_dtype: torch.dtype,
        pad_to_multiple_of: int | None = 8,
        bbox_regression_aux: bool = False,
        bbox_point_alignment_aux: bool = False,
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.cutoff_len = int(cutoff_len)
        self.compute_dtype = compute_dtype
        self.pad_to_multiple_of = pad_to_multiple_of
        self.bbox_regression_aux = bool(bbox_regression_aux)
        self.bbox_point_alignment_aux = bool(bbox_point_alignment_aux)

    @staticmethod
    def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        if messages and messages[0].get("role") == "system":
            return str(messages[0].get("content", "")), messages[1:]
        return "", messages

    def _encode_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[
        list[int],
        list[int],
        list[list[int]],
        list[list[float]],
        list[list[int]],
        list[list[float]],
    ]:
        system, turns = self._split_system(messages)
        return _encode_messages_example(
            messages=turns,
            system=system,
            point_clouds=[],
            template=self.template,
            tokenizer=self.tokenizer,
            cutoff_len=self.cutoff_len,
            return_bbox_regression=True,
            return_bbox_point_alignment=True,
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_messages(sample["messages"]) for sample in samples]
        max_len = max(len(item[0]) for item in encoded)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        batch_size = len(samples)
        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for index, item in enumerate(encoded):
            cur_input_ids, cur_labels = item[:2]
            length = len(cur_input_ids)
            input_ids[index, :length] = torch.tensor(cur_input_ids, dtype=torch.long)
            labels[index, :length] = torch.tensor(cur_labels, dtype=torch.long)
            attention_mask[index, :length] = 1

        max_point_len = max(sample["point_tokens"].shape[0] for sample in samples)
        point_dim = samples[0]["point_tokens"].shape[-1]
        point_token_features = torch.full(
            (batch_size, max_point_len, point_dim),
            fill_value=float("nan"),
            dtype=self.compute_dtype,
        )
        point_token_grid_coords = torch.full(
            (batch_size, max_point_len, 3),
            fill_value=float("nan"),
            dtype=torch.float32,
        )
        for index, sample in enumerate(samples):
            point_tokens = sample["point_tokens"].to(self.compute_dtype)
            point_grid_coords = sample["point_grid_coords"].to(torch.float32)
            if point_grid_coords.shape != (point_tokens.shape[0], 3):
                raise ValueError(
                    "point_grid_coords must stay aligned with point_tokens, "
                    f"got tokens={point_tokens.shape}, "
                    f"grid={point_grid_coords.shape}."
                )
            point_token_features[index, : point_tokens.shape[0]] = point_tokens
            point_token_grid_coords[index, : point_tokens.shape[0]] = (
                point_grid_coords
            )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "point_token_features": point_token_features,
            "point_token_grid_coords": point_token_grid_coords,
        }
        if self.bbox_regression_aux:
            batch_positions = [item[2] for item in encoded]
            batch_targets = [item[3] for item in encoded]
            self._add_bbox_regression_tensors(
                batch,
                batch_positions,
                batch_targets,
            )
        if self.bbox_point_alignment_aux:
            batch_positions = [item[4] for item in encoded]
            batch_bboxes = [item[5] for item in encoded]
            self._add_bbox_alignment_tensors(
                batch,
                batch_positions,
                batch_bboxes,
            )
        return batch

    @staticmethod
    def _allocate_token_position_tensors(
        batch_positions: list[list[list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch_positions)
        max_boxes = max(
            (len(sample_positions) for sample_positions in batch_positions),
            default=0,
        )
        max_numeric_tokens = max(
            (
                len(box_positions)
                for sample_positions in batch_positions
                for box_positions in sample_positions
            ),
            default=0,
        )
        return (
            torch.full(
                (batch_size, max_boxes, max_numeric_tokens),
                -1,
                dtype=torch.long,
            ),
            torch.zeros(
                (batch_size, max_boxes, max_numeric_tokens),
                dtype=torch.bool,
            ),
        )

    @classmethod
    def _add_bbox_regression_tensors(
        cls,
        batch: dict[str, torch.Tensor],
        batch_positions: list[list[list[int]]],
        batch_targets: list[list[list[float]]],
    ) -> None:
        position_tensor, position_mask = (
            cls._allocate_token_position_tensors(batch_positions)
        )
        batch_size, max_boxes = position_tensor.shape[:2]
        target_tensor = torch.zeros(
            (batch_size, max_boxes, 8),
            dtype=torch.float32,
        )
        target_mask = torch.zeros(
            (batch_size, max_boxes),
            dtype=torch.bool,
        )
        for batch_index, (sample_positions, sample_targets) in enumerate(
            zip(batch_positions, batch_targets)
        ):
            if len(sample_positions) != len(sample_targets):
                raise ValueError(
                    "bbox regression token/target count mismatch after "
                    "tokenization."
                )
            for box_index, (positions, target) in enumerate(
                zip(sample_positions, sample_targets)
            ):
                num_positions = len(positions)
                position_tensor[
                    batch_index, box_index, :num_positions
                ] = torch.as_tensor(positions, dtype=torch.long)
                position_mask[
                    batch_index, box_index, :num_positions
                ] = True
                target_tensor[batch_index, box_index] = torch.as_tensor(
                    target,
                    dtype=torch.float32,
                )
                target_mask[batch_index, box_index] = True
        batch["bbox_regression_token_positions"] = position_tensor
        batch["bbox_regression_token_mask"] = position_mask
        batch["bbox_regression_targets"] = target_tensor
        batch["bbox_regression_mask"] = target_mask

    @classmethod
    def _add_bbox_alignment_tensors(
        cls,
        batch: dict[str, torch.Tensor],
        batch_positions: list[list[list[int]]],
        batch_bboxes: list[list[list[float]]],
    ) -> None:
        position_tensor, position_mask = (
            cls._allocate_token_position_tensors(batch_positions)
        )
        batch_size, max_boxes = position_tensor.shape[:2]
        bbox_tensor = torch.full(
            (batch_size, max_boxes, 7),
            torch.nan,
            dtype=torch.float32,
        )
        bbox_mask = torch.zeros(
            (batch_size, max_boxes),
            dtype=torch.bool,
        )
        for batch_index, (sample_positions, sample_bboxes) in enumerate(
            zip(batch_positions, batch_bboxes)
        ):
            if len(sample_positions) != len(sample_bboxes):
                raise ValueError(
                    "bbox alignment token/box count mismatch after "
                    "tokenization."
                )
            for box_index, (positions, bbox) in enumerate(
                zip(sample_positions, sample_bboxes)
            ):
                num_positions = len(positions)
                position_tensor[
                    batch_index, box_index, :num_positions
                ] = torch.as_tensor(positions, dtype=torch.long)
                position_mask[
                    batch_index, box_index, :num_positions
                ] = True
                bbox_tensor[batch_index, box_index] = torch.as_tensor(
                    bbox,
                    dtype=torch.float32,
                )
                bbox_mask[batch_index, box_index] = True
        batch["bbox_point_alignment_token_positions"] = position_tensor
        batch["bbox_point_alignment_token_mask"] = position_mask
        batch["bbox_point_alignment_bboxes"] = bbox_tensor
        batch["bbox_point_alignment_box_mask"] = bbox_mask


class FilteredPointTokenTrainer(CustomSeq2SeqTrainer):
    def __init__(
        self,
        *args,
        shard_local_shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shard_local_shuffle = shard_local_shuffle

    def get_train_dataloader(self) -> DataLoader:
        if (
            not self.shard_local_shuffle
            or self.train_dataset is None
            or self.args.world_size != 1
        ):
            return super().get_train_dataloader()

        sampler = ShardLocalBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle_shards=True,
            shuffle_within_shard=True,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": self.train_dataset,
            "batch_sampler": sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                getattr(self.args, "dataloader_persistent_workers", False)
            )
            prefetch_factor = getattr(
                self.args,
                "dataloader_prefetch_factor",
                None,
            )
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**loader_kwargs)


def validate_training_config(
    config: dict[str, Any],
    has_eval: bool,
) -> None:
    required_fields = [
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_ratio",
        "save_steps",
        "save_total_limit",
        "dataloader_num_workers",
    ]
    if has_eval:
        required_fields.append("eval_steps")
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(
            "Training config must explicitly define trajectory/memory/throughput "
            f"fields; missing: {missing}"
        )
    if str(config["lr_scheduler_type"]).lower() != "cosine":
        raise ValueError("This workflow requires lr_scheduler_type: cosine.")
    warmup_ratio = float(config["warmup_ratio"])
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1].")
    if int(config["save_total_limit"]) <= 0:
        raise ValueError("save_total_limit must be positive.")
    reports = config.get("report_to", "none")
    if isinstance(reports, str):
        reports = [reports]
    if any(str(report).lower() == "wandb" for report in reports):
        missing_wandb = [
            key
            for key in ("wandb_project", "wandb_run_name")
            if not config.get(key)
        ]
        if missing_wandb:
            raise ValueError(
                f"W&B logging requires explicit fields: {missing_wandb}"
            )


def build_training_args(
    config: dict[str, Any],
    has_eval: bool,
) -> Seq2SeqTrainingArguments:
    validate_training_config(config, has_eval)
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in signature.parameters
        else "evaluation_strategy"
    )
    report_to = config.get("report_to", "none")
    if isinstance(report_to, str) and report_to.lower() == "none":
        report_to = []

    kwargs: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "do_train": True,
        "do_eval": has_eval,
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config.get("weight_decay", 0.0),
        "num_train_epochs": config.get("num_train_epochs", 1),
        "lr_scheduler_type": config["lr_scheduler_type"],
        "warmup_ratio": config["warmup_ratio"],
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "logging_steps": config.get("logging_steps", 10),
        "save_steps": config["save_steps"],
        "save_total_limit": config["save_total_limit"],
        "bf16": config.get("bf16", False),
        "fp16": config.get("fp16", False),
        "tf32": config.get("tf32", None),
        "overwrite_output_dir": config.get("overwrite_output_dir", False),
        "remove_unused_columns": False,
        "dataloader_num_workers": config["dataloader_num_workers"],
        "dataloader_pin_memory": config.get("dataloader_pin_memory", True),
        "dataloader_persistent_workers": config.get(
            "dataloader_persistent_workers",
            False,
        ),
        "dataloader_prefetch_factor": config.get(
            "dataloader_prefetch_factor",
            None,
        ),
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "ddp_timeout": config.get("ddp_timeout", 180000000),
        "report_to": report_to,
        "run_name": config.get("wandb_run_name", config.get("run_name", None)),
        "ignore_data_skip": config.get("ignore_data_skip", False),
        "seed": config.get("seed", 42),
    }
    kwargs[eval_key] = config.get("eval_strategy", "steps" if has_eval else "no")
    if has_eval:
        kwargs["eval_steps"] = config["eval_steps"]

    filtered = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    training_args = Seq2SeqTrainingArguments(**filtered)
    training_args.reset_scheduler_on_resume = bool(
        config.get("reset_scheduler_on_resume", False)
    )
    training_args.oom_split_batch_on_cuda_oom = bool(
        config.get("oom_split_batch_on_cuda_oom", False)
    )
    return training_args


def build_tokenizer_and_template(config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config.get("trust_remote_code", True),
        use_fast=config.get("use_fast_tokenizer", True),
        padding_side="right",
    )
    data_args = DataArguments(
        template=config.get("template", "spatiallm_qwen"),
        cutoff_len=config.get("cutoff_len", 8192),
        num_bins=config.get("num_bins", 1280),
        world_size=config.get("world_size", 16.0),
    )
    register_spatiallm_templates(
        cutoff_len=data_args.cutoff_len,
        num_bins=data_args.num_bins,
        world_size=data_args.world_size,
        do_augmentation=False,
        random_rotation=False,
        point_token_bbox_mask=False,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return tokenizer, template


def load_filtered_point_token_model(config: dict[str, Any]):
    model_path = config["model_name_or_path"]
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=config.get("trust_remote_code", True),
    )
    model_config.point_config["num_bins"] = config.get("num_bins", 1280)
    model_config.point_config["world_size"] = config.get("world_size", 16.0)
    model_config.point_config["max_point_tokens"] = None
    auxiliary_config_keys = (
        "bbox_regression_aux",
        "bbox_regression_loss_weight",
        "bbox_regression_loss_warmup_ratio",
        "bbox_regression_center_loss_weight",
        "bbox_regression_size_loss_weight",
        "bbox_regression_yaw_loss_weight",
        "bbox_regression_smooth_l1_beta",
        "bbox_regression_hidden_dim",
        "bbox_regression_bottleneck_dim",
        "bbox_point_alignment_aux",
        "bbox_point_alignment_loss_weight",
        "bbox_point_alignment_loss_warmup_ratio",
        "bbox_point_alignment_hidden_dim",
        "bbox_point_alignment_knn_k",
        "bbox_point_alignment_negative_distance_power",
        "bbox_point_alignment_detach_point_features",
    )
    for key in auxiliary_config_keys:
        if key in config:
            model_config.point_config[key] = config[key]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        trust_remote_code=config.get("trust_remote_code", True),
        torch_dtype=torch_dtype_from_name(config.get("model_torch_dtype", "auto")),
        low_cpu_mem_usage=config.get("low_cpu_mem_usage", False),
    )
    for auxiliary_key in (
        "bbox_regression_aux",
        "bbox_point_alignment_aux",
    ):
        expected = bool(config.get(auxiliary_key, False))
        actual = bool(getattr(model, auxiliary_key, False))
        if actual != expected:
            raise ValueError(
                f"Model {auxiliary_key}={actual} does not match config "
                f"{expected}."
            )

    if config.get("freeze_point_backbone", True):
        model.point_backbone.requires_grad_(False)
        model.set_point_backbone_dtype(torch.float32)
    if config.get("freeze_point_projector", True):
        model.point_proj.requires_grad_(False)

    if config.get("require_llm_only_training", False):
        trainable_point_parameters = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                name.startswith("point_backbone.")
                or name.startswith("point_proj.")
            )
        ]
        if trainable_point_parameters:
            raise ValueError(
                "LLM-only training requested, but point modules remain "
                f"trainable: {trainable_point_parameters[:10]}"
            )

    if not config.get("pure_bf16", False):
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.train()
    trainable, total = count_parameters(model)
    print(
        f"trainable params: {trainable:,} || all params: {total:,} || "
        f"trainable%: {100 * trainable / total:.4f}"
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SpatialLM stage-2 from scorer-filtered point-token cache."
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    required_epochs = config.get("required_num_train_epochs")
    if required_epochs is None and config.get("require_single_epoch", False):
        required_epochs = 1
    if (
        required_epochs is not None
        and float(config.get("num_train_epochs", 1))
        != float(required_epochs)
    ):
        raise ValueError(
            "Configured epoch-count guard failed: "
            f"required_num_train_epochs={required_epochs}, "
            f"num_train_epochs={config.get('num_train_epochs', 1)}."
        )
    configure_wandb_env(config)
    random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    if config.get("tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer, template = build_tokenizer_and_template(config)
    train_dataset = FilteredPointTokenDataset(
        Path(config["filtered_point_token_dir"]),
        max_samples=args.max_train_samples or config.get("max_train_samples"),
        max_point_tokens=config.get("max_point_tokens"),
        raw_point_token_threshold_exclusive=config.get(
            "raw_point_token_threshold_exclusive"
        ),
        shard_cache_size=config.get("shard_cache_size", 1),
        require_bbox_regression_aux=config.get(
            "bbox_regression_aux",
            False,
        ),
        require_bbox_point_alignment_aux=config.get(
            "bbox_point_alignment_aux",
            False,
        ),
    )
    eval_dir = config.get("eval_filtered_point_token_dir")
    eval_dataset = (
        FilteredPointTokenDataset(
            Path(eval_dir),
            max_samples=args.max_eval_samples or config.get("max_eval_samples"),
            max_point_tokens=config.get("max_point_tokens"),
            raw_point_token_threshold_exclusive=config.get(
                "raw_point_token_threshold_exclusive"
            ),
            shard_cache_size=config.get("shard_cache_size", 1),
            require_bbox_regression_aux=config.get(
                "bbox_regression_aux",
                False,
            ),
            require_bbox_point_alignment_aux=config.get(
                "bbox_point_alignment_aux",
                False,
            ),
        )
        if eval_dir
        else None
    )
    print(
        "Filtered train dataset selection: "
        f"{json.dumps(train_dataset.selection_summary, ensure_ascii=False)}",
        flush=True,
    )
    if eval_dataset is not None:
        print(
            "Filtered eval dataset selection: "
            f"{json.dumps(eval_dataset.selection_summary, ensure_ascii=False)}",
            flush=True,
        )
    lineage_report = validate_scorer_filtered_lineage(
        config,
        {
            "train": train_dataset,
            **({"eval": eval_dataset} if eval_dataset is not None else {}),
        },
    )

    compute_dtype = (
        torch.bfloat16
        if config.get("bf16", False) or config.get("pure_bf16", False)
        else torch.float16
        if config.get("fp16", False)
        else torch.float32
    )
    collator = FilteredPointTokenCollator(
        tokenizer=tokenizer,
        template=template,
        cutoff_len=config.get("cutoff_len", 8192),
        compute_dtype=compute_dtype,
        pad_to_multiple_of=8 if config.get("pad_to_multiple_of_8", True) else None,
        bbox_regression_aux=config.get("bbox_regression_aux", False),
        bbox_point_alignment_aux=config.get(
            "bbox_point_alignment_aux",
            False,
        ),
    )
    model = load_filtered_point_token_model(config)
    training_args = build_training_args(config, has_eval=eval_dataset is not None)
    effective_global_batch_size = (
        int(training_args.train_batch_size)
        * int(training_args.gradient_accumulation_steps)
        * int(training_args.world_size)
    )
    target_global_batch_size = config.get("target_global_batch_size")
    if (
        target_global_batch_size is not None
        and effective_global_batch_size != int(target_global_batch_size)
    ):
        raise ValueError(
            "Effective global batch size guard failed: "
            f"effective={effective_global_batch_size}, "
            f"target={target_global_batch_size}."
        )
    estimated_optimizer_steps = math.ceil(
        len(train_dataset) / effective_global_batch_size
    ) * math.ceil(float(training_args.num_train_epochs))
    print(
        "Optimization estimate: "
        f"samples={len(train_dataset)}, "
        f"global_batch={effective_global_batch_size}, "
        f"epochs={training_args.num_train_epochs}, "
        f"optimizer_steps≈{estimated_optimizer_steps}",
        flush=True,
    )
    runtime_wandb_config = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "effective_global_batch_size": effective_global_batch_size,
        "target_global_batch_size": target_global_batch_size,
        "bbox_regression_aux": config.get("bbox_regression_aux", False),
        "bbox_point_alignment_aux": config.get(
            "bbox_point_alignment_aux",
            False,
        ),
        "raw_point_token_threshold_exclusive": config.get(
            "raw_point_token_threshold_exclusive"
        ),
        "freeze_point_backbone": config.get("freeze_point_backbone", True),
        "freeze_point_projector": config.get(
            "freeze_point_projector",
            True,
        ),
        "require_llm_only_training": config.get(
            "require_llm_only_training",
            False,
        ),
        "filtered_train_selection": train_dataset.selection_summary,
        "filtered_eval_selection": (
            eval_dataset.selection_summary
            if eval_dataset is not None
            else None
        ),
        "scorer_filtered_lineage": lineage_report,
    }
    trainer = FilteredPointTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        finetuning_args=FinetuningArguments(
            pure_bf16=config.get("pure_bf16", False),
        ),
        shard_local_shuffle=config.get("shard_local_shuffle", True),
        callbacks=[WandbRuntimeConfigCallback(runtime_wandb_config)],
    )

    resume_from_checkpoint = resolve_resume_checkpoint(config)
    if (
        resume_from_checkpoint is not None
        and config.get("reset_scheduler_on_resume", False)
        and (Path(resume_from_checkpoint) / "optimizer.pt").is_file()
    ):
        raise ValueError(
            "reset_scheduler_on_resume=true is forbidden when the checkpoint "
            "contains optimizer state; preserve the original scheduler "
            "trajectory instead."
        )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
