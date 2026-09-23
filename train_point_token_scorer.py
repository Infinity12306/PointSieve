#!/usr/bin/env python3
"""Train a transformer scorer for GT-BBoxMask point-token prediction."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler

import spatiallm  # noqa: F401 - registers custom SpatialLM AutoClasses
from spatiallm.model.point_token_scorer import PointTokenScorer, ScorerConfig


DEFAULT_PROJECTOR_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "stage2_bboxmask"
    / "checkpoint-14392"
)


class PointTokenCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        max_samples: int | None = None,
        shard_cache_size: int = 2,
    ):
        self.cache_dir = cache_dir
        index_path = cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        with index_path.open("r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.metadata = self.index["metadata"]
        self.samples = self.index["samples"]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.shard_cache_size = max(1, shard_cache_size)
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.samples[index]
        shard = self._load_shard(entry["shard"])
        item = shard["items"][entry["item_index"]]
        return {
            "features": item["features"],
            "grid_coord": item["grid_coord"],
            "region_center_grid_coord": item["region_center_grid_coord"],
            "labels": item["labels"].float(),
            "scene_id": item["scene_id"],
            "point_cloud": item["point_cloud"],
        }

    @property
    def feature_dim(self) -> int:
        feature_dim = self.metadata.get("feature_dim")
        if feature_dim is not None:
            return int(feature_dim)
        sample = self[0]
        return int(sample["features"].shape[-1])

    @property
    def coord_scale(self) -> float:
        return float(self.metadata.get("reduced_grid_size", 80))

    @property
    def positive_count(self) -> int:
        return int(sum(int(sample["positive_count"]) for sample in self.samples))

    @property
    def token_count(self) -> int:
        return int(sum(int(sample["token_count"]) for sample in self.samples))


class ShardLocalBatchSampler(BatchSampler):
    """Build batches from one shard at a time to improve shard cache locality."""

    def __init__(
        self,
        dataset: PointTokenCacheDataset,
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


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_len = max(sample["features"].shape[0] for sample in samples)
    feature_dim = samples[0]["features"].shape[-1]
    feature_dtype = samples[0]["features"].dtype

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=feature_dtype)
    grid_coord = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    centers = torch.zeros(batch_size, 3, dtype=torch.float32)

    for i, sample in enumerate(samples):
        length = sample["features"].shape[0]
        features[i, :length] = sample["features"].to(feature_dtype)
        grid_coord[i, :length] = sample["grid_coord"].float()
        labels[i, :length] = sample["labels"].float()
        attention_mask[i, :length] = True
        centers[i] = sample["region_center_grid_coord"].float()

    return {
        "features": features,
        "grid_coord": grid_coord,
        "region_center_grid_coord": centers,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point-token mask scorer.")
    parser.add_argument("--train_cache_dir", type=Path, required=True)
    parser.add_argument("--eval_cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--projector_model_path",
        type=Path,
        default=DEFAULT_PROJECTOR_MODEL_PATH,
        help="SpatialLM checkpoint used to load the frozen point projector.",
    )
    parser.add_argument(
        "--projector_torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Dtype used when loading the frozen point projector.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_weight", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--dataloader_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dataloader_persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    parser.add_argument("--shard_cache_size", type=int, default=1)
    parser.add_argument(
        "--shard_local_shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Shuffle shard order and shuffle samples within each shard, then build "
            "each train batch from a single shard."
        ),
    )
    parser.add_argument(
        "--drop_last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop incomplete train batches.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument(
        "--debug_memory",
        action="store_true",
        help="Print detailed GPU memory usage around the first eval batch.",
    )
    parser.add_argument(
        "--disable_eval_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable PyTorch MHA/Transformer eval fastpath during evaluation. "
            "This avoids very large explicit attention buffers for long sequences."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--auto_resume_from_latest_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reset_scheduler_on_resume",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb_project", default="spatiallm-point-token-scorer")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_copy_from_run_id", default=None)
    parser.add_argument("--wandb_copy_from_project", default=None)
    parser.add_argument("--wandb_copy_from_entity", default=None)
    parser.add_argument("--wandb_copy_page_size", type=int, default=1000)

    raw_args = sys.argv[1:]
    if raw_args and Path(raw_args[0]).suffix.lower() in {".yaml", ".yml"}:
        if len(raw_args) != 1:
            parser.error("YAML config mode does not accept additional CLI overrides.")
        config_path = Path(raw_args[0])
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            parser.error(f"Expected YAML mapping: {config_path}")
        action_by_dest = {action.dest: action for action in parser._actions}
        unknown = sorted(set(config) - set(action_by_dest))
        if unknown:
            parser.error(f"Unknown YAML config keys: {unknown}")
        for key, value in config.items():
            action_by_dest[key].required = False
            parser.set_defaults(**{key: value})
        raw_args = []

    args = parser.parse_args(raw_args)
    for key in (
        "train_cache_dir",
        "eval_cache_dir",
        "output_dir",
        "projector_model_path",
    ):
        value = getattr(args, key)
        if value is not None and not isinstance(value, Path):
            setattr(args, key, Path(value))

    if args.num_train_epochs <= 0:
        parser.error("--num_train_epochs must be positive.")
    if args.per_device_train_batch_size <= 0:
        parser.error("--per_device_train_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive.")
    if args.lr_scheduler_type != "cosine":
        parser.error("--lr_scheduler_type must be cosine.")
    if args.warmup_ratio < 0 or args.warmup_ratio > 1:
        parser.error("--warmup_ratio must be in [0, 1].")
    if args.use_wandb and (not args.wandb_project or not args.wandb_run_name):
        parser.error("W&B requires explicit wandb_project and wandb_run_name.")
    if args.wandb_copy_page_size <= 0:
        parser.error("--wandb_copy_page_size must be positive.")
    if args.save_total_limit <= 0:
        parser.error("--save_total_limit must be positive.")
    if args.dataloader_prefetch_factor <= 0:
        parser.error("--dataloader_prefetch_factor must be positive.")
    return args


def load_frozen_point_projector(
    model_path: Path,
    dtype_name: str,
    device: torch.device,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(dtype_name),
    )
    point_projector = model.point_proj
    point_projector.requires_grad_(False)
    point_projector.eval()
    point_projector.to(device)
    return point_projector


@torch.no_grad()
def project_encoder_features(
    point_projector: nn.Module,
    encoder_features: torch.Tensor,
) -> torch.Tensor:
    projector_dtype = next(point_projector.parameters()).dtype
    point_tokens = point_projector(encoder_features.to(projector_dtype))
    return point_tokens.float()


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    mask = attention_mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def report_gpu_memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    print(
        f"{prefix}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
        f"peak_allocated={peak_allocated:.2f}GiB peak_reserved={peak_reserved:.2f}GiB"
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    point_projector: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    threshold: float,
    debug_memory: bool = False,
    disable_mha_fastpath: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    tp = fp = fn = tn = 0
    fastpath_enabled = None
    if disable_mha_fastpath and hasattr(torch.backends, "mha"):
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="eval", leave=False)):
            if debug_memory and batch_idx == 0:
                report_gpu_memory("eval before batch to device", device)
            batch = {key: value.to(device) for key, value in batch.items()}
            if debug_memory and batch_idx == 0:
                print(f"eval batch features shape={tuple(batch['features'].shape)}")
                report_gpu_memory("eval after batch to device", device)
            point_tokens = project_encoder_features(point_projector, batch["features"])
            if debug_memory and batch_idx == 0:
                print(f"eval point_tokens shape={tuple(point_tokens.shape)}")
                report_gpu_memory("eval after projector", device)
            logits = model(
                point_tokens,
                batch["grid_coord"],
                batch["region_center_grid_coord"],
                batch["attention_mask"],
            )
            if debug_memory and batch_idx == 0:
                report_gpu_memory("eval after scorer forward", device)
            loss = masked_bce_loss(
                logits,
                batch["labels"],
                batch["attention_mask"],
                pos_weight,
            )
            valid = batch["attention_mask"]
            pred = (torch.sigmoid(logits) >= threshold) & valid
            label = (batch["labels"] >= 0.5) & valid
            tp += int((pred & label).sum().item())
            fp += int((pred & ~label & valid).sum().item())
            fn += int((~pred & label & valid).sum().item())
            tn += int((~pred & ~label & valid).sum().item())
            tokens = int(valid.sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
    finally:
        if fastpath_enabled is not None:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    model.train()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: ScorerConfig,
    args: argparse.Namespace,
) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": asdict(config),
            "args": vars(args),
        },
        ckpt_dir / "scorer.pt",
    )
    checkpoints: list[tuple[int, Path]] = []
    for candidate in output_dir.glob("checkpoint-*"):
        suffix = candidate.name.removeprefix("checkpoint-")
        if candidate.is_dir() and suffix.isdigit():
            checkpoints.append((int(suffix), candidate))
    checkpoints.sort()
    for _, stale_checkpoint in checkpoints[: -args.save_total_limit]:
        shutil.rmtree(stale_checkpoint)


def latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for child in output_dir.glob("checkpoint-*"):
        scorer_path = child / "scorer.pt"
        suffix = child.name.removeprefix("checkpoint-")
        if child.is_dir() and suffix.isdigit() and scorer_path.is_file():
            checkpoints.append((int(suffix), child))
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None


def load_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_dir / "scorer.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    step = int(checkpoint["step"])
    print(
        f"Resumed scorer from {checkpoint_dir} at step {step}; "
        f"optimizer_state={'present' if optimizer_state is not None else 'missing'}."
    )
    return checkpoint


def copy_wandb_history_until_step(
    wandb_module,
    args: argparse.Namespace,
    max_step: int,
) -> None:
    if not args.wandb_copy_from_run_id:
        return
    entity = args.wandb_copy_from_entity or args.wandb_entity
    project = args.wandb_copy_from_project or args.wandb_project
    if not entity:
        raise ValueError("W&B clean-copy requires wandb_entity or source entity.")
    source = wandb_module.Api().run(
        f"{entity}/{project}/{args.wandb_copy_from_run_id}"
    )
    copied = 0
    for row in source.scan_history(page_size=args.wandb_copy_page_size):
        step = row.get("_step")
        if step is None or int(step) > max_step:
            continue
        payload = {
            key: value
            for key, value in row.items()
            if not key.startswith("_") and value is not None
        }
        if payload:
            wandb_module.log(payload, step=int(step))
            copied += 1
    print(f"Clean-copied W&B history through step {max_step}: rows={copied}")


def maybe_init_wandb(args: argparse.Namespace, resume_step: int | None = None):
    if not args.use_wandb:
        return None
    import wandb

    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": config,
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    wandb.init(**init_kwargs)
    if args.wandb_copy_from_run_id:
        if resume_step is None:
            raise RuntimeError(
                "wandb_copy_from_run_id requires a resumed local checkpoint."
            )
        copy_wandb_history_until_step(wandb, args, resume_step)
    return wandb


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    resume_checkpoint: Path | None = None
    if args.output_dir.exists() and args.overwrite_output_dir:
        shutil.rmtree(args.output_dir)
    elif args.output_dir.exists() and args.auto_resume_from_latest_checkpoint:
        resume_checkpoint = latest_checkpoint(args.output_dir)
        if resume_checkpoint is not None:
            print(f"Auto-resuming scorer from {resume_checkpoint}")
        elif any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is non-empty but has no scorer checkpoint: "
                f"{args.output_dir}"
            )
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. Enable auto-resume "
            "or choose a new output directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_dataset = PointTokenCacheDataset(
        args.train_cache_dir,
        max_samples=args.max_train_samples,
        shard_cache_size=args.shard_cache_size,
    )
    eval_dataset = (
        PointTokenCacheDataset(
            args.eval_cache_dir,
            max_samples=args.max_eval_samples,
            shard_cache_size=args.shard_cache_size,
        )
        if args.eval_cache_dir is not None
        else None
    )

    point_projector = load_frozen_point_projector(
        args.projector_model_path,
        args.projector_torch_dtype,
        device,
    )
    with torch.no_grad():
        dummy_context = torch.zeros(
            1,
            1,
            train_dataset.feature_dim,
            dtype=next(point_projector.parameters()).dtype,
            device=device,
        )
        dummy_point_tokens = point_projector(dummy_context)
        point_token_dim = int(dummy_point_tokens.shape[-1])
    print(
        "Loaded frozen point projector: "
        f"context_dim={train_dataset.feature_dim}, "
        f"point_token_dim={point_token_dim}, "
        f"dtype={next(point_projector.parameters()).dtype}"
    )

    config = ScorerConfig(
        encoder_feature_dim=train_dataset.feature_dim,
        point_token_dim=point_token_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        coord_scale=train_dataset.coord_scale,
    )
    model = PointTokenScorer(config).to(device)

    pos_weight_value = str(args.pos_weight).lower()
    if pos_weight_value == "auto":
        positive = train_dataset.positive_count
        negative = train_dataset.token_count - positive
        value = negative / max(positive, 1)
        pos_weight = torch.tensor(value, dtype=torch.float32, device=device)
        print(f"Using auto pos_weight={value:.4f}")
    elif pos_weight_value in {"none", "0"}:
        pos_weight = None
    else:
        pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32, device=device)

    dataloader_runtime_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": args.dataloader_pin_memory,
    }
    if args.num_workers > 0:
        dataloader_runtime_kwargs.update(
            {
                "persistent_workers": args.dataloader_persistent_workers,
                "prefetch_factor": args.dataloader_prefetch_factor,
            }
        )

    if args.shard_local_shuffle:
        train_batch_sampler = ShardLocalBatchSampler(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle_shards=True,
            shuffle_within_shard=True,
            drop_last=args.drop_last,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_batch,
            **dataloader_runtime_kwargs,
        )
        print(
            "Using shard-local train batching: "
            f"num_shards={len(train_batch_sampler.shards)}, "
            f"batch_size={args.per_device_train_batch_size}"
        )
    else:
        train_batch_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            drop_last=args.drop_last,
            **dataloader_runtime_kwargs,
        )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            **dataloader_runtime_kwargs,
        )
        if eval_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    global_step = 0
    checkpoint_state: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        checkpoint_state = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            device,
        )
        global_step = int(checkpoint_state["step"])
        if (
            args.reset_scheduler_on_resume
            and checkpoint_state.get("optimizer") is not None
        ):
            raise ValueError(
                "reset_scheduler_on_resume=true is forbidden because this "
                "checkpoint contains optimizer state. Resume the original "
                "scheduler trajectory instead."
            )

    micro_batches_per_epoch = len(train_loader)
    if micro_batches_per_epoch <= 0:
        raise RuntimeError("Train dataloader is empty.")
    steps_per_epoch = math.ceil(
        micro_batches_per_epoch / args.gradient_accumulation_steps
    )
    total_training_steps = steps_per_epoch * args.num_train_epochs
    remaining_steps = max(total_training_steps - global_step, 0)
    scheduler_uses_remaining_run = bool(
        checkpoint_state is not None and args.reset_scheduler_on_resume
    )
    scheduler_training_steps = (
        max(remaining_steps, 1)
        if scheduler_uses_remaining_run
        else max(total_training_steps, 1)
    )
    warmup_steps = int(scheduler_training_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=scheduler_training_steps,
    )
    if checkpoint_state is not None and not args.reset_scheduler_on_resume:
        saved_scheduler = checkpoint_state.get("scheduler")
        if saved_scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)
            for group, lr in zip(
                optimizer.param_groups,
                scheduler.get_last_lr(),
            ):
                group["lr"] = lr
        else:
            for _ in range(global_step):
                scheduler.step()
    print(
        "Scorer training schedule: "
        f"micro_batches_per_epoch={micro_batches_per_epoch}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_training_steps}, "
        f"start_step={global_step}, remaining_steps={remaining_steps}, "
        f"scheduler_training_steps={scheduler_training_steps}, "
        f"warmup_steps={warmup_steps}, "
        f"reset_on_resume={scheduler_uses_remaining_run}"
    )
    wandb = maybe_init_wandb(
        args, global_step if resume_checkpoint is not None else None
    )
    if global_step >= total_training_steps:
        print("No remaining scorer training steps.")
        if wandb is not None:
            wandb.finish()
        return

    start_epoch = min(global_step // steps_per_epoch, args.num_train_epochs)
    completed_steps_in_epoch = global_step % steps_per_epoch
    skip_micro_batches = min(
        completed_steps_in_epoch * args.gradient_accumulation_steps,
        micro_batches_per_epoch,
    )
    running_loss = 0.0
    running_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.num_train_epochs):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        progress = tqdm(
            desc=f"train epoch {epoch}",
            total=steps_per_epoch,
            initial=(
                completed_steps_in_epoch
                if epoch == start_epoch
                else 0
            ),
            unit="optimizer_step",
        )
        trained_micro_steps = 0
        for raw_micro_step, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and raw_micro_step <= skip_micro_batches:
                continue
            if global_step >= total_training_steps:
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            point_tokens = project_encoder_features(point_projector, batch["features"])
            logits = model(
                point_tokens,
                batch["grid_coord"],
                batch["region_center_grid_coord"],
                batch["attention_mask"],
            )
            loss = masked_bce_loss(
                logits,
                batch["labels"],
                batch["attention_mask"],
                pos_weight,
            )
            (loss / args.gradient_accumulation_steps).backward()
            tokens = int(batch["attention_mask"].sum().item())
            running_loss += float(loss.item()) * tokens
            running_tokens += tokens
            trained_micro_steps += 1

            if trained_micro_steps % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)
            progress.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step % args.logging_steps == 0:
                train_loss = running_loss / max(running_tokens, 1)
                log = {
                    "train/loss": train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch,
                }
                print(f"step={global_step} epoch={epoch} train_loss={train_loss:.6f}")
                if wandb is not None:
                    wandb.log(log, step=global_step)
                running_loss = 0.0
                running_tokens = 0

            if eval_loader is not None and global_step % args.eval_steps == 0:
                # The current train batch is no longer needed after backward,
                # optimizer.step(), and logging. Releasing these references before
                # eval avoids overlapping the largest train and eval batches.
                del batch, point_tokens, logits, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                metrics = evaluate(
                    model,
                    point_projector,
                    eval_loader,
                    device,
                    pos_weight,
                    args.threshold,
                    args.debug_memory,
                    args.disable_eval_mha_fastpath,
                )
                print(
                    "eval "
                    + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                )
                if wandb is not None:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)

            if global_step % args.save_steps == 0:
                save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    config,
                    args,
                )

        if (
            trained_micro_steps > 0
            and trained_micro_steps % args.gradient_accumulation_steps != 0
            and global_step < total_training_steps
        ):
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)
        progress.close()
        skip_micro_batches = 0
        completed_steps_in_epoch = 0

    save_checkpoint(
        args.output_dir,
        model,
        optimizer,
        scheduler,
        global_step,
        config,
        args,
    )
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
