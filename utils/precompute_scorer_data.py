#!/usr/bin/env python3
"""Precompute point-encoder context features and GT mask labels for scorer training."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from spatiallm.tuner.data.mm_plugin import SpatialLMPlugin


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "spatiallm"
DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "stage2_bboxmask"


@dataclass
class CacheItemMeta:
    epoch: int
    sample_index: int
    point_cloud_index: int
    scene_id: str
    point_cloud: str
    token_count: int
    positive_count: int
    shard: str
    item_index: int


@dataclass
class MessageItemMeta:
    epoch: int
    sample_index: int
    point_cloud_index: int
    scene_id: str
    point_cloud: str
    shard: str
    item_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a trained SpatialLM point encoder to precompute 512D context "
            "features and token-level GT BBoxMask labels."
        )
    )
    parser.add_argument("--dataset_json", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--message_output_dir",
        type=Path,
        default=None,
        help=(
            "Optional output dir for processed message cache. When set, messages "
            "are written with the same shard/item_index layout as the point-token "
            "cache, avoiding a second pass over the PCD files."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--epoch_indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit global epoch indices to process. Seeds still use "
            "these global epoch indices, so parallel workers do not duplicate "
            "augmentation when they process different epochs."
        ),
    )
    parser.add_argument(
        "--num_epoch_shards",
        type=int,
        default=1,
        help="Split the selected global epochs across this many workers.",
    )
    parser.add_argument(
        "--epoch_shard_index",
        type=int,
        default=0,
        help="Worker index for --num_epoch_shards.",
    )
    parser.add_argument("--shard_size", type=int, default=1250)
    parser.add_argument("--world_size", type=float, default=16.0)
    parser.add_argument("--num_bins", type=int, default=1280)
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float32",
        help=(
            "Model loading dtype. float32 preserves the original checkpoint "
            "weights before writing bf16 context features."
        ),
    )
    parser.add_argument(
        "--storage_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Point-encoder context feature dtype written to cache.",
    )
    parser.add_argument("--do_augmentation", action="store_true")
    parser.add_argument(
        "--random_rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bbox_regression_aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store numeric-token spans and continuous bbox-regression targets "
            "in the aligned processed-message cache."
        ),
    )
    parser.add_argument(
        "--bbox_point_alignment_aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store numeric-token spans and continuous bbox geometry for "
            "point-center alignment in the processed-message cache."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_epochs <= 0:
        parser.error("--num_epochs must be positive.")
    if args.num_epoch_shards <= 0:
        parser.error("--num_epoch_shards must be positive.")
    if args.epoch_shard_index < 0 or args.epoch_shard_index >= args.num_epoch_shards:
        parser.error("--epoch_shard_index must satisfy 0 <= index < num_epoch_shards.")
    if args.epoch_indices is not None:
        invalid_epochs = [epoch for epoch in args.epoch_indices if epoch < 0 or epoch >= args.num_epochs]
        if invalid_epochs:
            parser.error(
                "--epoch_indices must be within [0, num_epochs); "
                f"got invalid values {invalid_epochs}."
            )
    if args.shard_size <= 0:
        parser.error("--shard_size must be positive.")
    if args.world_size <= 0:
        parser.error("--world_size must be positive.")
    if args.num_bins <= 0:
        parser.error("--num_bins must be positive.")
    if args.bbox_expand_ratio < 0:
        parser.error("--bbox_expand_ratio must be non-negative.")
    return args


def selected_epochs(args: argparse.Namespace) -> list[int]:
    epochs = (
        list(dict.fromkeys(args.epoch_indices))
        if args.epoch_indices is not None
        else list(range(args.num_epochs))
    )
    return [
        epoch
        for ordinal, epoch in enumerate(epochs)
        if ordinal % args.num_epoch_shards == args.epoch_shard_index
    ]


def is_partial_epoch_run(args: argparse.Namespace) -> bool:
    return args.num_epoch_shards > 1 or args.epoch_indices is not None


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def storage_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def messages_from_sharegpt(sample: dict[str, Any]) -> list[dict[str, str]]:
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }
    messages = []
    for message in sample.get("conversations", []):
        role = role_map.get(str(message.get("from", "")))
        if role is None:
            raise ValueError(f"Unsupported conversation role: {message!r}")
        messages.append({"role": role, "content": str(message.get("value", ""))})
    return messages


def resolve_path(path_text: str, dataset_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else dataset_root / path


def scene_id_from_path(path_text: str) -> str:
    return Path(path_text).stem


def sample_seed(base_seed: int, epoch: int, sample_index: int, point_cloud_index: int) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, epoch, sample_index, point_cloud_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samples(dataset_json: Path, max_samples: int | None) -> list[dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"Expected JSON list: {dataset_json}")
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_spatiallm_model(args: argparse.Namespace):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.point_config["num_bins"] = args.num_bins
    config.point_config["world_size"] = args.world_size
    config.point_config["max_point_tokens"] = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.torch_dtype),
    )
    model.set_point_backbone_dtype(torch.float32)
    model.eval()
    model.requires_grad_(False)
    model.to(args.device)
    return model


def valid_bboxes(keep_bboxes: torch.Tensor) -> torch.Tensor:
    if keep_bboxes.numel() == 0:
        return keep_bboxes.reshape(0, 7)
    valid = torch.isfinite(keep_bboxes).all(dim=-1)
    return keep_bboxes[valid]


def point_token_bbox_overlap_labels(
    grid_coord: torch.Tensor,
    keep_bboxes: torch.Tensor,
    voxel_size: float,
) -> torch.Tensor:
    if grid_coord.numel() == 0:
        return torch.zeros(
            grid_coord.shape[0],
            dtype=torch.bool,
            device=grid_coord.device,
        )
    keep_bboxes = valid_bboxes(keep_bboxes)
    if keep_bboxes.shape[0] == 0:
        return torch.zeros(
            grid_coord.shape[0],
            dtype=torch.bool,
            device=grid_coord.device,
        )

    box_center = keep_bboxes[:, 0:3]
    box_size = keep_bboxes[:, 3:6].abs()
    box_angle_z = keep_bboxes[:, 6]

    half_voxel = float(voxel_size) * 0.5
    box_half = box_size * 0.5
    cos = torch.cos(box_angle_z)
    sin = torch.sin(box_angle_z)
    box_axis_x = torch.stack([cos, sin], dim=-1)
    box_axis_y = torch.stack([-sin, cos], dim=-1)

    hx = box_half[:, 0]
    hy = box_half[:, 1]
    hz = box_half[:, 2]
    voxel_center = (grid_coord.to(torch.float32) + 0.5) * float(voxel_size)
    delta = voxel_center[:, None, :] - box_center[None, :, :]
    delta_xy = delta[..., :2]

    z_overlap = delta[..., 2].abs() <= (half_voxel + hz)[None, :]
    box_radius_world_x = hx * box_axis_x[:, 0].abs() + hy * box_axis_y[:, 0].abs()
    box_radius_world_y = hx * box_axis_x[:, 1].abs() + hy * box_axis_y[:, 1].abs()
    overlap_world_x = delta[..., 0].abs() <= (half_voxel + box_radius_world_x)[None, :]
    overlap_world_y = delta[..., 1].abs() <= (half_voxel + box_radius_world_y)[None, :]

    delta_on_box_x = (delta_xy * box_axis_x[None, :, :]).sum(dim=-1)
    delta_on_box_y = (delta_xy * box_axis_y[None, :, :]).sum(dim=-1)
    voxel_radius_on_box_x = half_voxel * (
        box_axis_x[:, 0].abs() + box_axis_x[:, 1].abs()
    )
    voxel_radius_on_box_y = half_voxel * (
        box_axis_y[:, 0].abs() + box_axis_y[:, 1].abs()
    )
    overlap_box_x = delta_on_box_x.abs() <= (hx + voxel_radius_on_box_x)[None, :]
    overlap_box_y = delta_on_box_y.abs() <= (hy + voxel_radius_on_box_y)[None, :]

    return (
        z_overlap
        & overlap_world_x
        & overlap_world_y
        & overlap_box_x
        & overlap_box_y
    ).any(dim=1)


def compute_labels(
    grid_coord: torch.Tensor,
    keep_bboxes: torch.Tensor,
    final_voxel_size: float,
) -> torch.Tensor:
    keep_bboxes = valid_bboxes(keep_bboxes)
    if keep_bboxes.shape[0] == 0:
        return torch.zeros(grid_coord.shape[0], dtype=torch.float32, device=grid_coord.device)

    labels = point_token_bbox_overlap_labels(
        grid_coord,
        keep_bboxes,
        final_voxel_size,
    )
    return labels.to(torch.float32)


def encode_one_point_cloud(
    model,
    point_cloud: torch.Tensor,
    keep_bboxes: torch.Tensor,
    device: torch.device,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nan_mask = torch.isnan(point_cloud).any(dim=1)
    point_cloud = point_cloud[~nan_mask]
    if point_cloud.shape[0] == 0:
        raise ValueError("point cloud has no valid points after preprocessing")

    coords = point_cloud[:, :3].to(device=device, dtype=torch.int64)
    feats = point_cloud[:, 3:].to(device=device, dtype=torch.float32)
    input_dict = {
        "coord": feats[:, :3],
        "grid_coord": coords,
        "feat": feats,
        "batch": torch.zeros(coords.shape[0], dtype=torch.long, device=device),
        "return_grid_coord": True,
    }

    with torch.inference_mode():
        encoded = model.point_backbone(input_dict)
        context = encoded["context"].to(out_dtype)
        grid_coord = encoded["grid_coord"].to(torch.int32)

    keep_bboxes = keep_bboxes.to(device=device, dtype=torch.float32)
    labels = compute_labels(
        grid_coord,
        keep_bboxes,
        model.point_backbone.final_voxel_size,
    )
    if grid_coord.shape[0] == 0:
        center = torch.zeros(3, dtype=torch.float32, device=device)
    else:
        center = (grid_coord.float().amin(dim=0) + grid_coord.float().amax(dim=0)) * 0.5

    return (
        context.detach().cpu(),
        grid_coord.detach().cpu(),
        center.detach().cpu(),
        labels.detach().cpu(),
    )


def flush_shard(
    output_dir: Path,
    epoch: int,
    shard_index: int,
    items: list[dict[str, Any]],
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> str:
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    shard_rel = Path(f"epoch_{epoch:03d}") / f"shard_{shard_index:05d}.pt"
    shard_path = output_dir / shard_rel
    if shard_path.exists() and not overwrite:
        raise FileExistsError(f"Shard exists: {shard_path}. Use --overwrite to replace it.")
    torch.save({"metadata": metadata, "items": items}, shard_path)
    return str(shard_rel)


def flush_message_shard(
    output_dir: Path,
    epoch: int,
    shard_index: int,
    items: list[dict[str, Any]],
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> str:
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    shard_rel = Path(f"epoch_{epoch:03d}") / f"shard_{shard_index:05d}.pt"
    shard_path = output_dir / shard_rel
    if shard_path.exists() and not overwrite:
        raise FileExistsError(f"Message shard exists: {shard_path}. Use --overwrite to replace it.")
    torch.save({"metadata": metadata, "items": items}, shard_path)
    return str(shard_rel)


def write_index(path: Path, index: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Index exists: {path}. Use --overwrite to replace it.")
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def main() -> None:
    args = parse_args()
    epochs_to_process = selected_epochs(args)
    if not epochs_to_process:
        raise ValueError(
            "No epochs selected. Check --num_epochs, --epoch_indices, "
            "--num_epoch_shards, and --epoch_shard_index."
        )
    partial_epoch_run = is_partial_epoch_run(args)

    if args.output_dir.exists() and not partial_epoch_run:
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {args.output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.message_output_dir is not None:
        if args.message_output_dir.exists() and not partial_epoch_run:
            if not args.overwrite:
                raise FileExistsError(
                    f"Message output directory exists: {args.message_output_dir}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(args.message_output_dir)
        args.message_output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.dataset_json, args.max_samples)
    model = load_spatiallm_model(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dtype = storage_dtype(args.storage_dtype)
    plugin = SpatialLMPlugin(
        num_bins=args.num_bins,
        world_size=args.world_size,
        do_augmentation=args.do_augmentation,
        random_rotation=args.random_rotation,
        point_token_bbox_mask=True,
        point_token_bbox_expand_ratio=args.bbox_expand_ratio,
        bbox_regression_aux=args.bbox_regression_aux,
        bbox_point_alignment_aux=args.bbox_point_alignment_aux,
    )

    index_samples: list[CacheItemMeta] = []
    message_index_samples: list[MessageItemMeta] = []
    total_tokens = 0
    total_positive = 0
    total_items = 0
    shard_items: list[dict[str, Any]] = []
    message_shard_items: list[dict[str, Any]] = []
    shard_index = 0
    current_shard_rel = ""

    metadata = {
        "format": "point_token_scorer_cache_v2",
        "feature_type": "point_encoder_context",
        "projected": False,
        "dataset_json": str(args.dataset_json.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "model_path": str(args.model_path),
        "num_epochs": args.num_epochs,
        "selected_epochs": epochs_to_process,
        "num_epoch_shards": args.num_epoch_shards,
        "epoch_shard_index": args.epoch_shard_index,
        "partial_epoch_run": partial_epoch_run,
        "world_size": args.world_size,
        "num_bins": args.num_bins,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "seed": args.seed,
        "random_rotation": args.random_rotation,
        "do_augmentation": args.do_augmentation,
        "storage_dtype": args.storage_dtype,
        "feature_dim": None,
        "reduced_grid_size": int(model.point_backbone.reduced_grid_size),
        "final_voxel_size": float(model.point_backbone.final_voxel_size),
    }
    message_metadata = {
        "format": "point_token_cache_messages_v1",
        "source_cache_dir": str(args.output_dir.resolve()),
        "source_cache_format": metadata["format"],
        "dataset_json": str(args.dataset_json.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "world_size": args.world_size,
        "num_bins": args.num_bins,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "seed": args.seed,
        "random_rotation": args.random_rotation,
        "do_augmentation": args.do_augmentation,
        "bbox_regression_aux": args.bbox_regression_aux,
        "bbox_point_alignment_aux": args.bbox_point_alignment_aux,
        "created_with_point_token_cache": True,
    }

    for epoch in epochs_to_process:
        iterator = tqdm(
            enumerate(samples),
            total=len(samples),
            desc=f"precompute epoch {epoch}",
        )
        for sample_index, sample in iterator:
            point_clouds = sample.get("point_clouds", [])
            if isinstance(point_clouds, str):
                point_clouds = [point_clouds]
            messages = messages_from_sharegpt(sample)

            for point_cloud_index, point_cloud_text in enumerate(point_clouds):
                seed_all(sample_seed(args.seed, epoch, sample_index, point_cloud_index))
                pcd_path = resolve_path(str(point_cloud_text), args.dataset_root)
                mm_inputs = plugin._get_mm_inputs([messages], [str(pcd_path)])
                point_cloud = mm_inputs["point_clouds"][0]
                keep_bboxes = mm_inputs["point_token_keep_bboxes"][0]
                processed_messages = mm_inputs["messages"][0]

                features, grid_coord, center, labels = encode_one_point_cloud(
                    model,
                    point_cloud,
                    keep_bboxes,
                    device,
                    out_dtype,
                )
                if metadata["feature_dim"] is None:
                    metadata["feature_dim"] = int(features.shape[-1])

                item = {
                    "features": features,
                    "grid_coord": grid_coord,
                    "region_center_grid_coord": center,
                    "labels": labels.to(torch.uint8),
                    "sample_index": sample_index,
                    "point_cloud_index": point_cloud_index,
                    "scene_id": scene_id_from_path(str(point_cloud_text)),
                    "point_cloud": str(point_cloud_text),
                }
                shard_items.append(item)
                message_item = {
                    "messages": processed_messages,
                    "epoch": epoch,
                    "sample_index": sample_index,
                    "point_cloud_index": point_cloud_index,
                    "scene_id": scene_id_from_path(str(point_cloud_text)),
                    "point_cloud": str(point_cloud_text),
                }
                message_shard_items.append(message_item)

                positive_count = int(labels.sum().item())
                token_count = int(labels.numel())
                total_tokens += token_count
                total_positive += positive_count
                total_items += 1

                if len(shard_items) >= args.shard_size:
                    current_shard_rel = flush_shard(
                        args.output_dir,
                        epoch,
                        shard_index,
                        shard_items,
                        metadata,
                        args.overwrite,
                    )
                    for local_index, shard_item in enumerate(shard_items):
                        index_samples.append(
                            CacheItemMeta(
                                epoch=epoch,
                                sample_index=int(shard_item["sample_index"]),
                                point_cloud_index=int(shard_item["point_cloud_index"]),
                                scene_id=str(shard_item["scene_id"]),
                                point_cloud=str(shard_item["point_cloud"]),
                                token_count=int(shard_item["labels"].numel()),
                                positive_count=int(shard_item["labels"].sum().item()),
                                shard=current_shard_rel,
                                item_index=local_index,
                            )
                        )
                    if args.message_output_dir is not None:
                        flush_message_shard(
                            args.message_output_dir,
                            epoch,
                            shard_index,
                            message_shard_items,
                            message_metadata,
                            args.overwrite,
                        )
                        for local_index, message_item in enumerate(message_shard_items):
                            message_index_samples.append(
                                MessageItemMeta(
                                    epoch=epoch,
                                    sample_index=int(message_item["sample_index"]),
                                    point_cloud_index=int(message_item["point_cloud_index"]),
                                    scene_id=str(message_item["scene_id"]),
                                    point_cloud=str(message_item["point_cloud"]),
                                    shard=current_shard_rel,
                                    item_index=local_index,
                                )
                            )
                    shard_items = []
                    message_shard_items = []
                    shard_index += 1

        if shard_items:
            current_shard_rel = flush_shard(
                args.output_dir,
                epoch,
                shard_index,
                shard_items,
                metadata,
                args.overwrite,
            )
            for local_index, shard_item in enumerate(shard_items):
                index_samples.append(
                    CacheItemMeta(
                        epoch=epoch,
                        sample_index=int(shard_item["sample_index"]),
                        point_cloud_index=int(shard_item["point_cloud_index"]),
                        scene_id=str(shard_item["scene_id"]),
                        point_cloud=str(shard_item["point_cloud"]),
                        token_count=int(shard_item["labels"].numel()),
                        positive_count=int(shard_item["labels"].sum().item()),
                        shard=current_shard_rel,
                        item_index=local_index,
                    )
                )
            if args.message_output_dir is not None:
                flush_message_shard(
                    args.message_output_dir,
                    epoch,
                    shard_index,
                    message_shard_items,
                    message_metadata,
                    args.overwrite,
                )
                for local_index, message_item in enumerate(message_shard_items):
                    message_index_samples.append(
                        MessageItemMeta(
                            epoch=epoch,
                            sample_index=int(message_item["sample_index"]),
                            point_cloud_index=int(message_item["point_cloud_index"]),
                            scene_id=str(message_item["scene_id"]),
                            point_cloud=str(message_item["point_cloud"]),
                            shard=current_shard_rel,
                            item_index=local_index,
                        )
                    )
            shard_items = []
            message_shard_items = []
            shard_index += 1

    metadata["num_items"] = total_items
    metadata["total_tokens"] = total_tokens
    metadata["total_positive_tokens"] = total_positive
    metadata["positive_ratio"] = (
        float(total_positive / total_tokens) if total_tokens > 0 else 0.0
    )

    index = {
        "metadata": metadata,
        "samples": [asdict(item) for item in index_samples],
    }
    index_name = (
        f"index_epoch_shard_{args.epoch_shard_index:05d}_of_{args.num_epoch_shards:05d}.json"
        if partial_epoch_run
        else "index.json"
    )
    write_index(args.output_dir / index_name, index, args.overwrite)
    if args.message_output_dir is not None:
        message_metadata["num_items"] = total_items
        message_index = {
            "metadata": message_metadata,
            "samples": [asdict(item) for item in message_index_samples],
        }
        write_index(args.message_output_dir / index_name, message_index, args.overwrite)

    print(f"Wrote cache: {args.output_dir}")
    print(f"Wrote index: {args.output_dir / index_name}")
    if args.message_output_dir is not None:
        print(f"Wrote message cache: {args.message_output_dir}")
        print(f"Wrote message index: {args.message_output_dir / index_name}")
    print(f"items={total_items}, tokens={total_tokens}, positives={total_positive}")
    print(f"positive_ratio={metadata['positive_ratio']:.6f}")


if __name__ == "__main__":
    main()
