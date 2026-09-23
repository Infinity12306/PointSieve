#!/usr/bin/env python3
"""Create scorer-filtered point-token caches for stage-2 LLM training."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from precompute_point_token_scorer_data import (
    load_samples,
    messages_from_sharegpt,
    resolve_path,
    sample_seed,
)
from spatiallm.tuner.data.mm_plugin import SpatialLMPlugin
from spatiallm.model.point_token_scorer import (
    PointTokenScorer,
    ScorerConfig,
    load_frozen_point_projector,
    project_encoder_features,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CACHE_ROOT = REPO_ROOT / "artifacts" / "scorer_cache"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "filtered_cache"
DEFAULT_SCORER_PATH = REPO_ROOT / "artifacts" / "scorer" / "checkpoint-29488"
DEFAULT_PROJECTOR_MODEL_PATH = REPO_ROOT / "artifacts" / "stage2_bboxmask" / "checkpoint-14392"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "spatiallm"


@dataclass
class FilteredItemMeta:
    epoch: int
    sample_index: int
    point_cloud_index: int
    scene_id: str
    point_cloud: str
    raw_token_count: int
    token_count: int
    threshold_token_count: int
    positive_count: int
    shard: str
    item_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter precomputed point-token scorer cache with a trained scorer."
    )
    parser.add_argument("--input_cache_root", type=Path, default=DEFAULT_INPUT_CACHE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--message_cache_root",
        type=Path,
        default=None,
        help=(
            "Optional precomputed message cache root. When provided, the script "
            "loads processed messages from matching split/shard files instead of "
            "re-reading PCDs and reconstructing labels online."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--scorer_path", type=Path, default=DEFAULT_SCORER_PATH)
    parser.add_argument("--projector_model_path", type=Path, default=DEFAULT_PROJECTOR_MODEL_PATH)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset_json",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help=(
            "Override source JSON for a split, e.g. "
            "train=/path/spatiallm_stage2_bbox_train_20000.json. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_keep", type=int, default=4096)
    parser.add_argument("--min_keep", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=1250)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split each requested split into this many metadata shards for parallel processing.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard worker index. Must satisfy 0 <= shard_index < num_shards.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of cache items to score together within each loaded shard.",
    )
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base seed used to reconstruct processed messages. If omitted, use "
            "source cache metadata['seed'] when present, otherwise fall back to 0."
        ),
    )
    parser.add_argument("--num_bins", type=int, default=1280)
    parser.add_argument("--world_size", type=float, default=16.0)
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.1)
    parser.add_argument(
        "--bbox_regression_aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When reconstructing messages online, retain numeric-token spans "
            "and continuous bbox-regression targets."
        ),
    )
    parser.add_argument(
        "--bbox_point_alignment_aux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When reconstructing messages online, retain numeric-token spans "
            "and continuous bbox geometry for point-center alignment."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--projector_torch_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--storage_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--disable_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable PyTorch Transformer eval fastpath to avoid large attention buffers.",
    )
    parser.add_argument("--save_scores", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_keep <= 0:
        parser.error("--max_keep must be positive.")
    if args.min_keep < 0:
        parser.error("--min_keep must be non-negative.")
    if args.shard_size <= 0:
        parser.error("--shard_size must be positive.")
    if args.num_shards <= 0:
        parser.error("--num_shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard_index must satisfy 0 <= shard_index < num_shards.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max_items must be positive when set.")
    return args


def parse_dataset_json_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected SPLIT=PATH for --dataset_json, got: {value}")
        split, path = value.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"Empty split name in --dataset_json {value!r}")
        overrides[split] = Path(path).expanduser()
    return overrides


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def latest_scorer_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / "scorer.pt").is_file():
        return path / "scorer.pt"

    checkpoints: list[tuple[int, Path]] = []
    for candidate in path.glob("checkpoint-*"):
        if not candidate.is_dir() or not (candidate / "scorer.pt").is_file():
            continue
        suffix = candidate.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), candidate / "scorer.pt"))
    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]
    raise FileNotFoundError(f"No scorer.pt found under {path}")


def load_scorer(path: Path, device: torch.device) -> PointTokenScorer:
    scorer_path = latest_scorer_checkpoint(path)
    checkpoint = torch.load(scorer_path, map_location="cpu")
    config = ScorerConfig(**checkpoint["config"])
    scorer = PointTokenScorer(config)
    scorer.load_state_dict(checkpoint["model"])
    scorer.eval()
    scorer.requires_grad_(False)
    scorer.to(device=device, dtype=torch.float32)
    print(f"Loaded scorer: {scorer_path}")
    return scorer


def scorer_keep_indices(
    scores: torch.Tensor,
    threshold: float,
    min_keep: int,
    max_keep: int,
) -> torch.Tensor:
    num_tokens = int(scores.numel())
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    min_keep = min(max(min_keep, 0), num_tokens)
    max_keep = min(max_keep, num_tokens)
    if min_keep > max_keep:
        min_keep = max_keep

    keep_indices = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
    if keep_indices.numel() < min_keep:
        keep_indices = torch.topk(scores, k=min_keep).indices
    elif keep_indices.numel() > max_keep:
        selected_scores = scores[keep_indices]
        top_local = torch.topk(selected_scores, k=max_keep).indices
        keep_indices = keep_indices[top_local]
    return keep_indices.sort().values


def load_index(cache_dir: Path) -> dict[str, Any]:
    index_path = cache_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset_json(
    split: str,
    metadata: dict[str, Any],
    overrides: dict[str, Path],
    dataset_root: Path,
) -> Path:
    if split in overrides:
        return overrides[split]

    metadata_path = Path(str(metadata.get("dataset_json", "")))
    if metadata_path.is_file():
        return metadata_path

    if metadata_path.name:
        fallback = dataset_root / metadata_path.name
        if fallback.is_file():
            return fallback

    raise FileNotFoundError(
        f"Cannot resolve dataset JSON for split {split}. "
        "Pass --dataset_json split=/path/to/file.json."
    )


def processed_messages_for_item(
    sample: dict[str, Any],
    item_meta: dict[str, Any],
    plugin: SpatialLMPlugin,
    dataset_root: Path,
    seed: int,
) -> list[dict[str, str]]:
    messages = messages_from_sharegpt(sample)
    pcd_path = resolve_path(str(item_meta["point_cloud"]), dataset_root)
    np.random.seed(
        sample_seed(
            seed,
            int(item_meta["epoch"]),
            int(item_meta["sample_index"]),
            int(item_meta["point_cloud_index"]),
        )
    )
    mm_inputs = plugin._get_mm_inputs([messages], [str(pcd_path)])
    return mm_inputs["messages"][0]


def flush_shard(
    output_dir: Path,
    shard_index: int,
    items: list[dict[str, Any]],
    metadata: dict[str, Any],
    worker_shard_index: int | None = None,
    num_worker_shards: int | None = None,
) -> str:
    if worker_shard_index is None or num_worker_shards is None or num_worker_shards <= 1:
        shard_rel = f"shard_{shard_index:05d}.pt"
    else:
        shard_rel = (
            f"worker_{worker_shard_index:05d}_of_{num_worker_shards:05d}/"
            f"shard_{shard_index:05d}.pt"
        )
    (output_dir / shard_rel).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "items": items}, output_dir / shard_rel)
    return shard_rel


@torch.no_grad()
def filter_item(
    item: dict[str, Any],
    scorer: PointTokenScorer,
    point_projector: torch.nn.Module,
    device: torch.device,
    out_dtype: torch.dtype,
    threshold: float,
    min_keep: int,
    max_keep: int,
    save_scores: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    features = item["features"].unsqueeze(0).to(device, non_blocking=True)
    grid_coord = item["grid_coord"].float().unsqueeze(0).to(device, non_blocking=True)
    center = item["region_center_grid_coord"].float().unsqueeze(0).to(device, non_blocking=True)
    attention_mask = torch.ones(
        (1, features.shape[1]),
        dtype=torch.bool,
        device=device,
    )

    point_tokens = project_encoder_features(point_projector, features)
    logits = scorer(point_tokens, grid_coord, center, attention_mask)
    scores = torch.sigmoid(logits[0])
    threshold_count = int((scores >= threshold).sum().item())
    keep = scorer_keep_indices(scores, threshold, min_keep, max_keep)

    keep_cpu = keep.cpu()
    filtered = {
        "point_tokens": point_tokens[0, keep].to(out_dtype).cpu(),
        "grid_coord": item["grid_coord"][keep_cpu].to(torch.int32),
        "region_center_grid_coord": item["region_center_grid_coord"].float().cpu(),
        "sample_index": int(item["sample_index"]),
        "point_cloud_index": int(item["point_cloud_index"]),
        "scene_id": str(item["scene_id"]),
        "point_cloud": str(item["point_cloud"]),
        "raw_token_count": int(scores.numel()),
        "threshold_token_count": threshold_count,
    }
    if "labels" in item:
        labels = item["labels"][keep_cpu].to(torch.uint8)
        filtered["labels"] = labels
        positive_count = int(labels.sum().item())
    else:
        positive_count = 0
    if save_scores:
        filtered["scores"] = scores[keep].float().cpu()

    stats = {
        "raw_token_count": int(scores.numel()),
        "token_count": int(keep.numel()),
        "threshold_token_count": threshold_count,
        "positive_count": positive_count,
    }
    return filtered, stats


@torch.no_grad()
def filter_batch(
    items: list[dict[str, Any]],
    scorer: PointTokenScorer,
    point_projector: torch.nn.Module,
    device: torch.device,
    out_dtype: torch.dtype,
    threshold: float,
    min_keep: int,
    max_keep: int,
    save_scores: bool,
) -> list[tuple[dict[str, Any], dict[str, int]]]:
    if not items:
        return []

    batch_size = len(items)
    max_len = max(int(item["features"].shape[0]) for item in items)
    feature_dim = int(items[0]["features"].shape[-1])
    features = torch.zeros(
        (batch_size, max_len, feature_dim),
        dtype=items[0]["features"].dtype,
    )
    grid_coord = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    centers = torch.stack(
        [item["region_center_grid_coord"].float() for item in items],
        dim=0,
    )

    lengths: list[int] = []
    for row, item in enumerate(items):
        length = int(item["features"].shape[0])
        lengths.append(length)
        features[row, :length] = item["features"]
        grid_coord[row, :length] = item["grid_coord"].float()
        attention_mask[row, :length] = True

    features = features.to(device, non_blocking=True)
    grid_coord = grid_coord.to(device, non_blocking=True)
    centers = centers.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    point_tokens = project_encoder_features(point_projector, features)
    logits = scorer(point_tokens, grid_coord, centers, attention_mask)

    results: list[tuple[dict[str, Any], dict[str, int]]] = []
    for row, item in enumerate(items):
        length = lengths[row]
        cur_point_tokens = point_tokens[row, :length]
        scores = torch.sigmoid(logits[row, :length])
        threshold_count = int((scores >= threshold).sum().item())
        keep = scorer_keep_indices(scores, threshold, min_keep, max_keep)
        keep_cpu = keep.cpu()

        filtered = {
            "point_tokens": cur_point_tokens[keep].to(out_dtype).cpu(),
            "grid_coord": item["grid_coord"][keep_cpu].to(torch.int32),
            "region_center_grid_coord": item["region_center_grid_coord"].float().cpu(),
            "sample_index": int(item["sample_index"]),
            "point_cloud_index": int(item["point_cloud_index"]),
            "scene_id": str(item["scene_id"]),
            "point_cloud": str(item["point_cloud"]),
            "raw_token_count": int(scores.numel()),
            "threshold_token_count": threshold_count,
        }
        if "labels" in item:
            labels = item["labels"][keep_cpu].to(torch.uint8)
            filtered["labels"] = labels
            positive_count = int(labels.sum().item())
        else:
            positive_count = 0
        if save_scores:
            filtered["scores"] = scores[keep].float().cpu()

        stats = {
            "raw_token_count": int(scores.numel()),
            "token_count": int(keep.numel()),
            "threshold_token_count": threshold_count,
            "positive_count": positive_count,
        }
        results.append((filtered, stats))

    return results


def process_split(
    split: str,
    args: argparse.Namespace,
    dataset_overrides: dict[str, Path],
    scorer: PointTokenScorer,
    point_projector: torch.nn.Module,
    device: torch.device,
) -> None:
    input_dir = args.input_cache_root / split
    output_dir = args.output_root / split
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    partial_run = args.num_shards > 1
    if output_dir.exists() and not partial_run:
        if not args.overwrite:
            raise FileExistsError(
                f"Output split exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(input_dir)
    source_metadata = index["metadata"]
    samples_meta = index["samples"]
    if args.max_items is not None:
        samples_meta = samples_meta[: args.max_items]
    if partial_run:
        samples_meta = [
            meta
            for ordinal, meta in enumerate(samples_meta)
            if ordinal % args.num_shards == args.shard_index
        ]
    base_seed = int(source_metadata.get("seed", 0) if args.seed is None else args.seed)

    dataset_json = resolve_dataset_json(
        split,
        source_metadata,
        dataset_overrides,
        args.dataset_root,
    )
    message_cache_dir = args.message_cache_root / split if args.message_cache_root else None
    if message_cache_dir is not None and not message_cache_dir.is_dir():
        raise FileNotFoundError(message_cache_dir)
    message_metadata: dict[str, Any] = {}
    if message_cache_dir is not None:
        message_index = load_index(message_cache_dir)
        message_metadata = message_index.get("metadata") or {}
        message_source = message_metadata.get("source_cache_dir")
        if message_source is None:
            raise ValueError(
                f"Message cache has no source_cache_dir: {message_cache_dir}"
            )
        if Path(message_source).expanduser().resolve() != input_dir.resolve():
            raise ValueError(
                "Message/source cache lineage mismatch: "
                f"message_source={Path(message_source).expanduser().resolve()}, "
                f"feature_source={input_dir.resolve()}."
            )

    bbox_regression_aux = bool(
        message_metadata.get("bbox_regression_aux", args.bbox_regression_aux)
    )
    bbox_point_alignment_aux = bool(
        message_metadata.get(
            "bbox_point_alignment_aux",
            args.bbox_point_alignment_aux,
        )
    )

    raw_samples = None
    plugin = None
    if message_cache_dir is None:
        raw_samples = load_samples(dataset_json, max_samples=None)
        plugin = SpatialLMPlugin(
            num_bins=args.num_bins,
            world_size=args.world_size,
            do_augmentation=bool(source_metadata.get("do_augmentation", False)),
            random_rotation=bool(source_metadata.get("random_rotation", True)),
            point_token_bbox_mask=True,
            point_token_bbox_expand_ratio=args.bbox_expand_ratio,
            bbox_regression_aux=bbox_regression_aux,
            bbox_point_alignment_aux=bbox_point_alignment_aux,
        )

    by_shard: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for meta in samples_meta:
        by_shard.setdefault(str(meta["shard"]), []).append(meta)

    metadata = {
        "format": "filtered_point_token_cache_v1",
        "feature_type": "projected_point_tokens",
        "source_cache_dir": str(input_dir.resolve()),
        "source_cache_format": source_metadata.get("format"),
        "dataset_json": str(dataset_json.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "message_cache_dir": str(message_cache_dir.resolve()) if message_cache_dir else None,
        "projector_model_path": str(args.projector_model_path),
        "scorer_path": str(latest_scorer_checkpoint(args.scorer_path)),
        "threshold": args.threshold,
        "min_keep": args.min_keep,
        "max_keep": args.max_keep,
        "world_size": args.world_size,
        "num_bins": args.num_bins,
        "bbox_expand_ratio": args.bbox_expand_ratio,
        "seed": base_seed,
        "source_seed_present": "seed" in source_metadata,
        "random_rotation": bool(source_metadata.get("random_rotation", True)),
        "do_augmentation": bool(source_metadata.get("do_augmentation", False)),
        "bbox_regression_aux": bbox_regression_aux,
        "bbox_point_alignment_aux": bbox_point_alignment_aux,
        "storage_dtype": args.storage_dtype,
        "partial_run": partial_run,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "point_token_dim": None,
        "num_items": 0,
        "total_raw_tokens": 0,
        "total_threshold_tokens": 0,
        "total_kept_tokens": 0,
        "total_positive_tokens": 0,
    }

    output_items: list[dict[str, Any]] = []
    output_index: list[FilteredItemMeta] = []
    output_shard_index = 0
    out_dtype = dtype_from_name(args.storage_dtype)
    fastpath_enabled = None
    if args.disable_mha_fastpath and hasattr(torch.backends, "mha"):
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

    try:
        progress = tqdm(total=len(samples_meta), desc=f"filter {split}")
        for shard_rel, shard_metas in by_shard.items():
            shard = torch.load(input_dir / shard_rel, map_location="cpu")
            message_shard = (
                torch.load(message_cache_dir / shard_rel, map_location="cpu")
                if message_cache_dir is not None
                else None
            )
            for batch_start in range(0, len(shard_metas), args.batch_size):
                batch_metas = shard_metas[batch_start : batch_start + args.batch_size]
                batch_items = [
                    shard["items"][int(meta["item_index"])]
                    for meta in batch_metas
                ]
                batch_results = filter_batch(
                    batch_items,
                    scorer,
                    point_projector,
                    device,
                    out_dtype,
                    args.threshold,
                    args.min_keep,
                    args.max_keep,
                    args.save_scores,
                )

                for meta, (filtered, stats) in zip(batch_metas, batch_results):
                    filtered["epoch"] = int(meta["epoch"])
                    if message_shard is not None:
                        message_item = message_shard["items"][int(meta["item_index"])]
                        processed_messages = message_item["messages"]
                    else:
                        assert raw_samples is not None and plugin is not None
                        processed_messages = processed_messages_for_item(
                            raw_samples[int(meta["sample_index"])],
                            meta,
                            plugin,
                            args.dataset_root,
                            base_seed,
                        )
                    if bbox_regression_aux and not any(
                        "_bbox_regression" in message
                        for message in processed_messages
                    ):
                        raise ValueError(
                            "Filtered cache requested bbox regression metadata, "
                            f"but none was found for {meta}."
                        )
                    if bbox_point_alignment_aux and not any(
                        "_bbox_point_alignment" in message
                        for message in processed_messages
                    ):
                        raise ValueError(
                            "Filtered cache requested bbox point-alignment "
                            f"metadata, but none was found for {meta}."
                        )
                    filtered["messages"] = processed_messages

                    if metadata["point_token_dim"] is None:
                        metadata["point_token_dim"] = int(filtered["point_tokens"].shape[-1])

                    output_items.append(filtered)
                    metadata["num_items"] += 1
                    metadata["total_raw_tokens"] += stats["raw_token_count"]
                    metadata["total_threshold_tokens"] += stats["threshold_token_count"]
                    metadata["total_kept_tokens"] += stats["token_count"]
                    metadata["total_positive_tokens"] += stats["positive_count"]

                    if len(output_items) >= args.shard_size:
                        shard_out_rel = flush_shard(
                            output_dir,
                            output_shard_index,
                            output_items,
                            metadata,
                            args.shard_index if partial_run else None,
                            args.num_shards if partial_run else None,
                        )
                        for local_index, out_item in enumerate(output_items):
                            output_index.append(
                                FilteredItemMeta(
                                    epoch=int(out_item["epoch"]),
                                    sample_index=int(out_item["sample_index"]),
                                    point_cloud_index=int(out_item["point_cloud_index"]),
                                    scene_id=str(out_item["scene_id"]),
                                    point_cloud=str(out_item["point_cloud"]),
                                    raw_token_count=int(out_item["raw_token_count"]),
                                    token_count=int(out_item["point_tokens"].shape[0]),
                                    threshold_token_count=int(out_item["threshold_token_count"]),
                                    positive_count=int(
                                        out_item.get("labels", torch.empty(0)).sum().item()
                                        if "labels" in out_item
                                        else 0
                                    ),
                                    shard=shard_out_rel,
                                    item_index=local_index,
                                )
                            )
                        output_items = []
                        output_shard_index += 1
                    progress.update(1)

                del batch_items
                del batch_results
            del shard
            del message_shard
        progress.close()

        if output_items:
            shard_out_rel = flush_shard(
                output_dir,
                output_shard_index,
                output_items,
                metadata,
                args.shard_index if partial_run else None,
                args.num_shards if partial_run else None,
            )
            for local_index, out_item in enumerate(output_items):
                output_index.append(
                    FilteredItemMeta(
                        epoch=int(out_item["epoch"]),
                        sample_index=int(out_item["sample_index"]),
                        point_cloud_index=int(out_item["point_cloud_index"]),
                        scene_id=str(out_item["scene_id"]),
                        point_cloud=str(out_item["point_cloud"]),
                        raw_token_count=int(out_item["raw_token_count"]),
                        token_count=int(out_item["point_tokens"].shape[0]),
                        threshold_token_count=int(out_item["threshold_token_count"]),
                        positive_count=int(
                            out_item.get("labels", torch.empty(0)).sum().item()
                            if "labels" in out_item
                            else 0
                        ),
                        shard=shard_out_rel,
                        item_index=local_index,
                    )
                )

        index_out = {
            "metadata": metadata,
            "samples": [asdict(item) for item in output_index],
        }
        index_name = (
            f"index_shard_{args.shard_index:05d}_of_{args.num_shards:05d}.json"
            if partial_run
            else "index.json"
        )
        with (output_dir / index_name).open("w", encoding="utf-8") as handle:
            json.dump(index_out, handle, indent=2, ensure_ascii=False)

        kept_ratio = metadata["total_kept_tokens"] / max(metadata["total_raw_tokens"], 1)
        print(
            f"Wrote {split}: {output_dir / index_name} "
            f"items={metadata['num_items']} kept_ratio={kept_ratio:.4f}"
        )
    finally:
        if fastpath_enabled is not None:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)


def main() -> None:
    args = parse_args()
    dataset_overrides = parse_dataset_json_overrides(args.dataset_json)
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    scorer = load_scorer(args.scorer_path, device)
    point_projector = load_frozen_point_projector(
        args.projector_model_path,
        args.projector_torch_dtype,
        device,
    )

    for split in args.splits:
        process_split(split, args, dataset_overrides, scorer, point_projector, device)


if __name__ == "__main__":
    main()
