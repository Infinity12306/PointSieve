#!/usr/bin/env python3
"""Merge sharded filtered point-token cache indices into index.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge index_shard_*.json files produced by filter_point_tokens_with_scorer.py."
    )
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--pattern", default="index_shard_*.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "metadata" not in data or "samples" not in data:
        raise ValueError(f"Invalid filtered cache index: {path}")
    return data


def sample_sort_key(sample: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(sample.get("epoch", 0)),
        int(sample.get("sample_index", 0)),
        int(sample.get("point_cloud_index", 0)),
    )


def main() -> None:
    args = parse_args()
    paths = sorted(args.cache_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"No partial indices found under {args.cache_dir} matching {args.pattern!r}"
        )

    first = load_index(paths[0])
    metadata = dict(first["metadata"])
    merged_samples: list[dict[str, Any]] = []
    merged_from: list[str] = []
    total_fields = [
        "num_items",
        "total_raw_tokens",
        "total_threshold_tokens",
        "total_kept_tokens",
        "total_positive_tokens",
    ]
    totals = {key: 0 for key in total_fields}
    shard_indices: set[int] = set()
    num_shards: int | None = None

    for path in paths:
        data = load_index(path)
        part_metadata = data["metadata"]
        samples = data["samples"]
        merged_samples.extend(samples)
        merged_from.append(path.name)
        for key in total_fields:
            totals[key] += int(part_metadata.get(key, 0))
        if "shard_index" in part_metadata:
            shard_indices.add(int(part_metadata["shard_index"]))
        if "num_shards" in part_metadata:
            cur_num_shards = int(part_metadata["num_shards"])
            if num_shards is None:
                num_shards = cur_num_shards
            elif num_shards != cur_num_shards:
                raise ValueError(f"Mismatched num_shards in {path}: {cur_num_shards} != {num_shards}")

    if num_shards is not None and len(shard_indices) != num_shards:
        raise ValueError(
            f"Expected {num_shards} shard indices, found {len(shard_indices)}: {sorted(shard_indices)}"
        )

    merged_samples.sort(key=sample_sort_key)
    metadata.update(totals)
    metadata["partial_run"] = False
    metadata["merged_index"] = True
    metadata["merged_from"] = merged_from
    metadata["merged_shard_indices"] = sorted(shard_indices)
    metadata["kept_ratio"] = (
        float(totals["total_kept_tokens"] / totals["total_raw_tokens"])
        if totals["total_raw_tokens"] > 0
        else 0.0
    )
    metadata["positive_ratio_after_filter"] = (
        float(totals["total_positive_tokens"] / totals["total_kept_tokens"])
        if totals["total_kept_tokens"] > 0
        else 0.0
    )

    output_path = args.cache_dir / "index.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Index exists: {output_path}. Use --overwrite to replace it.")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "samples": merged_samples}, handle, indent=2)
    print(f"Wrote merged filtered cache index: {output_path}")


if __name__ == "__main__":
    main()
