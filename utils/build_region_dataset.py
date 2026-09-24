#!/usr/bin/env python3
"""Build two-stage SpatialLM training data with region supervision."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

from utils.generate_region_bboxes import (
    ObjectBbox,
    Region,
    expand_region,
    load_points_and_colors,
    make_regions,
    parse_bboxes,
    write_regions,
)
from utils.render_topdown import save_points_and_colors


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "spatiallm"

STAGE1_PROMPT = (
    "<point_cloud>Detect walls, doors, windows, regions. The reference code is as followed: "
    "@dataclass\n"
    "class Wall:\n"
    "    ax: int\n"
    "    ay: int\n"
    "    az: int\n"
    "    bx: int\n"
    "    by: int\n"
    "    bz: int\n"
    "    height: int\n"
    "    thickness: int\n\n"
    "@dataclass\n"
    "class Door:\n"
    "    wall_id: str\n"
    "    position_x: int\n"
    "    position_y: int\n"
    "    position_z: int\n"
    "    width: int\n"
    "    height: int\n\n"
    "@dataclass\n"
    "class Window:\n"
    "    wall_id: str\n"
    "    position_x: int\n"
    "    position_y: int\n"
    "    position_z: int\n"
    "    width: int\n"
    "    height: int\n\n"
    "@dataclass\n"
    "class Region:\n"
    "    position_x: int\n"
    "    position_y: int\n"
    "    position_z: int\n"
    "    scale_x: int\n"
    "    scale_y: int\n"
    "    scale_z: int"
)

STAGE2_PROMPT = (
    "<point_cloud>Detect bboxes. The reference code is as followed: "
    "@dataclass\n"
    "class Bbox:\n"
    "    class: str\n"
    "    position_x: int\n"
    "    position_y: int\n"
    "    position_z: int\n"
    "    angle_z: int\n"
    "    scale_x: int\n"
    "    scale_y: int\n"
    "    scale_z: int"
)


def clean_layout_text(layout_text: str) -> str:
    return layout_text.replace("<|layout_s|>", "").replace("<|layout_e|>", "").strip()


def scene_id_from_item(item: dict) -> str:
    point_clouds = item.get("point_clouds") or []
    if not point_clouds:
        raise ValueError("Training item has no point_clouds entry")
    return Path(point_clouds[0]).stem


def base_scene_id_from_item(item: dict) -> str:
    scene_id = scene_id_from_item(item)
    if "_region_" in scene_id:
        return scene_id.rsplit("_region_", 1)[0]
    return scene_id


def layout_text_from_item(item: dict) -> str:
    for conversation in item.get("conversations", []):
        if conversation.get("from") == "gpt":
            return conversation.get("value", "")
    return ""


def load_split_ids(split_csv: Path, split_name: str) -> set[str]:
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["id"] for row in reader if row.get("split") == split_name}


def load_scene_id_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_existing_training_items(json_path: Path, scene_ids: set[str]) -> list[dict]:
    if not scene_ids:
        return []
    with json_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return [
        item for item in items
        if base_scene_id_from_item(item) in scene_ids
    ]


def sample_items(
    source_json: Path,
    split_csv: Path,
    pcd_dir: Path,
    split_name: str,
    sample_size: int,
    seed: int,
    exclude_scene_ids: set[str] | None = None,
) -> list[dict]:
    with source_json.open("r", encoding="utf-8") as f:
        items = json.load(f)

    split_ids = load_split_ids(split_csv, split_name)
    exclude_scene_ids = exclude_scene_ids or set()
    eligible = []
    seen = set()
    for item in items:
        scene_id = scene_id_from_item(item)
        if scene_id in seen:
            continue
        if scene_id not in split_ids:
            continue
        if scene_id in exclude_scene_ids:
            continue
        if not (pcd_dir / f"{scene_id}.ply").is_file():
            continue
        eligible.append(item)
        seen.add(scene_id)

    if sample_size <= 0:
        return eligible

    if len(eligible) < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples, but only {len(eligible)} are eligible "
            f"after excluding {len(exclude_scene_ids)} scene ids."
        )

    rng = random.Random(seed)
    return rng.sample(eligible, sample_size)


def materialize_pcd(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def architecture_lines(layout_text: str) -> list[str]:
    lines = []
    for line in clean_layout_text(layout_text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("wall_", "door_", "window_")):
            lines.append(line)
    return lines


def format_region(index: int, region: Region) -> str:
    values = [
        region.position_x,
        region.position_y,
        region.position_z,
        region.scale_x,
        region.scale_y,
        region.scale_z,
    ]
    value_text = ",".join(repr(float(value)) for value in values)
    return f"region_{index}=Region({value_text})"


def format_bbox(index: int, bbox: ObjectBbox) -> str:
    values = [
        bbox.position_x,
        bbox.position_y,
        bbox.position_z,
        bbox.angle_z,
        bbox.scale_x,
        bbox.scale_y,
        bbox.scale_z,
    ]
    value_text = ",".join(repr(float(value)) for value in values)
    return f"bbox_{index}=Bbox({bbox.class_name},{value_text})"


def wrap_layout(lines: list[str]) -> str:
    return "<|layout_s|>" + "\n".join(lines) + "<|layout_e|>"


def make_sharegpt_item(point_cloud: str, prompt: str, label_lines: list[str]) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": wrap_layout(label_lines)},
        ],
        "point_clouds": [point_cloud],
    }


def relative_dataset_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return os.path.relpath(path, dataset_root)


def default_split_suffix(split_name: str, sample_size: int) -> str:
    if split_name == "train" and sample_size == 20000:
        return "train_20000"
    return split_name if sample_size <= 0 else f"{split_name}_{sample_size}"


def default_source_json(dataset_root: Path, split_name: str) -> Path:
    split_json = dataset_root / f"spatiallm_{split_name}.json"
    if split_json.exists():
        return split_json
    return dataset_root / "spatiallm_train.json"


def default_pcd_output_dir(dataset_root: Path, split_suffix: str) -> Path:
    if split_suffix == "train_20000":
        return dataset_root / "pcd_20000"
    return dataset_root / f"pcd_{split_suffix}"


def default_region_output_dir(dataset_root: Path, split_suffix: str) -> Path:
    if split_suffix == "train_20000":
        return dataset_root / "region_20000"
    return dataset_root / f"region_{split_suffix}"


def points_in_region(points: np.ndarray, region: Region) -> np.ndarray:
    center = np.array([region.position_x, region.position_y, region.position_z])
    half_size = np.array([region.scale_x, region.scale_y, region.scale_z]) * 0.5
    return np.all((points >= center - half_size) & (points <= center + half_size), axis=1)


def bbox_center_in_region(bbox: ObjectBbox, region: Region) -> bool:
    center = np.array([bbox.position_x, bbox.position_y, bbox.position_z])
    return bool(points_in_region(center[None, :], region)[0])


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_dataset_info(
    dataset_root: Path,
    stage1_json_name: str,
    stage2_json_name: str,
    stage1_dataset_name: str,
    stage2_dataset_name: str,
) -> None:
    info_path = dataset_root / "dataset_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    dataset_entry = lambda file_name: {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "point_clouds": "point_clouds",
        },
    }
    info[stage1_dataset_name] = dataset_entry(stage1_json_name)
    info[stage2_dataset_name] = dataset_entry(stage2_json_name)

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build two-stage hierarchical region training data."
    )
    parser.add_argument("--dataset_root", "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--train_json",
        "--train-json",
        "--source_json",
        "--source-json",
        type=Path,
        default=None,
        help="Source ShareGPT JSON. Defaults to spatiallm_{split}.json when present.",
    )
    parser.add_argument("--split_csv", "--split-csv", type=Path, default=None)
    parser.add_argument("--pcd_dir", "--pcd-dir", type=Path, default=None)
    parser.add_argument("--layout_dir", "--layout-dir", type=Path, default=None)
    parser.add_argument("--sample_size", "--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--exclude_scene_ids",
        "--exclude-scene-ids",
        type=Path,
        default=None,
        help=(
            "Optional text file with one scene id per line. These scenes are "
            "excluded from random sampling."
        ),
    )
    parser.add_argument(
        "--existing_stage1_json",
        "--existing-stage1-json",
        type=Path,
        default=None,
        help=(
            "Optional existing stage-1 JSON to prepend to the final output. "
            "Only scenes listed in --exclude_scene_ids are kept."
        ),
    )
    parser.add_argument(
        "--existing_stage2_json",
        "--existing-stage2-json",
        type=Path,
        default=None,
        help=(
            "Optional existing stage-2 JSON to prepend to the final output. "
            "Only scenes listed in --exclude_scene_ids are kept."
        ),
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--expand_fraction", "--expand-fraction", type=float, default=0.25)
    parser.add_argument(
        "--copy_mode",
        "--copy-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to materialize sampled scene PCDs into the split PCD directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min_region_points", "--min-region-points", type=int, default=1)
    parser.add_argument("--pcd_20000_dir", "--pcd-20000-dir", type=Path, default=None)
    parser.add_argument("--region_dir", "--region-dir", type=Path, default=None)
    parser.add_argument(
        "--stage1_json_name",
        "--stage1-json-name",
        default=None,
    )
    parser.add_argument(
        "--stage2_json_name",
        "--stage2-json-name",
        default=None,
    )
    parser.add_argument(
        "--stage1_dataset_name",
        "--stage1-dataset-name",
        default=None,
    )
    parser.add_argument(
        "--stage2_dataset_name",
        "--stage2-dataset-name",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    split_suffix = default_split_suffix(args.split, args.sample_size)
    source_json = args.train_json or default_source_json(dataset_root, args.split)
    split_csv = args.split_csv or dataset_root / "split.csv"
    pcd_dir = args.pcd_dir or dataset_root / "pcd"
    pcd_20000_dir = args.pcd_20000_dir or default_pcd_output_dir(
        dataset_root, split_suffix
    )
    region_dir = args.region_dir or default_region_output_dir(
        dataset_root, split_suffix
    )
    stage1_json_name = (
        args.stage1_json_name
        or f"spatiallm_stage1_region_{split_suffix}.json"
    )
    stage2_json_name = (
        args.stage2_json_name
        or f"spatiallm_stage2_bbox_{split_suffix}.json"
    )
    stage1_dataset_name = (
        args.stage1_dataset_name
        or f"spatiallm_stage1_region_{split_suffix}"
    )
    stage2_dataset_name = (
        args.stage2_dataset_name
        or f"spatiallm_stage2_bbox_{split_suffix}"
    )
    raw_region_dir = region_dir / "raw"
    expanded_region_dir = region_dir / "expanded"
    region_pcd_dir = region_dir / "pcd"

    raw_region_dir.mkdir(parents=True, exist_ok=True)
    expanded_region_dir.mkdir(parents=True, exist_ok=True)
    region_pcd_dir.mkdir(parents=True, exist_ok=True)

    excluded_scene_ids = load_scene_id_file(args.exclude_scene_ids)
    selected_items = sample_items(
        source_json=source_json,
        split_csv=split_csv,
        pcd_dir=pcd_dir,
        split_name=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        exclude_scene_ids=excluded_scene_ids,
    )

    if excluded_scene_ids and bool(args.existing_stage1_json) != bool(args.existing_stage2_json):
        raise ValueError(
            "--existing_stage1_json and --existing_stage2_json must be provided together."
        )

    stage1_items = (
        load_existing_training_items(args.existing_stage1_json, excluded_scene_ids)
        if args.existing_stage1_json is not None
        else []
    )
    stage2_items = (
        load_existing_training_items(args.existing_stage2_json, excluded_scene_ids)
        if args.existing_stage2_json is not None
        else []
    )
    existing_stage1_count = len(stage1_items)
    existing_stage2_count = len(stage2_items)
    metadata = []

    scene_ids_path = region_dir / "selected_scene_ids.txt"
    with scene_ids_path.open("w", encoding="utf-8") as f:
        for item in selected_items:
            f.write(scene_id_from_item(item) + "\n")

    combined_scene_ids_path = None
    if excluded_scene_ids:
        combined_scene_ids_path = region_dir / "combined_scene_ids.txt"
        combined_scene_ids = sorted(
            excluded_scene_ids | {scene_id_from_item(item) for item in selected_items}
        )
        with combined_scene_ids_path.open("w", encoding="utf-8") as f:
            for scene_id in combined_scene_ids:
                f.write(scene_id + "\n")

    for item_index, item in enumerate(selected_items, start=1):
        scene_id = scene_id_from_item(item)
        layout_text = layout_text_from_item(item)
        object_bboxes = parse_bboxes(layout_text)
        raw_regions = make_regions(object_bboxes, args.k)
        expanded_regions = [
            expand_region(region, args.expand_fraction) for region in raw_regions
        ]

        src_pcd_path = pcd_dir / f"{scene_id}.ply"
        sampled_pcd_path = pcd_20000_dir / f"{scene_id}.ply"
        materialize_pcd(src_pcd_path, sampled_pcd_path, args.copy_mode, args.overwrite)

        write_regions(raw_region_dir / f"{scene_id}.txt", raw_regions)
        write_regions(expanded_region_dir / f"{scene_id}.txt", expanded_regions)

        region_lines = [
            format_region(region_index, region)
            for region_index, region in enumerate(expanded_regions)
        ]
        stage1_label_lines = architecture_lines(layout_text) + region_lines
        stage1_items.append(
            make_sharegpt_item(
                point_cloud=relative_dataset_path(sampled_pcd_path, dataset_root),
                prompt=STAGE1_PROMPT,
                label_lines=stage1_label_lines,
            )
        )

        points, colors = load_points_and_colors(src_pcd_path)
        scene_metadata = {
            "scene_id": scene_id,
            "object_bbox_count": len(object_bboxes),
            "region_count": len(expanded_regions),
            "regions": [],
        }

        for region_index, region in enumerate(expanded_regions):
            mask = points_in_region(points, region)
            region_points = points[mask]
            region_colors = colors[mask]
            region_object_bboxes = [
                bbox for bbox in object_bboxes if bbox_center_in_region(bbox, region)
            ]

            region_entry = {
                "region_index": region_index,
                "point_count": int(region_points.shape[0]),
                "bbox_count": len(region_object_bboxes),
            }

            if (
                region_points.shape[0] >= args.min_region_points
                and len(region_object_bboxes) > 0
            ):
                region_pcd_name = f"{scene_id}_region_{region_index}.ply"
                save_points_and_colors(region_pcd_dir / region_pcd_name, region_points, region_colors)
                bbox_lines = [
                    format_bbox(bbox_index, bbox)
                    for bbox_index, bbox in enumerate(region_object_bboxes)
                ]
                stage2_items.append(
                    make_sharegpt_item(
                        point_cloud=relative_dataset_path(
                            region_pcd_dir / region_pcd_name,
                            dataset_root,
                        ),
                        prompt=STAGE2_PROMPT,
                        label_lines=bbox_lines,
                    )
                )
                region_entry["point_cloud"] = relative_dataset_path(
                    region_pcd_dir / region_pcd_name,
                    dataset_root,
                )
                region_entry["included"] = True
            else:
                region_entry["included"] = False

            scene_metadata["regions"].append(region_entry)

        metadata.append(scene_metadata)

        print(
            f"[{item_index}/{len(selected_items)}] {scene_id}: "
            f"{len(object_bboxes)} bboxes, {len(expanded_regions)} regions, "
            f"{len(stage2_items)} stage2 samples total"
        )

    stage1_json_path = dataset_root / stage1_json_name
    stage2_json_path = dataset_root / stage2_json_name
    write_json(stage1_json_path, stage1_items)
    write_json(stage2_json_path, stage2_items)

    metadata_path = region_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_size": args.sample_size,
                "split": args.split,
                "seed": args.seed,
                "k": args.k,
                "expand_fraction": args.expand_fraction,
                "source_json": str(source_json),
                "exclude_scene_ids": str(args.exclude_scene_ids)
                if args.exclude_scene_ids is not None
                else None,
                "excluded_scene_count": len(excluded_scene_ids),
                "new_scene_count": len(selected_items),
                "selected_scene_ids": str(scene_ids_path),
                "combined_scene_ids": str(combined_scene_ids_path)
                if combined_scene_ids_path is not None
                else None,
                "existing_stage1_json": str(args.existing_stage1_json)
                if args.existing_stage1_json is not None
                else None,
                "existing_stage2_json": str(args.existing_stage2_json)
                if args.existing_stage2_json is not None
                else None,
                "existing_stage1_count": existing_stage1_count,
                "existing_stage2_count": existing_stage2_count,
                "stage1_json": str(stage1_json_path),
                "stage2_json": str(stage2_json_path),
                "stage1_count": len(stage1_items),
                "stage2_count": len(stage2_items),
                "scenes": metadata,
            },
            f,
            indent=2,
        )
        f.write("\n")

    update_dataset_info(
        dataset_root=dataset_root,
        stage1_json_name=stage1_json_name,
        stage2_json_name=stage2_json_name,
        stage1_dataset_name=stage1_dataset_name,
        stage2_dataset_name=stage2_dataset_name,
    )

    print(f"Wrote sampled scene point clouds to {pcd_20000_dir}")
    print(f"Wrote newly selected scene ids to {scene_ids_path}")
    if combined_scene_ids_path is not None:
        print(f"Wrote combined scene ids to {combined_scene_ids_path}")
    print(f"Wrote raw regions to {raw_region_dir}")
    print(f"Wrote expanded regions to {expanded_region_dir}")
    print(f"Wrote region point clouds to {region_pcd_dir}")
    if existing_stage1_count or existing_stage2_count:
        print(
            f"Prepended existing training items: "
            f"{existing_stage1_count} stage1, {existing_stage2_count} stage2"
        )
    print(f"Wrote first-stage data to {stage1_json_path}")
    print(f"Wrote second-stage data to {stage2_json_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Updated {dataset_root / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
