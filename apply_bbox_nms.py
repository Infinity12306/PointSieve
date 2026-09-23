#!/usr/bin/env python3
"""Apply class-wise 3D bbox NMS to existing prediction txt files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from bbox import BBox3D
from bbox.metrics import iou_3d

from spatiallm import Layout
from spatiallm.layout.entity import Bbox


def bbox3d_from_bbox(bbox: Bbox, minimum_scale: float) -> BBox3D:
    return BBox3D(
        bbox.position_x,
        bbox.position_y,
        bbox.position_z,
        max(abs(bbox.scale_x), minimum_scale),
        max(abs(bbox.scale_y), minimum_scale),
        max(abs(bbox.scale_z), minimum_scale),
        euler_angles=[0, 0, bbox.angle_z],
        is_center=True,
    )


def classwise_nms(
    bboxes: list[Bbox],
    iou_threshold: float,
    minimum_scale: float,
) -> list[Bbox]:
    if iou_threshold <= 0 or len(bboxes) <= 1:
        return bboxes

    kept: list[Bbox] = []
    kept_boxes: list[BBox3D] = []
    for bbox in bboxes:
        current_box = bbox3d_from_bbox(bbox, minimum_scale)
        should_keep = True
        for kept_bbox, kept_box in zip(kept, kept_boxes):
            if bbox.class_name != kept_bbox.class_name:
                continue
            if iou_3d(current_box, kept_box) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(bbox)
            kept_boxes.append(current_box)

    for bbox_id, bbox in enumerate(kept):
        bbox.id = bbox_id
    return kept


def parse_class_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        source = source.strip()
        target = target.strip()
        if not separator or not source or not target:
            raise ValueError(
                "--class_alias must use a non-empty SOURCE=TARGET pair, "
                f"got {value!r}."
            )
        previous = aliases.get(source)
        if previous is not None and previous != target:
            raise ValueError(
                f"Conflicting aliases for {source!r}: {previous!r} and "
                f"{target!r}."
            )
        aliases[source] = target
    return aliases


def apply_class_aliases(
    bboxes: list[Bbox], class_aliases: dict[str, str]
) -> None:
    for bbox in bboxes:
        bbox.class_name = class_aliases.get(bbox.class_name, bbox.class_name)


def process_prediction_file(
    input_path: Path,
    output_path: Path,
    iou_threshold: float,
    minimum_scale: float,
    class_aliases: dict[str, str],
    skip_existing: bool,
) -> tuple[int, int]:
    if skip_existing and output_path.exists():
        layout = Layout(output_path.read_text(encoding="utf-8"))
        return len(layout.bboxes), len(layout.bboxes)

    layout = Layout(input_path.read_text(encoding="utf-8"))
    before_count = len(layout.bboxes)
    apply_class_aliases(layout.bboxes, class_aliases)
    layout.bboxes = classwise_nms(layout.bboxes, iou_threshold, minimum_scale)
    after_count = len(layout.bboxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(layout.to_language_string(), encoding="utf-8")
    return before_count, after_count


def copy_auxiliary_files(input_dir: Path, output_dir: Path, skip_existing: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(input_dir.iterdir()):
        if not input_path.is_file():
            continue
        output_path = output_dir / input_path.name
        if skip_existing and output_path.exists():
            continue
        shutil.copy2(input_path, output_path)


def resolve_input_dirs(input_dir: Path) -> tuple[Path, Path | None]:
    final_dir = input_dir / "final"
    stage1_dir = input_dir / "stage1"
    if final_dir.is_dir():
        return final_dir, stage1_dir if stage1_dir.is_dir() else None
    return input_dir, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Apply order-based class-wise 3D bbox NMS to prediction txt files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help=(
            "Input prediction dir. Can be a flat txt dir or a hierarchical root "
            "containing final/ and optionally stage1/."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help=(
            "Output prediction dir. For hierarchical input, writes output_dir/final "
            "and copies stage1 to output_dir/stage1."
        ),
    )
    parser.add_argument("--iou_threshold", type=float, required=True)
    parser.add_argument("--minimum_scale", type=float, default=1e-6)
    parser.add_argument(
        "--class_alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help=(
            "Canonicalize a predicted class before class-wise NMS. May be "
            "specified multiple times."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Do not rewrite output files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_aliases = parse_class_aliases(args.class_alias)
    source_pred_dir, source_stage1_dir = resolve_input_dirs(args.input_dir)
    if not source_pred_dir.is_dir():
        raise FileNotFoundError(source_pred_dir)

    if source_pred_dir == args.input_dir:
        target_pred_dir = args.output_dir
    else:
        target_pred_dir = args.output_dir / "final"

    pred_paths = sorted(source_pred_dir.glob("*.txt"))
    if not pred_paths:
        raise FileNotFoundError(f"No prediction txt files found in {source_pred_dir}")

    total_before = 0
    total_after = 0
    for pred_path in pred_paths:
        before_count, after_count = process_prediction_file(
            pred_path,
            target_pred_dir / pred_path.name,
            args.iou_threshold,
            args.minimum_scale,
            class_aliases,
            args.skip_existing,
        )
        total_before += before_count
        total_after += after_count

    if source_stage1_dir is not None:
        copy_auxiliary_files(
            source_stage1_dir,
            args.output_dir / "stage1",
            args.skip_existing,
        )

    print(f"input_prediction_dir={source_pred_dir}")
    print(f"output_prediction_dir={target_pred_dir}")
    if source_stage1_dir is not None:
        print(f"copied_stage1_dir={args.output_dir / 'stage1'}")
    print(f"iou_threshold={args.iou_threshold}")
    print(f"class_aliases={class_aliases}")
    print(f"files={len(pred_paths)}")
    print(f"bboxes_before={total_before}")
    print(f"bboxes_after={total_after}")
    print(f"bboxes_removed={total_before - total_after}")


if __name__ == "__main__":
    main()
