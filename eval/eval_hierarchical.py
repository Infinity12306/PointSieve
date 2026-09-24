#!/usr/bin/env python3
"""Evaluate hierarchical SpatialLM predictions.

This extends eval.py without changing it:
- object bbox precision/recall/F1 at IoU 0.25 and 0.50
- layout precision/recall/F1 at IoU 0.25 and 0.50
- SpatialLM-reported F1: scene mean per class, then macro mean across classes
- weighted summaries using GT label counts as weights
- first-stage region precision/recall/F1 at configurable IoU thresholds
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from bbox import BBox3D
from bbox.metrics import iou_3d
from scipy.optimize import linear_sum_assignment
from terminaltables import AsciiTable

from eval import eval_base as base_eval
from utils.generate_region_bboxes import (
    expand_region,
    make_regions,
    parse_bboxes,
)
from spatiallm import Layout
from spatiallm.layout.entity import Region

log = logging.getLogger(__name__)

OBJECT_THRESHOLDS = (0.25, 0.50)
LAYOUT_THRESHOLDS = (0.25, 0.50)
REGION_THRESHOLDS = (0.50, 0.75)
DEFAULT_LABEL_MAPPING = (
    Path(__file__).resolve().parents[1] / "benchmark_categories.tsv"
)


def format_score(value: float) -> str:
    return f"{value:.4f}"


def eval_tuple_to_dict(item: base_eval.EvalTuple) -> dict:
    return {
        "tp": int(item.tp),
        "num_pred": int(item.num_pred),
        "num_gt": int(item.num_gt),
        "precision": float(item.precision),
        "recall": float(item.recall),
        "f1": float(item.f1),
    }


def read_scene_ids(
    metadata: Path | None,
    gt_dir: Path,
    pred_dirs: Iterable[Path | None] = (),
) -> list[str]:
    if metadata is None:
        pred_scene_ids = set()
        for pred_dir in pred_dirs:
            if pred_dir is None:
                continue
            pred_scene_ids.update(path.stem for path in pred_dir.glob("*.txt"))
        if pred_scene_ids:
            return sorted(pred_scene_ids)
        return sorted(path.stem for path in gt_dir.glob("*.txt"))

    if metadata.suffix.lower() == ".txt":
        scene_ids = [
            line.strip()
            for line in metadata.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError(f"{metadata} contains duplicate scene ids.")
        return scene_ids

    with metadata.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"{metadata} must contain an 'id' column.")
        scene_ids = [row["id"] for row in reader]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"{metadata} contains duplicate scene ids.")
    return scene_ids


def read_text(path: Path, missing_pred: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    if missing_pred == "empty":
        log.warning("Missing prediction file, treating as empty: %s", path)
        return ""
    raise FileNotFoundError(path)


def load_layout(path: Path, missing_pred: str = "error") -> Layout:
    return Layout(read_text(path, missing_pred))


def aggregate(tuples: Iterable[base_eval.EvalTuple]) -> base_eval.EvalTuple:
    tuple_list = list(tuples)
    return base_eval.EvalTuple(
        tp=sum(item.tp for item in tuple_list),
        num_pred=sum(item.num_pred for item in tuple_list),
        num_gt=sum(item.num_gt for item in tuple_list),
    )


def spatiallm_reported_class_metric(
    tuples: Iterable[base_eval.EvalTuple],
) -> dict[str, float | int | None]:
    """Match SpatialLM eval.py: mean scene F1, excluding empty pred/GT pairs."""
    tuple_list = list(tuples)
    evaluated = [item for item in tuple_list if not item.masked]
    return {
        "f1": float(np.mean([item.f1 for item in evaluated])) if evaluated else None,
        "evaluated_scene_count": len(evaluated),
        "total_scene_count": len(tuple_list),
    }


def spatiallm_reported_metrics(
    classwise: dict[float, dict[str, list[base_eval.EvalTuple]]],
    class_names: list[str],
) -> dict[float, dict]:
    """Compute the class-equal macro F1 reported in the SpatialLM paper."""
    result = {}
    for threshold in sorted(classwise):
        classes = {
            class_name: spatiallm_reported_class_metric(
                classwise[threshold].get(class_name, [])
            )
            for class_name in class_names
        }
        class_f1_values = [
            item["f1"] for item in classes.values() if item["f1"] is not None
        ]
        result[threshold] = {
            "aggregation": "mean_scene_f1_per_class_then_mean_classes",
            "classes": classes,
            "macro_f1": (
                float(np.mean(class_f1_values)) if class_f1_values else None
            ),
            "evaluated_class_count": len(class_f1_values),
            "requested_class_count": len(class_names),
        }
    return result


def weighted_metric(
    class_metrics: dict[str, base_eval.EvalTuple], class_names: list[str]
) -> tuple[float, float, float, int, int]:
    support = sum(class_metrics[name].num_gt for name in class_names)
    predictions = sum(class_metrics[name].num_pred for name in class_names)
    if support == 0:
        return 0.0, 0.0, 0.0, support, predictions

    precision = (
        sum(class_metrics[name].precision * class_metrics[name].num_gt for name in class_names)
        / support
    )
    recall = (
        sum(class_metrics[name].recall * class_metrics[name].num_gt for name in class_names)
        / support
    )
    f1 = (
        sum(class_metrics[name].f1 * class_metrics[name].num_gt for name in class_names)
        / support
    )
    return precision, recall, f1, support, predictions


def summary_to_dict(
    class_metrics: dict[str, base_eval.EvalTuple],
    class_names: list[str],
) -> dict:
    weighted_p, weighted_r, weighted_f1, support, predictions = weighted_metric(
        class_metrics, class_names
    )
    micro = micro_metric(class_metrics, class_names)
    return {
        "weighted": {
            "num_gt": int(support),
            "num_pred": int(predictions),
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
        "micro": eval_tuple_to_dict(micro),
    }


def metrics_to_dict(
    metrics: dict[float, dict[str, base_eval.EvalTuple]],
    class_names: list[str],
    classwise: dict[float, dict[str, list[base_eval.EvalTuple]]] | None = None,
) -> dict:
    reported = (
        spatiallm_reported_metrics(classwise, class_names)
        if classwise is not None
        else None
    )
    result = {}
    for threshold in sorted(metrics):
        threshold_key = f"{threshold:.2f}"
        result[threshold_key] = {
            "classes": {
                class_name: eval_tuple_to_dict(metrics[threshold][class_name])
                for class_name in class_names
            },
            "summary": summary_to_dict(metrics[threshold], class_names),
        }
        if reported is not None:
            result[threshold_key]["spatiallm_reported"] = reported[threshold]
    return result


def region_metrics_to_dict(
    region_tuples: dict[float, list[base_eval.EvalTuple]],
) -> dict:
    return {
        f"{threshold:.2f}": eval_tuple_to_dict(aggregate(tuples))
        for threshold, tuples in sorted(region_tuples.items())
    }


def micro_metric(
    class_metrics: dict[str, base_eval.EvalTuple], class_names: list[str]
) -> base_eval.EvalTuple:
    return base_eval.EvalTuple(
        tp=sum(class_metrics[name].tp for name in class_names),
        num_pred=sum(class_metrics[name].num_pred for name in class_names),
        num_gt=sum(class_metrics[name].num_gt for name in class_names),
    )


def normalize_objects(
    layout: Layout,
    class_map: dict[str, str] | None,
    minimum_scale: float,
    object_classes: list[str],
) -> list:
    if class_map:
        layout.bboxes = base_eval.assign_class_map(layout.bboxes, class_map)
    else:
        for bbox in layout.bboxes:
            bbox.class_name = bbox.class_name.replace("_", " ")
    allowed_classes = set(object_classes)
    layout.bboxes = [
        bbox for bbox in layout.bboxes if bbox.class_name in allowed_classes
    ]
    base_eval.assign_minimum_scale(layout.bboxes, minimum_scale=minimum_scale)
    return layout.bboxes


def layout_instances(layout: Layout) -> list:
    wall_lookup = {wall.id: wall for wall in layout.walls}
    return list(filter(base_eval.is_valid_wall, layout.walls)) + list(
        filter(
            lambda entity: base_eval.is_valid_dw(entity, wall_lookup),
            layout.doors + layout.windows,
        )
    )


def evaluate_objects(
    scene_ids: list[str],
    gt_dir: Path,
    pred_dir: Path,
    class_map: dict[str, str] | None,
    object_classes: list[str],
    minimum_scale: float,
    missing_pred: str,
) -> dict[float, dict[str, list[base_eval.EvalTuple]]]:
    classwise = {
        threshold: defaultdict(list)
        for threshold in OBJECT_THRESHOLDS
    }

    for scene_id in scene_ids:
        pred_layout = load_layout(pred_dir / f"{scene_id}.txt", missing_pred)
        gt_layout = load_layout(gt_dir / f"{scene_id}.txt")
        pred_objects = normalize_objects(
            pred_layout,
            class_map,
            minimum_scale,
            object_classes,
        )
        gt_objects = normalize_objects(
            gt_layout,
            class_map,
            minimum_scale,
            object_classes,
        )

        for class_name in object_classes:
            pred_class = [
                entity for entity in pred_objects
                if base_eval.get_entity_class(entity) == class_name
            ]
            gt_class = [
                entity for entity in gt_objects
                if base_eval.get_entity_class(entity) == class_name
            ]
            for threshold in OBJECT_THRESHOLDS:
                classwise[threshold][class_name].append(
                    base_eval.calc_bbox_tp(pred_class, gt_class, threshold)
                )

    return classwise


def evaluate_layouts(
    scene_ids: list[str],
    gt_dir: Path,
    pred_dir: Path,
    missing_pred: str,
) -> dict[float, dict[str, list[base_eval.EvalTuple]]]:
    classwise = {
        threshold: defaultdict(list)
        for threshold in LAYOUT_THRESHOLDS
    }

    for scene_id in scene_ids:
        pred_layout = load_layout(pred_dir / f"{scene_id}.txt", missing_pred)
        gt_layout = load_layout(gt_dir / f"{scene_id}.txt")

        pred_wall_lookup = {wall.id: wall for wall in pred_layout.walls}
        gt_wall_lookup = {wall.id: wall for wall in gt_layout.walls}
        pred_layouts = layout_instances(pred_layout)
        gt_layouts = layout_instances(gt_layout)

        for class_name in base_eval.LAYOUTS:
            pred_class = [
                entity for entity in pred_layouts
                if base_eval.get_entity_class(entity) == class_name
            ]
            gt_class = [
                entity for entity in gt_layouts
                if base_eval.get_entity_class(entity) == class_name
            ]
            for threshold in LAYOUT_THRESHOLDS:
                classwise[threshold][class_name].append(
                    base_eval.calc_layout_tp(
                        pred_class,
                        gt_class,
                        pred_wall_lookup,
                        gt_wall_lookup,
                        threshold,
                    )
                )

    return classwise


def region_box(region: Region, minimum_scale: float = 1e-6) -> BBox3D:
    return BBox3D(
        region.position_x,
        region.position_y,
        region.position_z,
        max(region.scale_x, minimum_scale),
        max(region.scale_y, minimum_scale),
        max(region.scale_z, minimum_scale),
        euler_angles=[0, 0, 0],
        is_center=True,
    )


def calc_region_tp(
    pred_regions: list[Region],
    gt_regions: list[Region],
    iou_threshold: float,
) -> base_eval.EvalTuple:
    num_pred = len(pred_regions)
    num_gt = len(gt_regions)
    if num_pred == 0 or num_gt == 0:
        return base_eval.EvalTuple(0, num_pred, num_gt)

    pred_boxes = [region_box(region) for region in pred_regions]
    gt_boxes = [region_box(region) for region in gt_regions]
    iou_matrix = np.array(
        [
            iou_3d(pred_box, gt_box)
            for pred_box in pred_boxes
            for gt_box in gt_boxes
        ],
        dtype=np.float64,
    ).reshape(num_pred, num_gt)

    cost_matrix = np.full((num_pred, num_gt), base_eval.LARGE_COST_VALUE)
    valid_mask = iou_matrix >= iou_threshold
    cost_matrix[valid_mask] = -iou_matrix[valid_mask]
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    tp = int(np.sum(iou_matrix[row_indices, col_indices] >= iou_threshold))
    return base_eval.EvalTuple(tp, num_pred, num_gt)


def generated_region_to_layout_region(index: int, region) -> Region:
    return Region(
        id=index,
        position_x=region.position_x,
        position_y=region.position_y,
        position_z=region.position_z,
        scale_x=region.scale_x,
        scale_y=region.scale_y,
        scale_z=region.scale_z,
    )


def derive_gt_regions(gt_layout_text: str, k: int, expand_fraction: float) -> list[Region]:
    raw_regions = make_regions(parse_bboxes(gt_layout_text), k)
    expanded_regions = [
        expand_region(region, expand_fraction) for region in raw_regions
    ]
    return [
        generated_region_to_layout_region(index, region)
        for index, region in enumerate(expanded_regions)
    ]


def load_gt_regions(
    scene_id: str,
    gt_dir: Path,
    gt_region_dir: Path | None,
    k: int,
    expand_fraction: float,
) -> list[Region]:
    if gt_region_dir is not None:
        region_path = gt_region_dir / f"{scene_id}.txt"
        if region_path.exists():
            return load_layout(region_path).regions
        log.warning("GT region file missing, deriving from GT bboxes: %s", region_path)

    gt_layout_text = (gt_dir / f"{scene_id}.txt").read_text(encoding="utf-8")
    return derive_gt_regions(gt_layout_text, k, expand_fraction)


def evaluate_regions(
    scene_ids: list[str],
    gt_dir: Path,
    pred_dir: Path,
    gt_region_dir: Path | None,
    k: int,
    expand_fraction: float,
    missing_pred: str,
    thresholds: tuple[float, ...],
) -> dict[float, list[base_eval.EvalTuple]]:
    threshold_tuples = {threshold: [] for threshold in thresholds}

    for scene_id in scene_ids:
        pred_layout = load_layout(pred_dir / f"{scene_id}.txt", missing_pred)
        pred_regions = pred_layout.regions
        gt_regions = load_gt_regions(
            scene_id, gt_dir, gt_region_dir, k, expand_fraction
        )
        for threshold in thresholds:
            threshold_tuples[threshold].append(
                calc_region_tp(pred_regions, gt_regions, threshold)
            )

    return threshold_tuples


def aggregated_by_class(
    classwise: dict[float, dict[str, list[base_eval.EvalTuple]]],
    class_names: list[str],
) -> dict[float, dict[str, base_eval.EvalTuple]]:
    return {
        threshold: {
            class_name: aggregate(classwise[threshold].get(class_name, []))
            for class_name in class_names
        }
        for threshold in classwise
    }


def print_class_table(
    title: str,
    label_header: str,
    class_names: list[str],
    classwise: dict[float, dict[str, list[base_eval.EvalTuple]]],
    show_empty_classes: bool,
) -> dict[float, dict[str, base_eval.EvalTuple]]:
    metrics = aggregated_by_class(classwise, class_names)
    thresholds = sorted(metrics)
    headers = [label_header, "GT", "Pred"]
    for threshold in thresholds:
        headers.extend([f"P@{threshold:.2f}", f"R@{threshold:.2f}", f"F1@{threshold:.2f}"])

    table_data = [headers]
    for class_name in class_names:
        first = metrics[thresholds[0]][class_name]
        if not show_empty_classes and first.num_gt == 0 and first.num_pred == 0:
            continue

        row = [class_name, first.num_gt, first.num_pred]
        for threshold in thresholds:
            item = metrics[threshold][class_name]
            row.extend(
                [
                    format_score(item.precision),
                    format_score(item.recall),
                    format_score(item.f1),
                ]
            )
        table_data.append(row)

    print(f"\n{title}")
    print(AsciiTable(table_data).table)
    return metrics


def print_spatiallm_reported_table(
    title: str,
    label_header: str,
    class_names: list[str],
    classwise: dict[float, dict[str, list[base_eval.EvalTuple]]],
) -> dict[float, dict]:
    reported = spatiallm_reported_metrics(classwise, class_names)
    thresholds = sorted(reported)
    table_data = [[label_header] + [f"F1@{threshold:.2f}" for threshold in thresholds]]
    for class_name in class_names:
        row = [class_name]
        for threshold in thresholds:
            value = reported[threshold]["classes"][class_name]["f1"]
            row.append("n/a" if value is None else format_score(value))
        table_data.append(row)
    table_data.append(
        ["macro (SpatialLM reported)"]
        + [
            "n/a"
            if reported[threshold]["macro_f1"] is None
            else format_score(reported[threshold]["macro_f1"])
            for threshold in thresholds
        ]
    )
    print(f"\n{title}")
    print(AsciiTable(table_data).table)
    return reported


def print_summary_table(
    title: str,
    class_names: list[str],
    metrics: dict[float, dict[str, base_eval.EvalTuple]],
    classwise: dict[float, dict[str, list[base_eval.EvalTuple]]] | None = None,
) -> None:
    reported = (
        spatiallm_reported_metrics(classwise, class_names)
        if classwise is not None
        else None
    )
    table_data = [["Metric", "IoU", "GT", "Pred", "Precision", "Recall", "F1"]]
    for threshold in sorted(metrics):
        weighted_p, weighted_r, weighted_f1, support, predictions = weighted_metric(
            metrics[threshold], class_names
        )
        table_data.append(
            [
                "weighted",
                f"{threshold:.2f}",
                support,
                predictions,
                format_score(weighted_p),
                format_score(weighted_r),
                format_score(weighted_f1),
            ]
        )
        if reported is not None:
            macro_f1 = reported[threshold]["macro_f1"]
            table_data.append(
                [
                    "spatiallm_macro",
                    f"{threshold:.2f}",
                    "-",
                    "-",
                    "-",
                    "-",
                    "n/a" if macro_f1 is None else format_score(macro_f1),
                ]
            )

        micro = micro_metric(metrics[threshold], class_names)
        table_data.append(
            [
                "micro",
                f"{threshold:.2f}",
                micro.num_gt,
                micro.num_pred,
                format_score(micro.precision),
                format_score(micro.recall),
                format_score(micro.f1),
            ]
        )

    print(f"\n{title}")
    print(AsciiTable(table_data).table)


def print_region_table(region_tuples: dict[float, list[base_eval.EvalTuple]]) -> None:
    table_data = [["Stage1 Regions", "GT", "Pred", "Precision", "Recall", "F1"]]
    for threshold in sorted(region_tuples):
        item = aggregate(region_tuples[threshold])
        table_data.append(
            [
                f"IoU@{threshold:.2f}",
                item.num_gt,
                item.num_pred,
                format_score(item.precision),
                format_score(item.recall),
                format_score(item.f1),
            ]
        )
    print("\nStage1 Region Metrics")
    print(AsciiTable(table_data).table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Hierarchical SpatialLM evaluation script")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata CSV with an id column. Defaults to all *.txt in gt_dir.",
    )
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument(
        "--object_pred_dir",
        "--pred_dir",
        type=Path,
        default=None,
        help="Scene-level final object prediction directory.",
    )
    parser.add_argument(
        "--layout_pred_dir",
        type=Path,
        default=None,
        help="Layout prediction directory. Defaults to stage1_pred_dir, then object_pred_dir.",
    )
    parser.add_argument(
        "--stage1_pred_dir",
        type=Path,
        default=None,
        help="First-stage prediction directory containing Region lines.",
    )
    parser.add_argument(
        "--gt_region_dir",
        type=Path,
        default=None,
        help="Optional GT Region directory. If omitted, regions are derived from GT Bbox lines.",
    )
    parser.add_argument(
        "--label_mapping",
        type=Path,
        default=DEFAULT_LABEL_MAPPING if DEFAULT_LABEL_MAPPING.exists() else None,
    )
    parser.add_argument(
        "--no_label_mapping",
        action="store_true",
        help="Evaluate prediction/GT class names directly without a mapping TSV.",
    )
    parser.add_argument("--label_from", default="spatiallm59")
    parser.add_argument("--label_to", default="spatiallm20")
    parser.add_argument(
        "--object_classes",
        nargs="+",
        default=None,
        help=(
            "Object classes to evaluate. Defaults to eval.py's SpatialLM20 list. "
            "Use quoted values for class names containing spaces."
        ),
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--expand_fraction", type=float, default=0.25)
    parser.add_argument(
        "--region_iou_thresholds",
        nargs="+",
        type=float,
        default=list(REGION_THRESHOLDS),
        help=(
            "IoU thresholds for Stage-1 Region metrics "
            "(default: 0.50 0.75)."
        ),
    )
    parser.add_argument(
        "--minimum_scale",
        type=float,
        default=0.1,
        help=(
            "Clamp every predicted and GT object-box dimension to at least this "
            "many metres before IoU matching (SpatialLM eval.py default: 0.1)."
        ),
    )
    parser.add_argument(
        "--missing_pred",
        choices=["empty", "error"],
        default="empty",
        help="How to handle missing prediction files.",
    )
    parser.add_argument(
        "--show_empty_classes",
        action="store_true",
        help="Print classes with no GT and no predictions.",
    )
    parser.add_argument("--skip_layout", action="store_true")
    parser.add_argument("--skip_objects", action="store_true")
    parser.add_argument("--skip_regions", action="store_true")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to write all metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    region_iou_thresholds = tuple(args.region_iou_thresholds)
    if (
        not region_iou_thresholds
        or len(region_iou_thresholds) != len(set(region_iou_thresholds))
        or any(threshold <= 0.0 or threshold > 1.0 for threshold in region_iou_thresholds)
    ):
        raise ValueError(
            "--region_iou_thresholds must contain unique values in (0, 1]."
        )

    layout_pred_dir = args.layout_pred_dir or args.stage1_pred_dir or args.object_pred_dir
    scene_ids = read_scene_ids(
        args.metadata,
        args.gt_dir,
        (args.object_pred_dir, layout_pred_dir, args.stage1_pred_dir),
    )
    if not scene_ids:
        raise ValueError("No scenes to evaluate.")

    json_result = {
        "scene_count": len(scene_ids),
        "scene_ids": scene_ids,
        "gt_dir": str(args.gt_dir),
        "object_pred_dir": str(args.object_pred_dir) if args.object_pred_dir else None,
        "layout_pred_dir": str(layout_pred_dir) if layout_pred_dir else None,
        "stage1_pred_dir": str(args.stage1_pred_dir) if args.stage1_pred_dir else None,
        "gt_region_dir": str(args.gt_region_dir) if args.gt_region_dir else None,
        "object_minimum_scale": args.minimum_scale,
        "region_iou_thresholds": list(region_iou_thresholds),
        "spatiallm_reported_aggregation": (
            "mean_scene_f1_per_class_then_mean_classes"
        ),
    }
    object_classes = args.object_classes or list(base_eval.OBJECTS)
    if len(object_classes) != len(set(object_classes)):
        raise ValueError("--object_classes contains duplicate class names.")
    json_result["object_classes"] = object_classes

    class_map = None
    if args.label_mapping is not None and not args.no_label_mapping:
        class_map = base_eval.read_label_mapping(
            str(args.label_mapping), args.label_from, args.label_to
        )

    if layout_pred_dir is not None and not args.skip_layout:
        layout_classwise = evaluate_layouts(
            scene_ids,
            args.gt_dir,
            layout_pred_dir,
            args.missing_pred,
        )
        layout_metrics = print_class_table(
            "Layout Global-count Class Metrics",
            "Layouts",
            base_eval.LAYOUTS,
            layout_classwise,
            args.show_empty_classes,
        )
        print_spatiallm_reported_table(
            "Layout SpatialLM-reported Class F1",
            "Layouts",
            base_eval.LAYOUTS,
            layout_classwise,
        )
        print_summary_table(
            "Layout Summary",
            base_eval.LAYOUTS,
            layout_metrics,
            layout_classwise,
        )
        json_result["layout"] = metrics_to_dict(
            layout_metrics,
            base_eval.LAYOUTS,
            layout_classwise,
        )

    if args.object_pred_dir is not None and not args.skip_objects:
        object_classwise = evaluate_objects(
            scene_ids,
            args.gt_dir,
            args.object_pred_dir,
            class_map,
            object_classes,
            args.minimum_scale,
            args.missing_pred,
        )
        object_metrics = print_class_table(
            "Object Global-count Class Metrics",
            "Objects",
            object_classes,
            object_classwise,
            args.show_empty_classes,
        )
        print_spatiallm_reported_table(
            "Object SpatialLM-reported Class F1",
            "Objects",
            object_classes,
            object_classwise,
        )
        print_summary_table(
            "Object Summary",
            object_classes,
            object_metrics,
            object_classwise,
        )
        json_result["objects"] = metrics_to_dict(
            object_metrics,
            object_classes,
            object_classwise,
        )

    if args.stage1_pred_dir is not None and not args.skip_regions:
        region_tuples = evaluate_regions(
            scene_ids,
            args.gt_dir,
            args.stage1_pred_dir,
            args.gt_region_dir,
            args.k,
            args.expand_fraction,
            args.missing_pred,
            region_iou_thresholds,
        )
        print_region_table(region_tuples)
        json_result["regions"] = region_metrics_to_dict(region_tuples)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(json_result, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
