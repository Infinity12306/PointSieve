#!/usr/bin/env python3
"""Evaluate scene/region predictions by raw predicted-region point-token length."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval import eval_base as base_eval
from eval import eval_hierarchical as hierarchical_eval
from eval.apply_bbox_nms import apply_class_aliases, classwise_nms, parse_class_aliases
from spatiallm import Layout
from spatiallm.layout.entity import Region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Store scene-level metrics grouped by the largest raw region-token "
            "count, with optional region-level metrics for Stage-2 predictions."
        )
    )
    parser.add_argument("--raw_prediction_dir", type=Path, required=True)
    parser.add_argument("--scene_prediction_dir", type=Path, required=True)
    parser.add_argument("--stage1_pred_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional TXT/CSV scene manifest. When provided, require the "
            "Stage-1 predictions to match this exact split and use its order."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument(
        "--token_reference_method_name",
        default=None,
        help=(
            "Method whose Stage-2 region metadata supplies raw token counts. "
            "Defaults to --method_name. This must be set explicitly when "
            "evaluating a one-stage baseline against two-stage scene bins."
        ),
    )
    parser.add_argument(
        "--scene_only",
        action="store_true",
        help=(
            "Only evaluate scene-level predictions grouped by the maximum raw "
            "region-token count; do not read/evaluate per-region predictions."
        ),
    )
    parser.add_argument("--repeat_index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--token_boundaries",
        type=int,
        nargs=3,
        default=[1024, 1536, 2048],
    )
    parser.add_argument("--minimum_scale", type=float, default=0.1)
    parser.add_argument("--class_alias", action="append", default=[])
    parser.add_argument(
        "--label_mapping",
        type=Path,
        default=hierarchical_eval.DEFAULT_LABEL_MAPPING,
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
            "Object classes to evaluate. Defaults to eval.py's SpatialLM20 "
            "classes; required explicitly for ScanNet18."
        ),
    )
    parser.add_argument("--expected_scene_count", type=int, default=500)
    parser.add_argument("--expected_region_nms_iou", type=float, default=0.1)
    args = parser.parse_args()

    if sorted(args.token_boundaries) != args.token_boundaries:
        parser.error("--token_boundaries must be strictly increasing.")
    if len(set(args.token_boundaries)) != 3:
        parser.error("--token_boundaries must contain three distinct values.")
    if args.token_boundaries[0] <= 0:
        parser.error("--token_boundaries must be positive.")
    if args.minimum_scale <= 0:
        parser.error("--minimum_scale must be positive.")
    if args.expected_scene_count <= 0:
        parser.error("--expected_scene_count must be positive.")
    if args.object_classes is not None and (
        not args.object_classes
        or len(args.object_classes) != len(set(args.object_classes))
    ):
        parser.error("--object_classes must be a non-empty unique list.")
    return args


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def ordered_csv_fieldnames(
    rows: list[dict[str, Any]],
    fallback: list[str],
) -> list[str]:
    """Return the stable union of keys for heterogeneous provenance rows."""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames or fallback


def bin_definitions(boundaries: list[int]) -> list[dict[str, Any]]:
    first, second, third = boundaries
    return [
        {
            "key": f"lt_{first}",
            "label": f"<{first}",
            "lower_inclusive": None,
            "upper_exclusive": first,
        },
        {
            "key": f"ge_{first}_lt_{second}",
            "label": f"[{first}, {second})",
            "lower_inclusive": first,
            "upper_exclusive": second,
        },
        {
            "key": f"ge_{second}_lt_{third}",
            "label": f"[{second}, {third})",
            "lower_inclusive": second,
            "upper_exclusive": third,
        },
        {
            "key": f"ge_{third}",
            "label": f"[{third}, +inf)",
            "lower_inclusive": third,
            "upper_exclusive": None,
        },
    ]


def assign_bin(value: int, bins: list[dict[str, Any]]) -> str:
    for item in bins:
        lower = item["lower_inclusive"]
        upper = item["upper_exclusive"]
        if lower is not None and value < lower:
            continue
        if upper is not None and value >= upper:
            continue
        return str(item["key"])
    raise ValueError(f"Token count did not match a bin: {value}")


def bbox_center_in_region(bbox, region: Region) -> bool:
    center = np.asarray(
        [bbox.position_x, bbox.position_y, bbox.position_z],
        dtype=np.float64,
    )
    region_center = np.asarray(
        [region.position_x, region.position_y, region.position_z],
        dtype=np.float64,
    )
    region_scale = np.asarray(
        [region.scale_x, region.scale_y, region.scale_z],
        dtype=np.float64,
    )
    if np.any(region_scale <= 0):
        return False
    return bool(
        np.all(
            (center >= region_center - region_scale * 0.5)
            & (center <= region_center + region_scale * 0.5)
        )
    )


def evaluate_layout_pairs(
    layout_pairs: list[tuple[Layout, Layout]],
    class_map: dict[str, str] | None,
    object_classes: list[str],
    minimum_scale: float,
) -> dict[float, dict[str, list[base_eval.EvalTuple]]]:
    classwise = {
        threshold: defaultdict(list)
        for threshold in hierarchical_eval.OBJECT_THRESHOLDS
    }
    for pred_layout, gt_layout in layout_pairs:
        pred_objects = hierarchical_eval.normalize_objects(
            pred_layout,
            class_map,
            minimum_scale,
            object_classes,
        )
        gt_objects = hierarchical_eval.normalize_objects(
            gt_layout,
            class_map,
            minimum_scale,
            object_classes,
        )
        for class_name in object_classes:
            pred_class = [
                entity
                for entity in pred_objects
                if base_eval.get_entity_class(entity) == class_name
            ]
            gt_class = [
                entity
                for entity in gt_objects
                if base_eval.get_entity_class(entity) == class_name
            ]
            for threshold in hierarchical_eval.OBJECT_THRESHOLDS:
                classwise[threshold][class_name].append(
                    base_eval.calc_bbox_tp(
                        pred_class,
                        gt_class,
                        threshold,
                    )
                )
    return classwise


def subset_classwise(
    classwise: dict,
    indices: list[int],
    object_classes: list[str],
) -> dict:
    return {
        threshold: {
            class_name: [
                classwise[threshold][class_name][index]
                for index in indices
            ]
            for class_name in object_classes
        }
        for threshold in classwise
    }


def metrics_dict(classwise: dict, object_classes: list[str]) -> dict:
    aggregated = hierarchical_eval.aggregated_by_class(
        classwise,
        object_classes,
    )
    return hierarchical_eval.metrics_to_dict(
        aggregated,
        object_classes,
        classwise,
    )


def token_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": float(np.mean(values)),
        "max": max(values),
    }


def summary_rows(
    level: str,
    bin_reports: dict[str, dict[str, Any]],
    bins: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    count_key = "region_count" if level == "region" else "scene_count"
    for bin_info in bins:
        report = bin_reports[bin_info["key"]]
        for iou_key in ("0.25", "0.50"):
            metrics = report["metrics"][iou_key]
            for aggregation in ("micro", "weighted"):
                item = metrics["summary"][aggregation]
                rows.append(
                    {
                        "level": level,
                        "bin": bin_info["key"],
                        "bin_label": bin_info["label"],
                        "unit_count": report[count_key],
                        "iou": iou_key,
                        "aggregation": aggregation,
                        "precision": item["precision"],
                        "recall": item["recall"],
                        "f1": item["f1"],
                    }
                )
            rows.append(
                {
                    "level": level,
                    "bin": bin_info["key"],
                    "bin_label": bin_info["label"],
                    "unit_count": report[count_key],
                    "iou": iou_key,
                    "aggregation": (
                        "region_class_macro"
                        if level == "region"
                        else "spatiallm_scene_class_macro"
                    ),
                    "precision": None,
                    "recall": None,
                    "f1": metrics["spatiallm_reported"]["macro_f1"],
                }
            )
    return rows


def unit_metric_rows(
    unit_records: list[dict[str, Any]],
    classwise: dict,
    object_classes: list[str],
    id_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for unit_index, record in enumerate(unit_records):
        for threshold in hierarchical_eval.OBJECT_THRESHOLDS:
            item = hierarchical_eval.aggregate(
                classwise[threshold][class_name][unit_index]
                for class_name in object_classes
            )
            rows.append(
                {
                    **record,
                    "unit_id": record[id_key],
                    "iou": f"{threshold:.2f}",
                    "tp": item.tp,
                    "num_pred": item.num_pred,
                    "num_gt": item.num_gt,
                    "precision": item.precision,
                    "recall": item.recall,
                    "f1": item.f1,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    class_aliases = parse_class_aliases(args.class_alias)
    token_reference_method_name = (
        args.token_reference_method_name or args.method_name
    )
    bins = bin_definitions(args.token_boundaries)
    stage1_paths = sorted(args.stage1_pred_dir.glob("*.txt"))
    stage1_scene_ids = [path.stem for path in stage1_paths]
    if len(stage1_scene_ids) != args.expected_scene_count:
        raise ValueError(
            "Unexpected Stage-1 scene count: "
            f"expected={args.expected_scene_count}, "
            f"actual={len(stage1_scene_ids)}"
        )
    if len(stage1_scene_ids) != len(set(stage1_scene_ids)):
        raise ValueError("Stage-1 prediction directory has duplicate scene ids.")
    if args.metadata is not None:
        scene_ids = hierarchical_eval.read_scene_ids(
            args.metadata,
            args.gt_dir,
        )
        if set(scene_ids) != set(stage1_scene_ids):
            missing = sorted(set(scene_ids) - set(stage1_scene_ids))
            extra = sorted(set(stage1_scene_ids) - set(scene_ids))
            raise ValueError(
                "Stage-1 predictions do not match metadata split: "
                f"metadata={len(scene_ids)}, predictions={len(stage1_scene_ids)}, "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    else:
        scene_ids = stage1_scene_ids
    if len(scene_ids) != args.expected_scene_count:
        raise ValueError(
            "Unexpected metadata scene count: "
            f"expected={args.expected_scene_count}, actual={len(scene_ids)}"
        )
    missing_scene_predictions = [
        scene_id
        for scene_id in scene_ids
        if not (
            args.scene_prediction_dir / f"{scene_id}.txt"
        ).is_file()
    ]
    if missing_scene_predictions:
        raise FileNotFoundError(
            "Missing scene predictions: "
            f"{missing_scene_predictions[:5]}"
        )

    class_map = None
    if args.label_mapping is not None and not args.no_label_mapping:
        class_map = base_eval.read_label_mapping(
            str(args.label_mapping),
            args.label_from,
            args.label_to,
        )
    object_classes = args.object_classes or list(base_eval.OBJECTS)
    region_records: list[dict[str, Any]] = []
    region_layout_pairs: list[tuple[Layout, Layout]] = []
    scene_records: list[dict[str, Any]] = []
    total_region_count = 0

    for scene_id in scene_ids:
        stage1_layout = Layout(
            (
                args.stage1_pred_dir / f"{scene_id}.txt"
            ).read_text(encoding="utf-8")
        )
        gt_layout = Layout(
            (args.gt_dir / f"{scene_id}.txt").read_text(encoding="utf-8")
        )
        index_path = (
            args.raw_prediction_dir
            / "scene_region_index"
            / f"{scene_id}.json"
        )
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        scene_index = json.loads(index_path.read_text(encoding="utf-8"))
        region_ids = scene_index["region_ids"]
        if len(region_ids) != len(stage1_layout.regions):
            raise ValueError(
                f"Region count mismatch for {scene_id}: "
                f"stage1={len(stage1_layout.regions)}, "
                f"saved={len(region_ids)}"
            )
        total_region_count += len(region_ids)

        scene_raw_counts = []
        for expected_region_index, region_id in enumerate(region_ids):
            metadata_path = (
                args.raw_prediction_dir
                / "region_metadata"
                / f"{region_id}.json"
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            prediction_path = (
                args.raw_prediction_dir
                / "region_predictions"
                / f"{region_id}.txt"
            )
            if not args.scene_only and not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if int(metadata["region_index"]) != expected_region_index:
                raise ValueError(
                    f"Region index mismatch for {region_id}: "
                    f"{metadata['region_index']} != {expected_region_index}"
                )
            if not bool(metadata["prediction_is_region_nms_output"]):
                raise ValueError(f"Region prediction is not NMS output: {region_id}")
            if (
                float(metadata["region_bbox_nms_iou"])
                != args.expected_region_nms_iou
            ):
                raise ValueError(
                    f"Unexpected region NMS IoU for {region_id}: "
                    f"{metadata['region_bbox_nms_iou']}"
                )

            raw_token_count = int(metadata["raw_point_token_count"])
            token_bin = assign_bin(raw_token_count, bins)
            scene_raw_counts.append(raw_token_count)
            if args.scene_only:
                continue
            region = stage1_layout.regions[expected_region_index]
            region_gt_layout = Layout()
            region_gt_layout.bboxes = [
                copy.deepcopy(bbox)
                for bbox in gt_layout.bboxes
                if bbox_center_in_region(bbox, region)
            ]
            region_prediction = Layout(prediction_path.read_text(encoding="utf-8"))
            if class_aliases:
                apply_class_aliases(region_prediction.bboxes, class_aliases)
                region_prediction.bboxes = classwise_nms(
                    region_prediction.bboxes, args.expected_region_nms_iou,
                    float(metadata.get("region_bbox_nms_minimum_scale", 1.0e-6)),
                )
            region_layout_pairs.append(
                (
                    region_prediction,
                    region_gt_layout,
                )
            )
            region_records.append(
                {
                    **metadata,
                    "token_bin": token_bin,
                    "token_bin_label": next(
                        item["label"]
                        for item in bins
                        if item["key"] == token_bin
                    ),
                    "gt_bbox_count": len(region_gt_layout.bboxes),
                }
            )

        max_raw_token_count = max(scene_raw_counts, default=0)
        scene_bin = assign_bin(max_raw_token_count, bins)
        scene_records.append(
            {
                "scene_id": scene_id,
                "region_count": len(region_ids),
                "max_raw_point_token_count": max_raw_token_count,
                "token_bin": scene_bin,
                "token_bin_label": next(
                    item["label"]
                    for item in bins
                    if item["key"] == scene_bin
                ),
            }
        )

    region_classwise = None
    region_reports = None
    if not args.scene_only:
        region_classwise = evaluate_layout_pairs(
            region_layout_pairs,
            class_map,
            object_classes,
            args.minimum_scale,
        )
        region_reports = {}
        for bin_info in bins:
            indices = [
                index
                for index, record in enumerate(region_records)
                if record["token_bin"] == bin_info["key"]
            ]
            token_values = [
                int(region_records[index]["raw_point_token_count"])
                for index in indices
            ]
            region_reports[bin_info["key"]] = {
                **bin_info,
                "region_count": len(indices),
                "region_ids": [
                    region_records[index]["region_id"] for index in indices
                ],
                "raw_point_token_count": token_stats(token_values),
                "metrics": metrics_dict(
                    subset_classwise(
                        region_classwise,
                        indices,
                        object_classes,
                    ),
                    object_classes,
                ),
            }

    scene_classwise = hierarchical_eval.evaluate_objects(
        scene_ids,
        args.gt_dir,
        args.scene_prediction_dir,
        class_map,
        object_classes,
        args.minimum_scale,
        "error",
    )
    scene_reports = {}
    for bin_info in bins:
        indices = [
            index
            for index, record in enumerate(scene_records)
            if record["token_bin"] == bin_info["key"]
        ]
        token_values = [
            int(scene_records[index]["max_raw_point_token_count"])
            for index in indices
        ]
        scene_reports[bin_info["key"]] = {
            **bin_info,
            "scene_count": len(indices),
            "scene_ids": [
                scene_records[index]["scene_id"] for index in indices
            ],
            "max_raw_point_token_count": token_stats(token_values),
            "metrics": metrics_dict(
                subset_classwise(
                    scene_classwise,
                    indices,
                    object_classes,
                ),
                object_classes,
            ),
        }

    region_summary_rows = (
        []
        if region_reports is None
        else summary_rows(
            "region",
            region_reports,
            bins,
        )
    )
    scene_summary_rows = summary_rows(
        "scene_max_region",
        scene_reports,
        bins,
    )
    region_unit_rows = (
        []
        if region_classwise is None
        else unit_metric_rows(
            region_records,
            region_classwise,
            object_classes,
            "region_id",
        )
    )
    scene_unit_rows = unit_metric_rows(
        scene_records,
        scene_classwise,
        object_classes,
        "scene_id",
    )
    output_paths = {
        "json": args.output_dir / "token_bin_evaluation.json",
        "scene_summary_csv": (
            args.output_dir / "scene_max_region_bin_metrics.csv"
        ),
        "per_scene_csv": args.output_dir / "per_scene_metrics.csv",
    }
    if not args.scene_only:
        output_paths.update(
            {
                "region_summary_csv": (
                    args.output_dir / "region_bin_metrics.csv"
                ),
                "per_region_csv": args.output_dir / "per_region_metrics.csv",
            }
        )
    summary_fields = [
        "level",
        "bin",
        "bin_label",
        "unit_count",
        "iou",
        "aggregation",
        "precision",
        "recall",
        "f1",
    ]
    if not args.scene_only:
        atomic_write_csv(
            output_paths["region_summary_csv"],
            region_summary_rows,
            summary_fields,
        )
    atomic_write_csv(
        output_paths["scene_summary_csv"],
        scene_summary_rows,
        summary_fields,
    )
    if not args.scene_only:
        atomic_write_csv(
            output_paths["per_region_csv"],
            region_unit_rows,
            ordered_csv_fieldnames(region_unit_rows, ["unit_id"]),
        )
    atomic_write_csv(
        output_paths["per_scene_csv"],
        scene_unit_rows,
        ordered_csv_fieldnames(scene_unit_rows, ["unit_id"]),
    )
    atomic_write_json(
        output_paths["json"],
        {
            "format": (
                "hierarchical_scene_max_region_token_bins_v1"
                if args.scene_only
                else "hierarchical_stage2_raw_region_token_bins_v1"
            ),
            "method_name": args.method_name,
            "token_reference_method_name": token_reference_method_name,
            "token_reference_raw_prediction_dir": str(
                args.raw_prediction_dir
            ),
            "evaluation_scope": (
                "scene_only" if args.scene_only else "region_and_scene"
            ),
            "repeat_index": args.repeat_index,
            "seed": args.seed,
            "scene_count": len(scene_ids),
            "region_count": total_region_count,
            "token_count_definition": (
                "Raw Sonata output token count before model max-token cropping "
                "or scorer filtering."
            ),
            "bin_boundary_semantics": "left_closed_right_open",
            "bins": bins,
            "minimum_scale": args.minimum_scale,
            "metadata": str(args.metadata) if args.metadata else None,
            "label_mapping": (
                None
                if args.no_label_mapping or args.label_mapping is None
                else str(args.label_mapping)
            ),
            "object_classes": object_classes,
            "class_aliases": class_aliases,
            "region_level": (
                None
                if args.scene_only
                else {
                    "prediction_source": (
                        "Per-region predictions saved after inference-time "
                        "class-wise NMS."
                    ),
                    "cross_region_merge": False,
                    "region_nms_iou": args.expected_region_nms_iou,
                    "gt_assignment": (
                        "A GT bbox is assigned to every predicted region that "
                        "contains its center."
                    ),
                    "bins": region_reports,
                }
            ),
            "scene_level_by_max_region": {
                "prediction_source": (
                    "Standard scene-level final predictions after global NMS."
                ),
                "scene_bin_definition": (
                    "Bin of the maximum raw region point-token count in each "
                    "scene from token_reference_method_name; scenes with zero "
                    "predicted regions use count 0."
                ),
                "bins": scene_reports,
            },
            "outputs": {
                key: str(path)
                for key, path in output_paths.items()
            },
        },
    )
    print(f"Wrote token-bin evaluation: {output_paths['json']}")


if __name__ == "__main__":
    main()
