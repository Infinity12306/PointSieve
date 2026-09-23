#!/usr/bin/env python3
"""Aggregate three-stage hierarchical comparison JSONs into JSON/CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from formal_eval_checkpoint import resolve_method_checkpoints


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report every fixed-seed repeat plus population mean/std for "
            "Stage-1 layout/region and per-method Stage-2 object metrics."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    resolve_method_checkpoints(payload, REPO_ROOT)
    return payload


def absolute_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def repeat_name(repeat_index: int, seed: int) -> str:
    return f"repeat_{repeat_index:02d}_seed_{seed}"


def artifact_date_prefix(
    item: dict[str, Any],
    experiment: dict[str, Any],
) -> str:
    date_prefix = str(item.get("date_prefix", experiment["date_prefix"]))
    if re.fullmatch(r"\d{4}", date_prefix) is None:
        raise ValueError(
            f"Artifact date_prefix must use MMDD, got {date_prefix!r}."
        )
    return date_prefix


def read_eval_json(path: Path, expected_scene_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_count = int(payload.get("scene_count", -1))
    if actual_count != expected_scene_count:
        raise ValueError(
            f"Unexpected scene_count in {path}: "
            f"expected={expected_scene_count}, actual={actual_count}"
        )
    return payload


def required_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing metric path: {'.'.join(keys)}")
        current = current[key]
    if current is None or isinstance(current, bool):
        raise ValueError(f"Non-numeric metric at {'.'.join(keys)}: {current}")
    return float(current)


def collect_detection_metrics(
    payload: dict[str, Any],
    section: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for threshold in ("0.25", "0.50"):
        prefix = f"{section}.iou_{threshold}"
        for aggregation in ("micro", "weighted"):
            for metric_name in ("num_gt", "num_pred", "precision", "recall", "f1"):
                metrics[f"{prefix}.{aggregation}.{metric_name}"] = required_float(
                    payload,
                    (section, threshold, "summary", aggregation, metric_name),
                )
        metrics[f"{prefix}.micro.tp"] = required_float(
            payload,
            (section, threshold, "summary", "micro", "tp"),
        )
        metrics[f"{prefix}.spatiallm_macro_f1"] = required_float(
            payload,
            (section, threshold, "spatiallm_reported", "macro_f1"),
        )
    return metrics


def collect_region_metrics(
    payload: dict[str, Any],
    thresholds: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for threshold in thresholds:
        prefix = f"regions.iou_{threshold}"
        for metric_name in ("tp", "num_gt", "num_pred", "precision", "recall", "f1"):
            metrics[f"{prefix}.{metric_name}"] = required_float(
                payload,
                ("regions", threshold, metric_name),
            )
    return metrics


def collect_token_bin_metrics(
    payload: dict[str, Any],
    include_region_level: bool = True,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    section_specs = []
    if include_region_level:
        section_specs.append(
            (
                "region_level",
                "region_token_bins",
                "region_count",
                "raw_point_token_count",
            )
        )
    section_specs.append(
        (
            "scene_level_by_max_region",
            "scene_max_region_token_bins",
            "scene_count",
            "max_raw_point_token_count",
        )
    )
    for section_key, prefix, count_key, token_stats_key in section_specs:
        bin_reports = payload[section_key]["bins"]
        for bin_key, report in bin_reports.items():
            metric_prefix = f"{prefix}.{bin_key}"
            metrics[f"{metric_prefix}.unit_count"] = float(
                report[count_key]
            )
            for stat_name in ("min", "mean", "max"):
                value = report[token_stats_key][stat_name]
                if value is not None:
                    metrics[
                        f"{metric_prefix}.token_count.{stat_name}"
                    ] = float(value)
            for threshold in ("0.25", "0.50"):
                iou_prefix = f"{metric_prefix}.iou_{threshold}"
                for aggregation in ("micro", "weighted"):
                    for metric_name in (
                        "num_gt",
                        "num_pred",
                        "precision",
                        "recall",
                        "f1",
                    ):
                        metrics[
                            f"{iou_prefix}.{aggregation}.{metric_name}"
                        ] = float(
                            report["metrics"][threshold]["summary"][
                                aggregation
                            ][metric_name]
                        )
                macro_f1 = report["metrics"][threshold][
                    "spatiallm_reported"
                ]["macro_f1"]
                if macro_f1 is not None:
                    metrics[
                        f"{iou_prefix}.spatiallm_macro_f1"
                    ] = float(macro_f1)
    return metrics


def collect_scorer_filter_metrics(
    metadata_dir: Path,
    max_keep: int,
) -> dict[str, float]:
    """Measure scorer-only filtering after removing the hard max-token cap.

    ``kept_point_token_count`` is the number actually passed to Stage 2 and
    therefore already includes the ``max_keep`` cap.  To report the fraction
    removed by the score threshold alone, use ``min(raw, max_keep)`` as the
    baseline token population for each region.  This makes threshold 0.0
    report zero filtering: all tokens that survive the same max-token cap are
    retained by the threshold.  The raw/cap/kept totals are emitted alongside
    the ratio so the denominator remains auditable in JSON/CSV reports.
    """
    if max_keep <= 0:
        raise ValueError(f"max_keep must be positive, got {max_keep}.")
    metadata_paths = sorted(metadata_dir.glob("*.json"))
    if not metadata_paths:
        return {}

    raw_total = 0
    cap_baseline_total = 0
    kept_total = 0
    for metadata_path in metadata_paths:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            raw_count = int(payload["raw_point_token_count"])
            kept_count = int(payload["kept_point_token_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid token counts in {metadata_path}: {payload}"
            ) from exc
        if raw_count < 0 or kept_count < 0:
            raise ValueError(
                f"Token counts must be non-negative in {metadata_path}: "
                f"raw={raw_count}, kept={kept_count}"
            )
        if kept_count > min(raw_count, max_keep):
            raise ValueError(
                f"kept_point_token_count exceeds the configured cap in "
                f"{metadata_path}: raw={raw_count}, kept={kept_count}, "
                f"max_keep={max_keep}"
            )
        raw_total += raw_count
        cap_baseline_total += min(raw_count, max_keep)
        kept_total += kept_count

    filtered_ratio = (
        0.0
        if cap_baseline_total == 0
        else 1.0 - kept_total / cap_baseline_total
    )
    # Tiny negative values can only arise from malformed/inconsistent
    # metadata plus floating-point conversion; keep the report in [0, 1].
    filtered_ratio = min(max(filtered_ratio, 0.0), 1.0)
    return {
        "scorer.raw_point_token_count": float(raw_total),
        "scorer.max_keep_point_token_count": float(cap_baseline_total),
        "scorer.kept_point_token_count": float(kept_total),
        "scorer.filtered_point_token_ratio_after_max_keep": filtered_ratio,
    }


def mean_std(values: list[float], ddof: int) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    if not 0 <= ddof < len(values):
        raise ValueError(f"std ddof must satisfy 0 <= ddof < {len(values)}")
    if all(value == values[0] for value in values[1:]):
        return values[0], 0.0
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - ddof
    )
    return mean, math.sqrt(variance)


def aggregate_runs(
    runs: list[dict[str, Any]],
    ddof: int,
    allow_key_mismatch: bool = False,
) -> dict[str, dict[str, float | int]]:
    metric_names = list(runs[0]["metrics"])
    if allow_key_mismatch:
        common_names = set(metric_names)
        for run in runs[1:]:
            common_names.intersection_update(run["metrics"])
        metric_names = [
            metric_name
            for metric_name in metric_names
            if metric_name in common_names
        ]
    else:
        for run in runs[1:]:
            if list(run["metrics"]) != metric_names:
                raise ValueError("Metric keys differ across repeats.")
    result: dict[str, dict[str, float | int]] = {}
    for metric_name in metric_names:
        values = [float(run["metrics"][metric_name]) for run in runs]
        mean, std = mean_std(values, ddof)
        result[metric_name] = {
            "mean": mean,
            "std": std,
            "std_ddof": ddof,
            "repeat_count": len(values),
        }
    return result


def metric_value(run: dict[str, Any], metric_name: str) -> str:
    return f"{float(run['metrics'][metric_name]):.6f}"


def optional_metric_value(run: dict[str, Any], metric_name: str) -> str:
    if metric_name not in run["metrics"]:
        return "N/A"
    return metric_value(run, metric_name)


def aggregate_value(
    aggregate: dict[str, dict[str, float | int]],
    metric_name: str,
) -> str:
    item = aggregate[metric_name]
    return f"{float(item['mean']):.6f} ± {float(item['std']):.6f}"


def optional_aggregate_value(
    aggregate: dict[str, dict[str, float | int]],
    metric_name: str,
) -> str:
    if metric_name not in aggregate:
        return "N/A"
    return aggregate_value(aggregate, metric_name)


def append_markdown_table(
    lines: list[str],
    headers: list[str],
    rows: list[list[str]],
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def append_baseline_scene_token_bin_tables(
    lines: list[str],
    baseline_results: dict[str, Any],
    bin_items: list[tuple[str, str]],
) -> None:
    eligible_results = {
        name: result
        for name, result in baseline_results.items()
        if result.get("scene_token_bin_runs")
    }
    if not eligible_results:
        return

    lines.extend(
        [
            (
                "## One-stage SpatialLM baseline：按 two-stage 场景最大区域 "
                "raw point-token 数分桶：每次结果"
            ),
            "",
        ]
    )
    per_run_rows: list[list[str]] = []
    for baseline_name, baseline_result in eligible_results.items():
        reference_method = str(
            baseline_result["scene_token_bin_reference_method"]
        )
        for run in baseline_result["scene_token_bin_runs"]:
            for bin_key, bin_label in bin_items:
                metric_prefix = (
                    f"scene_max_region_token_bins.{bin_key}"
                )
                unit_count = int(
                    float(
                        run["metrics"][
                            f"{metric_prefix}.unit_count"
                        ]
                    )
                )
                for threshold in ("0.25", "0.50"):
                    prefix = f"{metric_prefix}.iou_{threshold}.micro"
                    macro_metric = (
                        f"{metric_prefix}.iou_{threshold}."
                        "spatiallm_macro_f1"
                    )
                    per_run_rows.append(
                        [
                            baseline_name,
                            reference_method,
                            str(run["repeat_index"]),
                            str(run["seed"]),
                            bin_label,
                            str(unit_count),
                            threshold,
                            metric_value(
                                run,
                                f"{prefix}.precision",
                            ),
                            metric_value(run, f"{prefix}.recall"),
                            metric_value(run, f"{prefix}.f1"),
                            optional_metric_value(run, macro_metric),
                        ]
                    )
    append_markdown_table(
        lines,
        [
            "Method",
            "Token reference",
            "Repeat",
            "Seed",
            "Bin",
            "Scenes",
            "IoU",
            "Micro P",
            "Micro R",
            "Micro F1",
            "Macro F1",
        ],
        per_run_rows,
    )

    lines.extend(
        [
            (
                "## One-stage SpatialLM baseline：按 two-stage 场景最大区域 "
                "raw point-token 数分桶：mean/std"
            ),
            "",
        ]
    )
    aggregate_rows: list[list[str]] = []
    for baseline_name, baseline_result in eligible_results.items():
        reference_method = str(
            baseline_result["scene_token_bin_reference_method"]
        )
        token_aggregate = baseline_result["scene_token_bin_aggregate"]
        for bin_key, bin_label in bin_items:
            metric_prefix = f"scene_max_region_token_bins.{bin_key}"
            for threshold in ("0.25", "0.50"):
                prefix = f"{metric_prefix}.iou_{threshold}.micro"
                macro_metric = (
                    f"{metric_prefix}.iou_{threshold}."
                    "spatiallm_macro_f1"
                )
                aggregate_rows.append(
                    [
                        baseline_name,
                        reference_method,
                        bin_label,
                        aggregate_value(
                            token_aggregate,
                            f"{metric_prefix}.unit_count",
                        ),
                        threshold,
                        aggregate_value(
                            token_aggregate,
                            f"{prefix}.precision",
                        ),
                        aggregate_value(
                            token_aggregate,
                            f"{prefix}.recall",
                        ),
                        aggregate_value(
                            token_aggregate,
                            f"{prefix}.f1",
                        ),
                        optional_aggregate_value(
                            token_aggregate,
                            macro_metric,
                        ),
                    ]
                )
    append_markdown_table(
        lines,
        [
            "Method",
            "Token reference",
            "Bin",
            "Scenes mean/std",
            "IoU",
            "Micro P",
            "Micro R",
            "Micro F1",
            "Macro F1",
        ],
        aggregate_rows,
    )


def build_markdown(
    result: dict[str, Any],
    experiment_name: str,
) -> str:
    stage1_runs = result["stage1"]["runs"]
    stage1_aggregate = result["stage1"]["aggregate"]
    stage1_skipped = bool(result["stage1"].get("skipped", False))
    baseline_results = result["one_stage_baselines"]
    method_results = result["stage2_methods"]
    region_iou_thresholds = [
        f"{float(value):.2f}"
        for value in result.get("region_iou_thresholds", (0.50, 0.75))
    ]

    if stage1_skipped:
        introduction = (
            "GT-region 输入直接提供 expanded regions，本报告仅汇总 Stage-2 object prediction；"
            f"共 {len(result.get('seeds', []))} 次固定 seed 重复，std 使用 "
            f"`ddof={result['std_ddof']}`。"
        )
    elif not baseline_results and not method_results:
        introduction = (
            f"共 {len(stage1_runs)} 次固定 seed 重复；std 使用 "
            f"`ddof={result['std_ddof']}`。本报告仅评测 Stage-1 Region "
            "proposal，不包含 Stage-2 object prediction。"
        )
    else:
        introduction = (
            f"共 {len(stage1_runs)} 次固定 seed 重复；std 使用 "
            f"`ddof={result['std_ddof']}`。每个 Stage-2 方法均复用相同 repeat "
            "对应的 Stage-1 预测，并使用相同 seed。One-stage baseline 独立处理"
            "完整场景点云，但使用同一组 seed。GT-region oracle 使用明确标注的GT区域。"
        )
    if result.get("development_only", False):
        introduction = (
            "仅用于开发集诊断/调参，不构成正式有效性验证。单次repeat的std=0"
            "不代表结果无波动；GT-region oracle不能作为可部署方法结果。\n\n"
            + introduction
        )

    lines = [
        f"# {experiment_name}",
        "",
        introduction,
        "",
    ]

    if not stage1_skipped and result["stage1"]["has_layout"]:
        lines.extend(["## Stage 1：每次 layout 指标", ""])
        layout_rows: list[list[str]] = []
        for run in stage1_runs:
            for threshold in ("0.25", "0.50"):
                prefix = f"layout.iou_{threshold}"
                layout_rows.append(
                    [
                        str(run["repeat_index"]),
                        str(run["seed"]),
                        threshold,
                        metric_value(run, f"{prefix}.micro.precision"),
                        metric_value(run, f"{prefix}.micro.recall"),
                        metric_value(run, f"{prefix}.micro.f1"),
                        metric_value(run, f"{prefix}.spatiallm_macro_f1"),
                    ]
                )
        append_markdown_table(
            lines,
            [
                "Repeat",
                "Seed",
                "IoU",
                "Micro P",
                "Micro R",
                "Micro F1",
                "Macro F1",
            ],
            layout_rows,
        )

        lines.extend(["## Stage 1：layout mean/std", ""])
        layout_aggregate_rows = []
        for threshold in ("0.25", "0.50"):
            prefix = f"layout.iou_{threshold}"
            layout_aggregate_rows.append(
                [
                    threshold,
                    aggregate_value(
                        stage1_aggregate,
                        f"{prefix}.micro.precision",
                    ),
                    aggregate_value(
                        stage1_aggregate,
                        f"{prefix}.micro.recall",
                    ),
                    aggregate_value(
                        stage1_aggregate,
                        f"{prefix}.micro.f1",
                    ),
                    aggregate_value(
                        stage1_aggregate,
                        f"{prefix}.spatiallm_macro_f1",
                    ),
                ]
            )
        append_markdown_table(
            lines,
            ["IoU", "Micro P", "Micro R", "Micro F1", "Macro F1"],
            layout_aggregate_rows,
        )

    if not stage1_skipped:
        lines.extend(["## Stage 1：每次 region 指标", ""])
        region_rows: list[list[str]] = []
        for run in stage1_runs:
            for threshold in region_iou_thresholds:
                prefix = f"regions.iou_{threshold}"
                region_rows.append(
                    [
                        str(run["repeat_index"]),
                        str(run["seed"]),
                        threshold,
                        metric_value(run, f"{prefix}.precision"),
                        metric_value(run, f"{prefix}.recall"),
                        metric_value(run, f"{prefix}.f1"),
                    ]
                )
        append_markdown_table(
            lines,
            ["Repeat", "Seed", "IoU", "Precision", "Recall", "F1"],
            region_rows,
        )

        lines.extend(["## Stage 1：region mean/std", ""])
        region_aggregate_rows = []
        for threshold in region_iou_thresholds:
            prefix = f"regions.iou_{threshold}"
            region_aggregate_rows.append(
                [
                    threshold,
                    aggregate_value(stage1_aggregate, f"{prefix}.precision"),
                    aggregate_value(stage1_aggregate, f"{prefix}.recall"),
                    aggregate_value(stage1_aggregate, f"{prefix}.f1"),
                ]
            )
        append_markdown_table(
            lines,
            ["IoU", "Precision", "Recall", "F1"],
            region_aggregate_rows,
        )

    baseline_layout_results = {
        name: value
        for name, value in baseline_results.items()
        if value["has_layout"]
    }
    if baseline_layout_results:
        lines.extend(
            [
                "## One-stage SpatialLM baseline：每次 layout 指标",
                "",
            ]
        )
        baseline_layout_rows: list[list[str]] = []
        for baseline_name, baseline_result in baseline_layout_results.items():
            for run in baseline_result["runs"]:
                for threshold in ("0.25", "0.50"):
                    prefix = f"layout.iou_{threshold}"
                    baseline_layout_rows.append(
                        [
                            baseline_name,
                            str(run["repeat_index"]),
                            str(run["seed"]),
                            threshold,
                            metric_value(
                                run,
                                f"{prefix}.micro.precision",
                            ),
                            metric_value(run, f"{prefix}.micro.recall"),
                            metric_value(run, f"{prefix}.micro.f1"),
                            metric_value(
                                run,
                                f"{prefix}.spatiallm_macro_f1",
                            ),
                        ]
                    )
        append_markdown_table(
            lines,
            [
                "Method",
                "Repeat",
                "Seed",
                "IoU",
                "Micro P",
                "Micro R",
                "Micro F1",
                "Macro F1",
            ],
            baseline_layout_rows,
        )

        lines.extend(
            [
                "## One-stage SpatialLM baseline：layout mean/std",
                "",
            ]
        )
        baseline_layout_aggregate_rows: list[list[str]] = []
        for baseline_name, baseline_result in baseline_layout_results.items():
            baseline_aggregate = baseline_result["aggregate"]
            for threshold in ("0.25", "0.50"):
                prefix = f"layout.iou_{threshold}"
                baseline_layout_aggregate_rows.append(
                    [
                        baseline_name,
                        threshold,
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.micro.precision",
                        ),
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.micro.recall",
                        ),
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.micro.f1",
                        ),
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.spatiallm_macro_f1",
                        ),
                    ]
                )
        append_markdown_table(
            lines,
            [
                "Method",
                "IoU",
                "Micro P",
                "Micro R",
                "Micro F1",
                "Macro F1",
            ],
            baseline_layout_aggregate_rows,
        )

    if baseline_results:
        lines.extend(
            ["## One-stage SpatialLM baseline：每次 object 指标", ""]
        )
        baseline_object_rows: list[list[str]] = []
        for baseline_name, baseline_result in baseline_results.items():
            for run in baseline_result["runs"]:
                for threshold in ("0.25", "0.50"):
                    prefix = f"objects.iou_{threshold}"
                    baseline_object_rows.append(
                        [
                            baseline_name,
                            str(run["repeat_index"]),
                            str(run["seed"]),
                            threshold,
                            metric_value(
                                run,
                                f"{prefix}.micro.precision",
                            ),
                            metric_value(run, f"{prefix}.micro.recall"),
                            metric_value(run, f"{prefix}.micro.f1"),
                            metric_value(
                                run,
                                f"{prefix}.spatiallm_macro_f1",
                            ),
                        ]
                    )
        append_markdown_table(
            lines,
            [
                "Method",
                "Repeat",
                "Seed",
                "IoU",
                "Micro P",
                "Micro R",
                "Micro F1",
                "Macro F1",
            ],
            baseline_object_rows,
        )

        lines.extend(
            [
                "## One-stage SpatialLM baseline：object mean/std",
                "",
            ]
        )
        baseline_object_aggregate_rows: list[list[str]] = []
        for baseline_name, baseline_result in baseline_results.items():
            baseline_aggregate = baseline_result["aggregate"]
            for threshold in ("0.25", "0.50"):
                prefix = f"objects.iou_{threshold}"
                baseline_object_aggregate_rows.append(
                    [
                        baseline_name,
                        threshold,
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.micro.precision",
                        ),
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.micro.recall",
                        ),
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.micro.f1",
                        ),
                        aggregate_value(
                            baseline_aggregate,
                            f"{prefix}.spatiallm_macro_f1",
                        ),
                    ]
                )
        append_markdown_table(
            lines,
            [
                "Method",
                "IoU",
                "Micro P",
                "Micro R",
                "Micro F1",
                "Macro F1",
            ],
            baseline_object_aggregate_rows,
        )

    lines.extend(["## Stage 2 object：每个方法、每次 Stage 1", ""])
    object_rows: list[list[str]] = []
    for method_name, method_result in method_results.items():
        for run in method_result["runs"]:
            for threshold in ("0.25", "0.50"):
                prefix = f"objects.iou_{threshold}"
                object_rows.append(
                    [
                        method_name,
                        str(run["repeat_index"]),
                        str(run["seed"]),
                        threshold,
                        metric_value(run, f"{prefix}.micro.precision"),
                        metric_value(run, f"{prefix}.micro.recall"),
                        metric_value(run, f"{prefix}.micro.f1"),
                        metric_value(run, f"{prefix}.spatiallm_macro_f1"),
                    ]
                )
    append_markdown_table(
        lines,
        [
            "Method",
            "Repeat",
            "Seed",
            "IoU",
            "Micro P",
            "Micro R",
            "Micro F1",
            "Macro F1",
        ],
        object_rows,
    )

    lines.extend(["## Stage 2 object：各方法 mean/std", ""])
    object_aggregate_rows: list[list[str]] = []
    for method_name, method_result in method_results.items():
        method_aggregate = method_result["aggregate"]
        for threshold in ("0.25", "0.50"):
            prefix = f"objects.iou_{threshold}"
            object_aggregate_rows.append(
                [
                    method_name,
                    threshold,
                    aggregate_value(
                        method_aggregate,
                        f"{prefix}.micro.precision",
                    ),
                    aggregate_value(
                        method_aggregate,
                        f"{prefix}.micro.recall",
                    ),
                    aggregate_value(method_aggregate, f"{prefix}.micro.f1"),
                    aggregate_value(
                        method_aggregate,
                        f"{prefix}.spatiallm_macro_f1",
                    ),
                ]
            )
    append_markdown_table(
        lines,
        ["Method", "IoU", "Micro P", "Micro R", "Micro F1", "Macro F1"],
        object_aggregate_rows,
    )

    filter_ratio_metric = (
        "scorer.filtered_point_token_ratio_after_max_keep"
    )
    filter_results = {
        method_name: method_result
        for method_name, method_result in method_results.items()
        if method_result["runs"]
        and filter_ratio_metric in method_result["runs"][0]["metrics"]
    }
    if filter_results:
        lines.extend(
            [
                "## Scorer point-token filtering：去除 max_keep 截断后的阈值筛除比例",
                "",
                (
                    "该比例按 `1 - kept / min(raw, max_keep)` 计算，先将每个 region "
                    "中由 `max_keep` 截断的 token 从分母中排除，因此 threshold 0.0 "
                    "应为 0。"
                ),
                "",
            ]
        )
        filter_rows: list[list[str]] = []
        for method_name, method_result in filter_results.items():
            for run in method_result["runs"]:
                filter_rows.append(
                    [
                        method_name,
                        str(run["repeat_index"]),
                        str(run["seed"]),
                        metric_value(run, filter_ratio_metric),
                    ]
                )
        append_markdown_table(
            lines,
            ["Method", "Repeat", "Seed", "Filtered ratio"],
            filter_rows,
        )

        filter_aggregate_rows: list[list[str]] = []
        for method_name, method_result in filter_results.items():
            filter_aggregate_rows.append(
                [
                    method_name,
                    aggregate_value(
                        method_result["aggregate"],
                        filter_ratio_metric,
                    ),
                ]
            )
        append_markdown_table(
            lines,
            ["Method", "Filtered ratio mean/std"],
            filter_aggregate_rows,
        )

    first, second, third = result["region_token_boundaries"]
    bin_items = [
        (f"lt_{first}", f"<{first}"),
        (f"ge_{first}_lt_{second}", f"[{first}, {second})"),
        (f"ge_{second}_lt_{third}", f"[{second}, {third})"),
        (f"ge_{third}", f"[{third}, +inf)"),
    ]
    append_baseline_scene_token_bin_tables(
        lines,
        baseline_results,
        bin_items,
    )
    token_level_specs = [
        (
            "region_token_bins",
            "Stage 2 object：按区域 raw point-token 数分桶",
        ),
        (
            "scene_max_region_token_bins",
            "Stage 2 object：按场景最大区域 raw point-token 数分桶",
        ),
    ]
    for metric_prefix, title in token_level_specs:
        lines.extend([f"## {title}：每次结果", ""])
        per_run_rows: list[list[str]] = []
        for method_name, method_result in method_results.items():
            if not method_result["token_bin_runs"]:
                continue
            for run in method_result["token_bin_runs"]:
                for bin_key, bin_label in bin_items:
                    unit_count = int(
                        float(
                            run["metrics"][
                                f"{metric_prefix}.{bin_key}.unit_count"
                            ]
                        )
                    )
                    for threshold in ("0.25", "0.50"):
                        prefix = (
                            f"{metric_prefix}.{bin_key}.iou_{threshold}.micro"
                        )
                        macro_metric = (
                            f"{metric_prefix}.{bin_key}.iou_{threshold}."
                            "spatiallm_macro_f1"
                        )
                        per_run_rows.append(
                            [
                                method_name,
                                str(run["repeat_index"]),
                                str(run["seed"]),
                                bin_label,
                                str(unit_count),
                                threshold,
                                metric_value(run, f"{prefix}.precision"),
                                metric_value(run, f"{prefix}.recall"),
                                metric_value(run, f"{prefix}.f1"),
                                optional_metric_value(run, macro_metric),
                            ]
                        )
        append_markdown_table(
            lines,
            [
                "Method",
                "Repeat",
                "Seed",
                "Bin",
                "Units",
                "IoU",
                "Micro P",
                "Micro R",
                "Micro F1",
                "Macro F1",
            ],
            per_run_rows,
        )

        lines.extend([f"## {title}：mean/std", ""])
        aggregate_rows: list[list[str]] = []
        for method_name, method_result in method_results.items():
            if not method_result["token_bin_runs"]:
                continue
            token_aggregate = method_result["token_bin_aggregate"]
            for bin_key, bin_label in bin_items:
                count_metric = f"{metric_prefix}.{bin_key}.unit_count"
                for threshold in ("0.25", "0.50"):
                    prefix = (
                        f"{metric_prefix}.{bin_key}.iou_{threshold}.micro"
                    )
                    macro_metric = (
                        f"{metric_prefix}.{bin_key}.iou_{threshold}."
                        "spatiallm_macro_f1"
                    )
                    aggregate_rows.append(
                        [
                            method_name,
                            bin_label,
                            aggregate_value(token_aggregate, count_metric),
                            threshold,
                            aggregate_value(
                                token_aggregate,
                                f"{prefix}.precision",
                            ),
                            aggregate_value(
                                token_aggregate,
                                f"{prefix}.recall",
                            ),
                            aggregate_value(
                                token_aggregate,
                                f"{prefix}.f1",
                            ),
                            optional_aggregate_value(
                                token_aggregate,
                                macro_metric,
                            ),
                        ]
                    )
        append_markdown_table(
            lines,
            [
                "Method",
                "Bin",
                "Units mean/std",
                "IoU",
                "Micro P",
                "Micro R",
                "Micro F1",
                "Macro F1",
            ],
            aggregate_rows,
        )
    return "\n".join(lines)


def write_csv(path: Path, result: dict[str, Any]) -> None:
    fieldnames = [
        "scope",
        "method",
        "repeat_index",
        "seed",
        "metric",
        "value",
        "mean",
        "std",
        "std_ddof",
        "source_json",
    ]
    rows: list[dict[str, Any]] = []
    scopes = [("stage1", "", result["stage1"])]
    scopes.extend(
        ("one_stage", baseline_name, baseline_result)
        for baseline_name, baseline_result in result[
            "one_stage_baselines"
        ].items()
    )
    scopes.extend(
        (
            "one_stage_scene_token_bins",
            baseline_name,
            {
                "runs": baseline_result["scene_token_bin_runs"],
                "aggregate": baseline_result[
                    "scene_token_bin_aggregate"
                ],
            },
        )
        for baseline_name, baseline_result in result[
            "one_stage_baselines"
        ].items()
        if baseline_result.get("scene_token_bin_runs")
    )
    scopes.extend(
        ("objects", method_name, method_result)
        for method_name, method_result in result["stage2_methods"].items()
    )
    scopes.extend(
        (
            "stage2_token_bins",
            method_name,
            {
                "runs": method_result["token_bin_runs"],
                "aggregate": method_result["token_bin_aggregate"],
            },
        )
        for method_name, method_result in result["stage2_methods"].items()
        if method_result["token_bin_runs"]
    )
    for scope, method_name, scope_result in scopes:
        aggregate = scope_result["aggregate"]
        for run in scope_result["runs"]:
            for metric_name, value in run["metrics"].items():
                aggregate_item = aggregate.get(metric_name)
                rows.append(
                    {
                        "scope": scope,
                        "method": method_name,
                        "repeat_index": run["repeat_index"],
                        "seed": run["seed"],
                        "metric": metric_name,
                        "value": value,
                        "mean": (
                            aggregate_item["mean"]
                            if aggregate_item is not None
                            else None
                        ),
                        "std": (
                            aggregate_item["std"]
                            if aggregate_item is not None
                            else None
                        ),
                        "std_ddof": (
                            aggregate_item["std_ddof"]
                            if aggregate_item is not None
                            else None
                        ),
                        "source_json": run["source_json"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    seeds = [int(seed) for seed in config["stage1"]["seeds"]]
    methods = config["methods"]
    baselines = config.get("one_stage_baselines", {})
    expected_scene_count = int(experiment["expected_scene_count"])
    ddof = int(config["evaluation"].get("std_ddof", 0))
    stage1_has_layout = not bool(
        config["evaluation"].get("stage1_skip_layout", False)
    )
    stage1_skipped = bool(config["evaluation"].get("stage1_skip_layout", False)) and all(
        str(method.get("region_source", "predicted")) == "gt_region"
        for method in methods.values()
    ) and not baselines
    output_root = absolute_path(experiment["output_root"])
    date_prefix = str(experiment["date_prefix"])
    stage1_date_prefix = artifact_date_prefix(config["stage1"], experiment)
    region_iou_thresholds = [
        f"{float(value):.2f}"
        for value in config["evaluation"].get(
            "region_iou_thresholds",
            (0.50, 0.75),
        )
    ]
    if (
        not region_iou_thresholds
        or len(region_iou_thresholds) != len(set(region_iou_thresholds))
    ):
        raise ValueError(
            "evaluation.region_iou_thresholds must contain unique values."
        )

    stage1_runs: list[dict[str, Any]] = []
    if not stage1_skipped:
        for repeat_index, seed in enumerate(seeds, start=1):
            name = repeat_name(repeat_index, seed)
            source_path = (
                output_root
                / "evaluation"
                / "stage1"
                / f"{stage1_date_prefix}_{name}.json"
            )
            payload = read_eval_json(source_path, expected_scene_count)
            metrics: dict[str, float] = {}
            if stage1_has_layout:
                metrics.update(collect_detection_metrics(payload, "layout"))
            metrics.update(collect_region_metrics(payload, region_iou_thresholds))
            stage1_runs.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "source_json": str(source_path),
                    "metrics": metrics,
                }
            )

    method_results: dict[str, Any] = {}
    for method_name in methods:
        skip_token_bins = bool(
            methods[method_name].get("skip_token_bins", False)
        )
        method_date_prefix = artifact_date_prefix(
            methods[method_name],
            experiment,
        )
        runs: list[dict[str, Any]] = []
        token_bin_runs: list[dict[str, Any]] = []
        for repeat_index, seed in enumerate(seeds, start=1):
            name = repeat_name(repeat_index, seed)
            source_path = (
                output_root
                / "evaluation"
                / "stage2"
                / method_name
                / f"{method_date_prefix}_{name}.json"
            )
            payload = read_eval_json(source_path, expected_scene_count)
            metrics = collect_detection_metrics(payload, "objects")
            scorer_max_keep = methods[method_name].get("scorer_max_keep")
            if scorer_max_keep is not None:
                metadata_dir = (
                    output_root
                    / "predictions"
                    / "stage2"
                    / method_name
                    / name
                    / "raw"
                    / "region_metadata"
                )
                metrics.update(
                    collect_scorer_filter_metrics(
                        metadata_dir,
                        int(scorer_max_keep),
                    )
                )
            runs.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "source_json": str(source_path),
                    "metrics": metrics,
                }
            )
            if not skip_token_bins:
                token_bin_source_path = (
                    output_root
                    / "evaluation"
                    / "stage2"
                    / method_name
                    / "token_bins"
                    / f"{method_date_prefix}_{name}"
                    / "token_bin_evaluation.json"
                )
                token_bin_payload = read_eval_json(
                    token_bin_source_path,
                    expected_scene_count,
                )
                token_bin_runs.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "source_json": str(token_bin_source_path),
                        "metrics": collect_token_bin_metrics(
                            token_bin_payload
                        ),
                    }
                )
        method_results[method_name] = {
            "checkpoint": str(methods[method_name]["checkpoint"]),
            "checkpoint_selection": methods[method_name].get(
                "checkpoint_selection"
            ),
            "method_type": str(methods[method_name]["type"]),
            "output_branch": str(methods[method_name]["output_branch"]),
            "region_source": str(
                methods[method_name].get("region_source", "predicted")
            ),
            "skip_token_bins": skip_token_bins,
            "runs": runs,
            "aggregate": aggregate_runs(runs, ddof),
            "token_bin_runs": token_bin_runs,
            "token_bin_aggregate": (
                {}
                if skip_token_bins
                else aggregate_runs(
                    token_bin_runs,
                    ddof,
                    allow_key_mismatch=True,
                )
            ),
        }

    baseline_results: dict[str, Any] = {}
    for baseline_name, baseline in baselines.items():
        baseline_date_prefix = artifact_date_prefix(
            baseline,
            experiment,
        )
        baseline_has_layout = not bool(
            baseline.get(
                "skip_layout",
                config["evaluation"].get(
                    "one_stage_skip_layout",
                    False,
                ),
            )
        )
        runs = []
        for repeat_index, seed in enumerate(seeds, start=1):
            name = repeat_name(repeat_index, seed)
            source_path = (
                output_root
                / "evaluation"
                / "one_stage"
                / baseline_name
                / f"{baseline_date_prefix}_{name}.json"
            )
            payload = read_eval_json(source_path, expected_scene_count)
            metrics = {}
            if baseline_has_layout:
                metrics.update(collect_detection_metrics(payload, "layout"))
            metrics.update(collect_detection_metrics(payload, "objects"))
            runs.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "source_json": str(source_path),
                    "metrics": metrics,
                }
            )
        baseline_result = {
            "checkpoint": str(baseline["checkpoint"]),
            "source_model": str(baseline["source_model"]),
            "has_layout": baseline_has_layout,
            "runs": runs,
            "aggregate": aggregate_runs(runs, ddof),
        }
        token_reference_method = baseline.get(
            "scene_token_bin_reference_method"
        )
        if token_reference_method is not None:
            scene_token_bin_runs = []
            for repeat_index, seed in enumerate(seeds, start=1):
                name = repeat_name(repeat_index, seed)
                source_path = (
                    output_root
                    / "evaluation"
                    / "one_stage"
                    / baseline_name
                    / "token_bins"
                    / f"{baseline_date_prefix}_{name}"
                    / "token_bin_evaluation.json"
                )
                payload = read_eval_json(
                    source_path,
                    expected_scene_count,
                )
                if payload.get("evaluation_scope") != "scene_only":
                    raise ValueError(
                        "Expected scene-only baseline token-bin evaluation: "
                        f"{source_path}"
                    )
                if (
                    payload.get("method_name") != baseline_name
                    or payload.get("token_reference_method_name")
                    != token_reference_method
                    or int(payload.get("repeat_index", -1))
                    != repeat_index
                    or int(payload.get("seed", -1)) != seed
                ):
                    raise ValueError(
                        "Baseline token-bin provenance mismatch at "
                        f"{source_path}."
                    )
                scene_token_bin_runs.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "source_json": str(source_path),
                        "metrics": collect_token_bin_metrics(
                            payload,
                            include_region_level=False,
                        ),
                    }
                )
            baseline_result.update(
                {
                    "scene_token_bin_reference_method": str(
                        token_reference_method
                    ),
                    "scene_token_bin_runs": scene_token_bin_runs,
                    "scene_token_bin_aggregate": aggregate_runs(
                        scene_token_bin_runs,
                        ddof,
                        allow_key_mismatch=True,
                    ),
                }
            )
        baseline_results[baseline_name] = baseline_result

    result = {
        "format": "hierarchical_repeated_comparison_v4",
        "experiment_name": str(experiment["name"]),
        "config": str(args.config.resolve()),
        "expected_scene_count": expected_scene_count,
        "repeat_count": len(seeds),
        "development_only": bool(experiment.get("development_only", False)),
        "seeds": seeds,
        "std_ddof": ddof,
        "region_token_boundaries": [
            int(value)
            for value in config["evaluation"]["region_token_boundaries"]
        ],
        "region_iou_thresholds": [
            float(value)
            for value in region_iou_thresholds
        ],
        "stage1": {
            "checkpoint": str(config["stage1"]["checkpoint"]),
            "has_layout": stage1_has_layout,
            "skipped": stage1_skipped,
            "reason": (
                "GT-region evaluation supplies GT expanded regions; no Stage-1 prediction/evaluation is produced."
                if stage1_skipped
                else None
            ),
            "runs": stage1_runs,
            "aggregate": aggregate_runs(stage1_runs, ddof) if stage1_runs else {},
        },
        "one_stage_baselines": baseline_results,
        "stage2_methods": method_results,
    }

    report_dir = output_root / "reports"
    report_stem = f"{date_prefix}_{experiment['name']}"
    json_path = report_dir / f"{report_stem}.json"
    csv_path = report_dir / f"{report_stem}.csv"
    markdown_path = report_dir / f"{report_stem}.md"
    atomic_write_text(
        json_path,
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    )
    write_csv(csv_path, result)
    atomic_write_text(
        markdown_path,
        build_markdown(result, str(experiment["name"])) + "\n",
    )
    print(f"Wrote aggregate JSON: {json_path}")
    print(f"Wrote long-form CSV: {csv_path}")
    print(f"Wrote Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
