#!/usr/bin/env python3
"""Orchestrate reusable Stage-1 repeats and fair fixed-seed Stage-2 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from eval.formal_eval_checkpoint import resolve_method_checkpoints


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_ORDER = (
    "stage1",
    "stage1_eval",
    "baseline",
    "stage2",
    "stage2_eval",
    "baseline_eval",
    "aggregate",
)
SCORER_METHOD_TYPES = ("conditional_scorer", "all_region_scorer")
ATTENTION_SCORER_METHOD_TYPES = ("attention_scorer",)
METHOD_TYPES = (
    "plain",
    *SCORER_METHOD_TYPES,
    *ATTENTION_SCORER_METHOD_TYPES,
    "bbox_regression",
)
REGION_SOURCES = ("predicted", "gt_region")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three fixed-seed Stage-1 predictions once, reuse them for all "
            "Stage-2 methods, and report every run plus mean/std."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=("all", "validate", *PHASE_ORDER),
        default=["all"],
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help=(
            "Comma-separated physical GPU ids. Each entry launches one shard; "
            "repeated ids intentionally launch multiple processes on one GPU."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional subset of method keys from the YAML.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        help="Optional subset of one_stage_baselines keys from the YAML.",
    )
    parser.add_argument(
        "--repeats",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional 1-based Stage-1 repeat indices. For example, "
            "--repeats 2 runs only the second configured seed."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print commands without writing or launching jobs.",
    )
    return parser.parse_args()


def absolute_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    resolve_method_checkpoints(payload, REPO_ROOT)
    return payload, text


def checkpoint_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def scorer_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    direct = path / "scorer.pt"
    if direct.is_file():
        return direct
    checkpoints: list[tuple[int, Path]] = []
    if path.is_dir():
        for candidate in path.glob("checkpoint-*"):
            scorer_file = candidate / "scorer.pt"
            suffix = candidate.name.removeprefix("checkpoint-")
            if scorer_file.is_file() and suffix.isdigit():
                checkpoints.append((int(suffix), scorer_file))
    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]
    raise FileNotFoundError(f"No scorer.pt found under {path}")


def scorer_checkpoint_exists(path: Path) -> bool:
    try:
        scorer_checkpoint_file(path)
    except FileNotFoundError:
        return False
    return True


def validate_scorer_point_module_lineage(
    method_name: str,
    method: dict[str, Any],
    scorer_path: Path,
) -> None:
    expected_source_value = method.get("scorer_point_module_source_path")
    if expected_source_value is None:
        return

    scorer_file = scorer_checkpoint_file(scorer_path)
    checkpoint = torch.load(
        scorer_file,
        map_location="cpu",
        weights_only=False,
    )
    scorer_args = checkpoint.get("args", {})
    actual_source_value = None
    if isinstance(scorer_args, dict):
        actual_source_value = scorer_args.get("model_name_or_path")
        if not actual_source_value:
            # Legacy cached-scorer checkpoints recorded the same frozen
            # point-backbone/projector provenance under projector_model_path.
            # Keep the resolved-path equality check below unchanged.
            actual_source_value = scorer_args.get("projector_model_path")
        if not actual_source_value:
            actual_source_value = scorer_args.get(
                "scorer_point_module_source_path"
            )
    if not actual_source_value and checkpoint.get("joint_training"):
        # Older joint scorer checkpoints did not copy their
        # point-module provenance into scorer.pt. The colocated Hugging Face
        # model config still records the exact Stage-2 initialization path.
        model_config_path = scorer_file.parent / "config.json"
        if model_config_path.is_file():
            model_config = json.loads(
                model_config_path.read_text(encoding="utf-8")
            )
            actual_source_value = model_config.get("_name_or_path")
    if not actual_source_value:
        raise ValueError(
            f"{method_name} scorer checkpoint does not record "
            "point-module lineage in scorer.pt or its colocated model "
            "config."
        )

    expected_source = absolute_path(expected_source_value).resolve()
    actual_source = absolute_path(actual_source_value).resolve()
    if actual_source != expected_source:
        raise ValueError(
            f"{method_name} scorer point-module lineage mismatch: "
            f"expected={expected_source}, actual={actual_source}."
        )


def validate_attention_scorer_pair(
    method_name: str,
    method: dict[str, Any],
) -> None:
    scorer_path = absolute_path(method["attention_scorer_path"])
    source_file = scorer_checkpoint_file(scorer_path)
    source = torch.load(source_file, map_location="cpu", weights_only=False)

    source_args = source.get("args", {})
    if not isinstance(source_args, dict):
        raise ValueError(f"{method_name} attention scorer args must be a mapping.")
    expected_args = (
        ("max_point_tokens", int(method["attention_max_point_tokens"]), None),
        ("budget", int(method["attention_budget"]), None),
        # Legacy static checkpoints predate the explicit metadata field.
        ("retention_bias_mode", "static", "static"),
    )
    for key, expected, legacy_default in expected_args:
        actual = source_args.get(key, legacy_default)
        if str(actual) != str(expected):
            raise ValueError(
                f"{method_name} attention scorer {key} mismatch: "
                f"expected={expected}, actual={actual}."
            )

    expected_scorer_step = method.get("attention_expected_scorer_step")
    if (
        expected_scorer_step is not None
        and int(source.get("step", -1)) != int(expected_scorer_step)
    ):
        raise ValueError(
            f"{method_name} attention scorer step mismatch: "
            f"expected={expected_scorer_step}, actual={source.get('step')}."
        )

    require_paired_scorer = bool(
        method.get("attention_require_paired_scorer", True)
    )
    model_path = absolute_path(method["checkpoint"])
    if not require_paired_scorer:
        if scorer_checkpoint_exists(model_path):
            raise ValueError(
                f"{method_name} is declared scorer-only but its Stage-2 "
                f"checkpoint contains a paired scorer.pt: {model_path}."
            )
        return

    paired_file = scorer_checkpoint_file(model_path)
    paired = torch.load(paired_file, map_location="cpu", weights_only=False)
    if source.get("config") != paired.get("config"):
        raise ValueError(
            f"{method_name} source and paired attention-scorer configs differ."
        )
    source_model = source.get("model")
    paired_model = paired.get("model")
    if not isinstance(source_model, dict) or not isinstance(paired_model, dict):
        raise ValueError(
            f"{method_name} attention scorer checkpoint has no model state."
        )
    if set(source_model) != set(paired_model):
        raise ValueError(
            f"{method_name} source and paired attention-scorer keys differ."
        )
    for key in source_model:
        if not torch.equal(source_model[key], paired_model[key]):
            raise ValueError(
                f"{method_name} paired attention scorer differs at {key}."
            )

    joint_training = paired.get("joint_training", {})
    expected_top_k = int(method["attention_top_k"])
    if (
        not isinstance(joint_training, dict)
        or joint_training.get("mode") != "frozen_hard_topk_lm"
        or int(joint_training.get("top_k", -1)) != expected_top_k
    ):
        raise ValueError(
            f"{method_name} paired scorer joint metadata mismatch: "
            f"expected mode=frozen_hard_topk_lm/top_k={expected_top_k}, "
            f"actual={joint_training}."
        )


def scene_ids_from_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = (
            payload.get("data")
            or payload.get("examples")
            or payload.get("items")
            or []
        )
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of examples in {path}")

    scene_ids: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        point_clouds = item.get("point_clouds") or item.get("point_cloud")
        if isinstance(point_clouds, list):
            if not point_clouds:
                continue
            raw_path = point_clouds[0]
        else:
            raw_path = point_clouds
        if not raw_path:
            continue
        scene_id = Path(str(raw_path)).stem
        if scene_id in seen:
            scene_id = f"{scene_id}_{index:06d}"
        scene_ids.append(scene_id)
        seen.add(scene_id)
    return scene_ids


def scene_ids_from_metadata(path: Path) -> list[str]:
    if path.suffix.lower() == ".txt":
        scene_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "id" not in (reader.fieldnames or []):
                raise ValueError(f"{path} must contain an 'id' column.")
            scene_ids = [row["id"] for row in reader if row.get("id")]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError(f"{path} contains duplicate scene ids.")
    return scene_ids


def validate_config(
    config: dict[str, Any],
    selected_methods: list[str] | None,
    selected_baselines: list[str] | None,
    *,
    require_stage2_methods: bool = True,
) -> tuple[list[str], list[str], list[int], list[str]]:
    experiment = config["experiment"]
    stage1 = config["stage1"]
    methods = config["methods"]
    baselines = config.get("one_stage_baselines", {})
    generation = config["generation"]
    evaluation = config["evaluation"]

    seeds = [int(seed) for seed in stage1["seeds"]]
    development_only = bool(experiment.get("development_only", False))
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("stage1.seeds must be unique non-negative seeds.")
    if not development_only and len(seeds) != 3:
        raise ValueError("stage1.seeds must contain exactly three unique non-negative seeds.")
    if not 0 <= int(evaluation.get("std_ddof", 0)) < len(seeds):
        raise ValueError("evaluation.std_ddof must be non-negative and less than repeat count.")
    region_iou_thresholds = [
        float(value)
        for value in evaluation.get("region_iou_thresholds", (0.50, 0.75))
    ]
    if (
        not region_iou_thresholds
        or len(region_iou_thresholds) != len(set(region_iou_thresholds))
        or any(threshold <= 0.0 or threshold > 1.0 for threshold in region_iou_thresholds)
    ):
        raise ValueError(
            "evaluation.region_iou_thresholds must contain unique values in (0, 1]."
        )
    if [
        int(value)
        for value in evaluation["region_token_boundaries"]
    ] != [1024, 1536, 2048]:
        raise ValueError(
            "evaluation.region_token_boundaries must be [1024, 1536, 2048]."
        )
    if int(generation["num_beams"]) != 1 and any(
        method["type"] == "bbox_regression" for method in methods.values()
    ):
        raise ValueError("bbox_regression requires generation.num_beams=1.")

    data_json = absolute_path(experiment["data_json"])
    dataset_root = absolute_path(experiment["dataset_root"])
    required_paths = [data_json, dataset_root, absolute_path(evaluation["gt_dir"])]
    metadata_value = experiment.get("metadata")
    if metadata_value is not None:
        required_paths.append(absolute_path(metadata_value))
    gt_region_value = evaluation.get("gt_region_dir")
    if gt_region_value is not None:
        required_paths.append(absolute_path(gt_region_value))
    no_label_mapping = bool(evaluation.get("no_label_mapping", False))
    object_classes = evaluation.get("object_classes")
    if no_label_mapping:
        if (
            not isinstance(object_classes, list)
            or not object_classes
            or len(object_classes) != len(set(object_classes))
        ):
            raise ValueError(
                "evaluation.object_classes must be a non-empty unique list "
                "when no_label_mapping=true."
            )
    else:
        for key in ("label_mapping", "label_from", "label_to"):
            if key not in evaluation:
                raise ValueError(f"evaluation.{key} is required.")
        required_paths.append(absolute_path(evaluation["label_mapping"]))
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    scene_ids = scene_ids_from_json(data_json)
    expected_scene_count = int(experiment["expected_scene_count"])
    if len(scene_ids) != expected_scene_count:
        raise ValueError(
            "Dataset scene count mismatch: "
            f"expected={expected_scene_count}, actual={len(scene_ids)}"
        )
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Resolved test scene ids are not unique.")
    if metadata_value is not None:
        metadata_scene_ids = scene_ids_from_metadata(
            absolute_path(metadata_value)
        )
        if set(metadata_scene_ids) != set(scene_ids):
            missing = sorted(set(metadata_scene_ids) - set(scene_ids))
            extra = sorted(set(scene_ids) - set(metadata_scene_ids))
            raise ValueError(
                "data_json and metadata scene sets differ: "
                f"metadata={len(metadata_scene_ids)}, json={len(scene_ids)}, "
                f"missing_from_json={missing[:5]}, extra_in_json={extra[:5]}"
            )
        if bool(experiment.get("require_scene_order_match", False)) and (
            metadata_scene_ids != scene_ids
        ):
            raise ValueError(
                "data_json and metadata contain the same scenes but in "
                "different orders while require_scene_order_match=true."
            )

    stage1_prompt_source = str(stage1.get("prompt_source", "default"))
    if stage1_prompt_source not in {"default", "data_json"}:
        raise ValueError(
            "stage1.prompt_source must be either 'default' or 'data_json'."
        )

    stage1_checkpoint = absolute_path(stage1["checkpoint"])
    stage1_config = checkpoint_config(stage1_checkpoint)
    expected_stage1_global_step = stage1.get("expected_global_step")
    if expected_stage1_global_step is not None:
        trainer_state_path = stage1_checkpoint / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(
                "Stage-1 checkpoint has no trainer_state.json: "
                f"{trainer_state_path}"
            )
        trainer_state = json.loads(
            trainer_state_path.read_text(encoding="utf-8")
        )
        actual_stage1_global_step = int(
            trainer_state.get("global_step", -1)
        )
        if actual_stage1_global_step != int(expected_stage1_global_step):
            raise ValueError(
                "Stage-1 global_step mismatch: "
                f"expected={expected_stage1_global_step}, "
                f"actual={actual_stage1_global_step}"
            )
    point_config = stage1_config["point_config"]
    expected_num_bins = int(stage1["num_bins"])
    expected_world_size = float(stage1["world_size"])
    actual_num_bins = int(point_config["num_bins"])
    actual_world_size = float(point_config.get("world_size", 32.0))
    if (actual_num_bins, actual_world_size) != (
        expected_num_bins,
        expected_world_size,
    ):
        raise ValueError(
            "Stage-1 preprocessing/checkpoint mismatch: "
            f"config=({expected_num_bins}, {expected_world_size}), "
            f"checkpoint=({actual_num_bins}, {actual_world_size})"
        )

    method_names = list(methods)
    if selected_methods is not None:
        unknown = sorted(set(selected_methods) - set(method_names))
        if unknown:
            raise ValueError(f"Unknown --methods entries: {unknown}")
        method_names = selected_methods
    if require_stage2_methods and not method_names:
        raise ValueError("At least one Stage-2 method must be selected.")

    for method_name in method_names:
        method = methods[method_name]
        aliases = method.get("class_aliases", {})
        if not isinstance(aliases, dict) or any(
            not isinstance(value, str) or not value.strip()
            for pair in aliases.items() for value in pair
        ):
            raise ValueError(f"{method_name}.class_aliases must map nonempty strings.")

        method_type = str(method["type"])
        if method_type not in METHOD_TYPES:
            raise ValueError(
                f"Unsupported type for {method_name}: {method_type}"
            )
        context_padding = float(method.get('region_context_padding', 0))
        if not math.isfinite(context_padding) or context_padding < 0:
            raise ValueError(f'{method_name}: region_context_padding must be finite and non-negative')
        if (context_padding or method.get('region_core_filter', False)) and method_type != 'plain':
            raise ValueError(f'{method_name}: context-halo diagnostics support plain methods only')
        for boolean_key in ['region_core_filter', 'no_cleanup']:
            if boolean_key in method and not isinstance(method[boolean_key], bool):
                raise ValueError(f'{method_name}.{boolean_key} must be a YAML boolean')
        if 'no_cleanup' in method and method_type != 'plain':
            raise ValueError(f'{method_name}: per-method cleanup control currently supports plain only')
        region_source = str(method.get("region_source", "predicted"))
        if region_source not in REGION_SOURCES:
            raise ValueError(
                f"Unsupported region_source for {method_name}: "
                f"{region_source!r}"
            )
        if region_source == "gt_region":
            if method_type not in {"plain", "all_region_scorer"}:
                raise ValueError(
                    f"{method_name} GT-region formal evaluation currently "
                    "requires type=plain or type=all_region_scorer."
                )
            if method_type == "all_region_scorer":
                gt_region_data_json = absolute_path(
                    method["gt_region_data_json"]
                )
                if not gt_region_data_json.is_file():
                    raise FileNotFoundError(gt_region_data_json)
            if not bool(method.get("skip_token_bins", False)):
                raise ValueError(
                    f"{method_name} must set skip_token_bins=true because "
                    "GT regions are not paired with predicted-region bins."
                )
        model_config = checkpoint_config(absolute_path(method["checkpoint"]))
        method_point_config = model_config["point_config"]
        expected_global_step = method.get("expected_global_step")
        if expected_global_step is not None:
            trainer_state_path = (
                absolute_path(method["checkpoint"]) / "trainer_state.json"
            )
            if not trainer_state_path.is_file():
                raise FileNotFoundError(
                    f"{method_name} checkpoint has no trainer_state.json: "
                    f"{trainer_state_path}"
                )
            trainer_state = json.loads(
                trainer_state_path.read_text(encoding="utf-8")
            )
            actual_global_step = int(trainer_state.get("global_step", -1))
            if actual_global_step != int(expected_global_step):
                raise ValueError(
                    f"{method_name} global_step mismatch: "
                    f"expected={expected_global_step}, "
                    f"actual={actual_global_step}"
                )
        expected_num_bins = method.get("expected_num_bins")
        actual_num_bins = int(method_point_config["num_bins"])
        if (
            expected_num_bins is not None
            and actual_num_bins != int(expected_num_bins)
        ):
            raise ValueError(
                f"{method_name} num_bins mismatch: "
                f"expected={expected_num_bins}, actual={actual_num_bins}"
            )
        output_branch = str(
            method.get(
                "output_branch",
                "mlp" if method_type == "bbox_regression" else "llm",
            )
        )
        expected_output_branch = (
            "mlp" if method_type == "bbox_regression" else "llm"
        )
        if output_branch != expected_output_branch:
            raise ValueError(
                f"{method_name} type={method_type} requires "
                f"output_branch={expected_output_branch}, got {output_branch}."
            )
        expected_world_size = float(method.get("expected_world_size", 16.0))
        actual_world_size = float(method_point_config.get("world_size", 32.0))
        if actual_world_size != expected_world_size:
            raise ValueError(
                f"{method_name} world_size mismatch: "
                f"expected={expected_world_size}, actual={actual_world_size}"
            )
        expected_max_tokens = method.get("expected_max_point_tokens")
        actual_max_tokens = method_point_config.get("max_point_tokens")
        if (
            expected_max_tokens is not None
            and (
                actual_max_tokens is None
                or int(actual_max_tokens) != int(expected_max_tokens)
            )
        ):
            raise ValueError(
                f"{method_name} max_point_tokens mismatch: "
                f"expected={expected_max_tokens}, "
                f"actual={actual_max_tokens}"
            )
        if method_type == "bbox_regression" and not bool(
            method_point_config.get("bbox_regression_aux", False)
        ):
            raise ValueError(f"{method_name} checkpoint has no bbox auxiliary head.")
        for expected_key, point_config_key, label in (
            (
                "expected_bbox_regression_aux",
                "bbox_regression_aux",
                "bbox regression auxiliary",
            ),
            (
                "expected_bbox_point_alignment_aux",
                "bbox_point_alignment_aux",
                "bbox point-alignment auxiliary",
            ),
        ):
            if expected_key not in method:
                continue
            expected_value = bool(method[expected_key])
            actual_value = bool(method_point_config.get(point_config_key, False))
            if actual_value != expected_value:
                raise ValueError(
                    f"{method_name} {label} mismatch: "
                    f"expected={expected_value}, actual={actual_value}"
                )
        if method_type in SCORER_METHOD_TYPES:
            scorer_path = absolute_path(method["scorer_path"])
            if not scorer_checkpoint_exists(scorer_path):
                raise FileNotFoundError(
                    f"Missing scorer checkpoint: {scorer_path}"
                )
            validate_scorer_point_module_lineage(
                method_name,
                method,
                scorer_path,
            )
            raw_threshold = int(
                method["scorer_raw_token_threshold_exclusive"]
            )
            raw_upper_threshold = method.get(
                "scorer_raw_token_upper_threshold_exclusive"
            )
            if raw_upper_threshold is not None:
                raw_upper_threshold = int(raw_upper_threshold)
            scorer_routing_scope = str(
                method.get("scorer_routing_scope", "region")
            )
            if scorer_routing_scope not in {"region", "scene_max"}:
                raise ValueError(
                    f"{method_name} scorer_routing_scope must be region or "
                    f"scene_max, got {scorer_routing_scope!r}."
                )
            scorer_max_keep = int(method.get("scorer_max_keep", 4096))
            if method_type == "all_region_scorer" and raw_threshold != 0:
                raise ValueError(
                    f"{method_name} type={method_type} must use "
                    "scorer_raw_token_threshold_exclusive=0."
                )
            if method_type == "conditional_scorer" and not (
                0 < raw_threshold <= scorer_max_keep
            ):
                raise ValueError(
                    f"{method_name} type={method_type} requires "
                    "0 < scorer_raw_token_threshold_exclusive <= "
                    f"scorer_max_keep ({scorer_max_keep}), got "
                    f"{raw_threshold}."
                )
            if (
                raw_upper_threshold is not None
                and raw_upper_threshold <= raw_threshold
            ):
                raise ValueError(
                    f"{method_name} requires "
                    "scorer_raw_token_upper_threshold_exclusive > "
                    "scorer_raw_token_threshold_exclusive, got "
                    f"{raw_upper_threshold} <= {raw_threshold}."
                )
            bypass_reference_method = method.get(
                "conditional_bypass_reference_method"
            )
            if (
                scorer_routing_scope == "scene_max"
                and bypass_reference_method is None
            ):
                raise ValueError(
                    f"{method_name} scene_max routing requires "
                    "conditional_bypass_reference_method."
                )
            if bypass_reference_method is not None:
                if method_type != "conditional_scorer":
                    raise ValueError(
                        f"{method_name} conditional bypass reuse requires "
                        "type=conditional_scorer."
                    )
                if bypass_reference_method not in methods:
                    raise ValueError(
                        f"{method_name} refers to unknown conditional bypass "
                        f"method {bypass_reference_method!r}."
                    )
                bypass_reference = methods[bypass_reference_method]
                if str(bypass_reference["type"]) != "plain":
                    raise ValueError(
                        f"{method_name} conditional bypass reference must be "
                        f"plain, got {bypass_reference['type']!r}."
                    )
                if absolute_path(bypass_reference["checkpoint"]) != absolute_path(
                    method["checkpoint"]
                ):
                    raise ValueError(
                        f"{method_name} conditional bypass reuse requires the "
                        "same Stage-2 checkpoint as its plain reference."
                    )
        if method_type in ATTENTION_SCORER_METHOD_TYPES:
            if region_source != "predicted":
                raise ValueError(
                    f"{method_name} attention_scorer requires predicted regions."
                )
            if str(method["attention_selection"]) != "hard_topk":
                raise ValueError(
                    f"{method_name} formal attention scorer must use hard_topk."
                )
            for key in (
                "attention_max_point_tokens",
                "attention_budget",
                "attention_top_k",
            ):
                if int(method[key]) <= 0:
                    raise ValueError(f"{method_name}.{key} must be positive.")
            if not bool(method["attention_disable_mha_fastpath"]):
                raise ValueError(
                    f"{method_name} must disable MHA fastpath for reproducibility."
                )
            validate_attention_scorer_pair(method_name, method)

    baseline_names = list(baselines)
    if selected_baselines is not None:
        unknown = sorted(set(selected_baselines) - set(baseline_names))
        if unknown:
            raise ValueError(f"Unknown --baselines entries: {unknown}")
        baseline_names = selected_baselines
    for baseline_name in baseline_names:
        baseline = baselines[baseline_name]

        class_aliases = baseline.get("class_aliases", {})
        if not isinstance(class_aliases, dict):
            raise ValueError(
                f"{baseline_name}.class_aliases must be a mapping."
            )
        for source, target in class_aliases.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"{baseline_name}.class_aliases contains an invalid "
                    f"source class: {source!r}."
                )
            if not isinstance(target, str) or not target.strip():
                raise ValueError(
                    f"{baseline_name}.class_aliases[{source!r}] must be a "
                    "non-empty string."
                )
        token_reference_method = baseline.get(
            "scene_token_bin_reference_method"
        )
        if (
            token_reference_method is not None
            and token_reference_method not in methods
        ):
            raise ValueError(
                f"{baseline_name}.scene_token_bin_reference_method refers "
                f"to unknown method {token_reference_method!r}."
            )
        prompt_source = str(baseline.get("prompt_source", "default"))
        if prompt_source not in {"default", "data_json"}:
            raise ValueError(
                f"{baseline_name}.prompt_source must be 'default' or 'data_json'."
            )
        baseline_data_json = absolute_path(
            baseline.get("data_json", experiment["data_json"])
        )
        if not baseline_data_json.is_file():
            raise FileNotFoundError(baseline_data_json)
        baseline_scene_ids = scene_ids_from_json(baseline_data_json)
        if set(baseline_scene_ids) != set(scene_ids):
            missing = sorted(set(scene_ids) - set(baseline_scene_ids))
            extra = sorted(set(baseline_scene_ids) - set(scene_ids))
            raise ValueError(
                f"{baseline_name} data_json scene set differs from the "
                "Stage-1 split: "
                f"stage1={len(scene_ids)}, baseline={len(baseline_scene_ids)}, "
                f"missing_from_baseline={missing[:5]}, "
                f"extra_in_baseline={extra[:5]}"
            )
        if bool(baseline.get("require_scene_order_match", False)) and (
            baseline_scene_ids != scene_ids
        ):
            raise ValueError(
                f"{baseline_name} data_json contains the same scenes as "
                "Stage 1 but in a different order while "
                "require_scene_order_match=true."
            )
        checkpoint = absolute_path(baseline["checkpoint"])
        baseline_config = checkpoint_config(checkpoint)
        point_config = baseline_config["point_config"]
        expected_num_bins = int(baseline["expected_num_bins"])
        actual_num_bins = int(point_config["num_bins"])
        expected_world_size = float(baseline["expected_world_size"])
        actual_world_size = float(point_config.get("world_size", 32.0))
        if (actual_num_bins, actual_world_size) != (
            expected_num_bins,
            expected_world_size,
        ):
            raise ValueError(
                f"{baseline_name} preprocessing/checkpoint mismatch: "
                f"expected=({expected_num_bins}, {expected_world_size}), "
                f"actual=({actual_num_bins}, {actual_world_size})"
            )
        if prompt_source == "default":
            code_template = absolute_path(baseline["code_template"])
            if not code_template.is_file():
                raise FileNotFoundError(code_template)
    return method_names, baseline_names, seeds, scene_ids


def parse_gpus(spec: str) -> list[str]:
    devices = [item.strip() for item in spec.split(",")]
    if not spec or not devices or any(not item for item in devices):
        raise ValueError(
            "--gpus (or CUDA_VISIBLE_DEVICES) must contain at least one GPU id."
        )
    return devices


def repeat_name(repeat_index: int, seed: int) -> str:
    return f"repeat_{repeat_index:02d}_seed_{seed}"


def select_repeats(
    seeds: list[int],
    selected_repeats: list[int] | None,
) -> list[tuple[int, int]]:
    if selected_repeats is None:
        selected_repeats = list(range(1, len(seeds) + 1))
    if len(selected_repeats) != len(set(selected_repeats)):
        raise ValueError("--repeats contains duplicate indices.")
    invalid = [
        repeat_index
        for repeat_index in selected_repeats
        if not 1 <= repeat_index <= len(seeds)
    ]
    if invalid:
        raise ValueError(
            f"--repeats must be between 1 and {len(seeds)}; got {invalid}"
        )
    return [
        (repeat_index, seeds[repeat_index - 1])
        for repeat_index in selected_repeats
    ]


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def child_environment(gpu: str, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run_shards(
    commands: list[list[str]],
    logs: list[Path],
    gpus: list[str],
    seed: int,
    dry_run: bool,
) -> None:
    for gpu, command, log_path in zip(gpus, commands, logs):
        print(
            f"[GPU {gpu}] {shell_join(command)} "
            f"> {shlex.quote(str(log_path))} 2>&1"
        )
    if dry_run:
        return

    processes: list[tuple[subprocess.Popen, Any, Path]] = []
    try:
        for gpu, command, log_path in zip(gpus, commands, logs):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            handle.write(f"$ {shell_join(command)}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=child_environment(gpu, seed),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, handle, log_path))
        failures = []
        for process, handle, log_path in processes:
            return_code = process.wait()
            handle.close()
            if return_code != 0:
                failures.append((return_code, log_path))
        if failures:
            details = ", ".join(
                f"{path} (exit={return_code})"
                for return_code, path in failures
            )
            raise RuntimeError(f"One or more inference shards failed: {details}")
    except BaseException:
        for process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            if not handle.closed:
                handle.close()
        raise


def run_logged(
    command: list[str],
    log_path: Path,
    dry_run: bool,
) -> None:
    print(f"{shell_join(command)} > {shlex.quote(str(log_path))} 2>&1")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shell_join(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit={result.returncode}; see {log_path}"
        )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def canonicalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if (
        normalized.get("format") == "one_stage_spatiallm_prediction_v1"
        and "prompt_source" not in normalized
    ):
        # Manifests created before prompt-source selection was introduced used
        # the default code-template prompt. Make that historical default
        # explicit for comparison without rewriting the existing artifact.
        normalized["prompt_source"] = "default"
    if (
        normalized.get("format") == "fixed_stage1_stage2_prediction_v1"
        and "stage1_prediction_dir" in normalized
        and "region_source" not in normalized
    ):
        # Fixed-Stage-1 manifests created before explicit region-source
        # provenance always used Stage-1 predictions. Normalize that legacy
        # field layout to the current schema while keeping all substantive
        # checkpoint, scorer, decoding, NMS, and scene-count guards strict.
        normalized["region_source"] = "predicted"
        normalized["region_source_dir"] = normalized.pop(
            "stage1_prediction_dir"
        )
        normalized["gt_region_data_json"] = None
    for optional_key in (
        "gt_region_data_json",
        "scorer_point_module_source_path",
    ):
        if normalized.get(optional_key) is None:
            normalized.pop(optional_key, None)
    return normalized


def ensure_manifest(path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        normalized_existing = canonicalize_manifest(existing)
        normalized_payload = canonicalize_manifest(payload)
        if normalized_existing != normalized_payload:
            differing_fields = sorted(
                key
                for key in set(normalized_existing) | set(normalized_payload)
                if (
                    key not in normalized_existing
                    or key not in normalized_payload
                    or normalized_existing[key] != normalized_payload[key]
                )
            )
            differing_values = {
                key: (
                    normalized_existing.get(key),
                    normalized_payload.get(key),
                )
                for key in differing_fields
            }
            raise ValueError(
                f"Provenance mismatch at {path}. Use a new output_root/method "
                "name instead of mixing checkpoints or decoding settings. "
                f"Differing top-level fields: {differing_fields}. "
                f"Existing/expected values: {differing_values}."
            )
        return
    if not dry_run:
        atomic_write_text(path, serialized)


def ensure_exact_txt_set(
    directory: Path,
    expected_scene_ids: list[str],
    label: str,
    suffix: str = ".txt",
) -> None:
    actual = {path.stem for path in directory.glob(f"*{suffix}")}
    expected = set(expected_scene_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"Incomplete {label}: expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def generation_args(config: dict[str, Any]) -> list[str]:
    generation = config["generation"]
    args = [
        "--inference_dtype",
        str(generation["inference_dtype"]),
        "--top_k",
        str(generation["top_k"]),
        "--top_p",
        str(generation["top_p"]),
        "--temperature",
        str(generation["temperature"]),
        "--num_beams",
        str(generation["num_beams"]),
        "--max_new_tokens",
        str(generation["max_new_tokens"]),
    ]
    if bool(generation["greedy"]):
        args.append("--greedy")
    if not bool(generation["deterministic"]):
        args.append("--no-deterministic")
    if bool(generation.get("no_cleanup", False)):
        args.append("--no_cleanup")
    return args


def common_eval_args(config: dict[str, Any]) -> list[str]:
    experiment = config["experiment"]
    evaluation = config["evaluation"]
    args = [
        "--gt_dir",
        str(absolute_path(evaluation["gt_dir"])),
        "--minimum_scale",
        str(evaluation["minimum_scale"]),
        "--missing_pred",
        str(evaluation["missing_pred"]),
        "--k",
        str(evaluation["region_k"]),
        "--expand_fraction",
        str(evaluation["region_expand_fraction"]),
    ]
    if experiment.get("metadata") is not None:
        args.extend(
            ["--metadata", str(absolute_path(experiment["metadata"]))]
        )
    if evaluation.get("gt_region_dir") is not None:
        args.extend(
            [
                "--gt_region_dir",
                str(absolute_path(evaluation["gt_region_dir"])),
            ]
        )
    region_iou_thresholds = evaluation.get("region_iou_thresholds")
    if region_iou_thresholds is not None:
        args.extend(
            [
                "--region_iou_thresholds",
                *[str(value) for value in region_iou_thresholds],
            ]
        )
    args.extend(object_class_args(config))
    return args


def object_class_args(config: dict[str, Any]) -> list[str]:
    evaluation = config["evaluation"]
    if bool(evaluation.get("no_label_mapping", False)):
        return [
            "--no_label_mapping",
            "--object_classes",
            *[str(item) for item in evaluation["object_classes"]],
        ]
    return [
        "--label_mapping",
        str(absolute_path(evaluation["label_mapping"])),
        "--label_from",
        str(evaluation["label_from"]),
        "--label_to",
        str(evaluation["label_to"]),
    ]


def result_table_log_root(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    return absolute_path(
        experiment.get("evaluation_log_root", experiment["log_root"])
    )


def stage1_phase(
    config: dict[str, Any],
    repeats: list[tuple[int, int]],
    scene_ids: list[str],
    gpus: list[str],
    dry_run: bool,
) -> None:
    experiment = config["experiment"]
    stage1 = config["stage1"]
    output_root = absolute_path(experiment["output_root"])
    log_root = absolute_path(experiment["log_root"])

    common_generation = generation_args(config)

    for repeat_index, seed in repeats:
        name = repeat_name(repeat_index, seed)
        prediction_dir = output_root / "predictions" / "stage1" / name
        manifest = {
            "format": "reusable_stage1_prediction_v1",
            "repeat_index": repeat_index,
            "seed": seed,
            "data_json": str(absolute_path(experiment["data_json"])),
            "dataset_root": str(absolute_path(experiment["dataset_root"])),
            "checkpoint": str(absolute_path(stage1["checkpoint"])),
            "num_bins": int(stage1["num_bins"]),
            "world_size": float(stage1["world_size"]),
            "prompt_source": str(stage1.get("prompt_source", "default")),
            "generation": config["generation"],
            "expected_scene_count": int(experiment["expected_scene_count"]),
        }
        if bool(experiment.get("lock_num_shards", False)):
            manifest["num_shards"] = len(gpus)
        ensure_manifest(
            prediction_dir / "run_manifest.json",
            manifest,
            dry_run,
        )

        commands = []
        logs = []
        for shard_index in range(len(gpus)):
            command = [
                sys.executable,
                str(REPO_ROOT / "inference.py"),
                "stage1",
                "--data_json",
                str(absolute_path(experiment["data_json"])),
                "--dataset_root",
                str(absolute_path(experiment["dataset_root"])),
                "--output_dir",
                str(prediction_dir),
                "--stage1_model_path",
                str(absolute_path(stage1["checkpoint"])),
                "--seed",
                str(seed),
                "--num_shards",
                str(len(gpus)),
                "--shard_index",
                str(shard_index),
                "--skip_existing",
                *common_generation,
            ]
            if str(stage1.get("prompt_source", "default")) == "data_json":
                command.append("--prompt_from_data_json")
            commands.append(command)
            logs.append(
                log_root
                / "inference"
                / f"stage1_{name}_shard{shard_index}.log"
            )
        run_shards(commands, logs, gpus, seed, dry_run)
        if not dry_run:
            ensure_exact_txt_set(prediction_dir, scene_ids, f"Stage 1 {name}")


def stage1_eval_phase(
    config: dict[str, Any],
    repeats: list[tuple[int, int]],
    scene_ids: list[str],
    dry_run: bool,
) -> None:
    experiment = config["experiment"]
    output_root = absolute_path(experiment["output_root"])
    log_root = result_table_log_root(config)

    stage1_artifact_name = str(
        config["stage1"].get("artifact_name", "stage1")
    )
    for repeat_index, seed in repeats:
        name = repeat_name(repeat_index, seed)
        prediction_dir = output_root / "predictions" / "stage1" / name
        if not dry_run:
            ensure_exact_txt_set(prediction_dir, scene_ids, f"Stage 1 {name}")
        output_json = (
            output_root
            / "evaluation"
            / "stage1"
            / f"{name}.json"
        )
        command = [
            sys.executable,
            str(REPO_ROOT / "eval.py"),
            "hierarchical",
            *common_eval_args(config),
            "--layout_pred_dir",
            str(prediction_dir),
            "--stage1_pred_dir",
            str(prediction_dir),
            "--skip_objects",
            "--output_json",
            str(output_json),
        ]
        if bool(config["evaluation"].get("stage1_skip_layout", False)):
            command.append("--skip_layout")
        log_path = (
            log_root
            / f"{stage1_artifact_name}_{name}.log"
        )
        run_logged(command, log_path, dry_run)


def baseline_phase(
    config: dict[str, Any],
    baseline_names: list[str],
    repeats: list[tuple[int, int]],
    scene_ids: list[str],
    gpus: list[str],
    dry_run: bool,
) -> None:
    experiment = config["experiment"]
    output_root = absolute_path(experiment["output_root"])
    log_root = absolute_path(experiment["log_root"])
    common_generation = generation_args(config)

    for baseline_name in baseline_names:
        baseline = config["one_stage_baselines"][baseline_name]

        baseline_artifact_name = baseline_name
        baseline_data_json = absolute_path(
            baseline.get("data_json", experiment["data_json"])
        )
        for repeat_index, seed in repeats:
            name = repeat_name(repeat_index, seed)
            prediction_dir = (
                output_root
                / "predictions"
                / "one_stage"
                / baseline_name
                / name
                / "raw"
            )
            manifest = {
                "format": "one_stage_spatiallm_prediction_v1",
                "baseline_name": baseline_name,
                "repeat_index": repeat_index,
                "seed": seed,
                "data_json": str(baseline_data_json),
                "dataset_root": str(absolute_path(experiment["dataset_root"])),
                "source_model": str(baseline["source_model"]),
                "checkpoint": str(absolute_path(baseline["checkpoint"])),
                "code_template": (
                    str(absolute_path(baseline["code_template"]))
                    if baseline.get("code_template") is not None
                    else None
                ),
                "prompt_source": str(
                    baseline.get("prompt_source", "default")
                ),
                "class_aliases": dict(baseline.get("class_aliases", {})),
                "generation": config["generation"],
                "expected_scene_count": int(experiment["expected_scene_count"]),
            }
            if bool(experiment.get("lock_num_shards", False)):
                manifest["num_shards"] = len(gpus)
            ensure_manifest(
                prediction_dir
                / "run_manifest.json",
                manifest,
                dry_run,
            )

            commands = []
            logs = []
            for shard_index in range(len(gpus)):
                command = [
                    sys.executable,
                    str(REPO_ROOT / "inference.py"),
                    "one_stage",
                    "--data_json",
                    str(baseline_data_json),
                    "--dataset_root",
                    str(absolute_path(experiment["dataset_root"])),
                    "--output_dir",
                    str(prediction_dir),
                    "--model_path",
                    str(absolute_path(baseline["checkpoint"])),
                    "--seed",
                    str(seed),
                    "--num_shards",
                    str(len(gpus)),
                    "--shard_index",
                    str(shard_index),
                    "--skip_existing",
                    *common_generation,
                ]
                if str(baseline.get("prompt_source", "default")) == "data_json":
                    command.append("--prompt_from_data_json")
                else:
                    command.extend(
                        [
                            "--code_template",
                            str(absolute_path(baseline["code_template"])),
                        ]
                    )
                commands.append(command)
                logs.append(
                    log_root
                    / "inference"
                    / (
                        f"{baseline_artifact_name}_{name}_"
                        f"shard{shard_index}.log"
                    )
                )
            run_shards(commands, logs, gpus, seed, dry_run)
            if not dry_run:
                ensure_exact_txt_set(
                    prediction_dir,
                    scene_ids,
                    f"{baseline_name} {name} raw predictions",
                )


def baseline_eval_phase(
    config: dict[str, Any],
    baseline_names: list[str],
    repeats: list[tuple[int, int]],
    scene_ids: list[str],
    dry_run: bool,
) -> None:
    experiment = config["experiment"]
    evaluation = config["evaluation"]
    postprocess = config["postprocess"]
    output_root = absolute_path(experiment["output_root"])
    process_log_root = absolute_path(experiment["log_root"])
    evaluation_log_root = result_table_log_root(config)
    nms_iou = str(postprocess["nms_iou"])

    for baseline_name in baseline_names:
        baseline = config["one_stage_baselines"][baseline_name]

        baseline_artifact_name = baseline_name
        for repeat_index, seed in repeats:
            name = repeat_name(repeat_index, seed)
            raw_dir = (
                output_root
                / "predictions"
                / "one_stage"
                / baseline_name
                / name
                / "raw"
            )
            if not dry_run:
                ensure_exact_txt_set(
                    raw_dir,
                    scene_ids,
                    f"{baseline_name} {name} raw predictions",
                )
            nms_dir = raw_dir.parent / f"NMS_{nms_iou}"
            nms_command = [
                sys.executable,
                str(REPO_ROOT / "eval.py"),
                "nms",
                "--input_dir",
                str(raw_dir),
                "--output_dir",
                str(nms_dir),
                "--iou_threshold",
                nms_iou,
                "--minimum_scale",
                str(postprocess["nms_minimum_scale"]),
            ]
            for source, target in baseline.get("class_aliases", {}).items():
                nms_command.extend(
                    ["--class_alias", f"{source}={target}"]
                )
            nms_log = (
                process_log_root
                / "nms"
                / (
                    f"{baseline_artifact_name}_{name}_"
                    f"NMS_{nms_iou}.log"
                )
            )
            run_logged(nms_command, nms_log, dry_run)
            if not dry_run:
                ensure_exact_txt_set(
                    nms_dir,
                    scene_ids,
                    f"{baseline_name} {name} NMS predictions",
                )

            output_json = (
                output_root
                / "evaluation"
                / "one_stage"
                / baseline_name
                / f"{name}.json"
            )
            eval_command = [
                sys.executable,
                str(REPO_ROOT / "eval.py"),
                "hierarchical",
                *common_eval_args(config),
                "--layout_pred_dir",
                str(nms_dir),
                "--object_pred_dir",
                str(nms_dir),
                "--skip_regions",
                "--output_json",
                str(output_json),
            ]
            if bool(
                baseline.get(
                    "skip_layout",
                    config["evaluation"].get(
                        "one_stage_skip_layout",
                        False,
                    ),
                )
            ):
                eval_command.append("--skip_layout")
            eval_log = (
                evaluation_log_root
                / f"{baseline_artifact_name}_{name}.log"
            )
            run_logged(eval_command, eval_log, dry_run)

            token_reference_method = baseline.get(
                "scene_token_bin_reference_method"
            )
            if token_reference_method is None:
                continue
            token_reference_raw_dir = (
                output_root
                / "predictions"
                / "stage2"
                / str(token_reference_method)
                / name
                / "raw"
            )
            if not dry_run:
                ensure_exact_txt_set(
                    token_reference_raw_dir / "scene_region_index",
                    scene_ids,
                    (
                        f"{baseline_name} {name} token-reference "
                        "region indexes"
                    ),
                    suffix=".json",
                )
            token_bin_output_dir = (
                output_root
                / "evaluation"
                / "one_stage"
                / baseline_name
                / "token_bins"
                / f"{name}"
            )
            token_bin_command = [
                sys.executable,
                str(REPO_ROOT / "eval.py"),
                "token_bins",
                "--raw_prediction_dir",
                str(token_reference_raw_dir),
                "--scene_prediction_dir",
                str(nms_dir),
                "--stage1_pred_dir",
                str(
                    output_root
                    / "predictions"
                    / "stage1"
                    / name
                ),
                "--gt_dir",
                str(absolute_path(evaluation["gt_dir"])),
                "--output_dir",
                str(token_bin_output_dir),
                "--method_name",
                baseline_name,
                "--token_reference_method_name",
                str(token_reference_method),
                "--scene_only",
                "--repeat_index",
                str(repeat_index),
                "--seed",
                str(seed),
                "--token_boundaries",
                *[
                    str(value)
                    for value in evaluation["region_token_boundaries"]
                ],
                "--minimum_scale",
                str(evaluation["minimum_scale"]),
                "--expected_scene_count",
                str(experiment["expected_scene_count"]),
                "--expected_region_nms_iou",
                str(postprocess["nms_iou"]),
                *object_class_args(config),
            ]
            for source, target in baseline.get("class_aliases", {}).items():
                token_bin_command.extend(["--class_alias", f"{source}={target}"])
            if experiment.get("metadata") is not None:
                token_bin_command.extend(
                    [
                        "--metadata",
                        str(absolute_path(experiment["metadata"])),
                    ]
                )
            token_bin_log = (
                process_log_root
                / "evaluation"
                / (
                    f"{baseline_artifact_name}_{name}_"
                    "scene_token_bins.log"
                )
            )
            run_logged(
                token_bin_command,
                token_bin_log,
                dry_run,
            )


def stage2_phase(
    config: dict[str, Any],
    method_names: list[str],
    repeats: list[tuple[int, int]],
    scene_ids: list[str],
    gpus: list[str],
    dry_run: bool,
) -> None:
    experiment = config["experiment"]
    stage1 = config["stage1"]
    output_root = absolute_path(experiment["output_root"])
    log_root = absolute_path(experiment["log_root"])
    common_generation = generation_args(config)

    for method_name in method_names:
        method = config["methods"][method_name]
        method_generation = dict(config['generation'])
        if 'no_cleanup' in method:
            method_generation['no_cleanup'] = method['no_cleanup']
        common_generation = generation_args({'generation': method_generation})

        method_artifact_name = method_name
        method_type = str(method["type"])
        region_source = str(method.get("region_source", "predicted"))
        for repeat_index, seed in repeats:
            name = repeat_name(repeat_index, seed)
            if region_source == "gt_region":
                stage1_dir = absolute_path(
                    config["evaluation"]["gt_region_dir"]
                )
            else:
                stage1_dir = output_root / "predictions" / "stage1" / name
            if not dry_run:
                ensure_exact_txt_set(
                    stage1_dir,
                    scene_ids,
                    (
                        f"GT regions for {method_name}"
                        if region_source == "gt_region"
                        else f"Stage 1 {name}"
                    ),
                )
            prediction_dir = (
                output_root
                / "predictions"
                / "stage2"
                / method_name
                / name
                / "raw"
            )
            manifest = {
                "format": "fixed_stage1_stage2_prediction_v1",
                "method_name": method_name,
                "method_type": method_type,
                "repeat_index": repeat_index,
                "seed": seed,
                "stage2_checkpoint": str(absolute_path(method["checkpoint"])),
                "scorer_path": (
                    str(absolute_path(method["scorer_path"]))
                    if method_type in SCORER_METHOD_TYPES
                    else None
                ),
                "scorer": (
                    {
                        key: method[key]
                        for key in (
                            "scorer_threshold",
                            "scorer_min_keep",
                            "scorer_max_keep",
                            "scorer_raw_token_threshold_exclusive",
                            "scorer_disable_mha_fastpath",
                        )
                    }
                    | {
                        "scorer_raw_token_upper_threshold_exclusive": (
                            method.get(
                                "scorer_raw_token_upper_threshold_exclusive"
                            )
                        ),
                        "scorer_routing_scope": method.get(
                            "scorer_routing_scope",
                            "region",
                        ),
                    }
                    if method_type in SCORER_METHOD_TYPES
                    else None
                ),
                "attention_scorer_path": (
                    str(absolute_path(method["attention_scorer_path"]))
                    if method_type in ATTENTION_SCORER_METHOD_TYPES
                    else None
                ),
                "attention_scorer": (
                    {
                        key: method[key]
                        for key in (
                            "attention_selection",
                            "attention_max_point_tokens",
                            "attention_budget",
                            "attention_top_k",
                            "attention_disable_mha_fastpath",
                        )
                    }
                    | {
                        "attention_require_paired_scorer": bool(
                            method.get(
                                "attention_require_paired_scorer",
                                True,
                            )
                        ),
                        "attention_expected_scorer_step": method.get(
                            "attention_expected_scorer_step"
                        ),
                    }
                    if method_type in ATTENTION_SCORER_METHOD_TYPES
                    else None
                ),
                "generation": config["generation"],
                "region_bbox_nms_iou": config["postprocess"]["nms_iou"],
                "class_aliases": dict(method.get("class_aliases", {})),
                "region_bbox_nms_minimum_scale": config["postprocess"][
                    "nms_minimum_scale"
                ],
                "expected_scene_count": int(experiment["expected_scene_count"]),
            }
            if (
                method_type in SCORER_METHOD_TYPES
                and method.get("scorer_point_module_source_path")
            ):
                manifest["scorer_point_module_source_path"] = str(
                    absolute_path(
                        method["scorer_point_module_source_path"]
                    )
                )
            if method.get('region_context_padding', 0) or method.get('region_core_filter', False):
                manifest['region_context_padding'] = float(method.get('region_context_padding', 0))
                manifest['region_core_filter'] = bool(method.get('region_core_filter', False))
                manifest['region_context_units'] = 'world_meters'
            if 'no_cleanup' in method:
                manifest['generation'] = dict(config['generation'], no_cleanup=method['no_cleanup'])
            bypass_reference_method = method.get(
                "conditional_bypass_reference_method"
            )
            if bypass_reference_method is not None:
                bypass_reference = config["methods"][
                    bypass_reference_method
                ]
                bypass_artifact_name = bypass_reference_method
                bypass_raw_prediction_dir = (
                    output_root
                    / "predictions"
                    / "stage2"
                    / bypass_artifact_name
                    / name
                    / "raw"
                )
                if not bypass_raw_prediction_dir.is_dir() and not dry_run:
                    raise FileNotFoundError(bypass_raw_prediction_dir)
                manifest["conditional_bypass_cache"] = {
                    "reference_method": bypass_reference_method,
                    "raw_prediction_dir": str(bypass_raw_prediction_dir),
                    "equivalence": (
                        "same_stage2_checkpoint_seed_and_region_postprocess"
                    ),
                }
            if region_source == "gt_region":
                manifest.update(
                    {
                        "region_source": region_source,
                        "region_source_dir": str(stage1_dir),
                    }
                )
                if method_type == "plain":
                    # Oracle changes only region geometry: keep the exact
                    # full-scene cleanup/cropping/decoding path used by plain
                    # predicted-region inference, not separately saved crops.
                    manifest["gt_region_pipeline"] = "scene_crop"
                else:
                    manifest["gt_region_data_json"] = str(
                        absolute_path(method["gt_region_data_json"])
                    )
            else:
                # Preserve the established manifest schema so existing
                # predicted-region artifacts remain resumable.
                manifest["stage1_prediction_dir"] = str(stage1_dir)
            if method.get("checkpoint_selection") is not None:
                manifest["checkpoint_selection"] = method[
                    "checkpoint_selection"
                ]
            if bool(
                experiment.get("lock_num_shards", False)
                or method.get("lock_num_shards", False)
            ):
                manifest["num_shards"] = len(gpus)
            ensure_manifest(
                prediction_dir / "run_manifest.json",
                manifest,
                dry_run,
            )

            commands = []
            logs = []
            for shard_index in range(len(gpus)):
                if region_source == "gt_region" and method_type != "plain":
                    command = [
                        sys.executable,
                        str(
                            REPO_ROOT
                            / "inference.py"
                        ),
                        "stage2",
                        "--data_json",
                        str(absolute_path(method["gt_region_data_json"])),
                        "--scene_data_json",
                        str(absolute_path(experiment["data_json"])),
                        "--dataset_root",
                        str(absolute_path(experiment["dataset_root"])),
                        "--gt_region_dir",
                        str(stage1_dir),
                        "--output_dir",
                        str(prediction_dir),
                        "--expected_scene_count",
                        str(experiment["expected_scene_count"]),
                        "--stage2_model_path",
                        str(absolute_path(method["checkpoint"])),
                        "--region_bbox_nms_iou",
                        str(config["postprocess"]["nms_iou"]),
                        "--region_bbox_nms_minimum_scale",
                        str(config["postprocess"]["nms_minimum_scale"]),
                        "--seed",
                        str(seed),
                        "--num_shards",
                        str(len(gpus)),
                        "--shard_index",
                        str(shard_index),
                        "--skip_existing",
                        *common_generation,
                    ]
                else:
                    command = [
                        sys.executable,
                        str(REPO_ROOT / "inference.py"),
                        "stage2",
                        "--data_json",
                        str(absolute_path(experiment["data_json"])),
                        "--dataset_root",
                        str(absolute_path(experiment["dataset_root"])),
                        "--stage1_pred_dir",
                        str(stage1_dir),
                        "--output_dir",
                        str(prediction_dir),
                        "--method",
                        method_type,
                        "--stage2_model_path",
                        str(absolute_path(method["checkpoint"])),
                        "--stage1_num_bins",
                        str(stage1["num_bins"]),
                        "--stage1_world_size",
                        str(stage1["world_size"]),
                        "--min_region_points",
                        str(config["generation"]["min_region_points"]),
                        "--region_bbox_nms_iou",
                        str(config["postprocess"]["nms_iou"]),
                        "--region_bbox_nms_minimum_scale",
                        str(config["postprocess"]["nms_minimum_scale"]),
                        "--seed",
                        str(seed),
                        "--num_shards",
                        str(len(gpus)),
                        "--shard_index",
                        str(shard_index),
                        "--skip_existing",
                        *common_generation,
                    ]
                if method.get('region_context_padding', 0):
                    command.extend(['--region_context_padding', str(method['region_context_padding'])])
                if method.get('region_core_filter', False):
                    command.append('--region_core_filter')
                if method_type in SCORER_METHOD_TYPES:
                    command.extend(
                        [
                            "--scorer_path",
                            str(absolute_path(method["scorer_path"])),
                            "--scorer_threshold",
                            str(method["scorer_threshold"]),
                            "--scorer_min_keep",
                            str(method["scorer_min_keep"]),
                            "--scorer_max_keep",
                            str(method["scorer_max_keep"]),
                            "--scorer_raw_token_threshold_exclusive",
                            str(
                                method[
                                    "scorer_raw_token_threshold_exclusive"
                                ]
                            ),
                        ]
                    )
                    if not bool(method["scorer_disable_mha_fastpath"]):
                        command.append("--no-scorer_disable_mha_fastpath")
                    raw_upper_threshold = method.get(
                        "scorer_raw_token_upper_threshold_exclusive"
                    )
                    if raw_upper_threshold is not None:
                        command.extend(
                            [
                                "--scorer_raw_token_upper_threshold_exclusive",
                                str(raw_upper_threshold),
                            ]
                        )
                    command.extend(
                        [
                            "--scorer_routing_scope",
                            str(method.get("scorer_routing_scope", "region")),
                        ]
                    )
                    if bypass_reference_method is not None:
                        command.extend(
                            [
                                "--conditional_bypass_raw_prediction_dir",
                                str(bypass_raw_prediction_dir),
                            ]
                        )
                if method_type in ATTENTION_SCORER_METHOD_TYPES:
                    command.extend(
                        [
                            "--attention_scorer_path",
                            str(
                                absolute_path(
                                    method["attention_scorer_path"]
                                )
                            ),
                            "--attention_selection",
                            str(method["attention_selection"]),
                            "--attention_max_point_tokens",
                            str(method["attention_max_point_tokens"]),
                            "--attention_budget",
                            str(method["attention_budget"]),
                            "--attention_top_k",
                            str(method["attention_top_k"]),
                        ]
                    )
                    if not bool(method["attention_disable_mha_fastpath"]):
                        command.append(
                            "--no-attention_disable_mha_fastpath"
                        )
                commands.append(command)
                logs.append(
                    log_root
                    / "inference"
                    / (
                        f"{method_artifact_name}_{name}_"
                        f"shard{shard_index}.log"
                    )
                )
            run_shards(commands, logs, gpus, seed, dry_run)
            if not dry_run:
                ensure_exact_txt_set(
                    prediction_dir / "stage1",
                    scene_ids,
                    f"{method_name} {name} copied region source",
                )
                ensure_exact_txt_set(
                    prediction_dir / "final",
                    scene_ids,
                    f"{method_name} {name} raw final",
                )
                if region_source == "predicted" or method_type == "plain":
                    ensure_exact_txt_set(
                        prediction_dir / "scene_region_index",
                        scene_ids,
                        f"{method_name} {name} region indexes",
                        suffix=".json",
                    )


def stage2_eval_phase(
    config: dict[str, Any],
    method_names: list[str],
    repeats: list[tuple[int, int]],
    scene_ids: list[str],
    dry_run: bool,
) -> None:
    experiment = config["experiment"]
    evaluation = config["evaluation"]
    postprocess = config["postprocess"]
    output_root = absolute_path(experiment["output_root"])
    process_log_root = absolute_path(experiment["log_root"])
    evaluation_log_root = result_table_log_root(config)
    nms_iou = str(postprocess["nms_iou"])

    for method_name in method_names:
        method = config["methods"][method_name]

        method_artifact_name = method_name
        skip_token_bins = bool(method.get("skip_token_bins", False))
        for repeat_index, seed in repeats:
            name = repeat_name(repeat_index, seed)
            raw_dir = (
                output_root
                / "predictions"
                / "stage2"
                / method_name
                / name
                / "raw"
            )
            if not dry_run:
                ensure_exact_txt_set(
                    raw_dir / "stage1",
                    scene_ids,
                    f"{method_name} {name} copied Stage 1",
                )
                ensure_exact_txt_set(
                    raw_dir / "final",
                    scene_ids,
                    f"{method_name} {name} raw final",
                )
            nms_dir = raw_dir.parent / f"NMS_{nms_iou}"
            nms_command = [
                sys.executable,
                str(REPO_ROOT / "eval.py"),
                "nms",
                "--input_dir",
                str(raw_dir),
                "--output_dir",
                str(nms_dir),
                "--iou_threshold",
                nms_iou,
                "--minimum_scale",
                str(postprocess["nms_minimum_scale"]),
            ]
            nms_log = (
                process_log_root
                / "nms"
                / (
                    f"{method_artifact_name}_{name}_"
                    f"NMS_{nms_iou}.log"
                )
            )
            for source, target in method.get("class_aliases", {}).items():
                nms_command.extend(["--class_alias", f"{source}={target}"])
            run_logged(nms_command, nms_log, dry_run)
            if not dry_run:
                ensure_exact_txt_set(
                    nms_dir / "final",
                    scene_ids,
                    f"{method_name} {name} NMS final",
                )

            output_json = (
                output_root
                / "evaluation"
                / "stage2"
                / method_name
                / f"{name}.json"
            )
            eval_command = [
                sys.executable,
                str(REPO_ROOT / "eval.py"),
                "hierarchical",
                *common_eval_args(config),
                "--object_pred_dir",
                str(nms_dir / "final"),
                "--skip_layout",
                "--skip_regions",
                "--output_json",
                str(output_json),
            ]
            eval_log = (
                evaluation_log_root
                / f"{method_artifact_name}_{name}.log"
            )
            run_logged(eval_command, eval_log, dry_run)

            if skip_token_bins:
                continue

            token_bin_output_dir = (
                output_root
                / "evaluation"
                / "stage2"
                / method_name
                / "token_bins"
                / f"{name}"
            )
            token_bin_command = [
                sys.executable,
                str(REPO_ROOT / "eval.py"),
                "token_bins",
                "--raw_prediction_dir",
                str(raw_dir),
                "--scene_prediction_dir",
                str(nms_dir / "final"),
                "--stage1_pred_dir",
                str(
                    output_root
                    / "predictions"
                    / "stage1"
                    / name
                ),
                "--gt_dir",
                str(absolute_path(evaluation["gt_dir"])),
                "--output_dir",
                str(token_bin_output_dir),
                "--method_name",
                method_name,
                "--repeat_index",
                str(repeat_index),
                "--seed",
                str(seed),
                "--token_boundaries",
                *[
                    str(value)
                    for value in evaluation["region_token_boundaries"]
                ],
                "--minimum_scale",
                str(evaluation["minimum_scale"]),
                "--expected_scene_count",
                str(experiment["expected_scene_count"]),
                "--expected_region_nms_iou",
                str(postprocess["nms_iou"]),
                *object_class_args(config),
            ]
            if experiment.get("metadata") is not None:
                token_bin_command.extend(
                    [
                        "--metadata",
                        str(absolute_path(experiment["metadata"])),
                    ]
                )
            token_bin_log = (
                process_log_root
                / "evaluation"
                / (
                    f"{method_artifact_name}_{name}_"
                    "token_bins.log"
                )
            )
            for source, target in method.get("class_aliases", {}).items():
                token_bin_command.extend(["--class_alias", f"{source}={target}"])
            run_logged(
                token_bin_command,
                token_bin_log,
                dry_run,
            )


def aggregate_phase(config_path: Path, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "eval.py"),
        "aggregate",
        "--config",
        str(config_path.resolve()),
    ]
    config, _ = load_config(config_path)
    experiment = config["experiment"]

    log_path = (
        absolute_path(experiment["log_root"])
        / "evaluation"
        / (
            f"{experiment['name']}_"
            "aggregate.log"
        )
    )
    run_logged(command, log_path, dry_run)


def snapshot_config(
    config: dict[str, Any],
    config_text: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    experiment = config["experiment"]
    digest = hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12]
    output_path = (
        absolute_path(experiment["output_root"])
        / "manifests"
        / (
            "config_"
            f"{digest}.yaml"
        )
    )
    if not output_path.exists():
        atomic_write_text(output_path, config_text)


def main() -> int:
    args = parse_args()
    config, config_text = load_config(args.config)

    phases = args.phase
    if "all" in phases:
        if len(phases) != 1:
            raise ValueError("--phase all cannot be combined with other phases.")
        phases = list(PHASE_ORDER)
    if "validate" in phases and len(phases) != 1:
        raise ValueError("--phase validate cannot be combined with other phases.")

    method_names, baseline_names, seeds, scene_ids = validate_config(
        config,
        args.methods,
        args.baselines,
        require_stage2_methods=bool(
            {"stage2", "stage2_eval"}.intersection(phases)
        ),
    )
    repeats = select_repeats(seeds, args.repeats)

    if "validate" in phases:
        print(
            "Validation passed: "
            f"scenes={len(scene_ids)}, repeats={repeats}, "
            f"methods={method_names}, baselines={baseline_names}"
        )
        return 0

    gpu_phases = {"stage1", "baseline", "stage2"}
    gpus = parse_gpus(args.gpus) if gpu_phases.intersection(phases) else []
    snapshot_config(config, config_text, args.dry_run)
    print(
        "Comparison setup: "
        f"phases={phases}, repeats={repeats}, methods={method_names}, "
        f"baselines={baseline_names}, "
        f"gpus={gpus or 'not needed'}, dry_run={args.dry_run}"
    )

    for phase in PHASE_ORDER:
        if phase not in phases:
            continue
        print(f"\n=== {phase} ===")
        if phase == "stage1":
            stage1_phase(
                config,
                repeats,
                scene_ids,
                gpus,
                args.dry_run,
            )
        elif phase == "stage1_eval":
            stage1_eval_phase(config, repeats, scene_ids, args.dry_run)
        elif phase == "baseline":
            baseline_phase(
                config,
                baseline_names,
                repeats,
                scene_ids,
                gpus,
                args.dry_run,
            )
        elif phase == "baseline_eval":
            baseline_eval_phase(
                config,
                baseline_names,
                repeats,
                scene_ids,
                args.dry_run,
            )
        elif phase == "stage2":
            stage2_phase(
                config,
                method_names,
                repeats,
                scene_ids,
                gpus,
                args.dry_run,
            )
        elif phase == "stage2_eval":
            stage2_eval_phase(
                config,
                method_names,
                repeats,
                scene_ids,
                args.dry_run,
            )
        elif phase == "aggregate":
            aggregate_phase(args.config, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
