#!/usr/bin/env python3
"""Run Stage 2 from an immutable directory of Stage-1 predictions."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from tqdm import tqdm
from transformers import set_seed

from apply_bbox_nms import classwise_nms as postprocess_classwise_nms
from build_hierarchical_region_dataset import STAGE2_PROMPT
from inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    center_crop_point_arrays,
    decode_bbox_regression_layout,
    decode_generated_layout,
    final_layout_from_parts,
    generate_layout_text,
    generate_layout_text_with_bbox_regression,
    load_model_and_tokenizer,
    load_scenes,
    model_world_size,
    points_in_region,
    prepare_point_arrays,
    prepare_scene_point_cloud,
    prompt_with_point_token,
)
from inference_hierarchical_scorer_rawgt1024_conditional import (
    install_conditional_scorer_point_filter,
    load_point_token_scorer,
    print_conditional_summary,
)
from inference_hierarchical_attention_scorer import (
    install_attention_scorer_bias,
    install_attention_scorer_topk,
    load_attention_scorer,
    validate_checkpoint_args,
)
from spatiallm import Layout
from spatiallm.layout.entity import Bbox


SCORER_METHODS = ("conditional_scorer", "all_region_scorer")
ATTENTION_SCORER_METHODS = ("attention_scorer",)
METHODS = (
    "plain",
    *SCORER_METHODS,
    *ATTENTION_SCORER_METHODS,
    "bbox_regression",
)


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


def region_with_context(region, padding: float):
    """Add an input-only halo in world meters; never change the target ROI."""
    if not math.isfinite(padding) or padding < 0:
        raise ValueError('Region context padding must be finite and non-negative')
    if padding == 0:
        return region
    return replace(region, **{f'scale_{axis}': getattr(region, f'scale_{axis}') + 2 * padding
                              for axis in 'xyz'})


def filter_bboxes_to_region_core(bboxes, region):
    """Match training's center-in-region target convention without reading GT."""
    return [bbox for bbox in bboxes if all(
        abs(getattr(bbox, f'position_{axis}') - getattr(region, f'position_{axis}'))
        <= getattr(region, f'scale_{axis}') / 2 + 1e-6 for axis in 'xyz')]


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    set_seed(seed)
    if not deterministic:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def install_point_token_count_recorder(model) -> None:
    """Record Sonata output length before model-side max-token cropping."""

    original_forward_point_cloud = model.forward_point_cloud
    model._fixed_region_last_raw_point_token_count = None
    model._fixed_region_last_kept_point_token_count = None

    def record_raw_token_count(_module, _inputs, output):
        context = output.get("context") if isinstance(output, dict) else output
        if not torch.is_tensor(context) or context.ndim < 2:
            raise TypeError(
                "Cannot determine raw point-token count from point-backbone "
                f"output type={type(output).__name__}."
            )
        model._fixed_region_last_raw_point_token_count = int(context.shape[0])

    model._fixed_region_point_backbone_hook = (
        model.point_backbone.register_forward_hook(record_raw_token_count)
    )

    def counted_forward_point_cloud(
        self,
        point_cloud: torch.Tensor,
        device,
        dtype,
        point_token_keep_bboxes=None,
        return_grid_coord: bool = False,
    ):
        result = original_forward_point_cloud(
            point_cloud,
            device,
            dtype,
            point_token_keep_bboxes,
            return_grid_coord=return_grid_coord,
        )
        point_tokens = result[0] if return_grid_coord else result
        self._fixed_region_last_kept_point_token_count = int(
            point_tokens.shape[1]
        )
        return result

    model.forward_point_cloud = MethodType(counted_forward_point_cloud, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one Stage-2 method on already-persisted Stage-1 "
            "regions. No Stage-1 model weights are loaded."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data_json", type=Path)
    input_group.add_argument("--point_cloud", type=Path)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage1_pred_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--stage2_model_path", required=True)
    parser.add_argument("--stage1_num_bins", type=int, default=1280)
    parser.add_argument("--stage1_world_size", type=float, default=32.0)
    parser.add_argument("--scorer_path", type=Path)
    parser.add_argument("--scorer_threshold", type=float, default=0.5)
    parser.add_argument("--scorer_max_keep", type=int, default=4096)
    parser.add_argument("--scorer_min_keep", type=int, default=1)
    parser.add_argument(
        "--scorer_raw_token_threshold_exclusive",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--scorer_raw_token_upper_threshold_exclusive",
        type=int,
        help=(
            "Optional exclusive upper raw-token routing bound. When set, "
            "conditional_scorer filters only regions strictly between the "
            "lower and upper bounds."
        ),
    )
    parser.add_argument(
        "--scorer_routing_scope",
        choices=("region", "scene_max"),
        default="region",
        help=(
            "Apply the raw-token routing window independently per region, or "
            "route the whole scene from the maximum cached region token count. "
            "scene_max requires --conditional_bypass_raw_prediction_dir."
        ),
    )
    parser.add_argument(
        "--scorer_disable_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--conditional_bypass_raw_prediction_dir",
        type=Path,
        help=(
            "Optional raw prediction directory from the identical Stage-2 "
            "checkpoint/seed. For conditional_scorer only, regions whose "
            "recorded raw token count is outside the configured routing "
            "window reuse these cached predictions exactly; filtered regions "
            "are still inferred normally."
        ),
    )
    parser.add_argument("--scorer_debug", action="store_true")
    parser.add_argument("--attention_scorer_path", type=Path)
    parser.add_argument(
        "--attention_selection",
        choices=("soft_bias", "hard_topk"),
        default="hard_topk",
    )
    parser.add_argument(
        "--attention_max_point_tokens",
        type=int,
        default=3200,
    )
    parser.add_argument("--attention_budget", type=int, default=512)
    parser.add_argument("--attention_top_k", type=int, default=512)
    parser.add_argument(
        "--attention_disable_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attention_debug", action="store_true")
    parser.add_argument("--inference_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no_cleanup", action="store_true")
    parser.add_argument("--min_region_points", type=int, default=1)
    parser.add_argument("--region_context_padding", type=float, default=0.0,
                        help="Input-only context halo on every ROI face, in world meters.")
    parser.add_argument("--region_core_filter", action="store_true",
                        help="Keep only decoded bboxes whose centers lie in the original ROI.")
    parser.add_argument("--region_bbox_nms_iou", type=float, default=0.1)
    parser.add_argument(
        "--region_bbox_nms_minimum_scale",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.stage1_num_bins <= 0 or args.stage1_world_size <= 0:
        parser.error("Stage-1 num_bins and world_size must be positive.")
    if args.min_region_points < 1:
        parser.error("--min_region_points must be positive.")
    if not math.isfinite(args.region_context_padding) or args.region_context_padding < 0:
        parser.error("--region_context_padding must be finite and non-negative.")
    if (args.region_context_padding or args.region_core_filter) and args.method != "plain":
        parser.error("Context-halo diagnostics currently support --method plain only.")
    if args.region_bbox_nms_iou < 0:
        parser.error("--region_bbox_nms_iou must be non-negative.")
    if args.region_bbox_nms_minimum_scale <= 0:
        parser.error("--region_bbox_nms_minimum_scale must be positive.")
    if args.method in ATTENTION_SCORER_METHODS:
        if args.attention_scorer_path is None:
            parser.error(
                "--attention_scorer_path is required for attention_scorer."
            )
        if args.attention_max_point_tokens <= 0:
            parser.error(
                "--attention_max_point_tokens must be positive."
            )
        if args.attention_budget <= 0:
            parser.error("--attention_budget must be positive.")
        if args.attention_top_k <= 0:
            parser.error("--attention_top_k must be positive.")
    if args.num_shards < 1:
        parser.error("--num_shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must satisfy 0 <= index < num_shards.")
    if args.method in SCORER_METHODS and args.scorer_path is None:
        parser.error(f"--method {args.method} requires --scorer_path.")
    if args.method not in SCORER_METHODS and args.scorer_path is not None:
        parser.error("--scorer_path is valid only for scorer methods.")
    if (
        args.conditional_bypass_raw_prediction_dir is not None
        and args.method != "conditional_scorer"
    ):
        parser.error(
            "--conditional_bypass_raw_prediction_dir requires "
            "--method conditional_scorer."
        )
    if (
        args.conditional_bypass_raw_prediction_dir is not None
        and not args.conditional_bypass_raw_prediction_dir.is_dir()
    ):
        parser.error(
            "--conditional_bypass_raw_prediction_dir is not a directory: "
            f"{args.conditional_bypass_raw_prediction_dir}"
        )
    if (
        args.scorer_routing_scope == "scene_max"
        and args.conditional_bypass_raw_prediction_dir is None
    ):
        parser.error(
            "--scorer_routing_scope scene_max requires "
            "--conditional_bypass_raw_prediction_dir."
        )
    if args.scorer_max_keep <= 0 or args.scorer_min_keep < 0:
        parser.error("Invalid scorer min/max keep values.")
    if args.scorer_raw_token_threshold_exclusive < 0:
        parser.error("--scorer_raw_token_threshold_exclusive must be non-negative.")
    if (
        args.scorer_raw_token_upper_threshold_exclusive is not None
        and args.scorer_raw_token_upper_threshold_exclusive
        <= args.scorer_raw_token_threshold_exclusive
    ):
        parser.error(
            "--scorer_raw_token_upper_threshold_exclusive must be greater "
            "than --scorer_raw_token_threshold_exclusive."
        )
    if (
        args.method == "all_region_scorer"
        and args.scorer_raw_token_threshold_exclusive != 0
    ):
        parser.error(
            "--method all_region_scorer requires "
            "--scorer_raw_token_threshold_exclusive 0."
        )
    if args.method == "bbox_regression" and args.num_beams != 1:
        parser.error("--method bbox_regression requires --num_beams 1.")
    return args


def cached_conditional_bypass_region_output(
    args: argparse.Namespace,
    region_id: str,
    scene_id: str,
    region_index: int,
    scene_apply_scorer: bool | None = None,
) -> dict[str, Any] | None:
    """Load an exactly equivalent cached bypass prediction when configured."""

    source_dir = args.conditional_bypass_raw_prediction_dir
    if source_dir is None:
        return None
    metadata_path = source_dir / "region_metadata" / f"{region_id}.json"
    prediction_path = source_dir / "region_predictions" / f"{region_id}.txt"
    if not metadata_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(
            f"Missing conditional-bypass cache for {region_id}: "
            f"{metadata_path}, {prediction_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_token_count = int(metadata["raw_point_token_count"])
    upper_threshold = args.scorer_raw_token_upper_threshold_exclusive
    apply_scorer = (
        raw_token_count > args.scorer_raw_token_threshold_exclusive
        and (
            upper_threshold is None
            or raw_token_count < upper_threshold
        )
    )
    if scene_apply_scorer is False:
        apply_scorer = False
    if apply_scorer:
        return None

    if str(metadata.get("method")) != "plain":
        raise ValueError(
            f"Conditional-bypass cache for {region_id} is not plain inference."
        )
    if int(metadata.get("seed", -1)) != args.seed:
        raise ValueError(
            f"Conditional-bypass cache seed mismatch for {region_id}: "
            f"{metadata.get('seed')} != {args.seed}"
        )
    cached_model = Path(str(metadata.get("stage2_model_path", ""))).resolve()
    requested_model = Path(args.stage2_model_path).resolve()
    if cached_model != requested_model:
        raise ValueError(
            f"Conditional-bypass model mismatch for {region_id}: "
            f"{cached_model} != {requested_model}"
        )
    if float(metadata.get("region_bbox_nms_iou", -1.0)) != float(
        args.region_bbox_nms_iou
    ):
        raise ValueError(
            f"Conditional-bypass region NMS mismatch for {region_id}."
        )
    if float(metadata.get("region_bbox_nms_minimum_scale", -1.0)) != float(
        args.region_bbox_nms_minimum_scale
    ):
        raise ValueError(
            f"Conditional-bypass minimum scale mismatch for {region_id}."
        )

    prediction_text = prediction_path.read_text(encoding="utf-8")
    region_layout = Layout(prediction_text)
    expected_bbox_count = int(metadata["bbox_count"])
    if len(region_layout.bboxes) != expected_bbox_count:
        raise ValueError(
            f"Conditional-bypass bbox count mismatch for {region_id}: "
            f"{len(region_layout.bboxes)} != {expected_bbox_count}"
        )
    output = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "method",
            "seed",
            "stage2_model_path",
            "region_bbox_nms_iou",
            "region_bbox_nms_minimum_scale",
            "prediction_is_region_nms_output",
        }
    }
    output.update(
        {
            "region_id": region_id,
            "scene_id": scene_id,
            "region_index": region_index,
            "kept_point_token_count": raw_token_count,
            "prediction_text": prediction_text,
            "conditional_bypass_prediction_reused": True,
            "conditional_bypass_source_dir": str(source_dir.resolve()),
        }
    )
    return output


def conditional_scene_route_applies_scorer(
    args: argparse.Namespace,
    scene_id: str,
    region_count: int,
) -> bool | None:
    """Resolve GT-free scene routing from the paired plain-cache metadata."""

    if args.scorer_routing_scope == "region":
        return None
    source_dir = args.conditional_bypass_raw_prediction_dir
    if source_dir is None:
        raise ValueError("scene_max routing requires a plain prediction cache.")
    expected_region_ids = [
        f"{scene_id}__region_{region_index:04d}"
        for region_index in range(region_count)
    ]
    index_path = source_dir / "scene_region_index" / f"{scene_id}.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing conditional-bypass scene index: {index_path}"
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    if index_payload.get("region_ids") != expected_region_ids:
        raise ValueError(
            f"Conditional-bypass region ids mismatch for scene {scene_id}."
        )
    raw_token_counts = []
    for region_id in expected_region_ids:
        metadata_path = (
            source_dir / "region_metadata" / f"{region_id}.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing conditional-bypass metadata: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_token_counts.append(int(metadata["raw_point_token_count"]))
    scene_max_raw_token_count = max(raw_token_counts, default=0)
    upper_threshold = args.scorer_raw_token_upper_threshold_exclusive
    return (
        scene_max_raw_token_count
        > args.scorer_raw_token_threshold_exclusive
        and (
            upper_threshold is None
            or scene_max_raw_token_count < upper_threshold
        )
    )


def predict_scene(
    scene,
    stage2_model,
    stage2_tokenizer,
    args: argparse.Namespace,
) -> tuple[Layout, Layout, list[dict[str, Any]]]:
    stage1_path = args.stage1_pred_dir / f"{scene.scene_id}.txt"
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Missing Stage-1 prediction: {stage1_path}")
    stage1_layout = Layout(stage1_path.read_text(encoding="utf-8"))
    scene_apply_scorer = conditional_scene_route_applies_scorer(
        args,
        scene.scene_id,
        len(stage1_layout.regions),
    )

    scene_pcd = prepare_scene_point_cloud(
        scene.pcd_path,
        args.stage1_num_bins,
        args.no_cleanup,
        world_size=args.stage1_world_size,
    )
    num_bins_stage2 = int(stage2_model.config.point_config["num_bins"])
    world_size_stage2 = model_world_size(stage2_model)
    stage2_prompt = prompt_with_point_token(STAGE2_PROMPT)

    all_bboxes: list[Bbox] = []
    region_outputs: list[dict[str, Any]] = []
    for region_index, region in enumerate(stage1_layout.regions):
        region_id = f"{scene.scene_id}__region_{region_index:04d}"
        cached_region_output = cached_conditional_bypass_region_output(
            args,
            region_id,
            scene.scene_id,
            region_index,
            scene_apply_scorer,
        )
        if cached_region_output is not None:
            cached_region_layout = Layout(
                str(cached_region_output["prediction_text"])
            )
            all_bboxes.extend(cached_region_layout.bboxes)
            region_outputs.append(cached_region_output)
            continue
        crop_region = region_with_context(region, args.region_context_padding)
        mask = points_in_region(scene_pcd.points, crop_region)
        region_points = scene_pcd.points[mask]
        region_colors = scene_pcd.colors[mask]
        point_count_before_center_crop = int(region_points.shape[0])
        if region_points.shape[0] < args.min_region_points:
            region_outputs.append(
                {
                    "region_id": region_id,
                    "scene_id": scene.scene_id,
                    "region_index": region_index,
                    "runnable": False,
                    "skip_reason": "too_few_points",
                    "point_count_before_center_crop": (
                        point_count_before_center_crop
                    ),
                    "point_count": int(region_points.shape[0]),
                    "raw_point_token_count": 0,
                    "kept_point_token_count": 0,
                    "bbox_count_before_nms": 0,
                    "bbox_count": 0,
                    "prediction_text": Layout().to_language_string(),
                }
            )
            continue

        region_points, region_colors = center_crop_point_arrays(
            region_points,
            region_colors,
            world_size_stage2,
        )
        if region_points.shape[0] < args.min_region_points:
            region_outputs.append(
                {
                    "region_id": region_id,
                    "scene_id": scene.scene_id,
                    "region_index": region_index,
                    "runnable": False,
                    "skip_reason": "too_few_points_after_center_crop",
                    "point_count_before_center_crop": (
                        point_count_before_center_crop
                    ),
                    "point_count": int(region_points.shape[0]),
                    "raw_point_token_count": 0,
                    "kept_point_token_count": 0,
                    "bbox_count_before_nms": 0,
                    "bbox_count": 0,
                    "prediction_text": Layout().to_language_string(),
                }
            )
            continue

        region_pcd = prepare_point_arrays(
            region_points,
            region_colors,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        stage2_model._fixed_region_last_raw_point_token_count = None
        stage2_model._fixed_region_last_kept_point_token_count = None
        if args.method == "bbox_regression":
            generated, bbox_predictions = generate_layout_text_with_bbox_regression(
                stage2_model,
                stage2_tokenizer,
                stage2_prompt,
                region_pcd.input_tensor,
                args,
            )
            region_layout = decode_bbox_regression_layout(
                generated,
                bbox_predictions,
                region_pcd.min_extent,
                world_size_stage2,
            )
        else:
            generated = generate_layout_text(
                stage2_model,
                stage2_tokenizer,
                stage2_prompt,
                region_pcd.input_tensor,
                args,
            )
            region_layout = decode_generated_layout(
                generated,
                region_pcd.min_extent,
                num_bins_stage2,
                world_size=world_size_stage2,
            )
        raw_token_count = (
            stage2_model._fixed_region_last_raw_point_token_count
        )
        kept_token_count = (
            stage2_model._fixed_region_last_kept_point_token_count
        )
        if raw_token_count is None or kept_token_count is None:
            raise RuntimeError(
                f"Point-token counts were not captured for {region_id}."
            )
        bbox_count_before_core_filter = len(region_layout.bboxes)
        if args.region_core_filter:
            region_layout.bboxes = filter_bboxes_to_region_core(region_layout.bboxes, region)
        bbox_count_before_nms = len(region_layout.bboxes)
        region_layout.bboxes = postprocess_classwise_nms(
            region_layout.bboxes,
            args.region_bbox_nms_iou,
            args.region_bbox_nms_minimum_scale,
        )
        all_bboxes.extend(region_layout.bboxes)
        region_outputs.append(
            {
                "region_id": region_id,
                "scene_id": scene.scene_id,
                "region_index": region_index,
                "runnable": True,
                "skip_reason": "",
                "point_count_before_center_crop": (
                    point_count_before_center_crop
                ),
                "point_count": int(region_points.shape[0]),
                "raw_point_token_count": int(raw_token_count),
                "kept_point_token_count": int(kept_token_count),
                "bbox_count_before_nms": bbox_count_before_nms,
                **({'bbox_count_before_core_filter': bbox_count_before_core_filter}
                   if args.region_core_filter else {}),
                "bbox_count": len(region_layout.bboxes),
                "prediction_text": region_layout.to_language_string(),
            }
        )

    return (
        stage1_layout,
        final_layout_from_parts(stage1_layout, all_bboxes),
        region_outputs,
    )


def scene_region_outputs_complete(
    output_dir: Path,
    scene_id: str,
) -> bool:
    index_path = output_dir / "scene_region_index" / f"{scene_id}.json"
    if not index_path.is_file():
        return False
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    region_ids = payload.get("region_ids")
    if not isinstance(region_ids, list):
        return False
    return all(
        (output_dir / "region_predictions" / f"{region_id}.txt").is_file()
        and (output_dir / "region_metadata" / f"{region_id}.json").is_file()
        for region_id in region_ids
    )


def write_region_outputs(
    output_dir: Path,
    scene_id: str,
    region_outputs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    region_ids = []
    for region_output in region_outputs:
        region_id = str(region_output["region_id"])
        region_ids.append(region_id)
        prediction_text = str(region_output["prediction_text"])
        metadata = {
            key: value
            for key, value in region_output.items()
            if key != "prediction_text"
        }
        metadata.update(
            {
                "method": args.method,
                "seed": args.seed,
                "stage2_model_path": args.stage2_model_path,
                "region_bbox_nms_iou": args.region_bbox_nms_iou,
                "region_bbox_nms_minimum_scale": (
                    args.region_bbox_nms_minimum_scale
                ),
                "prediction_is_region_nms_output": True,
            }
        )
        if args.region_context_padding or args.region_core_filter:
            metadata.update(region_context_padding=args.region_context_padding,
                            region_core_filter=args.region_core_filter,
                            region_context_units='world_meters')
        atomic_write_text(
            output_dir / "region_predictions" / f"{region_id}.txt",
            prediction_text,
        )
        atomic_write_json(
            output_dir / "region_metadata" / f"{region_id}.json",
            metadata,
        )
    atomic_write_json(
        output_dir / "scene_region_index" / f"{scene_id}.json",
        {
            "scene_id": scene_id,
            "region_ids": region_ids,
            "region_count": len(region_ids),
            "method": args.method,
            "seed": args.seed,
        },
    )


def main() -> int:
    args = parse_args()
    configure_reproducibility(args.seed, args.deterministic)
    if not args.stage1_pred_dir.is_dir():
        raise NotADirectoryError(args.stage1_pred_dir)

    scenes = load_scenes(args)
    if not scenes:
        raise ValueError("No scenes selected for fixed-Stage-1 inference.")

    stage2_model, stage2_tokenizer = load_model_and_tokenizer(
        args.stage2_model_path,
        args.inference_dtype,
        args.device,
    )
    if args.method in SCORER_METHODS:
        scorer = load_point_token_scorer(args.scorer_path, args.device)
        install_conditional_scorer_point_filter(stage2_model, scorer, args)
    elif args.method in ATTENTION_SCORER_METHODS:
        scorer, checkpoint = load_attention_scorer(
            args.attention_scorer_path,
            args.device,
        )
        validate_checkpoint_args(checkpoint, args)
        if args.attention_selection == "hard_topk":
            install_attention_scorer_topk(stage2_model, scorer, args)
        else:
            install_attention_scorer_bias(stage2_model, scorer, args)
        install_point_token_count_recorder(stage2_model)
    else:
        install_point_token_count_recorder(stage2_model)
    if args.method == "bbox_regression":
        if not getattr(stage2_model, "bbox_regression_aux", False):
            raise ValueError(
                "The selected bbox_regression checkpoint has no bbox auxiliary head."
            )

    stage1_output_dir = args.output_dir / "stage1"
    final_output_dir = args.output_dir / "final"
    stage1_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "region_predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "region_metadata").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scene_region_index").mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc=f"Fixed Stage-1 / {args.method} Stage-2"):
        stage1_output_path = stage1_output_dir / f"{scene.scene_id}.txt"
        final_output_path = final_output_dir / f"{scene.scene_id}.txt"
        if (
            args.skip_existing
            and stage1_output_path.exists()
            and final_output_path.exists()
            and scene_region_outputs_complete(args.output_dir, scene.scene_id)
        ):
            continue
        try:
            stage1_layout, final_layout, region_outputs = predict_scene(
                scene,
                stage2_model,
                stage2_tokenizer,
                args,
            )
            atomic_write_text(
                stage1_output_path,
                stage1_layout.to_language_string(),
            )
            atomic_write_text(
                final_output_path,
                final_layout.to_language_string(),
            )
            write_region_outputs(
                args.output_dir,
                scene.scene_id,
                region_outputs,
                args,
            )
            (args.output_dir / "errors" / f"{scene.scene_id}.txt").unlink(
                missing_ok=True
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((scene.scene_id, str(exc)))
            atomic_write_text(
                args.output_dir / "errors" / f"{scene.scene_id}.txt",
                str(exc),
            )

    if args.method in SCORER_METHODS:
        print_conditional_summary(
            stage2_model,
            args.scorer_raw_token_threshold_exclusive,
            args.scorer_raw_token_upper_threshold_exclusive,
        )
    if failures:
        print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        for scene_id, error in failures[:10]:
            print(f"{scene_id}: {error}", file=sys.stderr)
        return 1

    print(
        "Wrote fixed-Stage-1 Stage-2 predictions: "
        f"method={args.method}, output_dir={args.output_dir}, seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
