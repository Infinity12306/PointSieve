#!/usr/bin/env python3
"""Two-stage hierarchical SpatialLM inference for evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from bbox import BBox3D
from bbox.metrics import iou_3d
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from utils.build_region_dataset import STAGE1_PROMPT, STAGE2_PROMPT
from utils.inference_base import preprocess_point_cloud
from utils.render_topdown import save_points_and_colors
from spatiallm import Layout
from spatiallm.layout.entity import Bbox, Region
from spatiallm.pcd import cleanup_pcd, get_points_and_colors, load_o3d_pcd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "spatiallm"
POINT_PROMPT = "<|point_start|><|point_pad|><|point_end|>"
LAYOUT_START = "<|layout_s|>"
LAYOUT_END = "<|layout_e|>"


@dataclass
class SceneInput:
    scene_id: str
    pcd_path: Path


@dataclass
class PreparedPointCloud:
    input_tensor: torch.Tensor
    min_extent: np.ndarray
    points: np.ndarray
    colors: np.ndarray


@dataclass
class RegionPrediction:
    index: int
    region: Region
    point_count: int
    bboxes: list[Bbox]
    skipped: bool
    skip_reason: str = ""
    points: np.ndarray | None = None
    colors: np.ndarray | None = None


@dataclass
class HierarchicalPrediction:
    scene_id: str
    stage1_text: str
    final_text: str
    stage1_layout: Layout
    final_layout: Layout
    region_predictions: list[RegionPrediction]


def latest_checkpoint(path: Path) -> Path:
    checkpoints = []
    for candidate in path.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), candidate))
    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]
    return path


def resolve_model_path(value: str | Path) -> str:
    text = str(value)
    path = Path(text)
    if path.exists():
        return str(latest_checkpoint(path))
    return text


def prompt_with_point_token(prompt: str) -> str:
    return prompt.replace("<point_cloud>", POINT_PROMPT)


def clean_generated_text(text: str) -> str:
    return text.replace(LAYOUT_START, "").replace(LAYOUT_END, "").strip()


def discard_trailing_truncated_bbox_anchor(
    generated_text: str,
    anchor_hidden_states: list[torch.Tensor],
    *,
    generated_token_count: int,
    max_new_tokens: int,
) -> bool:
    """Drop only an unparseable final bbox anchor cut off by max_new_tokens."""
    if generated_token_count < max_new_tokens or not anchor_hidden_states:
        return False

    parsed_layout = Layout(generated_text)
    if len(anchor_hidden_states) != len(parsed_layout.bboxes) + 1:
        return False

    lines = generated_text.rstrip().splitlines()
    if not lines:
        return False
    trailing_line = lines[-1]
    if re.search(r"bbox_\d+=Bbox\([^,\n]+,", trailing_line) is None:
        return False
    if Layout(trailing_line).bboxes:
        return False

    anchor_hidden_states.pop()
    return True


def bbox_numeric_char_spans(content: str) -> list[list[tuple[int, int]]]:
    """Return the seven numeric-field spans for every complete serialized bbox."""
    number = r"([^,\n()]+)"
    pattern = re.compile(
        r"bbox_\d+=Bbox\([^,\n]+,"
        + ",".join([number] * 7)
        + r"\)"
    )
    return [
        [match.span(group_index) for group_index in range(1, 8)]
        for match in pattern.finditer(content)
    ]


def pool_generated_bbox_numeric_hidden_states(
    tokenizer,
    generated_ids: torch.Tensor,
    captured_token_ids: list[int],
    captured_token_hidden_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Mean-pool generated hidden states inside all seven bbox numeric fields."""
    if len(captured_token_ids) != len(captured_token_hidden_states):
        raise RuntimeError(
            "Captured generated token/hidden-state count mismatch: "
            f"tokens={len(captured_token_ids)}, "
            f"hidden_states={len(captured_token_hidden_states)}."
        )
    if not captured_token_ids:
        return []

    raw_generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoding = tokenizer(
        raw_generated_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    reencoded_ids = torch.as_tensor(
        encoding["input_ids"],
        dtype=generated_ids.dtype,
        device=generated_ids.device,
    )
    if (
        reencoded_ids.shape != generated_ids.shape
        or not torch.equal(reencoded_ids, generated_ids)
    ):
        raise RuntimeError(
            "Generated text does not round-trip to the original token IDs; "
            "cannot align bbox numeric hidden states safely."
        )

    offsets = encoding["offset_mapping"]
    pooled_hidden_states: list[torch.Tensor] = []
    for box_spans in bbox_numeric_char_spans(raw_generated_text):
        numeric_token_positions: list[int] = []
        for span_start, span_end in box_spans:
            field_positions = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if (
                    token_end > token_start
                    and token_start >= span_start
                    and token_end <= span_end
                )
            ]
            if not field_positions:
                raise RuntimeError(
                    "No generated tokenizer token was fully contained in bbox "
                    f"numeric span [{span_start}, {span_end})."
                )
            if field_positions[-1] >= len(captured_token_hidden_states):
                raise RuntimeError(
                    "A complete generated bbox numeric field has no matching "
                    "autoregressive hidden state."
                )
            numeric_token_positions.extend(field_positions)

        selected_hidden_states = torch.stack(
            [
                captured_token_hidden_states[position]
                for position in numeric_token_positions
            ]
        )
        pooled_hidden_states.append(selected_hidden_states.mean(dim=0))

    return pooled_hidden_states


def make_conversation(model, prompt: str) -> list[dict[str, str]]:
    if model.config.model_type == "spatiallm_qwen":
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def generate_layout_text(
    model,
    tokenizer,
    prompt: str,
    point_cloud: torch.Tensor,
    args: argparse.Namespace,
) -> str:
    if args.seed >= 0:
        set_seed(args.seed)

    conversation = make_conversation(model, prompt)
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "point_clouds": point_cloud,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": not args.greedy,
        "use_cache": True,
    }
    if not args.greedy:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)

    generated_ids = output_ids[0, input_ids.shape[1] :]
    return clean_generated_text(
        tokenizer.decode(generated_ids, skip_special_tokens=True)
    )


def generate_layout_text_with_bbox_regression(
    model,
    tokenizer,
    prompt: str,
    point_cloud: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[str, np.ndarray]:
    """Generate normal tokens and regress boxes from the configured LLM states."""
    if args.num_beams != 1:
        raise ValueError(
            "BBox-regression hidden-state capture currently requires --num_beams 1."
        )
    if not getattr(model, "bbox_regression_aux", False):
        raise ValueError("The Stage-2 model does not enable bbox_regression_aux.")
    if not hasattr(model, "bbox_regression_head"):
        raise ValueError("The Stage-2 model has no bbox_regression_head.")
    feature_source = getattr(
        args,
        "bbox_regression_feature_source",
        "anchor",
    )
    if feature_source not in {"anchor", "numeric_token_mean"}:
        raise ValueError(
            f"Unsupported bbox regression feature source: {feature_source}."
        )
    if args.seed >= 0:
        set_seed(args.seed)

    conversation = make_conversation(model, prompt)
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "point_clouds": point_cloud,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": not args.greedy,
        "use_cache": True,
    }
    if not args.greedy:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    generated_prefix_ids: list[int] = []
    anchor_hidden_states: list[torch.Tensor] = []
    generated_token_hidden_states: list[torch.Tensor] = []
    anchor_pattern = re.compile(r"bbox_\d+=Bbox\([^,\n]+,$")
    original_forward = model.forward
    had_instance_forward = "forward" in model.__dict__
    previous_instance_forward = model.__dict__.get("forward")

    @wraps(type(model).forward)
    def forward_with_bbox_capture(self, *forward_args, **forward_kwargs):
        current_input_ids = forward_kwargs.get("input_ids")
        past_key_values = forward_kwargs.get("past_key_values")
        if past_key_values is None:
            has_cached_tokens = False
        elif hasattr(past_key_values, "get_seq_length"):
            has_cached_tokens = int(past_key_values.get_seq_length()) > 0
        else:
            has_cached_tokens = bool(past_key_values)
        caller_requested_hidden_states = bool(
            forward_kwargs.get("output_hidden_states", False)
        )
        forward_kwargs["output_hidden_states"] = True
        outputs = original_forward(*forward_args, **forward_kwargs)

        if has_cached_tokens and current_input_ids is not None:
            if current_input_ids.shape != (1, 1):
                raise RuntimeError(
                    "Expected one autoregressive token while capturing bbox "
                    "regression hidden states, "
                    f"got input_ids shape {tuple(current_input_ids.shape)}."
                )
            token_id = int(current_input_ids.item())
            generated_prefix_ids.append(token_id)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    "Model did not return hidden states for bbox regression."
                )
            current_hidden_state = hidden_states[-1][0, -1].detach()
            if feature_source == "numeric_token_mean":
                generated_token_hidden_states.append(current_hidden_state)
            else:
                token_text = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if "," in token_text:
                    prefix_text = tokenizer.decode(
                        generated_prefix_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if anchor_pattern.search(prefix_text):
                        anchor_hidden_states.append(current_hidden_state.cpu())

        if not caller_requested_hidden_states:
            outputs.hidden_states = None
        return outputs

    model.forward = MethodType(forward_with_bbox_capture, model)
    try:
        with torch.inference_mode():
            output_ids = model.generate(**generate_kwargs)
    finally:
        if had_instance_forward:
            model.forward = previous_instance_forward
        else:
            del model.forward

    generated_ids = output_ids[0, input_ids.shape[1] :]
    captured_ids = torch.as_tensor(
        generated_prefix_ids,
        dtype=generated_ids.dtype,
        device=generated_ids.device,
    )
    if (
        captured_ids.numel() > generated_ids.numel()
        or not torch.equal(generated_ids[: captured_ids.numel()], captured_ids)
    ):
        raise RuntimeError(
            "Captured autoregressive token sequence does not match generate() output."
        )

    generated_text = clean_generated_text(
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    if feature_source == "anchor":
        discard_trailing_truncated_bbox_anchor(
            generated_text,
            anchor_hidden_states,
            generated_token_count=int(generated_ids.numel()),
            max_new_tokens=args.max_new_tokens,
        )
        regression_hidden_states = anchor_hidden_states
    else:
        regression_hidden_states = pool_generated_bbox_numeric_hidden_states(
            tokenizer,
            generated_ids,
            generated_prefix_ids,
            generated_token_hidden_states,
        )

    if not regression_hidden_states:
        return generated_text, np.empty((0, 8), dtype=np.float32)

    head_dtype = next(model.bbox_regression_head.parameters()).dtype
    hidden_batch = torch.stack(regression_hidden_states).to(
        device=model.device,
        dtype=head_dtype,
    )
    with torch.inference_mode():
        predictions = model.bbox_regression_head(hidden_batch).float().cpu().numpy()
    return generated_text, predictions


def decode_bbox_regression_layout(
    generated_text: str,
    predictions: np.ndarray,
    min_extent: np.ndarray,
    world_size: float,
) -> Layout:
    """Keep LLM classes/counts but replace all seven box properties with MLP output."""
    layout = Layout(generated_text)
    if len(layout.bboxes) != len(predictions):
        raise ValueError(
            "Generated bbox/MLP anchor count mismatch: "
            f"parsed_bboxes={len(layout.bboxes)}, anchors={len(predictions)}."
        )
    if predictions.size == 0:
        return layout
    if predictions.shape != (len(layout.bboxes), 8):
        raise ValueError(
            f"Expected bbox predictions with shape ({len(layout.bboxes)}, 8), "
            f"got {predictions.shape}."
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("BBox regression head produced non-finite values.")

    centers = np.clip(predictions[:, :3], 0.0, 1.0) * world_size
    sizes = np.exp(np.clip(predictions[:, 3:6], -20.0, 20.0))
    yaws = 0.5 * np.arctan2(predictions[:, 6], predictions[:, 7])
    for bbox, center, size, yaw in zip(
        layout.bboxes,
        centers,
        sizes,
        yaws,
    ):
        bbox.position_x, bbox.position_y, bbox.position_z = center.tolist()
        bbox.scale_x, bbox.scale_y, bbox.scale_z = size.tolist()
        bbox.angle_z = float(yaw)
    layout.translate(min_extent)
    return layout


def prepare_scene_point_cloud(
    pcd_path: Path,
    num_bins: int,
    no_cleanup: bool,
    world_size: float = 32.0,
) -> PreparedPointCloud:
    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")

    pcd = load_o3d_pcd(str(pcd_path))
    grid_size = Layout.get_grid_size(num_bins, world_size=world_size)
    if not no_cleanup:
        pcd = cleanup_pcd(pcd, voxel_size=grid_size)

    points, colors = get_points_and_colors(pcd)
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud has no points: {pcd_path}")

    return prepare_point_arrays(points, colors, num_bins, world_size=world_size)


def prepare_point_arrays(
    points: np.ndarray,
    colors: np.ndarray,
    num_bins: int,
    world_size: float = 32.0,
) -> PreparedPointCloud:
    if points.shape[0] == 0:
        raise ValueError("Cannot prepare an empty point cloud.")

    grid_size = Layout.get_grid_size(num_bins, world_size=world_size)
    min_extent = np.min(points, axis=0)
    input_tensor = preprocess_point_cloud(points, colors, grid_size, num_bins)
    return PreparedPointCloud(
        input_tensor=input_tensor,
        min_extent=min_extent,
        points=points,
        colors=colors,
    )


def decode_generated_layout(
    text: str,
    min_extent: np.ndarray,
    num_bins: int,
    world_size: float = 32.0,
) -> Layout:
    layout = Layout(text)
    layout.undiscretize_and_unnormalize(num_bins=num_bins, world_size=world_size)
    layout.translate(min_extent)
    return layout


def model_world_size(model) -> float:
    return float(model.config.point_config.get("world_size", 32.0))


def center_crop_point_arrays(
    points: np.ndarray,
    colors: np.ndarray,
    world_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    extent = max_bound - min_bound
    if world_size >= 32.0 or np.all(extent <= world_size):
        return points, colors

    crop_center = (min_bound + max_bound) * 0.5
    crop_half_size = np.full(3, world_size * 0.5, dtype=np.float64)
    crop_min_bound = crop_center - crop_half_size
    crop_max_bound = crop_center + crop_half_size
    crop_mask = np.all(
        (points >= crop_min_bound) & (points <= crop_max_bound),
        axis=1,
    )
    if not np.any(crop_mask):
        nearest_index = int(np.argmin(np.sum((points - crop_center) ** 2, axis=1)))
        crop_mask[nearest_index] = True

    return points[crop_mask], colors[crop_mask]


def points_in_region(points: np.ndarray, region: Region) -> np.ndarray:
    center = np.array([region.position_x, region.position_y, region.position_z])
    scale = np.array([region.scale_x, region.scale_y, region.scale_z])
    if np.any(scale <= 0):
        return np.zeros(points.shape[0], dtype=bool)
    half_size = scale * 0.5
    return np.all((points >= center - half_size) & (points <= center + half_size), axis=1)


def bbox3d_from_bbox(bbox: Bbox) -> BBox3D:
    return BBox3D(
        bbox.position_x,
        bbox.position_y,
        bbox.position_z,
        max(bbox.scale_x, 1e-6),
        max(bbox.scale_y, 1e-6),
        max(bbox.scale_z, 1e-6),
        euler_angles=[0, 0, bbox.angle_z],
        is_center=True,
    )


def classwise_nms(bboxes: list[Bbox], iou_threshold: float) -> list[Bbox]:
    if iou_threshold <= 0 or len(bboxes) <= 1:
        return bboxes

    kept: list[Bbox] = []
    kept_boxes: list[BBox3D] = []
    for bbox in bboxes:
        current_box = bbox3d_from_bbox(bbox)
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
    return kept


def final_layout_from_parts(stage1_layout: Layout, bboxes: list[Bbox]) -> Layout:
    final_layout = Layout()
    final_layout.walls = stage1_layout.walls
    final_layout.doors = stage1_layout.doors
    final_layout.windows = stage1_layout.windows
    final_layout.bboxes = bboxes
    for bbox_id, bbox in enumerate(final_layout.bboxes):
        bbox.id = bbox_id
    return final_layout


def load_gt_region_layout(scene_id: str, gt_region_dir: Path) -> Layout:
    region_path = gt_region_dir / f"{scene_id}.txt"
    if not region_path.exists():
        raise FileNotFoundError(f"GT region file not found: {region_path}")
    return Layout(region_path.read_text(encoding="utf-8"))


def predict_hierarchical_scene(
    scene: SceneInput,
    stage1_model,
    stage1_tokenizer,
    stage2_model,
    stage2_tokenizer,
    args: argparse.Namespace,
) -> HierarchicalPrediction:
    stage2_prompt = prompt_with_point_token(STAGE2_PROMPT)
    num_bins_stage2 = stage2_model.config.point_config["num_bins"]
    world_size_stage2 = model_world_size(stage2_model)

    gt_region_dir = getattr(args, "gt_region_dir", None)
    world_size_stage1 = (
        world_size_stage2
        if gt_region_dir is not None
        else model_world_size(stage1_model)
    )
    num_bins_scene = (
        num_bins_stage2
        if gt_region_dir is not None
        else stage1_model.config.point_config["num_bins"]
    )
    scene_pcd = prepare_scene_point_cloud(
        scene.pcd_path,
        num_bins_scene,
        args.no_cleanup,
        world_size=world_size_stage1,
    )
    if gt_region_dir is not None:
        stage1_layout = load_gt_region_layout(scene.scene_id, gt_region_dir)
    else:
        stage1_prompt = prompt_with_point_token(STAGE1_PROMPT)
        stage1_generated = generate_layout_text(
            stage1_model,
            stage1_tokenizer,
            stage1_prompt,
            scene_pcd.input_tensor,
            args,
        )
        stage1_layout = decode_generated_layout(
            stage1_generated,
            scene_pcd.min_extent,
            num_bins_scene,
            world_size=world_size_stage1,
        )

    region_predictions: list[RegionPrediction] = []
    all_bboxes: list[Bbox] = []
    for region_index, region in enumerate(stage1_layout.regions):
        mask = points_in_region(scene_pcd.points, region)
        region_points = scene_pcd.points[mask]
        region_colors = scene_pcd.colors[mask]

        if region_points.shape[0] < args.min_region_points:
            region_predictions.append(
                RegionPrediction(
                    index=region_index,
                    region=region,
                    point_count=int(region_points.shape[0]),
                    bboxes=[],
                    skipped=True,
                    skip_reason="too_few_points",
                    points=region_points,
                    colors=region_colors,
                )
            )
            continue

        region_points, region_colors = center_crop_point_arrays(
            region_points,
            region_colors,
            world_size_stage2,
        )

        if region_points.shape[0] < args.min_region_points:
            region_predictions.append(
                RegionPrediction(
                    index=region_index,
                    region=region,
                    point_count=int(region_points.shape[0]),
                    bboxes=[],
                    skipped=True,
                    skip_reason="too_few_points_after_center_crop",
                    points=region_points,
                    colors=region_colors,
                )
            )
            continue

        region_pcd = prepare_point_arrays(
            region_points,
            region_colors,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        stage2_generated = generate_layout_text(
            stage2_model,
            stage2_tokenizer,
            stage2_prompt,
            region_pcd.input_tensor,
            args,
        )
        region_layout = decode_generated_layout(
            stage2_generated,
            region_pcd.min_extent,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        all_bboxes.extend(region_layout.bboxes)
        region_predictions.append(
            RegionPrediction(
                index=region_index,
                region=region,
                point_count=int(region_points.shape[0]),
                bboxes=region_layout.bboxes,
                skipped=False,
                points=region_points,
                colors=region_colors,
            )
        )

    all_bboxes = classwise_nms(all_bboxes, args.bbox_nms_iou)
    final_layout = final_layout_from_parts(stage1_layout, all_bboxes)
    return HierarchicalPrediction(
        scene_id=scene.scene_id,
        stage1_text=stage1_layout.to_language_string(),
        final_text=final_layout.to_language_string(),
        stage1_layout=stage1_layout,
        final_layout=final_layout,
        region_predictions=region_predictions,
    )


def predict_hierarchical_scene_with_bbox_regression(
    scene: SceneInput,
    stage1_model,
    stage1_tokenizer,
    stage2_model,
    stage2_tokenizer,
    args: argparse.Namespace,
) -> tuple[HierarchicalPrediction, HierarchicalPrediction]:
    """Return paired LLM-number and MLP-number predictions from one generation."""
    stage2_prompt = prompt_with_point_token(STAGE2_PROMPT)
    num_bins_stage2 = stage2_model.config.point_config["num_bins"]
    world_size_stage2 = model_world_size(stage2_model)

    gt_region_dir = getattr(args, "gt_region_dir", None)
    world_size_stage1 = (
        world_size_stage2
        if gt_region_dir is not None
        else model_world_size(stage1_model)
    )
    num_bins_scene = (
        num_bins_stage2
        if gt_region_dir is not None
        else stage1_model.config.point_config["num_bins"]
    )
    scene_pcd = prepare_scene_point_cloud(
        scene.pcd_path,
        num_bins_scene,
        args.no_cleanup,
        world_size=world_size_stage1,
    )
    if gt_region_dir is not None:
        stage1_layout = load_gt_region_layout(scene.scene_id, gt_region_dir)
    else:
        stage1_prompt = prompt_with_point_token(STAGE1_PROMPT)
        stage1_generated = generate_layout_text(
            stage1_model,
            stage1_tokenizer,
            stage1_prompt,
            scene_pcd.input_tensor,
            args,
        )
        stage1_layout = decode_generated_layout(
            stage1_generated,
            scene_pcd.min_extent,
            num_bins_scene,
            world_size=world_size_stage1,
        )

    llm_region_predictions: list[RegionPrediction] = []
    mlp_region_predictions: list[RegionPrediction] = []
    all_llm_bboxes: list[Bbox] = []
    all_mlp_bboxes: list[Bbox] = []
    for region_index, region in enumerate(stage1_layout.regions):
        mask = points_in_region(scene_pcd.points, region)
        region_points = scene_pcd.points[mask]
        region_colors = scene_pcd.colors[mask]

        skip_reason = ""
        if region_points.shape[0] < args.min_region_points:
            skip_reason = "too_few_points"
        else:
            region_points, region_colors = center_crop_point_arrays(
                region_points,
                region_colors,
                world_size_stage2,
            )
            if region_points.shape[0] < args.min_region_points:
                skip_reason = "too_few_points_after_center_crop"

        if skip_reason:
            for predictions in (llm_region_predictions, mlp_region_predictions):
                predictions.append(
                    RegionPrediction(
                        index=region_index,
                        region=region,
                        point_count=int(region_points.shape[0]),
                        bboxes=[],
                        skipped=True,
                        skip_reason=skip_reason,
                        points=region_points,
                        colors=region_colors,
                    )
                )
            continue

        region_pcd = prepare_point_arrays(
            region_points,
            region_colors,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        stage2_generated, bbox_predictions = (
            generate_layout_text_with_bbox_regression(
                stage2_model,
                stage2_tokenizer,
                stage2_prompt,
                region_pcd.input_tensor,
                args,
            )
        )
        llm_region_layout = decode_generated_layout(
            stage2_generated,
            region_pcd.min_extent,
            num_bins_stage2,
            world_size=world_size_stage2,
        )
        mlp_region_layout = decode_bbox_regression_layout(
            stage2_generated,
            bbox_predictions,
            region_pcd.min_extent,
            world_size_stage2,
        )
        all_llm_bboxes.extend(llm_region_layout.bboxes)
        all_mlp_bboxes.extend(mlp_region_layout.bboxes)
        llm_region_predictions.append(
            RegionPrediction(
                index=region_index,
                region=region,
                point_count=int(region_points.shape[0]),
                bboxes=llm_region_layout.bboxes,
                skipped=False,
                points=region_points,
                colors=region_colors,
            )
        )
        mlp_region_predictions.append(
            RegionPrediction(
                index=region_index,
                region=region,
                point_count=int(region_points.shape[0]),
                bboxes=mlp_region_layout.bboxes,
                skipped=False,
                points=region_points,
                colors=region_colors,
            )
        )

    all_llm_bboxes = classwise_nms(all_llm_bboxes, args.bbox_nms_iou)
    all_mlp_bboxes = classwise_nms(all_mlp_bboxes, args.bbox_nms_iou)
    llm_final_layout = final_layout_from_parts(stage1_layout, all_llm_bboxes)
    mlp_final_layout = final_layout_from_parts(stage1_layout, all_mlp_bboxes)
    common = {
        "scene_id": scene.scene_id,
        "stage1_text": stage1_layout.to_language_string(),
        "stage1_layout": stage1_layout,
    }
    return (
        HierarchicalPrediction(
            **common,
            final_text=llm_final_layout.to_language_string(),
            final_layout=llm_final_layout,
            region_predictions=llm_region_predictions,
        ),
        HierarchicalPrediction(
            **common,
            final_text=mlp_final_layout.to_language_string(),
            final_layout=mlp_final_layout,
            region_predictions=mlp_region_predictions,
        ),
    )


def resolve_point_cloud_path(raw_path: str, dataset_root: Path, json_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    dataset_candidate = dataset_root / path
    if dataset_candidate.exists():
        return dataset_candidate

    json_candidate = json_dir / path
    if json_candidate.exists():
        return json_candidate

    return dataset_candidate


def scenes_from_json(data_json: Path, dataset_root: Path) -> list[SceneInput]:
    data = json.loads(data_json.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("examples") or data.get("items") or []
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples in {data_json}")

    scenes: list[SceneInput] = []
    seen: set[str] = set()
    for index, item in enumerate(data):
        point_clouds = item.get("point_clouds") or item.get("point_cloud")
        if isinstance(point_clouds, list):
            if not point_clouds:
                continue
            raw_point_cloud = point_clouds[0]
        else:
            raw_point_cloud = point_clouds
        if not raw_point_cloud:
            continue

        pcd_path = resolve_point_cloud_path(
            str(raw_point_cloud), dataset_root, data_json.parent
        )
        scene_id = pcd_path.stem
        if scene_id in seen:
            scene_id = f"{scene_id}_{index:06d}"
        scenes.append(SceneInput(scene_id=scene_id, pcd_path=pcd_path))
        seen.add(scene_id)
    return scenes


def scenes_from_point_cloud(point_cloud: Path) -> list[SceneInput]:
    if point_cloud.is_file():
        return [SceneInput(scene_id=point_cloud.stem, pcd_path=point_cloud)]
    if point_cloud.is_dir():
        return [
            SceneInput(scene_id=pcd_path.stem, pcd_path=pcd_path)
            for pcd_path in sorted(point_cloud.glob("*.ply"))
        ]
    raise FileNotFoundError(point_cloud)


def apply_subset_args(
    scenes: list[SceneInput],
    start_index: int,
    end_index: int | None,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> list[SceneInput]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < --num_shards")

    scenes = scenes[start_index:end_index]
    if num_shards > 1:
        scenes = [
            scene for index, scene in enumerate(scenes)
            if index % num_shards == shard_index
        ]
    if limit is not None:
        scenes = scenes[:limit]
    return scenes


def load_model_and_tokenizer(model_path: str, dtype: str, device: str):
    resolved_path = resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=True,
    )
    model.to(device)
    model.set_point_backbone_dtype(torch.float32)
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Two-stage hierarchical SpatialLM inference")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--data_json",
        type=Path,
        help="Stage1-style JSON dataset, e.g. spatiallm_stage1_region_val.json.",
    )
    input_group.add_argument(
        "-p",
        "--point_cloud",
        type=Path,
        help="PLY file or directory of PLY files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Output root. Writes output_dir/stage1 and output_dir/final.",
    )
    parser.add_argument(
        "--bbox_regression_output_dir",
        type=Path,
        default=None,
        help=(
            "Optional paired output root whose bbox properties come directly "
            "from bbox_regression_head. Normal LLM token generation still runs "
            "once and --output_dir receives its ordinary decoded boxes."
        ),
    )
    parser.add_argument(
        "--bbox_regression_feature_source",
        choices=("anchor", "numeric_token_mean"),
        default="anchor",
        help=(
            "Hidden feature used by bbox_regression_head. 'anchor' preserves "
            "historical comma-anchor checkpoints; 'numeric_token_mean' pools "
            "all generated tokens inside the seven complete bbox numeric fields."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Base directory for relative point_cloud paths in JSON.",
    )
    parser.add_argument(
        "--stage1_model_path",
        default="saves/hierarchical/stage1_regions",
    )
    parser.add_argument(
        "--stage2_model_path",
        default="saves/hierarchical/stage2_bboxes",
    )
    parser.add_argument(
        "--gt_region_dir",
        type=Path,
        default=None,
        help=(
            "Use per-scene GT Region txt files from this directory instead of "
            "loading/running the stage-1 model. This isolates stage-2 bbox inference."
        ),
    )
    parser.add_argument("--inference_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_cleanup", action="store_true")
    parser.add_argument("--min_region_points", type=int, default=1)
    parser.add_argument(
        "--bbox_nms_iou",
        type=float,
        default=0.0,
        help="Class-wise bbox NMS IoU threshold. <=0 disables NMS.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--save_region_pcds",
        action="store_true",
        help="Also save cropped predicted-region PCDs under output_dir/regions.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def load_scenes(args: argparse.Namespace) -> list[SceneInput]:
    if args.data_json is not None:
        scenes = scenes_from_json(args.data_json, args.dataset_root)
    else:
        scenes = scenes_from_point_cloud(args.point_cloud)
    return apply_subset_args(
        scenes,
        args.start_index,
        args.end_index,
        args.limit,
        args.num_shards,
        args.shard_index,
    )


def write_prediction_outputs(
    prediction: HierarchicalPrediction,
    output_dir: Path,
) -> None:
    stage1_dir = output_dir / "stage1"
    final_dir = output_dir / "final"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    (stage1_dir / f"{prediction.scene_id}.txt").write_text(
        prediction.stage1_text,
        encoding="utf-8",
    )
    (final_dir / f"{prediction.scene_id}.txt").write_text(
        prediction.final_text,
        encoding="utf-8",
    )


def write_region_debug_pcds(
    prediction: HierarchicalPrediction,
    output_dir: Path,
) -> None:
    region_dir = output_dir / "regions" / prediction.scene_id
    region_dir.mkdir(parents=True, exist_ok=True)
    for region_prediction in prediction.region_predictions:
        if region_prediction.points is None or region_prediction.colors is None:
            continue
        if region_prediction.points.shape[0] == 0:
            continue
        save_points_and_colors(
            region_dir / f"region_{region_prediction.index}.ply",
            region_prediction.points,
            region_prediction.colors,
        )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.bbox_regression_output_dir is not None:
        args.bbox_regression_output_dir.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes(args)
    if not scenes:
        raise ValueError("No scenes found for inference.")

    stage1_model = None
    stage1_tokenizer = None
    if args.gt_region_dir is None:
        stage1_model, stage1_tokenizer = load_model_and_tokenizer(
            args.stage1_model_path,
            args.inference_dtype,
            args.device,
        )
    elif not args.gt_region_dir.is_dir():
        raise NotADirectoryError(f"GT region directory not found: {args.gt_region_dir}")

    stage2_model, stage2_tokenizer = load_model_and_tokenizer(
        args.stage2_model_path,
        args.inference_dtype,
        args.device,
    )
    if args.bbox_regression_output_dir is not None:
        if args.num_beams != 1:
            raise ValueError(
                "--bbox_regression_output_dir requires --num_beams 1."
            )
        if not getattr(stage2_model, "bbox_regression_aux", False):
            raise ValueError(
                "Paired bbox-regression output requires a bbox auxiliary checkpoint."
            )
    elif args.bbox_regression_feature_source != "anchor":
        raise ValueError(
            "--bbox_regression_feature_source only applies together with "
            "--bbox_regression_output_dir."
        )

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc="Hierarchical inference"):
        final_path = args.output_dir / "final" / f"{scene.scene_id}.txt"
        stage1_path = args.output_dir / "stage1" / f"{scene.scene_id}.txt"
        paired_paths_exist = True
        if args.bbox_regression_output_dir is not None:
            paired_paths_exist = (
                (
                    args.bbox_regression_output_dir
                    / "final"
                    / f"{scene.scene_id}.txt"
                ).exists()
                and (
                    args.bbox_regression_output_dir
                    / "stage1"
                    / f"{scene.scene_id}.txt"
                ).exists()
            )
        if (
            args.skip_existing
            and final_path.exists()
            and stage1_path.exists()
            and paired_paths_exist
        ):
            continue

        try:
            if args.bbox_regression_output_dir is None:
                prediction = predict_hierarchical_scene(
                    scene,
                    stage1_model,
                    stage1_tokenizer,
                    stage2_model,
                    stage2_tokenizer,
                    args,
                )
                write_prediction_outputs(prediction, args.output_dir)
                if args.save_region_pcds:
                    write_region_debug_pcds(prediction, args.output_dir)
            else:
                llm_prediction, mlp_prediction = (
                    predict_hierarchical_scene_with_bbox_regression(
                        scene,
                        stage1_model,
                        stage1_tokenizer,
                        stage2_model,
                        stage2_tokenizer,
                        args,
                    )
                )
                write_prediction_outputs(llm_prediction, args.output_dir)
                write_prediction_outputs(
                    mlp_prediction,
                    args.bbox_regression_output_dir,
                )
                if args.save_region_pcds:
                    write_region_debug_pcds(llm_prediction, args.output_dir)
            successful_output_dirs = [args.output_dir]
            if args.bbox_regression_output_dir is not None:
                successful_output_dirs.append(args.bbox_regression_output_dir)
            for successful_output_dir in successful_output_dirs:
                (
                    successful_output_dir
                    / "errors"
                    / f"{scene.scene_id}.txt"
                ).unlink(missing_ok=True)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((scene.scene_id, str(exc)))
            error_dirs = [args.output_dir / "errors"]
            if args.bbox_regression_output_dir is not None:
                error_dirs.append(args.bbox_regression_output_dir / "errors")
            for error_dir in error_dirs:
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / f"{scene.scene_id}.txt").write_text(
                    str(exc),
                    encoding="utf-8",
                )

    if failures:
        print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        for scene_id, error in failures[:10]:
            print(f"{scene_id}: {error}", file=sys.stderr)
        return 1

    print(f"Wrote hierarchical predictions to {args.output_dir}")
    if args.bbox_regression_output_dir is not None:
        print(
            "Wrote paired bbox-regression predictions to "
            f"{args.bbox_regression_output_dir} using feature source "
            f"{args.bbox_regression_feature_source}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
