#!/usr/bin/env python3
"""Hierarchical inference with scorer filtering only for long Stage-2 regions.

This replaces ``deprecated_inference_hierarchical_scorer.py`` for
RawTokenGT1024 evaluation. After Sonata encoding, regions with a raw
point-token count strictly greater than the configured threshold are filtered
by the scorer; shorter regions bypass the scorer and pass every encoded point
token directly to the LLM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import MethodType

import torch
from tqdm import tqdm

from inference.inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    load_model_and_tokenizer,
    load_scenes,
    predict_hierarchical_scene,
    write_prediction_outputs,
    write_region_debug_pcds,
)
from spatiallm.model import PointBackboneType
from spatiallm.model.point_token_scorer import PointTokenScorer, ScorerConfig


DEFAULT_RAW_TOKEN_THRESHOLD_EXCLUSIVE = 1024
DEFAULT_SCORER_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "scorer"
)


def latest_scorer_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if (path / "scorer.pt").is_file():
        return path / "scorer.pt"

    checkpoints = []
    for candidate in path.glob("checkpoint-*"):
        scorer_path = candidate / "scorer.pt"
        suffix = candidate.name.removeprefix("checkpoint-")
        if candidate.is_dir() and scorer_path.is_file() and suffix.isdigit():
            checkpoints.append((int(suffix), scorer_path))
    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]
    raise FileNotFoundError(f"No scorer.pt found under {path}")


def load_point_token_scorer(path: Path, device: str) -> PointTokenScorer:
    scorer_path = latest_scorer_checkpoint(path)
    checkpoint = torch.load(scorer_path, map_location="cpu")
    scorer = PointTokenScorer(ScorerConfig(**checkpoint["config"]))
    scorer.load_state_dict(checkpoint["model"])
    scorer.eval()
    scorer.requires_grad_(False)
    scorer.to(device=device, dtype=torch.float32)
    print(f"Loaded point-token scorer: {scorer_path}")
    return scorer


def scorer_keep_indices(
    scores: torch.Tensor,
    threshold: float,
    min_keep: int,
    max_keep: int,
) -> torch.Tensor:
    token_count = int(scores.numel())
    if token_count == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if max_keep <= 0:
        raise ValueError("--scorer_max_keep must be positive.")
    if min_keep < 0:
        raise ValueError("--scorer_min_keep must be non-negative.")

    min_keep = min(min_keep, token_count)
    max_keep = min(max_keep, token_count)
    if min_keep > max_keep:
        min_keep = max_keep
    keep_indices = torch.nonzero(
        scores >= threshold,
        as_tuple=False,
    ).flatten()
    if keep_indices.numel() < min_keep:
        keep_indices = torch.topk(scores, k=min_keep).indices
    elif keep_indices.numel() > max_keep:
        selected_scores = scores[keep_indices]
        keep_indices = keep_indices[
            torch.topk(selected_scores, k=max_keep).indices
        ]
    return keep_indices.sort().values


def install_conditional_scorer_point_filter(
    stage2_model,
    scorer: PointTokenScorer,
    args: argparse.Namespace,
) -> None:
    """Filter only regions whose raw encoded token count exceeds the threshold."""

    scorer_device = torch.device(args.device)
    raw_threshold = int(args.scorer_raw_token_threshold_exclusive)
    raw_upper_threshold = getattr(
        args,
        "scorer_raw_token_upper_threshold_exclusive",
        None,
    )
    if raw_upper_threshold is not None:
        raw_upper_threshold = int(raw_upper_threshold)

    stage2_model._fixed_region_last_raw_point_token_count = None
    stage2_model._fixed_region_last_kept_point_token_count = None
    stage2_model._conditional_scorer_last_applied = None
    stage2_model._conditional_scorer_total_regions = 0
    stage2_model._conditional_scorer_filtered_regions = 0
    stage2_model._conditional_scorer_bypassed_regions = 0
    stage2_model._conditional_scorer_total_raw_tokens = 0
    stage2_model._conditional_scorer_total_kept_tokens = 0

    def conditional_scorer_forward_point_cloud(
        self,
        point_cloud: torch.Tensor,
        device,
        dtype,
        point_token_keep_bboxes=None,
        return_grid_coord: bool = False,
    ):
        if self.point_backbone_type != PointBackboneType.SONATA:
            raise NotImplementedError(
                "Point-token scorer inference currently supports Sonata only."
            )

        self.point_backbone.to(torch.float32)
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        if point_cloud.shape[0] == 0:
            raise ValueError("Point cloud has no valid points after removing NaNs.")

        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        input_dict = {
            "coord": feats[:, :3].to(device),
            "grid_coord": coords.to(device),
            "feat": feats.to(device),
            "batch": torch.zeros(
                coords.shape[0],
                dtype=torch.long,
                device=device,
            ),
            "return_grid_coord": True,
        }

        with torch.inference_mode():
            encoded = self.point_backbone(input_dict)
            context = encoded["context"]
            grid_coord = encoded["grid_coord"].to(torch.int32)
            if context.shape[0] == 0:
                raise ValueError("Point encoder produced zero point tokens.")

            projector_dtype = next(self.point_proj.parameters()).dtype
            point_tokens = self.point_proj(context.to(projector_dtype))
            raw_token_count = int(point_tokens.shape[0])
            apply_scorer = (
                raw_token_count > raw_threshold
                and (
                    raw_upper_threshold is None
                    or raw_token_count < raw_upper_threshold
                )
            )

            if apply_scorer:
                center = (
                    grid_coord.float().amin(dim=0)
                    + grid_coord.float().amax(dim=0)
                ) * 0.5
                attention_mask = torch.ones(
                    1,
                    grid_coord.shape[0],
                    dtype=torch.bool,
                    device=scorer_device,
                )

                fastpath_enabled = None
                if (
                    args.scorer_disable_mha_fastpath
                    and hasattr(torch.backends, "mha")
                ):
                    fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
                    torch.backends.mha.set_fastpath_enabled(False)
                try:
                    logits = scorer(
                        point_tokens.float().unsqueeze(0).to(scorer_device),
                        grid_coord.float().unsqueeze(0).to(scorer_device),
                        center.float().unsqueeze(0).to(scorer_device),
                        attention_mask,
                    ).squeeze(0)
                finally:
                    if fastpath_enabled is not None:
                        torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

                scores = torch.sigmoid(logits)
                keep_indices = scorer_keep_indices(
                    scores,
                    args.scorer_threshold,
                    args.scorer_min_keep,
                    args.scorer_max_keep,
                ).to(point_tokens.device)
            else:
                keep_indices = torch.arange(
                    raw_token_count,
                    dtype=torch.long,
                    device=point_tokens.device,
                )

            kept_token_count = int(keep_indices.numel())
            self._fixed_region_last_raw_point_token_count = raw_token_count
            self._fixed_region_last_kept_point_token_count = kept_token_count
            self._conditional_scorer_last_applied = apply_scorer
            self._conditional_scorer_total_regions += 1
            self._conditional_scorer_total_raw_tokens += raw_token_count
            self._conditional_scorer_total_kept_tokens += kept_token_count
            if apply_scorer:
                self._conditional_scorer_filtered_regions += 1
            else:
                self._conditional_scorer_bypassed_regions += 1

            if args.scorer_debug:
                branch = "filtered" if apply_scorer else "bypassed"
                print(
                    "conditional scorer tokens: "
                    f"raw={raw_token_count}, keep={kept_token_count}, "
                    f"raw_threshold_exclusive={raw_threshold}, branch={branch}, "
                    f"raw_upper_threshold_exclusive={raw_upper_threshold}, "
                    f"score_threshold={args.scorer_threshold}"
                )

            selected_tokens = point_tokens[keep_indices].to(dtype).unsqueeze(0)
            if return_grid_coord:
                return selected_tokens, grid_coord[keep_indices]
            return selected_tokens

    stage2_model.forward_point_cloud = MethodType(
        conditional_scorer_forward_point_cloud,
        stage2_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical inference that applies scorer filtering only when a "
            "Stage-2 region has raw_point_token_count > threshold."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--data_json",
        type=Path,
        help="Stage1-style JSON dataset, e.g. spatiallm_stage1_region_test.json.",
    )
    input_group.add_argument(
        "-p",
        "--point_cloud",
        type=Path,
        help="PLY file or directory of PLY files.",
    )
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--stage1_model_path",
        default="artifacts/stage1",
        help="Stage-1 checkpoint used to predict regions.",
    )
    parser.add_argument(
        "--stage2_model_path",
        default="artifacts/stage2_scorer_filtered",
        help="Stage-2 checkpoint used to predict object bboxes.",
    )
    parser.add_argument(
        "--scorer_path",
        type=Path,
        default=DEFAULT_SCORER_PATH,
        help="Scorer checkpoint dir, scorer.pt file, or checkpoint parent.",
    )
    parser.add_argument("--scorer_threshold", type=float, default=0.5)
    parser.add_argument("--scorer_max_keep", type=int, default=4096)
    parser.add_argument("--scorer_min_keep", type=int, default=1)
    parser.add_argument(
        "--scorer_raw_token_threshold_exclusive",
        type=int,
        default=DEFAULT_RAW_TOKEN_THRESHOLD_EXCLUSIVE,
        help=(
            "Apply scorer filtering only when the raw encoded point-token "
            "count is strictly greater than this value."
        ),
    )
    parser.add_argument(
        "--scorer_raw_token_upper_threshold_exclusive",
        type=int,
        help=(
            "Optional exclusive upper routing bound. Regions at or above "
            "this raw-token count bypass scorer filtering."
        ),
    )
    parser.add_argument(
        "--scorer_disable_mha_fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scorer_debug", action="store_true")
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
    parser.add_argument("--bbox_nms_iou", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--save_region_pcds", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    if args.scorer_max_keep <= 0:
        parser.error("--scorer_max_keep must be positive.")
    if args.scorer_min_keep < 0:
        parser.error("--scorer_min_keep must be non-negative.")
    if args.scorer_raw_token_threshold_exclusive < 0:
        parser.error(
            "--scorer_raw_token_threshold_exclusive must be non-negative."
        )
    if (
        args.scorer_raw_token_upper_threshold_exclusive is not None
        and args.scorer_raw_token_upper_threshold_exclusive
        <= args.scorer_raw_token_threshold_exclusive
    ):
        parser.error(
            "--scorer_raw_token_upper_threshold_exclusive must be greater "
            "than --scorer_raw_token_threshold_exclusive."
        )
    if args.num_shards < 1:
        parser.error("--num_shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must satisfy 0 <= index < num_shards.")
    return args


def print_conditional_summary(
    stage2_model,
    raw_threshold: int,
    raw_upper_threshold: int | None = None,
) -> None:
    total_regions = int(stage2_model._conditional_scorer_total_regions)
    filtered_regions = int(stage2_model._conditional_scorer_filtered_regions)
    bypassed_regions = int(stage2_model._conditional_scorer_bypassed_regions)
    total_raw_tokens = int(stage2_model._conditional_scorer_total_raw_tokens)
    total_kept_tokens = int(stage2_model._conditional_scorer_total_kept_tokens)
    print(
        "Conditional scorer summary: "
        f"raw_threshold_exclusive={raw_threshold}, "
        f"raw_upper_threshold_exclusive={raw_upper_threshold}, "
        f"regions={total_regions}, filtered_regions={filtered_regions}, "
        f"bypassed_regions={bypassed_regions}, "
        f"raw_tokens={total_raw_tokens}, kept_tokens={total_kept_tokens}"
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(args)
    if not scenes:
        raise ValueError("No scenes found for inference.")

    stage1_model, stage1_tokenizer = load_model_and_tokenizer(
        args.stage1_model_path,
        args.inference_dtype,
        args.device,
    )
    stage2_model, stage2_tokenizer = load_model_and_tokenizer(
        args.stage2_model_path,
        args.inference_dtype,
        args.device,
    )
    scorer = load_point_token_scorer(args.scorer_path, args.device)
    install_conditional_scorer_point_filter(stage2_model, scorer, args)

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc="Hierarchical conditional scorer inference"):
        final_path = args.output_dir / "final" / f"{scene.scene_id}.txt"
        stage1_path = args.output_dir / "stage1" / f"{scene.scene_id}.txt"
        if args.skip_existing and final_path.exists() and stage1_path.exists():
            continue

        try:
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
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((scene.scene_id, str(exc)))
            error_dir = args.output_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / f"{scene.scene_id}.txt").write_text(
                str(exc),
                encoding="utf-8",
            )

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

    print(f"Wrote conditional scorer predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
