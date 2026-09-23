from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM


@dataclass
class ScorerConfig:
    encoder_feature_dim: int
    point_token_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    coord_scale: float
    score_head_bias: bool = True


class PointTokenScorer(nn.Module):
    """Score final Sonata tokens using projected features and grid positions."""

    def __init__(self, config: ScorerConfig):
        super().__init__()
        self.config = config
        self.coord_scale = float(config.coord_scale)
        input_dim = config.point_token_dim + 6
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.score_head = nn.Linear(
            config.hidden_dim,
            1,
            bias=config.score_head_bias,
        )

    def forward(
        self,
        point_tokens: torch.Tensor,
        grid_coord: torch.Tensor,
        region_center_grid_coord: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        denom = max(self.coord_scale - 1.0, 1.0)
        grid_norm = grid_coord.float() / denom
        center_norm = region_center_grid_coord.float()[:, None, :] / denom
        center_norm = center_norm.expand(-1, point_tokens.shape[1], -1)
        x = torch.cat([point_tokens.float(), grid_norm, center_norm], dim=-1)
        x = self.input_mlp(x) + self.position_mlp(grid_norm)
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        return self.score_head(x).squeeze(-1)


def load_frozen_point_projector(
    model_path,
    dtype_name: str,
    device: torch.device,
) -> nn.Module:
    """Load only the trained SpatialLM projector needed by offline scorers."""
    dtype = (
        "auto"
        if dtype_name == "auto"
        else {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_name]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    point_projector = model.point_proj
    point_projector.requires_grad_(False)
    point_projector.eval()
    point_projector.to(device)
    return point_projector


@torch.no_grad()
def project_encoder_features(
    point_projector: nn.Module,
    encoder_features: torch.Tensor,
) -> torch.Tensor:
    """Project cached point-encoder context into SpatialLM point tokens."""
    projector_dtype = next(point_projector.parameters()).dtype
    return point_projector(encoder_features.to(projector_dtype)).float()


def packed_point_tokens_to_padded(
    point_tokens: torch.Tensor,
    grid_coord: torch.Tensor,
    token_batch: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert packed final tokens to a differentiable padded scorer batch."""
    if point_tokens.ndim != 2:
        raise ValueError(f"point_tokens must be [T, C], got {tuple(point_tokens.shape)}")
    if grid_coord.shape != (point_tokens.shape[0], 3):
        raise ValueError(
            "grid_coord must align with point_tokens and have shape [T, 3], "
            f"got {tuple(grid_coord.shape)}"
        )
    if token_batch.shape != (point_tokens.shape[0],):
        raise ValueError(
            "token_batch must align with point_tokens and have shape [T], "
            f"got {tuple(token_batch.shape)}"
        )

    counts = torch.bincount(token_batch.long(), minlength=batch_size)
    if torch.any(counts == 0):
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Encoded point-token batch has empty samples: {empty}")

    max_tokens = int(counts.max().item())
    starts = torch.cumsum(counts, dim=0) - counts
    local_index = torch.arange(
        point_tokens.shape[0], device=point_tokens.device, dtype=torch.long
    ) - starts[token_batch.long()]

    padded_tokens = point_tokens.new_zeros(
        (batch_size, max_tokens, point_tokens.shape[-1])
    )
    padded_grid = torch.zeros(
        (batch_size, max_tokens, 3),
        dtype=grid_coord.dtype,
        device=grid_coord.device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_tokens), dtype=torch.bool, device=point_tokens.device
    )
    padded_tokens[token_batch, local_index] = point_tokens
    padded_grid[token_batch, local_index] = grid_coord
    attention_mask[token_batch, local_index] = True

    grid_float = padded_grid.float()
    min_grid = grid_float.masked_fill(~attention_mask[..., None], torch.inf).amin(dim=1)
    max_grid = grid_float.masked_fill(~attention_mask[..., None], -torch.inf).amax(dim=1)
    region_centers = (min_grid + max_grid) * 0.5
    return padded_tokens, padded_grid, attention_mask, region_centers


@torch.no_grad()
def batched_point_token_bbox_overlap_labels(
    grid_coord: torch.Tensor,
    token_mask: torch.Tensor,
    gt_bboxes: torch.Tensor,
    voxel_size: float,
) -> torch.Tensor:
    """Build GT labels for padded tokens and padded yaw-only object bboxes on GPU."""
    if grid_coord.ndim != 3 or grid_coord.shape[-1] != 3:
        raise ValueError(f"grid_coord must be [B, T, 3], got {tuple(grid_coord.shape)}")
    if token_mask.shape != grid_coord.shape[:2]:
        raise ValueError("token_mask must have shape [B, T].")
    if gt_bboxes.ndim != 3 or gt_bboxes.shape[0] != grid_coord.shape[0] or gt_bboxes.shape[-1] != 7:
        raise ValueError(f"gt_bboxes must be [B, K, 7], got {tuple(gt_bboxes.shape)}")
    if gt_bboxes.shape[1] == 0:
        return torch.zeros_like(token_mask)

    boxes = gt_bboxes.to(device=grid_coord.device, dtype=torch.float32)
    box_valid = torch.isfinite(boxes).all(dim=-1)
    boxes = torch.nan_to_num(boxes)
    box_center = boxes[..., 0:3]
    box_half = boxes[..., 3:6].abs() * 0.5
    box_angle_z = boxes[..., 6]

    cos = torch.cos(box_angle_z)
    sin = torch.sin(box_angle_z)
    box_axis_x = torch.stack([cos, sin], dim=-1)
    box_axis_y = torch.stack([-sin, cos], dim=-1)

    half_voxel = float(voxel_size) * 0.5
    voxel_center = (grid_coord.float() + 0.5) * float(voxel_size)
    delta = voxel_center[:, :, None, :] - box_center[:, None, :, :]
    delta_xy = delta[..., :2]

    hx = box_half[..., 0]
    hy = box_half[..., 1]
    hz = box_half[..., 2]
    z_overlap = delta[..., 2].abs() <= half_voxel + hz[:, None, :]

    box_radius_world_x = hx * box_axis_x[..., 0].abs() + hy * box_axis_y[..., 0].abs()
    box_radius_world_y = hx * box_axis_x[..., 1].abs() + hy * box_axis_y[..., 1].abs()
    overlap_world_x = delta[..., 0].abs() <= half_voxel + box_radius_world_x[:, None, :]
    overlap_world_y = delta[..., 1].abs() <= half_voxel + box_radius_world_y[:, None, :]

    delta_on_box_x = (delta_xy * box_axis_x[:, None, :, :]).sum(dim=-1)
    delta_on_box_y = (delta_xy * box_axis_y[:, None, :, :]).sum(dim=-1)
    voxel_radius_on_box_x = half_voxel * (
        box_axis_x[..., 0].abs() + box_axis_x[..., 1].abs()
    )
    voxel_radius_on_box_y = half_voxel * (
        box_axis_y[..., 0].abs() + box_axis_y[..., 1].abs()
    )
    overlap_box_x = delta_on_box_x.abs() <= hx[:, None, :] + voxel_radius_on_box_x[:, None, :]
    overlap_box_y = delta_on_box_y.abs() <= hy[:, None, :] + voxel_radius_on_box_y[:, None, :]

    overlap = (
        z_overlap
        & overlap_world_x
        & overlap_world_y
        & overlap_box_x
        & overlap_box_y
        & box_valid[:, None, :]
    )
    return overlap.any(dim=-1) & token_mask


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_weight: float | torch.Tensor | None,
) -> torch.Tensor:
    weight = None
    if pos_weight is not None:
        weight = torch.as_tensor(
            pos_weight,
            dtype=torch.float32,
            device=logits.device,
        )
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), labels.float(), reduction="none", pos_weight=weight
    )
    valid = attention_mask.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def scorer_keep_indices(
    scores: torch.Tensor,
    threshold: float,
    min_keep: int,
    max_keep: int,
) -> torch.Tensor:
    """Threshold scores, enforce keep bounds, and preserve Sonata token order."""
    num_tokens = int(scores.numel())
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    min_keep = min(max(int(min_keep), 0), num_tokens)
    max_keep = min(max(int(max_keep), min_keep), num_tokens)

    keep_indices = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
    if keep_indices.numel() < min_keep:
        keep_indices = torch.topk(scores, k=min_keep).indices
    elif keep_indices.numel() > max_keep:
        selected_scores = scores[keep_indices]
        top_local = torch.topk(selected_scores, k=max_keep).indices
        keep_indices = keep_indices[top_local]
    return keep_indices.sort().values


def score_and_hard_topk_point_tokens(
    scorer: PointTokenScorer,
    point_tokens: torch.Tensor,
    grid_coord: torch.Tensor,
    token_batch: torch.Tensor,
    batch_size: int,
    top_k: int,
    max_point_tokens: int | None,
    detach_scorer_input: bool = True,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """Use a frozen scorer for order-preserving hard top-k token selection."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    cropped_tokens: list[torch.Tensor] = []
    cropped_grids: list[torch.Tensor] = []
    region_centers: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        sample_mask = token_batch == batch_index
        sample_tokens = point_tokens[sample_mask]
        sample_grid = grid_coord[sample_mask]
        if sample_tokens.shape[0] == 0:
            raise ValueError(f"Encoded point-token batch has empty sample {batch_index}.")

        grid_float = sample_grid.float()
        region_centers.append(
            (grid_float.amin(dim=0) + grid_float.amax(dim=0)) * 0.5
        )
        if max_point_tokens is not None and sample_tokens.shape[0] > max_point_tokens:
            start = (sample_tokens.shape[0] - max_point_tokens) // 2
            end = start + max_point_tokens
            sample_tokens = sample_tokens[start:end]
            sample_grid = sample_grid[start:end]
        cropped_tokens.append(sample_tokens)
        cropped_grids.append(sample_grid)

    max_tokens = max(tokens.shape[0] for tokens in cropped_tokens)
    padded_tokens = point_tokens.new_zeros(
        (batch_size, max_tokens, point_tokens.shape[-1])
    )
    padded_grid = torch.zeros(
        (batch_size, max_tokens, 3),
        dtype=grid_coord.dtype,
        device=grid_coord.device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_tokens),
        dtype=torch.bool,
        device=point_tokens.device,
    )
    for batch_index, (sample_tokens, sample_grid) in enumerate(
        zip(cropped_tokens, cropped_grids)
    ):
        count = sample_tokens.shape[0]
        padded_tokens[batch_index, :count] = sample_tokens
        padded_grid[batch_index, :count] = sample_grid
        attention_mask[batch_index, :count] = True

    scorer_tokens = padded_tokens.detach() if detach_scorer_input else padded_tokens
    if scorer.training:
        scorer.eval()
    with torch.no_grad():
        logits = scorer(
            scorer_tokens,
            padded_grid,
            torch.stack(region_centers),
            attention_mask,
        )

    selected: list[torch.Tensor] = []
    kept_count = logits.new_zeros(())
    for batch_index, sample_tokens in enumerate(cropped_tokens):
        count = sample_tokens.shape[0]
        keep_count = min(top_k, count)
        keep = torch.topk(logits[batch_index, :count], k=keep_count).indices
        keep = keep.sort().values
        selected.append(sample_tokens[keep].unsqueeze(0))
        kept_count = kept_count + keep_count

    valid_logits = logits[attention_mask]
    valid_count = attention_mask.sum()
    metrics: dict[str, torch.Tensor] = {
        "scorer_keep_ratio": (kept_count / valid_count.clamp_min(1)).detach(),
        "scorer_tokens": valid_count.detach(),
        "scorer_kept_tokens": kept_count.detach(),
        "scorer_score_mean": valid_logits.mean().detach(),
        "scorer_score_std": valid_logits.float().std(unbiased=False).detach(),
    }
    return selected, metrics


def score_and_threshold_point_tokens(
    scorer: PointTokenScorer,
    point_tokens: torch.Tensor,
    grid_coord: torch.Tensor,
    token_batch: torch.Tensor,
    batch_size: int,
    threshold: float,
    min_keep: int,
    max_keep: int,
    raw_token_threshold_exclusive: int = 0,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Conditionally filter a packed batch and preserve token/grid order.

    Samples with ``raw_token_count <= raw_token_threshold_exclusive`` bypass
    scorer selection and retain every token.  The default value ``0`` keeps
    the historical all-region behavior because a valid encoded sample always
    contains at least one point token.
    """
    if raw_token_threshold_exclusive < 0:
        raise ValueError("raw_token_threshold_exclusive must be non-negative.")
    if raw_token_threshold_exclusive > max_keep:
        raise ValueError(
            "raw_token_threshold_exclusive must be no larger than max_keep; "
            "otherwise a bypassed sample could exceed the configured token cap."
        )
    padded_tokens, padded_grid, attention_mask, centers = (
        packed_point_tokens_to_padded(
            point_tokens,
            grid_coord,
            token_batch,
            batch_size,
        )
    )
    if scorer.training:
        scorer.eval()
    with torch.no_grad():
        logits = scorer(
            padded_tokens.detach(),
            padded_grid,
            centers,
            attention_mask,
        )
        probabilities = torch.sigmoid(logits)

    selected_tokens: list[torch.Tensor] = []
    selected_grids: list[torch.Tensor] = []
    kept_count = logits.new_zeros(())
    threshold_count = logits.new_zeros(())
    applied_regions = logits.new_zeros(())
    for batch_index in range(batch_size):
        valid_count = int(attention_mask[batch_index].sum().item())
        valid_scores = probabilities[batch_index, :valid_count]
        if valid_count > raw_token_threshold_exclusive:
            keep = scorer_keep_indices(
                valid_scores,
                threshold,
                min_keep,
                max_keep,
            )
            applied_regions = applied_regions + 1
        else:
            keep = torch.arange(
                valid_count,
                dtype=torch.long,
                device=valid_scores.device,
            )
        selected_tokens.append(
            padded_tokens[batch_index, keep].unsqueeze(0)
        )
        selected_grids.append(padded_grid[batch_index, keep])
        kept_count = kept_count + keep.numel()
        threshold_count = threshold_count + (
            valid_scores >= threshold
        ).sum()

    valid_logits = logits[attention_mask]
    valid_count = attention_mask.sum()
    region_count = logits.new_tensor(float(batch_size)).clamp_min(1)
    metrics: dict[str, torch.Tensor] = {
        "scorer_keep_ratio": (
            kept_count / valid_count.clamp_min(1)
        ).detach(),
        "scorer_threshold_keep_ratio": (
            threshold_count / valid_count.clamp_min(1)
        ).detach(),
        "scorer_tokens": valid_count.detach(),
        "scorer_kept_tokens": kept_count.detach(),
        "scorer_score_mean": torch.sigmoid(valid_logits).mean().detach(),
        "scorer_score_std": torch.sigmoid(valid_logits).float().std(
            unbiased=False
        ).detach(),
        "scorer_applied_region_ratio": (
            applied_regions / region_count
        ).detach(),
        "scorer_bypassed_region_ratio": (
            (region_count - applied_regions) / region_count
        ).detach(),
    }
    return selected_tokens, selected_grids, metrics


def score_and_select_point_tokens(
    scorer: PointTokenScorer,
    point_tokens: torch.Tensor,
    grid_coord: torch.Tensor,
    token_batch: torch.Tensor,
    gt_bboxes: torch.Tensor,
    voxel_size: float,
    threshold: float,
    min_keep: int,
    max_keep: int,
    pos_weight: float | None,
    detach_scorer_input: bool = True,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Score a packed batch, supervise with GT mask, and hard-filter LLM tokens."""
    batch_size = int(gt_bboxes.shape[0])
    padded_tokens, padded_grid, attention_mask, centers = packed_point_tokens_to_padded(
        point_tokens, grid_coord, token_batch, batch_size
    )
    scorer_tokens = padded_tokens.detach() if detach_scorer_input else padded_tokens
    logits = scorer(scorer_tokens, padded_grid, centers, attention_mask)
    gt_labels = batched_point_token_bbox_overlap_labels(
        padded_grid, attention_mask, gt_bboxes, voxel_size
    )
    scorer_loss = masked_bce_with_logits(
        logits, gt_labels, attention_mask, pos_weight
    )

    probabilities = torch.sigmoid(logits.detach())
    selected: list[torch.Tensor] = []
    kept_count = logits.new_zeros(())
    for batch_index in range(batch_size):
        valid_scores = probabilities[batch_index][attention_mask[batch_index]]
        keep = scorer_keep_indices(valid_scores, threshold, min_keep, max_keep)
        selected.append(padded_tokens[batch_index, keep].unsqueeze(0))
        kept_count = kept_count + keep.numel()

    valid = attention_mask
    predictions = (probabilities >= threshold) & valid
    targets = gt_labels & valid
    true_positive = (predictions & targets).sum()
    false_positive = (predictions & ~targets & valid).sum()
    false_negative = (~predictions & targets).sum()
    precision = true_positive.float() / (true_positive + false_positive).clamp_min(1)
    recall = true_positive.float() / (true_positive + false_negative).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    metrics: dict[str, torch.Tensor] = {
        "scorer_loss": scorer_loss.detach(),
        "scorer_precision": precision.detach(),
        "scorer_recall": recall.detach(),
        "scorer_f1": f1.detach(),
        "scorer_positive_ratio": (
            targets.sum().float() / valid.sum().clamp_min(1)
        ).detach(),
        "scorer_keep_ratio": (
            kept_count / valid.sum().clamp_min(1)
        ).detach(),
        "scorer_tokens": valid.sum().detach(),
        "scorer_kept_tokens": kept_count.detach(),
    }
    return selected, scorer_loss, metrics
