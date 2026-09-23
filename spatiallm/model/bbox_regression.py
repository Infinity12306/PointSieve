"""Numeric-token-pooled continuous bbox regression utilities for SpatialLM."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class BboxRegressionHead(nn.Module):
    """Predict center, log-size and doubled-angle yaw from pooled bbox tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 8),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


def shift_bbox_token_positions(
    token_positions: torch.Tensor,
    point_start_end_token_pos: Sequence[tuple],
) -> torch.Tensor:
    """Map pre-expansion token positions to sequences containing point tokens."""
    shifted = token_positions.clone()
    for batch_index, (_, num_patches, point_end_pos) in enumerate(
        point_start_end_token_pos
    ):
        valid = shifted[batch_index] >= point_end_pos
        shifted[batch_index, valid] += int(num_patches) - 1
    return shifted


def pool_bbox_numeric_hidden_states(
    hidden_states: torch.Tensor,
    token_positions: torch.Tensor,
    token_mask: torch.Tensor,
    point_start_end_token_pos: Sequence[tuple],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the last-layer states of all seven numeric bbox fields."""
    shifted_positions = shift_bbox_token_positions(
        token_positions,
        point_start_end_token_pos,
    )
    valid_token_mask = token_mask & (shifted_positions >= 0)
    valid_token_mask = valid_token_mask & (
        shifted_positions < hidden_states.shape[1]
    )
    safe_positions = shifted_positions.clamp(
        min=0,
        max=hidden_states.shape[1] - 1,
    )
    batch_indices = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
    )[:, None, None]
    gathered = hidden_states[batch_indices, safe_positions]
    weights = valid_token_mask.to(gathered.dtype).unsqueeze(-1)
    pooled = (gathered * weights).sum(dim=2)
    pooled = pooled / weights.sum(dim=2).clamp_min(1.0)
    requested_counts = token_mask.sum(dim=-1)
    complete_numeric_mask = (requested_counts > 0) & (
        valid_token_mask.sum(dim=-1) == requested_counts
    )
    return pooled, complete_numeric_mask


def compute_bbox_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    center_weight: float,
    size_weight: float,
    yaw_weight: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute V-DETR-inspired Smooth L1 terms over valid continuous boxes."""
    valid_predictions = predictions[mask].float()
    valid_targets = targets[mask].float()
    if valid_predictions.numel() == 0:
        zero = predictions.float().sum() * 0.0
        return zero, {
            "center_loss": zero.detach(),
            "size_loss": zero.detach(),
            "yaw_loss": zero.detach(),
            "bbox_loss": zero.detach(),
            "num_boxes": zero.detach(),
        }

    center_loss = F.smooth_l1_loss(
        valid_predictions[:, :3],
        valid_targets[:, :3],
        beta=beta,
        reduction="mean",
    )
    size_loss = F.smooth_l1_loss(
        valid_predictions[:, 3:6],
        valid_targets[:, 3:6],
        beta=beta,
        reduction="mean",
    )
    yaw_loss = F.smooth_l1_loss(
        valid_predictions[:, 6:8],
        valid_targets[:, 6:8],
        beta=beta,
        reduction="mean",
    )
    bbox_loss = (
        center_weight * center_loss
        + size_weight * size_loss
        + yaw_weight * yaw_loss
    )
    metrics = {
        "center_loss": center_loss.detach(),
        "size_loss": size_loss.detach(),
        "yaw_loss": yaw_loss.detach(),
        "bbox_loss": bbox_loss.detach(),
        "num_boxes": mask.sum().detach(),
    }
    return bbox_loss, metrics


def apply_bbox_regression_auxiliary_loss(
    model: nn.Module,
    hidden_states: torch.Tensor,
    loss: torch.Tensor | None,
    point_start_end_token_pos: Sequence[tuple],
    numeric_token_positions: torch.Tensor | None,
    numeric_token_mask: torch.Tensor | None,
    targets: torch.Tensor | None,
    mask: torch.Tensor | None,
    loss_scale: float | torch.Tensor,
    accumulation_divisor: float | torch.Tensor,
) -> torch.Tensor | None:
    """Pool numeric-token states, regress boxes and add the weighted loss."""
    model._last_bbox_regression_metrics = {}
    if (
        not getattr(model, "bbox_regression_aux", False)
        or numeric_token_positions is None
        or numeric_token_mask is None
        or targets is None
        or mask is None
    ):
        return loss

    pooled_hidden_states, valid_numeric_mask = pool_bbox_numeric_hidden_states(
        hidden_states,
        numeric_token_positions,
        numeric_token_mask,
        point_start_end_token_pos,
    )
    valid_mask = mask & valid_numeric_mask
    predictions = model.bbox_regression_head(pooled_hidden_states)
    bbox_loss, metrics = compute_bbox_regression_loss(
        predictions,
        targets,
        valid_mask,
        center_weight=model.bbox_regression_center_loss_weight,
        size_weight=model.bbox_regression_size_loss_weight,
        yaw_weight=model.bbox_regression_yaw_loss_weight,
        beta=model.bbox_regression_smooth_l1_beta,
    )
    scale = torch.as_tensor(
        loss_scale,
        dtype=bbox_loss.dtype,
        device=bbox_loss.device,
    )
    weighted_bbox_loss = (
        bbox_loss * model.bbox_regression_loss_weight * scale
    )
    divisor = torch.as_tensor(
        accumulation_divisor,
        dtype=bbox_loss.dtype,
        device=bbox_loss.device,
    ).clamp_min(1.0)
    optimization_bbox_loss = weighted_bbox_loss / divisor
    total_loss = (
        optimization_bbox_loss
        if loss is None
        else loss + optimization_bbox_loss
    )
    model._last_bbox_regression_metrics = {
        **metrics,
        "loss_scale": scale.detach(),
        "weighted_bbox_loss": weighted_bbox_loss.detach(),
        "optimization_bbox_loss": optimization_bbox_loss.detach(),
    }
    return total_loss
