"""BBox numeric-token to point-token geometric alignment utilities."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from spatiallm.model.bbox_regression import pool_bbox_numeric_hidden_states


class BboxPointAlignmentHead(nn.Module):
    """Map pooled LLM bbox-number hidden states into point-token space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


def yaw_box_overlap_matrix(
    boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return exact pairwise overlap for yaw-only 3D boxes via vectorized SAT."""
    if boxes.ndim != 2 or boxes.shape[-1] != 7:
        raise ValueError(f"boxes must have shape [M, 7], got {tuple(boxes.shape)}")
    if boxes.shape[0] == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=boxes.device)

    boxes = boxes.float()
    centers = boxes[:, :3]
    half_size = boxes[:, 3:6].abs() * 0.5
    yaw = boxes[:, 6]
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    axis_x = torch.stack([cos, sin], dim=-1)
    axis_y = torch.stack([-sin, cos], dim=-1)

    # delta[i, j] points from box i to box j.
    delta_xy = centers[None, :, :2] - centers[:, None, :2]
    delta_z = centers[None, :, 2] - centers[:, None, 2]

    dot_xx = torch.einsum("id,jd->ij", axis_x, axis_x).abs()
    dot_xy = torch.einsum("id,jd->ij", axis_x, axis_y).abs()
    dot_yx = torch.einsum("id,jd->ij", axis_y, axis_x).abs()
    dot_yy = torch.einsum("id,jd->ij", axis_y, axis_y).abs()

    delta_on_ix = torch.einsum("ijd,id->ij", delta_xy, axis_x).abs()
    delta_on_iy = torch.einsum("ijd,id->ij", delta_xy, axis_y).abs()
    delta_on_jx = torch.einsum("ijd,jd->ij", delta_xy, axis_x).abs()
    delta_on_jy = torch.einsum("ijd,jd->ij", delta_xy, axis_y).abs()

    hx = half_size[:, 0]
    hy = half_size[:, 1]
    hz = half_size[:, 2]
    overlap_ix = delta_on_ix <= (
        hx[:, None] + hx[None, :] * dot_xx + hy[None, :] * dot_xy + eps
    )
    overlap_iy = delta_on_iy <= (
        hy[:, None] + hx[None, :] * dot_yx + hy[None, :] * dot_yy + eps
    )
    overlap_jx = delta_on_jx <= (
        hx[None, :] + hx[:, None] * dot_xx + hy[:, None] * dot_yx + eps
    )
    overlap_jy = delta_on_jy <= (
        hy[None, :] + hx[:, None] * dot_xy + hy[:, None] * dot_yy + eps
    )
    overlap_z = delta_z.abs() <= (hz[:, None] + hz[None, :] + eps)
    return overlap_ix & overlap_iy & overlap_jx & overlap_jy & overlap_z


def apply_bbox_point_alignment_auxiliary_loss(
    model: nn.Module,
    hidden_states: torch.Tensor,
    loss: torch.Tensor | None,
    point_start_end_token_pos: Sequence[tuple],
    numeric_token_positions: torch.Tensor | None,
    numeric_token_mask: torch.Tensor | None,
    bboxes: torch.Tensor | None,
    bbox_mask: torch.Tensor | None,
    point_features: Sequence[torch.Tensor] | None,
    point_grid_coords: Sequence[torch.Tensor] | None,
    loss_scale: float | torch.Tensor,
    accumulation_divisor: float | torch.Tensor,
) -> torch.Tensor | None:
    """Align bbox-number hidden states with KNN center point-token features."""
    model._last_bbox_point_alignment_metrics = {}
    if (
        not getattr(model, "bbox_point_alignment_aux", False)
        or numeric_token_positions is None
        or numeric_token_mask is None
        or bboxes is None
        or bbox_mask is None
    ):
        return loss
    if point_features is None or point_grid_coords is None:
        raise RuntimeError(
            "BBox point alignment requires projected point features and matching "
            "final Sonata grid coordinates."
        )
    if len(point_features) != hidden_states.shape[0] or len(point_grid_coords) != hidden_states.shape[0]:
        raise ValueError("Point feature/grid lists must align with the local batch.")

    pooled_hidden, complete_span_mask = pool_bbox_numeric_hidden_states(
        hidden_states,
        numeric_token_positions,
        numeric_token_mask,
        point_start_end_token_pos,
    )
    valid_bbox_mask = bbox_mask & complete_span_mask
    mapped_bbox_features = model.bbox_point_alignment_head(pooled_hidden)

    zero = mapped_bbox_features.float().sum() * 0.0
    positive_loss_sum = zero
    negative_loss_sum = zero
    positive_similarity_sum = zero.detach()
    negative_similarity_sum = zero.detach()
    knn_distance_sum = zero.detach()
    positive_count = 0
    negative_anchor_count = 0
    negative_pair_count = 0

    knn_k = int(model.bbox_point_alignment_knn_k)
    distance_power = float(model.bbox_point_alignment_negative_distance_power)
    detach_point_features = bool(
        model.bbox_point_alignment_detach_point_features
    )
    voxel_size = float(model.point_backbone.final_voxel_size)

    for batch_index in range(hidden_states.shape[0]):
        sample_valid = valid_bbox_mask[batch_index]
        if not torch.any(sample_valid):
            continue

        sample_boxes = bboxes[batch_index, sample_valid].float()
        finite_boxes = torch.isfinite(sample_boxes).all(dim=-1)
        if not torch.any(finite_boxes):
            continue
        sample_boxes = sample_boxes[finite_boxes]
        sample_bbox_features = mapped_bbox_features[batch_index, sample_valid]
        sample_bbox_features = sample_bbox_features[finite_boxes]

        sample_point_features = point_features[batch_index]
        sample_grid = point_grid_coords[batch_index]
        if sample_point_features.ndim != 2:
            raise ValueError(
                "Each point feature tensor must have shape [T, C], got "
                f"{tuple(sample_point_features.shape)}"
            )
        if sample_grid.shape != (sample_point_features.shape[0], 3):
            raise ValueError(
                "Each point grid tensor must align with point features, got "
                f"features={tuple(sample_point_features.shape)}, "
                f"grid={tuple(sample_grid.shape)}"
            )
        if sample_point_features.shape[0] == 0:
            raise ValueError("BBox point alignment received an empty point-token set.")

        point_centers = (sample_grid.float() + 0.5) * voxel_size
        bbox_centers = sample_boxes[:, :3]
        center_to_point = torch.cdist(bbox_centers, point_centers)
        current_k = min(knn_k, sample_point_features.shape[0])
        knn_distances, knn_indices = torch.topk(
            center_to_point,
            k=current_k,
            dim=-1,
            largest=False,
            sorted=False,
        )
        center_point_features = sample_point_features[knn_indices].mean(dim=1)
        if detach_point_features:
            center_point_features = center_point_features.detach()

        bbox_embeddings = F.normalize(
            sample_bbox_features.float(),
            p=2,
            dim=-1,
        )
        point_embeddings = F.normalize(
            center_point_features.float(),
            p=2,
            dim=-1,
        )
        similarities = bbox_embeddings @ point_embeddings.transpose(0, 1)

        diagonal = similarities.diagonal()
        positive_loss_sum = positive_loss_sum + ((1.0 - diagonal) ** 2).sum()
        positive_similarity_sum = positive_similarity_sum + diagonal.detach().sum()
        knn_distance_sum = (
            knn_distance_sum + knn_distances.detach().mean(dim=-1).sum()
        )
        num_boxes = int(sample_boxes.shape[0])
        positive_count += num_boxes

        if num_boxes <= 1:
            continue
        overlap = yaw_box_overlap_matrix(sample_boxes)
        eye = torch.eye(num_boxes, dtype=torch.bool, device=overlap.device)
        negative_mask = (~overlap) & (~eye)
        center_distances = torch.cdist(bbox_centers, bbox_centers)
        raw_weights = center_distances.pow(distance_power) * negative_mask.float()
        weight_sums = raw_weights.sum(dim=-1, keepdim=True)
        valid_negative_anchor = weight_sums.squeeze(-1) > 0
        if not torch.any(valid_negative_anchor):
            continue

        weights = raw_weights / weight_sums.clamp_min(torch.finfo(torch.float32).eps)
        per_anchor_negative_loss = (weights * similarities.square()).sum(dim=-1)
        per_anchor_negative_similarity = (
            weights * similarities.detach()
        ).sum(dim=-1)
        negative_loss_sum = (
            negative_loss_sum
            + per_anchor_negative_loss[valid_negative_anchor].sum()
        )
        negative_similarity_sum = (
            negative_similarity_sum
            + per_anchor_negative_similarity[valid_negative_anchor].sum()
        )
        negative_anchor_count += int(valid_negative_anchor.sum().item())
        negative_pair_count += int(negative_mask.sum().item())

    if positive_count == 0:
        alignment_loss = zero
        positive_loss = zero
        negative_loss = zero
        positive_similarity = zero.detach()
        negative_similarity = zero.detach()
        mean_knn_distance = zero.detach()
    else:
        positive_loss = positive_loss_sum / float(positive_count)
        positive_similarity = positive_similarity_sum / float(positive_count)
        mean_knn_distance = knn_distance_sum / float(positive_count)
        if negative_anchor_count > 0:
            negative_loss = negative_loss_sum / float(negative_anchor_count)
            negative_similarity = (
                negative_similarity_sum / float(negative_anchor_count)
            )
        else:
            negative_loss = zero
            negative_similarity = zero.detach()
        alignment_loss = positive_loss + negative_loss

    scale = torch.as_tensor(
        loss_scale,
        dtype=alignment_loss.dtype,
        device=alignment_loss.device,
    )
    weighted_alignment_loss = (
        alignment_loss * model.bbox_point_alignment_loss_weight * scale
    )
    divisor = torch.as_tensor(
        accumulation_divisor,
        dtype=alignment_loss.dtype,
        device=alignment_loss.device,
    ).clamp_min(1.0)
    optimization_alignment_loss = weighted_alignment_loss / divisor
    total_loss = (
        optimization_alignment_loss
        if loss is None
        else loss + optimization_alignment_loss
    )
    model._last_bbox_point_alignment_metrics = {
        "positive_loss": positive_loss.detach(),
        "negative_loss": negative_loss.detach(),
        "alignment_loss": alignment_loss.detach(),
        "positive_similarity": positive_similarity.detach(),
        "distance_weighted_negative_similarity": negative_similarity.detach(),
        "mean_knn_distance": mean_knn_distance.detach(),
        "num_boxes": alignment_loss.new_tensor(float(positive_count)),
        "num_negative_anchors": alignment_loss.new_tensor(
            float(negative_anchor_count)
        ),
        "num_negative_pairs": alignment_loss.new_tensor(
            float(negative_pair_count)
        ),
        "loss_scale": scale.detach(),
        "weighted_alignment_loss": weighted_alignment_loss.detach(),
        "optimization_alignment_loss": optimization_alignment_loss.detach(),
    }
    return total_loss
