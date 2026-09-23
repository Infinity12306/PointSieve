from enum import Enum

import torch


class PointBackboneType(Enum):
    SCENESCRIPT = "scenescript"
    SONATA = "sonata"


class ProjectorType(Enum):
    LINEAR = "linear"
    MLP = "mlp"


def center_crop_point_tokens(
    point_features: torch.Tensor,
    max_point_tokens: int | None,
) -> torch.Tensor:
    """Center-crop the point-token dimension, removing both sequence ends."""
    if max_point_tokens is None:
        return point_features
    if max_point_tokens <= 0:
        raise ValueError("max_point_tokens must be positive when configured.")

    num_point_tokens = point_features.shape[-2]
    if num_point_tokens <= max_point_tokens:
        return point_features

    excess = num_point_tokens - max_point_tokens
    trim_start = excess // 2
    return point_features[..., trim_start : trim_start + max_point_tokens, :]
