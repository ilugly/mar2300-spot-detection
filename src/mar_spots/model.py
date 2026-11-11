"""A small fully convolutional baseline for point annotations."""

from math import ceil, floor, isfinite

import numpy as np
import torch
from torch import nn

from .dataset import Point, validate_bounds


def point_target(
    shape: tuple[int, int], points: tuple[Point, ...], radius: float
) -> np.ndarray:
    """Create a binary target with a configurable disk around each center."""
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Expected a positive (height, width)")
    if not isfinite(radius) or radius < 1:
        raise ValueError("radius must be at least one pixel")
    validate_bounds(points, shape)
    height, width = shape
    target = np.zeros(shape, dtype=np.float32)
    for point in points:
        left = max(0, ceil(point.x - radius))
        right = min(width, floor(point.x + radius) + 1)
        top = max(0, ceil(point.y - radius))
        bottom = min(height, floor(point.y + radius) + 1)
        rows, columns = np.ogrid[top:bottom, left:right]
        disk = np.hypot(rows - point.y, columns - point.x) <= radius
        target[top:bottom, left:right][disk] = 1
    return target


class SpotCNN(nn.Sequential):
    """Map normalized float tensors of shape (N, 1, H, W) to pixel logits."""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 1, kernel_size=1),
        )


def spot_loss(logits: torch.Tensor, targets: torch.Tensor, positive_weight: float) -> torch.Tensor:
    """Weighted binary cross-entropy for sparse targets in normalized image patches."""
    if not isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and greater than zero")
    weight = logits.new_tensor(positive_weight)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
