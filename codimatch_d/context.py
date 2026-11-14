"""Project pooling kernels adapted from the MuSc neighborhood rule."""

from collections.abc import Sequence

import numpy as np


def _positive_int(value: int) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0


def pool_grid(features: np.ndarray, grid: tuple[int, int], degree: int) -> np.ndarray:
    """Average in-bounds neighbors for each patch."""
    if len(grid) != 2 or not all(_positive_int(v) for v in grid):
        raise ValueError("grid must be two positive integers")
    if not _positive_int(degree) or degree % 2 != 1:
        raise ValueError("degree must be a positive odd integer")
    values = np.array(features, dtype=np.float32, order="C", copy=True)
    height, width = map(int, grid)
    if values.ndim != 2 or values.shape[0] != height * width or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("features must be finite (height*width, D), D > 0")
    if degree == 1:
        return values
    plane = values.reshape(height, width, -1)
    summed = np.zeros_like(plane)
    counts = np.zeros((height, width, 1), dtype=np.float32)
    radius = degree // 2
    for dy in range(-min(radius, height - 1), min(radius, height - 1) + 1):
        y0, y1 = max(0, -dy), min(height, height - dy)
        for dx in range(-min(radius, width - 1), min(radius, width - 1) + 1):
            x0, x1 = max(0, -dx), min(width, width - dx)
            summed[y0:y1, x0:x1] += plane[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            counts[y0:y1, x0:x1] += 1
    return np.ascontiguousarray((summed / counts).reshape(values.shape), dtype=np.float32)


def pool_reference_views(features: np.ndarray, grids: Sequence[tuple[int, int]],
                         degree: int) -> np.ndarray:
    """Pool each reference view on its own grid."""
    if not grids:
        raise ValueError("at least one reference-view grid is required")
    if any(len(g) != 2 or not all(_positive_int(v) for v in g) for g in grids):
        raise ValueError("all view grids must contain two positive integers")
    sizes = [int(h) * int(w) for h, w in grids]
    if features.ndim != 2 or sum(sizes) != len(features):
        raise ValueError("view grids must cover every reference feature exactly once")
    parts = []
    start = 0
    for grid, size in zip(grids, sizes, strict=True):
        parts.append(pool_grid(features[start:start + size], grid, degree))
        start += size
    return np.concatenate(parts, axis=0)
