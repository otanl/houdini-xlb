"""Pure NumPy rasterization helpers for Houdini connected geometry."""

from __future__ import annotations

import numpy as np


def _grid_centers(ny: int, nx: int, length_x: float, length_y: float) -> np.ndarray:
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * length_x / nx
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * length_y / ny
    xx, yy = np.meshgrid(xs, ys)
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


def _inside_convex(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(cross >= -1e-9, axis=1) | np.all(cross <= 1e-9, axis=1)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) <= 2:
        return unique
    ordered = sorted(map(tuple, unique))

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def rasterize_points(
    points: np.ndarray,
    classes: np.ndarray,
    ny: int,
    nx: int,
    length_x: float,
    length_y: float,
) -> np.ndarray:
    """Rasterize connected pieces to a world-coordinate height map."""
    if min(ny, nx) < 2 or min(length_x, length_y) <= 0:
        raise ValueError("raster dimensions and domain lengths must be positive")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    classes = np.asarray(classes).reshape(-1)
    if len(points) != len(classes):
        raise ValueError("points and classes must have equal length")
    heightmap = np.zeros((ny, nx), dtype=np.float32)
    flat = heightmap.ravel()
    grid = _grid_centers(ny, nx, length_x, length_y)
    for class_id in np.unique(classes):
        piece = points[classes == class_id]
        hull = _convex_hull(piece[:, :2])
        if len(hull) < 3:
            continue
        mask = _inside_convex(grid, hull)
        flat[mask] = np.maximum(flat[mask], float(piece[:, 2].max()))
    return heightmap
