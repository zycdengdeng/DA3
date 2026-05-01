"""2.5-D LiDAR ground-height grid for AA-HAD road snapping.

A single planar ground from RANSAC is too coarse for an intersection
(crown for drainage, multiple meeting streets at different elevations,
sidewalks). Instead we build a **2.5-D height map** in world XY:

    grid[i, j] = low quantile of LiDAR Z over cell (i, j)

For an AA-HAD candidate point at world XY ``(x, y)``:

* If the cell at ``(x, y)`` has enough LiDAR samples, look up its
  ground height ``z_g``.
* If the AA-HAD predicted ``z_aahad`` lies within ``max_dz`` of
  ``z_g``, it is treated as a road point and **snapped to ``z_g``**.
* Otherwise it is left as-is (not ground — possibly a wall, sign, etc.).

This pulls the road-curvature artefact (DA3 over-predicts depth near
the camera, road appears to bow upward) onto the actual LiDAR-measured
road surface, while leaving everything else untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GroundHeightGrid:
    """A 2.5-D ground-height map with regular XY binning."""

    cell_size: float
    x_min: float
    y_min: float
    height: np.ndarray  # (n_y, n_x) float32, NaN where no data
    count: np.ndarray  # (n_y, n_x) int32, samples per cell

    @property
    def shape_xy(self) -> tuple[int, int]:
        return int(self.height.shape[1]), int(self.height.shape[0])

    def query(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each ``(x, y)`` row, return ``(z_ground, valid)``."""
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must be (N, 2); got {xy.shape}")
        ix = np.floor((xy[:, 0] - self.x_min) / self.cell_size).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y_min) / self.cell_size).astype(np.int64)
        H, W = self.height.shape
        in_grid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        z = np.full(xy.shape[0], np.nan, dtype=np.float32)
        if in_grid.any():
            z_in = self.height[iy[in_grid], ix[in_grid]]
            z[in_grid] = z_in
        valid = in_grid & np.isfinite(z)
        return z, valid


def build_ground_height_grid(
    lidar_xyz: np.ndarray,
    cell_size: float,
    *,
    low_quantile: float = 0.1,
    min_pts_per_cell: int = 3,
    z_clip: tuple[float, float] | None = None,
) -> GroundHeightGrid:
    """Bin world-frame LiDAR points in XY and report a low-quantile Z
    per cell.

    Parameters
    ----------
    lidar_xyz : (N, 3) world XYZ.
    cell_size : metres (typical 0.25–0.5 for intersection roads).
    low_quantile : 0.0–1.0; fraction of points whose Z is at or below
        the reported value. ``0.1`` picks the 10th percentile, which
        suppresses overhanging structures (trees, signs) while staying
        on the road surface.
    min_pts_per_cell : cells with fewer samples report NaN.
    z_clip : optional ``(z_lo, z_hi)`` filter applied before binning.
        Keeps obvious non-ground returns (sky, tall buildings) out of
        the grid.

    Returns
    -------
    :class:`GroundHeightGrid`
    """
    if lidar_xyz.ndim != 2 or lidar_xyz.shape[1] != 3:
        raise ValueError(f"lidar_xyz must be (N, 3); got {lidar_xyz.shape}")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    if not (0.0 < low_quantile < 1.0):
        raise ValueError("low_quantile must be in (0, 1)")

    pts = np.asarray(lidar_xyz, dtype=np.float64)
    if z_clip is not None:
        z_lo, z_hi = z_clip
        m = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
        pts = pts[m]
    if pts.shape[0] == 0:
        return GroundHeightGrid(
            cell_size=cell_size, x_min=0.0, y_min=0.0,
            height=np.zeros((1, 1), dtype=np.float32) * np.nan,
            count=np.zeros((1, 1), dtype=np.int32),
        )

    x_min = float(np.floor(pts[:, 0].min() / cell_size) * cell_size)
    y_min = float(np.floor(pts[:, 1].min() / cell_size) * cell_size)
    x_max = float(np.ceil(pts[:, 0].max() / cell_size) * cell_size)
    y_max = float(np.ceil(pts[:, 1].max() / cell_size) * cell_size)
    n_x = max(1, int(round((x_max - x_min) / cell_size)))
    n_y = max(1, int(round((y_max - y_min) / cell_size)))

    ix = np.floor((pts[:, 0] - x_min) / cell_size).astype(np.int64).clip(0, n_x - 1)
    iy = np.floor((pts[:, 1] - y_min) / cell_size).astype(np.int64).clip(0, n_y - 1)
    flat = iy * n_x + ix
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    z_s = pts[order, 2]

    height = np.full((n_y, n_x), np.nan, dtype=np.float32)
    count = np.zeros((n_y, n_x), dtype=np.int32)
    if flat_s.size > 0:
        diff = np.diff(flat_s, prepend=flat_s[0] - 1)
        starts = np.flatnonzero(diff != 0)
        ends = np.concatenate([starts[1:], [flat_s.size]])
        for s, e in zip(starts, ends):
            n = int(e - s)
            cell_id = int(flat_s[s])
            iy_c, ix_c = divmod(cell_id, n_x)
            count[iy_c, ix_c] = n
            if n >= min_pts_per_cell:
                height[iy_c, ix_c] = float(np.quantile(z_s[s:e], low_quantile))

    return GroundHeightGrid(
        cell_size=cell_size, x_min=x_min, y_min=y_min,
        height=height, count=count,
    )


def snap_to_ground(
    points: np.ndarray,
    grid: GroundHeightGrid,
    *,
    max_dz: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap road-like points onto the LiDAR-derived ground.

    For each point, look up the cell's ground Z. If the input Z lies
    within ``max_dz`` of the grid Z, replace input Z with grid Z.
    Otherwise the point is left unchanged (treated as non-road).

    Returns
    -------
    snapped : (N, 3) float32 — points with Z possibly replaced.
    is_ground : (N,) bool — which points were snapped.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {points.shape}")
    out = points.astype(np.float32, copy=True)
    if points.size == 0:
        return out, np.zeros(0, dtype=bool)
    z_g, valid = grid.query(points[:, :2])
    near = valid & (np.abs(points[:, 2] - z_g) <= float(max_dz))
    out[near, 2] = z_g[near]
    return out, near


__all__ = [
    "GroundHeightGrid",
    "build_ground_height_grid",
    "snap_to_ground",
]
