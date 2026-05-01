"""Tests for the 2.5-D LiDAR ground-height grid + snap."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.pipeline.ground_height_grid import (
    build_ground_height_grid,
    snap_to_ground,
)


def test_build_grid_low_quantile_picks_floor():
    """In a single XY cell, 80% of points are at z=0 (ground) and
    20% at z=2 (overhang). Low-quantile 0.1 should pick ~0."""
    rng = np.random.default_rng(0)
    n_ground = 80
    n_over = 20
    pts_ground = np.column_stack([
        rng.uniform(0, 0.2, size=n_ground),
        rng.uniform(0, 0.2, size=n_ground),
        rng.normal(0.0, 0.01, size=n_ground),
    ])
    pts_over = np.column_stack([
        rng.uniform(0, 0.2, size=n_over),
        rng.uniform(0, 0.2, size=n_over),
        np.full(n_over, 2.0),
    ])
    pts = np.vstack([pts_ground, pts_over])

    grid = build_ground_height_grid(
        pts, cell_size=0.5, low_quantile=0.1, min_pts_per_cell=10,
    )
    z, valid = grid.query(np.array([[0.1, 0.1]]))
    assert valid[0]
    assert abs(float(z[0])) < 0.05


def test_build_grid_marks_sparse_cells_invalid():
    """Cells with fewer than min_pts_per_cell samples report NaN."""
    pts = np.array([
        [0.1, 0.1, 0.0],
        [10.0, 10.0, 5.0],  # alone in its cell
    ])
    grid = build_ground_height_grid(
        pts, cell_size=0.5, low_quantile=0.1, min_pts_per_cell=3,
    )
    z, valid = grid.query(np.array([[10.0, 10.0]]))
    assert not valid[0]


def test_build_grid_z_clip_applied_before_binning():
    pts = np.array([
        [0.1, 0.1, 0.0],
        [0.1, 0.1, 0.0],
        [0.1, 0.1, 0.0],
        [0.1, 0.1, 50.0],   # tall building, should be clipped
    ])
    grid = build_ground_height_grid(
        pts, cell_size=0.5, low_quantile=0.1,
        min_pts_per_cell=2, z_clip=(-1.0, 5.0),
    )
    z, valid = grid.query(np.array([[0.1, 0.1]]))
    assert valid[0]
    assert abs(float(z[0])) < 0.1


def test_query_outside_grid_marks_invalid():
    pts = np.array([
        [0.1, 0.1, 0.0],
        [0.2, 0.2, 0.0],
        [0.3, 0.3, 0.0],
    ])
    grid = build_ground_height_grid(pts, cell_size=0.5, min_pts_per_cell=1)
    z, valid = grid.query(np.array([[100.0, 100.0]]))
    assert not valid[0]


def test_snap_to_ground_pulls_close_points():
    rng = np.random.default_rng(0)
    n = 1000
    lidar = np.column_stack([
        rng.uniform(0, 5, size=n),
        rng.uniform(0, 5, size=n),
        np.full(n, 2.0),  # all ground at z=2
    ])
    grid = build_ground_height_grid(
        lidar, cell_size=0.5, low_quantile=0.1, min_pts_per_cell=2,
    )

    # AA-HAD points: one near ground, one far above.
    aahad = np.array([
        [2.5, 2.5, 2.3],   # 0.3 above ground -> snap
        [2.5, 2.5, 5.0],   # 3.0 above -> leave alone
        [2.5, 2.5, 1.7],   # 0.3 below -> snap
    ])
    snapped, is_ground = snap_to_ground(aahad, grid, max_dz=0.5)
    assert is_ground.tolist() == [True, False, True]
    assert abs(float(snapped[0, 2]) - 2.0) < 0.05
    assert abs(float(snapped[1, 2]) - 5.0) < 1e-3
    assert abs(float(snapped[2, 2]) - 2.0) < 0.05


def test_snap_to_ground_rejects_outside_grid():
    lidar = np.array([
        [0.1, 0.1, 0.0],
        [0.2, 0.2, 0.0],
        [0.3, 0.3, 0.0],
    ])
    grid = build_ground_height_grid(lidar, cell_size=0.5, min_pts_per_cell=1)
    far = np.array([[1000.0, 1000.0, 0.1]])
    snapped, is_ground = snap_to_ground(far, grid, max_dz=0.5)
    assert not is_ground[0]
    assert abs(float(snapped[0, 2]) - 0.1) < 1e-3


def test_snap_to_ground_rejects_bad_input():
    grid = build_ground_height_grid(np.array([[0, 0, 0]]), cell_size=0.5, min_pts_per_cell=1)
    with pytest.raises(ValueError):
        snap_to_ground(np.zeros((5, 2)), grid)


def test_build_grid_rejects_bad_inputs():
    pts = np.array([[0, 0, 0]])
    with pytest.raises(ValueError):
        build_ground_height_grid(pts, cell_size=0.0)
    with pytest.raises(ValueError):
        build_ground_height_grid(pts, cell_size=0.5, low_quantile=0.0)
    with pytest.raises(ValueError):
        build_ground_height_grid(np.zeros((5, 2)), cell_size=0.5)
