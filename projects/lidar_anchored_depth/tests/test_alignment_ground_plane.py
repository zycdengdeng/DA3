"""Tests for ``alignment.ground_plane`` (B5)."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.alignment.ground_plane import (
    fit_ground_plane_ransac,
    ground_depth_map,
    ray_plane_intersection,
)


def test_ransac_recovers_horizontal_plane_at_origin():
    """Z = 0 plane → normal (0, 0, 1), offset 0."""
    rng = np.random.default_rng(0)
    n = 1000
    pts = np.stack(
        [
            rng.uniform(-50, 50, n),
            rng.uniform(0, 100, n),
            rng.normal(0, 0.05, n),  # a little noise on Z
        ],
        axis=1,
    )
    plane = fit_ground_plane_ransac(
        pts, distance_threshold=0.2, n_iterations=200, seed=0,
    )
    np.testing.assert_allclose(plane.normal, [0, 0, 1], atol=0.05)
    assert abs(plane.offset) < 0.05
    assert plane.inlier_count > 0.95 * n


def test_ransac_robust_to_outliers():
    rng = np.random.default_rng(0)
    n_in = 800
    n_out = 200
    inliers = np.stack(
        [
            rng.uniform(-30, 30, n_in),
            rng.uniform(0, 80, n_in),
            rng.normal(0, 0.05, n_in),
        ],
        axis=1,
    )
    outliers = np.stack(
        [
            rng.uniform(-30, 30, n_out),
            rng.uniform(0, 80, n_out),
            rng.uniform(2.0, 8.0, n_out),  # well above ground
        ],
        axis=1,
    )
    pts = np.concatenate([inliers, outliers], axis=0)
    plane = fit_ground_plane_ransac(
        pts, distance_threshold=0.2, n_iterations=500, seed=0,
    )
    np.testing.assert_allclose(plane.normal, [0, 0, 1], atol=0.05)
    assert plane.inlier_count >= 0.9 * n_in
    assert plane.inlier_count < n_in + n_out  # outliers actually rejected


def test_ransac_recovers_tilted_plane():
    """A plane tilted by arctan(0.1) about world X (≈5.7°) → recovered."""
    rng = np.random.default_rng(0)
    n = 1000
    xs = rng.uniform(-30, 30, n)
    ys = rng.uniform(0, 80, n)
    zs = 0.1 * ys + rng.normal(0, 0.05, n)  # plane normal ∝ (0, -0.1, 1)
    pts = np.stack([xs, ys, zs], axis=1)
    plane = fit_ground_plane_ransac(
        pts, distance_threshold=0.2, n_iterations=500, seed=0,
    )
    # Expected normal: (0, -0.1, 1) normalized
    expected = np.array([0.0, -0.1, 1.0]) / np.linalg.norm([0.0, -0.1, 1.0])
    cos_sim = float(plane.normal @ expected)
    assert cos_sim > 0.99


def test_ray_plane_intersection_known():
    """Ray from (0,0,5) along (0,1,-1) hits Z=0 plane at t = 5."""
    from lidar_anchored_depth.alignment.ground_plane import GroundPlane

    plane = GroundPlane(
        normal=np.array([0.0, 0.0, 1.0]),
        offset=0.0,
        inlier_count=100, inlier_ratio=1.0, residual_std=0.0,
    )
    origin = np.array([0.0, 0.0, 5.0])
    dirs = np.array([[0.0, 1.0, -1.0]])
    t = ray_plane_intersection(origin, dirs, plane)
    assert t[0] == pytest.approx(5.0)


def test_ground_depth_map_recovers_synthetic_ground(synthetic_camera):
    """Render a Z = 0 plane through the synthetic camera and check depths."""
    K, T_wc = synthetic_camera
    from lidar_anchored_depth.alignment.ground_plane import GroundPlane

    plane = GroundPlane(
        normal=np.array([0.0, 0.0, 1.0]),
        offset=0.0,
        inlier_count=100, inlier_ratio=1.0, residual_std=0.0,
    )
    H, W = 1080, 1920
    depth = ground_depth_map(plane, K, T_wc, (H, W))
    # Check: principal point row should have a valid depth (camera looks
    # forward + 25° down so the principal axis hits the ground).
    cy = int(K[1, 2])
    cx = int(K[0, 2])
    assert np.isfinite(depth[cy, cx])
    assert depth[cy, cx] > 0
    # Bottom of image (looking nearly straight down) → smaller depth
    bottom_depth = depth[H - 50, cx]
    top_depth = depth[cy - 200, cx]
    assert np.isfinite(bottom_depth)
    assert np.isfinite(top_depth)
    assert bottom_depth < top_depth


def test_ground_depth_map_non_ground_mask_is_nan(synthetic_camera):
    K, T_wc = synthetic_camera
    from lidar_anchored_depth.alignment.ground_plane import GroundPlane

    plane = GroundPlane(
        normal=np.array([0.0, 0.0, 1.0]), offset=0.0,
        inlier_count=100, inlier_ratio=1.0, residual_std=0.0,
    )
    H, W = 1080, 1920
    non_ground = np.zeros((H, W), dtype=bool)
    non_ground[100:200, 200:400] = True
    depth = ground_depth_map(
        plane, K, T_wc, (H, W), non_ground_mask=non_ground,
    )
    assert np.all(np.isnan(depth[100:200, 200:400]))
