"""Tests for Stage 5: LiDAR-priority depth completion + FOV cull."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.pipeline.lidar_completion import (
    CameraView,
    lidar_priority_fill,
    points_in_any_camera_fov,
    points_in_camera_fov,
)


def _identity_view(H: int = 100, W: int = 200, fx: float = 100.0) -> CameraView:
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    T_wc = np.eye(4)
    return CameraView(cam_id="0", K=K, T_wc=T_wc, image_hw=(H, W))


def test_fov_cull_keeps_in_frustum_and_drops_behind():
    view = _identity_view()
    pts = np.array([
        [0.0, 0.0, 5.0],   # in front, on axis
        [0.0, 0.0, -3.0],  # behind
        [10.0, 0.0, 5.0],  # tan = 2 * fx / W = 1 → outside
        [0.5, 0.0, 5.0],   # in front, slightly off axis
    ])
    mask = points_in_camera_fov(pts, view)
    assert mask[0]
    assert not mask[1]
    assert not mask[2]
    assert mask[3]


def test_fov_cull_respects_z_max():
    view = _identity_view()
    pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 50.0]])
    assert points_in_camera_fov(pts, view, z_max=10.0).tolist() == [True, False]


def test_fov_any_is_union_over_views():
    v1 = _identity_view()
    v2 = _identity_view()
    # Tilt v2 so a point at (0, 0, -5) becomes "in front" of it.
    R_180 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.0]])
    v2 = CameraView(cam_id="1", K=v1.K, T_wc=R_180, image_hw=v1.image_hw)
    pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
    mask = points_in_any_camera_fov(pts, [v1, v2])
    assert mask.tolist() == [True, True]


def test_fov_empty_points_returns_empty_mask():
    view = _identity_view()
    assert points_in_camera_fov(np.zeros((0, 3)), view).shape == (0,)


# --- LiDAR-priority fill ---------------------------------------------------

def test_priority_fill_lidar_dominates_shared_voxel():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aahad = np.array([[0.02, 0.02, 0.02]])  # same voxel at 0.05m
    aahad_rgb = np.array([[100, 200, 50]], dtype=np.uint8)
    xyz, rgb, src = lidar_priority_fill(lidar, aahad, aahad_rgb, voxel_size=0.05)
    assert xyz.shape[0] == 1  # only LiDAR survives
    assert src.tolist() == [0]


def test_priority_fill_aahad_fills_empty_voxel():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aahad = np.array([[5.0, 5.0, 5.0]])  # different voxel
    aahad_rgb = np.array([[100, 200, 50]], dtype=np.uint8)
    xyz, rgb, src = lidar_priority_fill(lidar, aahad, aahad_rgb, voxel_size=0.05)
    assert xyz.shape[0] == 2  # LiDAR + AA-HAD survive
    assert sorted(src.tolist()) == [0, 1]


def test_priority_fill_keeps_lidar_in_output():
    lidar = np.random.default_rng(0).normal(size=(10, 3))
    aahad = np.array([[100.0, 100.0, 100.0]])
    aahad_rgb = np.array([[1, 2, 3]], dtype=np.uint8)
    xyz, rgb, src = lidar_priority_fill(lidar, aahad, aahad_rgb, voxel_size=0.05)
    # All 10 LiDAR + 1 AA-HAD
    assert xyz.shape[0] == 11
    assert (src[:10] == 0).all()
    assert src[10] == 1


def test_priority_fill_lidar_color_default_is_light_grey():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aahad = np.array([[5.0, 5.0, 5.0]])
    aahad_rgb = np.array([[10, 20, 30]], dtype=np.uint8)
    xyz, rgb, src = lidar_priority_fill(lidar, aahad, aahad_rgb, voxel_size=0.05)
    # First row is LiDAR -> default grey 180
    assert tuple(rgb[0]) == (180, 180, 180)
    assert tuple(rgb[1]) == (10, 20, 30)


def test_priority_fill_lidar_color_custom():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aahad = np.array([[5.0, 5.0, 5.0]])
    aahad_rgb = np.array([[10, 20, 30]], dtype=np.uint8)
    xyz, rgb, src = lidar_priority_fill(
        lidar, aahad, aahad_rgb, voxel_size=0.05,
        lidar_color=(240, 240, 240),
    )
    assert tuple(rgb[0]) == (240, 240, 240)


def test_priority_fill_max_dist_filter_drops_far_aahad():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aahad = np.array([
        [10.0, 0.0, 0.0],   # 10 m from LiDAR -> drop with max_dist=2
        [0.5, 0.0, 0.0],    # in different voxel than LiDAR (size 0.05),
                            # 0.5 m away -> keep with max_dist=2
    ])
    aahad_rgb = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8)
    xyz, rgb, src = lidar_priority_fill(
        lidar, aahad, aahad_rgb, voxel_size=0.05, max_dist_to_lidar=2.0,
    )
    n_aa = int((src == 1).sum())
    assert n_aa == 1


def test_priority_fill_outlier_reference_lidar_keeps_aahad_near_ground():
    """Regression: when ``lidar_xyz`` has been thinned (e.g. ground band
    removed via --lidar-skip-ground) the outlier-rejection KD-tree
    should be built on the **un-thinned** reference cloud so AA-HAD
    ground points aren't falsely flagged as outliers just because the
    surviving backbone no longer covers the road surface.
    """
    # Backbone: a single LiDAR point 5 m above origin (mimics what
    # survives skip-ground: vertical structure tops only).
    lidar_thinned = np.array([[0.0, 0.0, 5.0]])
    # Reference: includes the ground LiDAR point that was dropped.
    lidar_full = np.array([
        [0.0, 0.0, 5.0],
        [0.5, 0.0, 0.0],   # ground LiDAR — gone from backbone
    ])
    # AA-HAD ground point near the dropped ground LiDAR.
    aahad = np.array([[0.6, 0.0, 0.0]])
    aahad_rgb = np.array([[1, 2, 3]], dtype=np.uint8)

    # Without the reference, max_dist=2 drops the ground AA-HAD because
    # nearest LiDAR is the 5 m elevated point.
    _, _, src_no_ref = lidar_priority_fill(
        lidar_thinned, aahad, aahad_rgb,
        voxel_size=0.05, max_dist_to_lidar=2.0,
    )
    assert int((src_no_ref == 1).sum()) == 0  # AA-HAD dropped

    # With the un-thinned reference, the ground LiDAR is in the tree
    # so the AA-HAD ground point passes (nearest LiDAR ≈ 0.1 m).
    _, _, src_with_ref = lidar_priority_fill(
        lidar_thinned, aahad, aahad_rgb,
        voxel_size=0.05, max_dist_to_lidar=2.0,
        outlier_reference_lidar=lidar_full,
    )
    assert int((src_with_ref == 1).sum()) == 1  # AA-HAD kept


def test_priority_fill_empty_lidar_returns_aahad_only():
    aahad = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    aahad_rgb = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    xyz, rgb, src = lidar_priority_fill(
        np.zeros((0, 3)), aahad, aahad_rgb, voxel_size=0.05,
    )
    assert xyz.shape == (2, 3)
    assert (src == 1).all()


def test_priority_fill_empty_aahad_returns_lidar_only():
    lidar = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    xyz, rgb, src = lidar_priority_fill(
        lidar, np.zeros((0, 3)), None, voxel_size=0.05,
    )
    assert xyz.shape == (2, 3)
    assert (src == 0).all()
    # LiDAR is now always coloured (default light grey) even when
    # aahad_rgb is None — fixes the "yellow LiDAR-only objects" bug.
    assert rgb is not None
    assert tuple(rgb[0]) == (180, 180, 180)


def test_priority_fill_lidar_grey_when_aahad_has_no_colors():
    """When AA-HAD points exist but aahad_rgb is None, LiDAR should
    still be grey (not inherit any fallback)."""
    lidar = np.array([[0.0, 0.0, 0.0]])
    aahad = np.array([[5.0, 5.0, 5.0]])
    xyz, rgb, src = lidar_priority_fill(
        lidar, aahad, None, voxel_size=0.05,
    )
    assert rgb is not None
    # LiDAR row stays grey 180, AA-HAD row gets fallback 220 220 100
    assert tuple(rgb[0]) == (180, 180, 180)
    assert tuple(rgb[1]) == (220, 220, 100)


def test_priority_fill_rejects_bad_inputs():
    with pytest.raises(ValueError):
        lidar_priority_fill(np.zeros((5, 2)), np.zeros((1, 3)), None, 0.05)
    with pytest.raises(ValueError):
        lidar_priority_fill(np.zeros((5, 3)), np.zeros((1, 3)), None, 0.0)
