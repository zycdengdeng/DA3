"""Tests for ``alignment.height_anchor`` (M1 HAD-mask + shared solver).

The headline claim is that **two height anchors uniquely determine
``(a, b)``**. We verify by building a synthetic scene where the true
(a, b) is known and the solver recovers it to ~1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.alignment.height_anchor import (
    had_mask,
    height_interval_from_lidar_in_mask,
    mask_top_bottom_pixels,
    solve_affine_from_height_anchors,
)
from lidar_anchored_depth.alignment.projection import (
    bbox_3d_corners,
    world_to_image,
)
from lidar_anchored_depth.data.base import DynamicObject

from _synthetic import (  # type: ignore[import-not-found]
    synthetic_d_image_consistent_with_mask,
    synthetic_mask_from_world_box,
)


# --------------------------------------------------------------------- #
# Closed-form solver
# --------------------------------------------------------------------- #
def test_solver_recovers_known_affine(synthetic_camera, synthetic_object):
    """Build a scene where z = a_true · d̃ + b_true exactly, then solve."""
    K, T_wc = synthetic_camera
    obj = synthetic_object
    a_true, b_true = 0.3, 1.5

    # Top corner (#4) and bottom corner (#0) of the bbox in world frame
    corners = bbox_3d_corners(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2],
        obj.yaw,
    )
    # Use the front-left-top (#4) and front-left-bottom (#0)
    p_top_world = corners[4]
    p_bot_world = corners[0]

    uv_top, z_cam_top, _ = world_to_image(
        p_top_world.reshape(1, 3), K, T_wc, dist=None
    )
    uv_bot, z_cam_bot, _ = world_to_image(
        p_bot_world.reshape(1, 3), K, T_wc, dist=None
    )
    assert uv_top.shape[0] == 1 and uv_bot.shape[0] == 1

    # Construct d̃ such that z_cam = a_true · d̃ + b_true
    d_top = (z_cam_top[0] - b_true) / a_true
    d_bot = (z_cam_bot[0] - b_true) / a_true

    sol = solve_affine_from_height_anchors(
        K=K, T_wc=T_wc,
        uv_top=uv_top[0], uv_bot=uv_bot[0],
        d_pred_top=d_top, d_pred_bot=d_bot,
        Z_max=p_top_world[2], Z_min=p_bot_world[2],
    )
    assert sol is not None
    a, b = sol
    assert a == pytest.approx(a_true, abs=1e-9)
    assert b == pytest.approx(b_true, abs=1e-9)


def test_solver_singular_returns_none(synthetic_camera):
    """Same pixel + same d̃ → singular system → None."""
    K, T_wc = synthetic_camera
    uv = np.array([100.0, 100.0])
    sol = solve_affine_from_height_anchors(
        K=K, T_wc=T_wc,
        uv_top=uv, uv_bot=uv,
        d_pred_top=2.0, d_pred_bot=2.0,
        Z_max=1.5, Z_min=0.0,
    )
    assert sol is None


# --------------------------------------------------------------------- #
# mask_top_bottom_pixels
# --------------------------------------------------------------------- #
def test_mask_topbot_simple_rectangle():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:50, 30:70] = True
    out = mask_top_bottom_pixels(mask)
    assert out is not None
    uv_top, uv_bot = out
    assert uv_top[1] == 20
    assert uv_bot[1] == 49
    # u coordinate is row centroid → center of the row range
    assert uv_top[0] == pytest.approx((30 + 69) / 2.0)


def test_mask_topbot_empty_returns_none():
    out = mask_top_bottom_pixels(np.zeros((100, 100), dtype=bool))
    assert out is None


# --------------------------------------------------------------------- #
# height_interval_from_lidar_in_mask
# --------------------------------------------------------------------- #
def test_height_interval_recovers_object_extent(
    synthetic_camera, synthetic_object, synthetic_object_lidar
):
    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920

    # Project the object's LiDAR onto image, build its mask
    mask = synthetic_mask_from_world_box(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )
    assert mask.any()

    uv, _, in_front = world_to_image(synthetic_object_lidar, K, T_wc, dist=None)
    interval = height_interval_from_lidar_in_mask(
        lidar_world=synthetic_object_lidar,
        pixel_uv=uv, in_front=in_front, mask=mask,
        percentile_lo=2.0, percentile_hi=98.0, min_points=5,
    )
    assert interval is not None
    # Synthetic object centered at z=0.75, height=1.5 → [0, 1.5]
    assert interval.Z_min == pytest.approx(0.0, abs=0.1)
    assert interval.Z_max == pytest.approx(1.5, abs=0.1)
    assert interval.n_points >= 50


# --------------------------------------------------------------------- #
# Full had_mask path
# --------------------------------------------------------------------- #
def test_had_mask_recovers_known_affine(
    synthetic_camera, synthetic_object, synthetic_object_lidar
):
    """End-to-end M1: build a synthetic scene with z = a · d̃ + b, recover."""
    K, T_wc = synthetic_camera
    obj = synthetic_object
    H, W = 1080, 1920
    a_true, b_true = 0.25, 1.0

    mask = synthetic_mask_from_world_box(
        obj.xyz[0], obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )
    assert mask.any()
    masks = mask[None, ...]

    d_image = synthetic_d_image_consistent_with_mask(
        mask=mask, K=K, T_wc=T_wc,
        Z_min=obj.Z_min, Z_max=obj.Z_max,
        a_true=a_true, b_true=b_true,
    )

    # Use 0/100 percentiles so the LiDAR-derived [Z_min, Z_max] equals the
    # actual sample extent (≈ exact bbox extent for dense sampling). The
    # closed-form solver itself is verified exactly in
    # ``test_solver_recovers_known_affine``; here we additionally
    # exercise the LiDAR-in-mask + topbot-pixel + apply path.
    result = had_mask(
        d_pred_image=d_image, K=K, T_wc=T_wc,
        masks=masks, lidar_world=synthetic_object_lidar,
        percentile_lo=0.0, percentile_hi=100.0, min_lidar_per_mask=5,
    )

    assert result.n_masks_solved == 1
    anchor = result.anchors[0]
    # ``a`` is the slope and is well-determined by the height range.
    # ``b`` is amplified by the system's conditioning (small d̃_top − d̃_bot
    # for a distant small object), so we use a loose tolerance here. The
    # closed-form correctness is asserted at machine precision in
    # test_solver_recovers_known_affine.
    assert anchor.a == pytest.approx(a_true, rel=0.05)
    assert anchor.b == pytest.approx(b_true, abs=0.5)
    np.testing.assert_array_equal(result.coverage_mask, mask)


def test_had_mask_skips_masks_without_enough_lidar(synthetic_camera):
    """Mask with no LiDAR-in-mask should be skipped (no anchors)."""
    K, T_wc = synthetic_camera
    H, W = 1080, 1920
    d_image = np.full((H, W), 1.0, dtype=np.float32)
    masks = np.zeros((1, H, W), dtype=bool)
    masks[0, 100:200, 100:200] = True
    # No LiDAR points
    lidar = np.zeros((0, 3), dtype=np.float32)
    result = had_mask(
        d_pred_image=d_image, K=K, T_wc=T_wc,
        masks=masks, lidar_world=lidar,
        min_lidar_per_mask=5,
    )
    assert result.n_masks_attempted == 1
    assert result.n_masks_solved == 0
    assert len(result.anchors) == 0
