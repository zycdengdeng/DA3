"""Tests for ``alignment.bbox_anchor`` (M2 / M3 AA-HAD)."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.alignment.bbox_anchor import (
    aa_had,
    match_bboxes_to_masks,
    points_in_oriented_bbox,
)
from lidar_anchored_depth.data.base import DynamicObject

from _synthetic import (  # type: ignore[import-not-found]
    synthetic_d_image_consistent_with_mask,
    synthetic_mask_from_world_box,
)


# --------------------------------------------------------------------- #
# Bbox ↔ mask matching
# --------------------------------------------------------------------- #
def test_match_picks_high_iou_pair(synthetic_camera, synthetic_object):
    K, T_wc = synthetic_camera
    H, W = 1080, 1920
    mask = synthetic_mask_from_world_box(
        synthetic_object.xyz[0], synthetic_object.xyz[1], synthetic_object.xyz[2],
        synthetic_object.lwh[0], synthetic_object.lwh[1], synthetic_object.lwh[2],
        synthetic_object.yaw, K, T_wc, (H, W),
    )
    masks = mask[None, ...]
    matches = match_bboxes_to_masks(
        [synthetic_object], masks, K, T_wc, dist=None,
        dt_seconds=0.0, iou_threshold=0.3,
    )
    assert len(matches) == 1
    assert matches[0].iou > 0.9


def test_match_rejects_when_iou_too_low(synthetic_camera, synthetic_object):
    """Mask elsewhere in the image → IoU 0 → no match."""
    K, T_wc = synthetic_camera
    H, W = 1080, 1920
    mask = np.zeros((H, W), dtype=bool)
    mask[10:50, 10:50] = True  # far from where the bbox projects
    masks = mask[None, ...]
    matches = match_bboxes_to_masks(
        [synthetic_object], masks, K, T_wc, dist=None,
        dt_seconds=0.0, iou_threshold=0.3,
    )
    assert len(matches) == 0


def test_match_motion_compensation_recovers_iou_when_dt_known(
    synthetic_camera, synthetic_object,
):
    """Object moved, mask is at the moved position; correct dt restores IoU."""
    K, T_wc = synthetic_camera
    H, W = 1080, 1920

    # Original object at LiDAR timestamp; mask is built from where the
    # object is at camera timestamp (5 m later in +X)
    obj = synthetic_object
    obj.velocity_xy[:] = [5.0 / 0.1, 0.0]  # 50 m/s vx → moves 5 m in 100 ms
    dt = 0.1

    # Mask is at the object's CAMERA-time position
    mask_at_camera_t = synthetic_mask_from_world_box(
        obj.xyz[0] + 5.0, obj.xyz[1], obj.xyz[2],
        obj.lwh[0], obj.lwh[1], obj.lwh[2], obj.yaw,
        K, T_wc, (H, W),
    )
    masks = mask_at_camera_t[None, ...]

    # Without compensation: bbox projects to LiDAR-time pose → low IoU
    no_comp = match_bboxes_to_masks(
        [obj], masks, K, T_wc, dist=None, dt_seconds=0.0,
        iou_threshold=0.3,
    )
    # With compensation: IoU recovers
    with_comp = match_bboxes_to_masks(
        [obj], masks, K, T_wc, dist=None, dt_seconds=dt,
        iou_threshold=0.3,
    )

    assert len(with_comp) == 1
    assert with_comp[0].iou > 0.5
    # Either no match without comp, or markedly worse IoU
    if no_comp:
        assert no_comp[0].iou < with_comp[0].iou


# --------------------------------------------------------------------- #
# Full AA-HAD recovery
# --------------------------------------------------------------------- #
def test_aa_had_recovers_known_affine(synthetic_camera, synthetic_object):
    """The headline test: build a scene with z = a · d̃ + b and recover."""
    K, T_wc = synthetic_camera
    H, W = 1080, 1920
    a_true, b_true = 0.4, 2.0

    obj = synthetic_object
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

    result = aa_had(
        d_pred_image=d_image, K=K, T_wc=T_wc,
        masks=masks, dynamic_objects=[obj],
        dt_seconds=0.0, iou_threshold=0.3,
    )

    assert result.n_masks_solved == 1
    a_rec = result.anchors[0].a
    b_rec = result.anchors[0].b
    assert a_rec == pytest.approx(a_true, rel=0.05)
    assert b_rec == pytest.approx(b_true, abs=0.1)


def test_points_in_oriented_bbox_axis_aligned():
    """Yaw=0 bbox at origin: simple axis-aligned containment check."""
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 0.0]),
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )
    pts = np.array(
        [
            [0.0, 0.0, 0.0],     # centre
            [1.9, 0.9, 0.7],     # just inside
            [2.1, 0.0, 0.0],     # just outside on X
            [0.0, 1.1, 0.0],     # just outside on Y
            [0.0, 0.0, 0.8],     # just outside on Z
        ],
        dtype=np.float64,
    )
    inside = points_in_oriented_bbox(pts, obj)
    assert inside.tolist() == [True, True, False, False, False]


def test_points_in_oriented_bbox_yaw_90_degrees():
    """Yaw=90° rotates the bbox; what was on X is now on Y."""
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 0.0]),
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=np.pi / 2,
    )
    pts = np.array(
        [
            [0.0, 1.9, 0.0],    # in bbox-local +X (length axis), inside
            [0.0, 2.1, 0.0],    # outside the rotated length axis
            [0.9, 0.0, 0.0],    # in bbox-local +Y (width axis), inside
            [1.1, 0.0, 0.0],    # outside the rotated width axis
        ],
        dtype=np.float64,
    )
    inside = points_in_oriented_bbox(pts, obj)
    assert inside.tolist() == [True, False, True, False]


def test_points_in_oriented_bbox_expand():
    """Expand parameter loosens the half-extent."""
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 0.0]),
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )
    pts = np.array([[2.05, 0.0, 0.0]], dtype=np.float64)
    assert points_in_oriented_bbox(pts, obj).tolist() == [False]
    assert points_in_oriented_bbox(pts, obj, expand=0.1).tolist() == [True]


def test_points_in_oriented_bbox_expand_xyz_per_axis():
    """expand_xyz overrides isotropic expand; negative ez shrinks Z."""
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 0.0]),
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )
    # Point just past the +X edge, exactly on +Y edge, just below -Z bottom
    pts = np.array(
        [
            [2.10, 0.0, 0.0],   # past +X by 0.10
            [0.0, 1.05, 0.0],   # past +Y by 0.05
            [0.0, 0.0, -0.80],  # past -Z by 0.05 (h/2 = 0.75)
        ],
        dtype=np.float64,
    )
    # Asymmetric: expand X by 0.15, Y by 0.10, shrink Z by 0.10
    out = points_in_oriented_bbox(
        pts, obj, expand_xyz=(0.15, 0.10, -0.10),
    )
    assert out.tolist() == [True, True, False]


def test_points_in_oriented_bbox_z_local_min_offset_drops_ground():
    """z_local_min_offset cuts the bottom band of the bbox."""
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 0.0]),
        lwh=np.array([4.0, 2.0, 1.5]),  # h/2 = 0.75
        yaw=0.0,
    )
    pts = np.array(
        [
            [0.0, 0.0, 0.7],   # near top — kept
            [0.0, 0.0, -0.5],  # mid — kept
            [0.0, 0.0, -0.70], # bottom 5cm of bbox — DROPPED with offset 0.10
            [0.0, 0.0, -0.74], # very bottom — DROPPED
        ],
        dtype=np.float64,
    )
    # Without offset: all in bbox
    assert points_in_oriented_bbox(pts, obj).tolist() == [True, True, True, True]
    # With 0.10m ground cut: cut at z >= -0.75 + 0.10 = -0.65, so only
    # the two bottom-most points are dropped.
    assert points_in_oriented_bbox(
        pts, obj, z_local_min_offset=0.10
    ).tolist() == [True, True, False, False]


def test_points_in_oriented_bbox_empty():
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 0.0]),
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )
    out = points_in_oriented_bbox(np.zeros((0, 3)), obj)
    assert out.shape == (0,)


def test_aa_had_no_match_no_solve(synthetic_camera, synthetic_object):
    """Mask in the wrong place → no match → no solve."""
    K, T_wc = synthetic_camera
    H, W = 1080, 1920
    d_image = np.full((H, W), 1.0, dtype=np.float32)
    masks = np.zeros((1, H, W), dtype=bool)
    masks[0, 5:30, 5:30] = True
    result = aa_had(
        d_pred_image=d_image, K=K, T_wc=T_wc,
        masks=masks, dynamic_objects=[synthetic_object],
        dt_seconds=0.0, iou_threshold=0.3,
    )
    assert result.n_masks_solved == 0
    assert len(result.anchors) == 0
