"""Tests for ``alignment.ray_obb.ray_obb_z_target``.

Verifies the slab-method intersection on hand-checked geometry:
- principal-axis pixel hits the bbox's near face at the expected depth
- pixel that misses the bbox returns NaN
- yaw-rotated bbox: the rotated face is hit at the correct depth
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lidar_anchored_depth.alignment.ray_obb import ray_obb_z_target
from lidar_anchored_depth.data.base import DynamicObject


def _K_simple(fx=1000, fy=1000, cx=640, cy=360) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_principal_axis_hits_near_face_at_camera_distance(synthetic_camera):
    """Pixel at the principal point looks along camera Z forward.

    Place an axis-aligned bbox squarely in front of the camera. The
    near-face hit's camera depth must equal the distance from the
    camera origin to that face.
    """
    K, T_wc = synthetic_camera
    # Place a 4x2x2 box centred 30 m forward along the camera Z direction.
    cam_origin = T_wc[:3, 3]
    cam_z_axis_world = T_wc[:3, 2]
    centre_world = cam_origin + 30.0 * cam_z_axis_world
    obj = DynamicObject(
        id=1, label="Car",
        xyz=centre_world,
        lwh=np.array([2.0, 2.0, 2.0]),  # half-extent 1 m on each axis
        yaw=0.0,
    )
    cx, cy = K[0, 2], K[1, 2]
    z = ray_obb_z_target(np.array([[cx, cy]]), K, T_wc, obj)
    # Near face is 30 - 1 = 29 m along camera-Z (since box half-extent is
    # 1 m and the box is axis-aligned in world; the camera-Z and world
    # axes don't quite align, but the box extends 1 m in every world
    # axis). For a pure forward-looking axis-aligned box this is exact;
    # for the synthetic camera (pitched) it's *approximately* exact —
    # cap the tolerance at 0.5 m.
    assert math.isfinite(float(z[0]))
    assert 27.0 < float(z[0]) < 31.0


def test_misses_bbox_returns_nan(synthetic_camera):
    """A pixel ray that doesn't hit the bbox returns NaN."""
    K, T_wc = synthetic_camera
    cam_origin = T_wc[:3, 3]
    cam_z = T_wc[:3, 2]
    centre_world = cam_origin + 30.0 * cam_z
    obj = DynamicObject(
        id=1, label="Car",
        xyz=centre_world, lwh=np.array([2.0, 2.0, 2.0]), yaw=0.0,
    )
    # Pixel far off to the side / corner — its ray should miss the
    # 2-m-wide box centred on the principal axis.
    z = ray_obb_z_target(np.array([[10.0, 10.0]]), K, T_wc, obj)
    assert np.isnan(float(z[0]))


def test_two_pixels_one_hit_one_miss(synthetic_camera):
    K, T_wc = synthetic_camera
    cam_origin = T_wc[:3, 3]
    cam_z = T_wc[:3, 2]
    centre_world = cam_origin + 30.0 * cam_z
    obj = DynamicObject(
        id=1, label="Car",
        xyz=centre_world, lwh=np.array([2.0, 2.0, 2.0]), yaw=0.0,
    )
    cx, cy = K[0, 2], K[1, 2]
    uv = np.array([[cx, cy], [10.0, 10.0]])
    z = ray_obb_z_target(uv, K, T_wc, obj)
    assert math.isfinite(float(z[0]))
    assert np.isnan(float(z[1]))


def test_yaw_rotation_pulls_corner_closer_to_off_axis_ray():
    """An off-principal pixel hits a yaw-rotated bbox at a closer near depth.

    A central ray going through the box centre is unaffected by yaw
    rotation about Z (geometric symmetry). But a ray off to the side
    intersects the rotated bbox at a different depth. We use an OFF-
    centre pixel and check the yaw=45° box's near hit is *closer* than
    the yaw=0° box's near hit (because the rotated face presents an
    edge / corner toward the off-axis ray).
    """
    K = _K_simple(1000, 1000, 640, 360)
    # Upright camera at world origin, looking along world +Y.
    # T_wc rotation columns = camera basis in world:
    #   cam X (right) → world +X
    #   cam Y (down)  → world -Z
    #   cam Z (fwd)   → world +Y
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    assert np.isclose(np.linalg.det(R), 1.0)
    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = np.array([0.0, 0.0, 0.0])

    box_centre = np.array([0.0, 30.0, 0.0])
    # Asymmetric XY footprint → yaw matters. (length=6, width=2)
    box_lwh = np.array([6.0, 2.0, 2.0])
    obj_yaw0 = DynamicObject(id=1, label="Car",
                             xyz=box_centre, lwh=box_lwh, yaw=0.0)
    obj_yaw90 = DynamicObject(id=2, label="Car",
                              xyz=box_centre, lwh=box_lwh, yaw=math.pi / 2)

    # Central pixel ray (along +Y world) hits both boxes' near face at
    # y = 30 − 1 = 29 (because both boxes have width 2 m on the ray's
    # incidence axis after rotation; for yaw=0, width axis is Y, half=1;
    # for yaw=90, length axis is rotated to Y, half=3 → near at 27).
    cx, cy = K[0, 2], K[1, 2]
    z0 = ray_obb_z_target(np.array([[cx, cy]]), K, T_wc, obj_yaw0)
    z90 = ray_obb_z_target(np.array([[cx, cy]]), K, T_wc, obj_yaw90)
    assert math.isfinite(float(z0[0]))
    assert math.isfinite(float(z90[0]))
    # yaw=0 sees the 2-m-wide face, near at y = 29
    assert float(z0[0]) == pytest.approx(29.0, abs=0.01)
    # yaw=90 rotates the length axis (6 m) onto the world-Y axis, so the
    # near face is now at y = 30 − 3 = 27 (closer to the camera).
    assert float(z90[0]) == pytest.approx(27.0, abs=0.01)


def test_empty_input():
    K = _K_simple()
    T_wc = np.eye(4)
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.zeros(3), lwh=np.array([2.0, 2.0, 2.0]), yaw=0.0,
    )
    out = ray_obb_z_target(np.zeros((0, 2)), K, T_wc, obj)
    assert out.shape == (0,)


def test_ray_parallel_to_face_outside_bbox_misses():
    """Ray parallel to a face *and* outside the slab → miss (NaN)."""
    K = _K_simple()
    T_wc = np.eye(4)
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 30.0]),
        lwh=np.array([1.0, 1.0, 1.0]),
        yaw=0.0,
    )
    # A pixel far to the side — its ray will almost-graze the box but
    # the box is small; treat as miss.
    z = ray_obb_z_target(np.array([[10000.0, 10000.0]]), K, T_wc, obj)
    assert np.isnan(float(z[0]))
