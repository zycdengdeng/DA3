"""Tests for ``alignment.projection`` — geometry sanity (no algorithms).

The math here is also exercised through ``viz.overlay`` (because we now
re-export the projection primitives), so this file focuses on the new
APIs introduced for HAD: ``pixel_ray_to_world``, ``project_3d_bbox``,
``world_z_at_pixel_unit_depth``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lidar_anchored_depth.alignment.projection import (
    BBOX_EDGES,
    bbox_3d_corners,
    bbox_uv_aabb,
    invert_T,
    pixel_ray_to_world,
    pixel_to_camera_ray,
    project_3d_bbox,
    project_camera_to_image,
    world_to_camera,
    world_to_image,
    world_z_at_pixel_unit_depth,
)
from lidar_anchored_depth.data.base import DynamicObject


# --------------------------------------------------------------------- #
# invert_T round-trip
# --------------------------------------------------------------------- #
def test_invert_T_round_trip():
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R = -R
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [1.0, -2.5, 7.0]
    np.testing.assert_allclose(invert_T(T) @ T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T @ invert_T(T), np.eye(4), atol=1e-12)


# --------------------------------------------------------------------- #
# pixel_to_camera_ray / pixel_ray_to_world
# --------------------------------------------------------------------- #
def test_pixel_to_camera_ray_principal_point():
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
    rays = pixel_to_camera_ray(np.array([[640, 360]]), K)
    np.testing.assert_allclose(rays[0], [0, 0, 1], atol=1e-12)


def test_pixel_to_camera_ray_off_axis():
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
    # Pixel (840, 360) → x_n = (840 - 640) / 1000 = 0.2
    rays = pixel_to_camera_ray(np.array([[840, 360]]), K)
    np.testing.assert_allclose(rays[0], [0.2, 0.0, 1.0], atol=1e-12)


def test_pixel_ray_to_world_origin_is_camera_centre():
    K = np.eye(3)
    T_wc = np.eye(4)
    T_wc[:3, 3] = [3.0, 4.0, 5.0]
    origin, _ = pixel_ray_to_world(np.array([[0.5, 0.5]]), K, T_wc)
    np.testing.assert_allclose(origin, [3.0, 4.0, 5.0])


# --------------------------------------------------------------------- #
# world_z_at_pixel_unit_depth — the math HAD relies on
# --------------------------------------------------------------------- #
def test_world_z_decomposition_against_full_path(synthetic_camera):
    """Z_world(u, v, z_cam) should equal α(u,v)·z_cam + β computed exactly."""
    K, T_wc = synthetic_camera
    rng = np.random.default_rng(0)
    uv = np.stack(
        [rng.uniform(0, 1920, 50), rng.uniform(0, 1080, 50)], axis=1
    )
    z_cams = rng.uniform(5.0, 80.0, 50)

    alphas, beta = world_z_at_pixel_unit_depth(uv, K, T_wc)

    # Full path: backproject pixel to camera ray (with z=1), scale by z_cam,
    # transform to world, take Z component
    rays_cam = pixel_to_camera_ray(uv, K)
    P_cam = rays_cam * z_cams[:, None]
    P_h = np.hstack([P_cam, np.ones((P_cam.shape[0], 1))])
    P_world = (T_wc @ P_h.T).T[:, :3]
    Z_world_actual = P_world[:, 2]

    Z_world_decomposed = alphas * z_cams + beta
    np.testing.assert_allclose(Z_world_actual, Z_world_decomposed, atol=1e-9)


# --------------------------------------------------------------------- #
# project_3d_bbox
# --------------------------------------------------------------------- #
def test_project_3d_bbox_in_view(synthetic_camera, synthetic_object):
    K, T_wc = synthetic_camera
    uv8, P_cam = project_3d_bbox(synthetic_object, K, T_wc)
    assert uv8 is not None
    assert uv8.shape == (8, 2)
    assert (P_cam[:, 2] > 0.1).all()
    # Bbox AABB is non-degenerate
    u_min, v_min, u_max, v_max = bbox_uv_aabb(uv8)
    assert u_max > u_min
    assert v_max > v_min


def test_project_3d_bbox_behind_camera_returns_none(synthetic_camera):
    K, T_wc = synthetic_camera
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, -10.0, 0.0]),  # behind camera (Y < 5)
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )
    uv8, _ = project_3d_bbox(obj, K, T_wc)
    assert uv8 is None
