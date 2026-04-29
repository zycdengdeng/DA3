"""Tests for ``viz.overlay`` projection and bbox geometry.

Geometry is verified against hand-checked toy cases. The drawing
functions are smoke-tested for output shape / dtype only — pixel-level
appearance is verified by ``preview_frame.py`` on real data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lidar_anchored_depth.data.base import DynamicObject
from lidar_anchored_depth.viz.overlay import (
    BBOX_EDGES,
    bbox_3d_corners,
    bbox_corners_from_object,
    color_for_label,
    draw_bbox_3d_overlay,
    draw_lidar_overlay,
    project_camera_to_image,
    world_to_camera,
    world_to_image,
)


# --------------------------------------------------------------------- #
# world_to_camera
# --------------------------------------------------------------------- #
def test_world_to_camera_identity():
    pts = np.array([[1, 2, 3], [-4, -5, -6]], dtype=np.float64)
    out = world_to_camera(pts, np.eye(4))
    np.testing.assert_allclose(out, pts)


def test_world_to_camera_pure_translation():
    pts = np.array([[1, 2, 3]], dtype=np.float64)
    T_wc = np.eye(4)
    T_wc[:3, 3] = [10, 20, 30]  # camera origin at (10,20,30) in world
    # World point (1,2,3) → camera point = world - camera_origin = (-9, -18, -27)
    out = world_to_camera(pts, T_wc)
    np.testing.assert_allclose(out, [[-9, -18, -27]])


def test_world_to_camera_empty():
    out = world_to_camera(np.zeros((0, 3)), np.eye(4))
    assert out.shape == (0, 3)


# --------------------------------------------------------------------- #
# project_camera_to_image
# --------------------------------------------------------------------- #
def _K_simple(fx=1000, fy=1000, cx=640, cy=360) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_project_no_distortion_principal_point():
    K = _K_simple()
    pts = np.array([[0, 0, 5]], dtype=np.float64)
    uv, in_front = project_camera_to_image(pts, K)
    np.testing.assert_allclose(uv, [[640, 360]])
    assert in_front.tolist() == [True]


def test_project_no_distortion_off_axis():
    K = _K_simple(1000, 1000, 640, 360)
    # Point at (1, 0, 5) should project to (640 + 1000*1/5, 360) = (840, 360)
    pts = np.array([[1.0, 0.0, 5.0]], dtype=np.float64)
    uv, _ = project_camera_to_image(pts, K)
    np.testing.assert_allclose(uv, [[840, 360]])


def test_project_filters_behind_camera():
    K = _K_simple()
    pts = np.array(
        [[0, 0, 5.0], [0, 0, -1.0], [1, 1, 0.05]],  # behind, very close
        dtype=np.float64,
    )
    uv, in_front = project_camera_to_image(pts, K, z_min=0.1)
    # Only the first point passes
    assert in_front.tolist() == [True, False, False]
    assert uv.shape == (1, 2)


def test_project_all_behind_returns_empty():
    K = _K_simple()
    pts = np.array([[0, 0, -1.0], [0, 0, -2.0]], dtype=np.float64)
    uv, in_front = project_camera_to_image(pts, K)
    assert uv.shape == (0, 2)
    assert in_front.tolist() == [False, False]


def test_world_to_image_full_path():
    K = _K_simple(1000, 1000, 640, 360)
    T_wc = np.eye(4)  # camera at world origin, identity orientation
    pts = np.array([[1.0, 0.0, 5.0]], dtype=np.float64)
    uv, z, in_front = world_to_image(pts, K, T_wc)
    np.testing.assert_allclose(uv, [[840, 360]])
    np.testing.assert_allclose(z, [5.0])
    assert in_front.tolist() == [True]


# --------------------------------------------------------------------- #
# bbox_3d_corners
# --------------------------------------------------------------------- #
def test_bbox_corners_axis_aligned():
    corners = bbox_3d_corners(0, 0, 0, 2, 4, 6, yaw=0)
    # Length=2 along X, width=4 along Y, height=6 along Z
    # Bottom face at z=-3, top at z=+3
    # Corners 0..3 at z=-3
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners[0], [-1, -2, -3])
    np.testing.assert_allclose(corners[1], [+1, -2, -3])
    np.testing.assert_allclose(corners[2], [+1, +2, -3])
    np.testing.assert_allclose(corners[3], [-1, +2, -3])
    # Top
    np.testing.assert_allclose(corners[4], [-1, -2, +3])
    np.testing.assert_allclose(corners[7], [-1, +2, +3])


def test_bbox_corners_translation():
    corners = bbox_3d_corners(10, 20, 30, 2, 2, 2, yaw=0)
    # Center is (10,20,30); corners shift by (10,20,30)
    np.testing.assert_allclose(corners[0], [9, 19, 29])
    np.testing.assert_allclose(corners[6], [11, 21, 31])


def test_bbox_corners_yaw_90deg():
    # Rotate yaw by 90deg → length axis (x) maps to world Y
    corners = bbox_3d_corners(0, 0, 0, 4, 2, 2, yaw=math.pi / 2)
    # Bottom-front-left (-2, -1, -1) rotated 90deg about z → (1, -2, -1)
    np.testing.assert_allclose(corners[0], [1, -2, -1], atol=1e-12)
    # Bottom-back-right (+2, +1, -1) rotated → (-1, +2, -1)
    np.testing.assert_allclose(corners[2], [-1, +2, -1], atol=1e-12)


def test_bbox_edges_count():
    assert len(BBOX_EDGES) == 12


def test_bbox_corners_from_object():
    obj = DynamicObject(
        id=1,
        label="Car",
        xyz=np.array([5.0, 0.0, 1.0]),
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )
    c1 = bbox_corners_from_object(obj)
    c2 = bbox_3d_corners(5, 0, 1, 4, 2, 1.5, 0)
    np.testing.assert_allclose(c1, c2)


# --------------------------------------------------------------------- #
# Drawing — smoke tests (cv2 required)
# --------------------------------------------------------------------- #
@pytest.fixture
def blank_img():
    return np.zeros((360, 640, 3), dtype=np.uint8)


def test_draw_lidar_overlay_smoke(blank_img):
    cv2 = pytest.importorskip("cv2")
    uv = np.array([[100, 100], [200, 200], [-5, 100], [99999, 50]],
                  dtype=np.float64)
    depth = np.array([5.0, 10.0, 7.0, 30.0])
    out = draw_lidar_overlay(blank_img, uv, depth, radius=2)
    assert out.shape == blank_img.shape
    assert out.dtype == np.uint8
    # In-frame points should have written non-zero pixels somewhere
    assert (out != 0).any()


def test_draw_lidar_overlay_empty_returns_copy(blank_img):
    cv2 = pytest.importorskip("cv2")
    out = draw_lidar_overlay(
        blank_img, np.zeros((0, 2)), np.zeros((0,)),
    )
    np.testing.assert_array_equal(out, blank_img)
    assert out is not blank_img  # must be a copy


def test_draw_bbox_3d_overlay_smoke(blank_img):
    cv2 = pytest.importorskip("cv2")
    K = _K_simple()
    T_wc = np.eye(4)
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, 10.0]),
        lwh=np.array([2.0, 2.0, 2.0]),
        yaw=0.0,
    )
    out, drew = draw_bbox_3d_overlay(blank_img, obj, K, T_wc)
    assert drew is True
    assert (out != 0).any()


def test_draw_bbox_3d_overlay_skips_when_behind_camera(blank_img):
    cv2 = pytest.importorskip("cv2")
    K = _K_simple()
    T_wc = np.eye(4)
    obj = DynamicObject(
        id=1, label="Car",
        xyz=np.array([0.0, 0.0, -5.0]),  # behind the camera
        lwh=np.array([2.0, 2.0, 2.0]),
        yaw=0.0,
    )
    out, drew = draw_bbox_3d_overlay(blank_img, obj, K, T_wc)
    assert drew is False
    np.testing.assert_array_equal(out, blank_img)


# --------------------------------------------------------------------- #
# Color palette
# --------------------------------------------------------------------- #
def test_color_for_label_known():
    assert color_for_label("Car") == (0, 255, 0)


def test_color_for_label_unknown_default():
    assert color_for_label("Spaceship") == (0, 255, 255)
