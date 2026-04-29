"""Tests for ``data.calibration``.

Verifies the parser produces a valid SE(3) ``T_wc`` (inverse of the
``virtualLidarToCam`` transform stored in ``calib.json``) and that the
ZYX-Euler helper matches a known 90° rotation.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from lidar_anchored_depth.data.calibration import (
    PINHOLE_CAMERA_IDS,
    PINHOLE_FOLDER_TO_CAMID,
    euler_zyx_to_R,
    load_scene_calibration,
)
from lidar_anchored_depth.data.calibration import (
    _invert_T,
    _rodrigues_to_R,
    _rt_to_T,
)


@pytest.fixture
def fake_calib_path():
    fake = {
        "imgSize": {"fish": [1280, 1280], "notFish": [1280, 720]},
        "lidar": {
            "0": {
                "name": "rad0_1.pcd",
                "lidarToVirtualLidar": {
                    "rotateMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "trans": [10.0, 20.0, 5.0],
                },
            },
        },
        "camera": {
            "3": {
                "name": "cam3_1.png",
                "isFish": 0,
                "intri": [1200, 0, 640, 0, 1200, 360, 0, 0, 1],
                "distor": [-0.2, 0.05, 0.001, -0.001, 0.0],
                "virtualLidarToCam": {
                    "rotate": [0.0, 0.0, math.pi / 2],
                    "trans": [1.0, 2.0, 3.0],
                },
            },
            "2": {
                "name": "cam2_1.png",
                "isFish": 1,
                "intri": [445, 0, 628, 0, 438, 648, 0, 0, 1],
                "distor": [-0.03, -0.005, 0.0, 0.0],
                "virtualLidarToCam": {
                    "rotate": [0.0, 0.0, 0.0],
                    "trans": [0.0, 0.0, 0.0],
                },
            },
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(fake, f)
        path = f.name
    yield path
    os.unlink(path)


def test_pinhole_camera_ids_match_dataset_guide():
    assert PINHOLE_CAMERA_IDS == ("0", "3", "6", "9")


def test_folder_to_camid_mapping_matches_dataset_guide():
    assert PINHOLE_FOLDER_TO_CAMID == {
        "pinhole0": "3",
        "pinhole1": "6",
        "pinhole2": "9",
        "pinhole3": "0",
    }


def test_euler_zyx_yaw_only():
    R = euler_zyx_to_R(0.0, 0.0, math.pi / 2)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    np.testing.assert_allclose(R, expected, atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_euler_zyx_intrinsic_order():
    """ZYX intrinsic: R = Rz @ Ry @ Rx; checked by composing pieces."""
    R = euler_zyx_to_R(0.1, 0.2, 0.3)
    # Reconstruct manually
    cr, sr = math.cos(0.1), math.sin(0.1)
    cp, sp = math.cos(0.2), math.sin(0.2)
    cy, sy = math.cos(0.3), math.sin(0.3)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    np.testing.assert_allclose(R, Rz @ Ry @ Rx, atol=1e-12)


def test_rt_invert_roundtrip():
    R = _rodrigues_to_R([0.1, -0.2, 0.3])
    t = np.array([1.5, -0.7, 4.0])
    T = _rt_to_T(R, t)
    np.testing.assert_allclose(_invert_T(T) @ T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T @ _invert_T(T), np.eye(4), atol=1e-12)


def test_load_scene_calibration_basic(fake_calib_path):
    calib = load_scene_calibration(fake_calib_path)
    assert sorted(calib.cameras.keys()) == ["2", "3"]
    assert sorted(calib.pinhole_cameras.keys()) == ["3"]
    assert calib.img_size_pinhole_hw == (720, 1280)
    assert calib.img_size_fisheye_hw == (1280, 1280)


def test_T_wc_inverts_virtual_lidar_to_cam(fake_calib_path):
    calib = load_scene_calibration(fake_calib_path)
    cam3 = calib.cameras["3"]
    R_v2c = _rodrigues_to_R([0.0, 0.0, math.pi / 2])
    T_v2c = _rt_to_T(R_v2c, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cam3.T_wc @ T_v2c, np.eye(4), atol=1e-12)


def test_lidar_T_wL(fake_calib_path):
    calib = load_scene_calibration(fake_calib_path)
    L = calib.lidars["0"]
    np.testing.assert_allclose(L.T_wL[:3, :3], np.eye(3))
    np.testing.assert_allclose(L.T_wL[:3, 3], [10.0, 20.0, 5.0])


def test_distortion_preserved_for_pinhole_and_fisheye(fake_calib_path):
    calib = load_scene_calibration(fake_calib_path)
    assert calib.cameras["3"].dist.shape == (5,)
    assert calib.cameras["2"].dist.shape == (4,)
