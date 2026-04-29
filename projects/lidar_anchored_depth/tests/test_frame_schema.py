"""Smoke tests for the Frame schema."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.data.base import Frame


def _ok_kwargs():
    return dict(
        frame_id="x",
        image=np.zeros((32, 64, 3), dtype=np.uint8),
        K=np.eye(3),
        T_wc=np.eye(4),
        lidar_world=np.zeros((10, 3), dtype=np.float32),
    )


def test_minimum_frame_constructs():
    f = Frame(**_ok_kwargs())
    assert f.hw == (32, 64)
    np.testing.assert_allclose(f.T_cw, np.eye(4))


def test_image_shape_validated():
    with pytest.raises(ValueError, match="image must be"):
        Frame(**{**_ok_kwargs(), "image": np.zeros((32, 64), dtype=np.uint8)})


def test_image_dtype_validated():
    with pytest.raises(ValueError, match="image must be uint8"):
        Frame(**{**_ok_kwargs(), "image": np.zeros((32, 64, 3), dtype=np.float32)})


def test_intrinsics_shape_validated():
    with pytest.raises(ValueError, match="K must be"):
        Frame(**{**_ok_kwargs(), "K": np.eye(4)})


def test_T_wc_shape_validated():
    with pytest.raises(ValueError, match="T_wc must be"):
        Frame(**{**_ok_kwargs(), "T_wc": np.eye(3)})


def test_lidar_shape_validated():
    with pytest.raises(ValueError, match="lidar_world"):
        Frame(**{**_ok_kwargs(), "lidar_world": np.zeros((10, 4), dtype=np.float32)})


def test_gt_depth_shape_validated():
    with pytest.raises(ValueError, match="gt_depth"):
        Frame(**{**_ok_kwargs(), "gt_depth": np.zeros((10, 10), dtype=np.float32)})


def test_sam_masks_shape_validated():
    with pytest.raises(ValueError, match="sam_masks"):
        Frame(**{**_ok_kwargs(), "sam_masks": np.zeros((3, 10, 10), dtype=bool)})
