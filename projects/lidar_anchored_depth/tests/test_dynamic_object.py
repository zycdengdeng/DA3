"""Tests for DynamicObject and AA-HAD-related Frame fields."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.data.base import DynamicObject, Frame


def _ok_frame_kwargs(**overrides):
    base = dict(
        frame_id="x",
        image=np.zeros((32, 64, 3), dtype=np.uint8),
        K=np.eye(3),
        T_wc=np.eye(4),
        lidar_world=np.zeros((10, 3), dtype=np.float32),
    )
    base.update(overrides)
    return base


def _ok_object_kwargs(**overrides):
    base = dict(
        id=1,
        label="Car",
        xyz=np.array([10.0, 20.0, -1.0], dtype=np.float64),
        lwh=np.array([4.5, 2.0, 1.5], dtype=np.float64),
        yaw=0.0,
    )
    base.update(overrides)
    return base


def test_dynamic_object_height_properties():
    o = DynamicObject(**_ok_object_kwargs())
    assert o.Z_min == pytest.approx(-1.0 - 0.75)
    assert o.Z_max == pytest.approx(-1.0 + 0.75)
    assert o.Z_max - o.Z_min == pytest.approx(o.lwh[2])


def test_dynamic_object_predict_xyz_static():
    o = DynamicObject(**_ok_object_kwargs())
    out = o.predict_xyz(0.05)
    np.testing.assert_allclose(out, o.xyz)
    # original is not mutated
    np.testing.assert_allclose(o.xyz, [10.0, 20.0, -1.0])


def test_dynamic_object_predict_xyz_moving():
    o = DynamicObject(
        **_ok_object_kwargs(velocity_xy=np.array([5.0, -2.0], dtype=np.float64)),
    )
    out = o.predict_xyz(0.04)  # 40 ms
    np.testing.assert_allclose(out, [10.0 + 0.2, 20.0 - 0.08, -1.0])


def test_frame_accepts_dynamic_objects():
    objs = [
        DynamicObject(**_ok_object_kwargs(id=1)),
        DynamicObject(**_ok_object_kwargs(id=2, xyz=np.array([0.0, 5.0, -1.5]))),
    ]
    f = Frame(**_ok_frame_kwargs(dynamic_objects=objs, image_ts_ms=1000, lidar_ts_ms=950))
    assert len(f.dynamic_objects) == 2
    assert f.image_ts_ms - f.lidar_ts_ms == 50


def test_dynamic_object_xyz_shape_validated():
    bad = DynamicObject(**_ok_object_kwargs(xyz=np.zeros(2)))
    with pytest.raises(ValueError, match="xyz"):
        Frame(**_ok_frame_kwargs(dynamic_objects=[bad]))


def test_dynamic_object_lwh_shape_validated():
    bad = DynamicObject(**_ok_object_kwargs(lwh=np.zeros(2)))
    with pytest.raises(ValueError, match="lwh"):
        Frame(**_ok_frame_kwargs(dynamic_objects=[bad]))


def test_dynamic_object_velocity_shape_validated():
    bad = DynamicObject(
        **_ok_object_kwargs(velocity_xy=np.zeros(3, dtype=np.float64))
    )
    with pytest.raises(ValueError, match="velocity"):
        Frame(**_ok_frame_kwargs(dynamic_objects=[bad]))


def test_frame_without_dynamic_objects_still_valid():
    f = Frame(**_ok_frame_kwargs())
    assert f.dynamic_objects is None
    assert f.image_ts_ms is None
    assert f.lidar_ts_ms is None
