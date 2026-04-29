"""Tests for ``data.roadside_v2x`` (path resolution + annotation parsing)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lidar_anchored_depth.data.roadside_v2x import (
    PCD_TIMESTAMP_TOLERANCE_MS,
    STATIC_FIXTURE_CLASSES,
    RoadsideV2XLoader,
)


def test_static_fixture_classes_documented():
    assert STATIC_FIXTURE_CLASSES == frozenset(
        {"Bollards", "Crash_bucket", "Cone"}
    )


def test_pcd_tolerance_default():
    assert PCD_TIMESTAMP_TOLERANCE_MS == 500


def test_pinhole_image_path_uses_folder_mapping():
    p = RoadsideV2XLoader.pinhole_image_path(Path("/scene"), "3", 1742877031036)
    assert p == Path(
        "/scene/road/cameras/pinhole0/cam3_1742877031036.png"
    )
    p = RoadsideV2XLoader.pinhole_image_path(Path("/scene"), "0", 1742877031036)
    assert p == Path(
        "/scene/road/cameras/pinhole3/cam0_1742877031036.png"
    )


def test_pinhole_image_path_rejects_fisheye():
    with pytest.raises(ValueError, match="not a pinhole"):
        RoadsideV2XLoader.pinhole_image_path(Path("/scene"), "2", 0)


def test_merged_pcd_path_aligned_default():
    p = RoadsideV2XLoader.merged_pcd_path(Path("/scene"), 12345)
    assert p == Path("/scene/road_labels/merged_pcd_all/12345.pcd")


def test_merged_pcd_path_fallback():
    p = RoadsideV2XLoader.merged_pcd_path(
        Path("/scene"), 12345, prefer_aligned=False
    )
    assert p == Path("/scene/road/lidar/merged_pcd/12345.pcd")


def test_annotation_json_path():
    p = RoadsideV2XLoader.annotation_json_path(Path("/scene"), "1742877031036")
    assert p == Path(
        "/scene/road_labels/interpolation_labels/1742877031036.json"
    )


def test_constructor_rejects_non_pinhole():
    with pytest.raises(ValueError, match="not a pinhole"):
        RoadsideV2XLoader(data_root="/tmp", cameras=["2"])


def test_parse_object_dynamic_class():
    loader = RoadsideV2XLoader(data_root="/tmp")
    raw = {
        "id": 1, "label": "Car",
        "x": -31.197, "y": -9.21, "z": -1.71,
        "length": 5.41, "width": 2.06, "height": 1.49,
        "roll": 0.0, "pitch": 0.0, "yaw": 3.083,
        "occlusion": 0, "num_points": 697,
        "vx": 0.5, "vy": -0.1,
    }
    obj, is_static = loader._parse_object(raw, scene_id="008")
    assert is_static is False
    assert obj.id == 1
    assert obj.label == "Car"
    np.testing.assert_allclose(obj.xyz, [-31.197, -9.21, -1.71])
    np.testing.assert_allclose(obj.lwh, [5.41, 2.06, 1.49])
    assert obj.yaw == pytest.approx(3.083)
    np.testing.assert_allclose(obj.velocity_xy, [0.5, -0.1])
    assert obj.occlusion == 0
    assert obj.num_points == 697
    assert obj.Z_min == pytest.approx(-1.71 - 1.49 / 2)
    assert obj.Z_max == pytest.approx(-1.71 + 1.49 / 2)


def test_parse_object_static_fixture_flagged():
    loader = RoadsideV2XLoader(data_root="/tmp")
    for cls in ["Bollards", "Crash_bucket", "Cone"]:
        raw = {
            "id": 99, "label": cls,
            "x": 0.0, "y": 0.0, "z": -1.0,
            "length": 0.5, "width": 0.5, "height": 1.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "occlusion": 0, "num_points": 50,
        }
        _, is_static = loader._parse_object(raw, scene_id="008")
        assert is_static is True, f"{cls} should be static fixture"


def test_filter_dynamic_drops_occluded():
    loader = RoadsideV2XLoader(data_root="/tmp", occlusion_max=2)
    raw_visible = {
        "id": 1, "label": "Car",
        "x": 0, "y": 0, "z": 0, "length": 4, "width": 2, "height": 1.5,
        "yaw": 0, "occlusion": 0, "num_points": 100,
    }
    raw_heavily_occluded = {**raw_visible, "occlusion": 2}
    obj_v, _ = loader._parse_object(raw_visible, "x")
    obj_o, _ = loader._parse_object(raw_heavily_occluded, "x")
    assert loader._filter_dynamic(obj_v) is True
    assert loader._filter_dynamic(obj_o) is False


def test_filter_dynamic_drops_low_point_count():
    loader = RoadsideV2XLoader(data_root="/tmp", min_num_points=20)
    raw_full = {
        "id": 1, "label": "Car",
        "x": 0, "y": 0, "z": 0, "length": 4, "width": 2, "height": 1.5,
        "yaw": 0, "occlusion": 0, "num_points": 100,
    }
    raw_sparse = {**raw_full, "num_points": 5}
    obj_f, _ = loader._parse_object(raw_full, "x")
    obj_s, _ = loader._parse_object(raw_sparse, "x")
    assert loader._filter_dynamic(obj_f) is True
    assert loader._filter_dynamic(obj_s) is False


def test_global_key_pairs_scene_with_id():
    loader = RoadsideV2XLoader(data_root="/tmp")
    assert loader._global_key("008", 17) == ("008", 17)
    # cross-scene id collision is disambiguated by the scene component
    assert loader._global_key("008", 1) != loader._global_key("009", 1)


def test_index_not_built_returns_zero_length():
    loader = RoadsideV2XLoader(data_root="/tmp")
    assert len(loader) == 0


def test_get_frame_not_implemented_at_stage12():
    loader = RoadsideV2XLoader(data_root="/tmp")
    with pytest.raises(NotImplementedError):
        loader.get_frame(0)


def test_read_annotation_entry_roundtrip(tmp_path):
    payload = {
        "timestamp": "1742877031036",
        "pcd_file": "1742877031036.pcd",
        "object": [{"id": 1, "label": "Car", "x": 0, "y": 0, "z": 0,
                    "length": 4, "width": 2, "height": 1.5, "yaw": 0,
                    "occlusion": 0, "num_points": 100}],
    }
    json_path = tmp_path / "x.json"
    json_path.write_text(json.dumps(payload))
    loader = RoadsideV2XLoader(data_root=str(tmp_path))
    out = loader._read_annotation_entry(json_path)
    assert out == payload
