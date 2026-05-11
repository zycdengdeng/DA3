"""Tests for the dual labels-source feature.

Two label folders co-exist in the V2X dataset:

* ``road_labels/interpolation_labels/<ts>.json`` — 10 Hz interpolated
  bboxes; smooth across video frames but the interpolation jitter
  bleeds dynamic-object LiDAR into the "static" cloud at bbox edges.
* ``road_labels/merged_pcd_all/<ts>.json`` — 1 Hz hand-labeled bboxes;
  the static accumulator wants these so its bbox cull is exact.

These tests verify the loader honours the ``labels_source`` switch
and the higher-level configs default to the right split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_scene_tree(
    root: Path,
    *,
    interp_ts: list[int],
    merged_ts: list[int],
) -> Path:
    """Build a minimal V2X scene directory with both label folders
    populated. Returns the scene_dir."""
    scene_dir = root / "008_dummy_scene"
    (scene_dir / "road_labels" / "interpolation_labels").mkdir(parents=True)
    (scene_dir / "road_labels" / "merged_pcd_all").mkdir(parents=True)
    for ts in interp_ts:
        (scene_dir / "road_labels" / "interpolation_labels" / f"{ts}.json"
         ).write_text(json.dumps({"timestamp": ts, "objects": []}))
    for ts in merged_ts:
        (scene_dir / "road_labels" / "merged_pcd_all" / f"{ts}.json"
         ).write_text(json.dumps({"timestamp": ts, "objects": []}))
    # support_info/ calib.json is needed at data_root level by the
    # property accessor, but the index-build path only needs scene dirs
    # — we never trigger calibration loading here.
    return scene_dir


def test_loader_rejects_unknown_labels_source(tmp_path):
    from lidar_anchored_depth.data import RoadsideV2XLoader
    with pytest.raises(ValueError, match="labels_source"):
        RoadsideV2XLoader(tmp_path, labels_source="bogus")


def test_loader_default_labels_source_is_interpolation(tmp_path):
    from lidar_anchored_depth.data import RoadsideV2XLoader
    loader = RoadsideV2XLoader(tmp_path)
    assert loader.labels_source == "interpolation"


def test_loader_picks_interpolation_folder_by_default(tmp_path):
    """With default settings, the loader's scene index should pick
    up the 10 Hz interpolation ts, not the 1 Hz hand-label ts."""
    from lidar_anchored_depth.data import RoadsideV2XLoader

    _make_scene_tree(
        tmp_path,
        interp_ts=[1000, 1100, 1200, 1300, 1400],
        merged_ts=[1000, 1400],
    )
    loader = RoadsideV2XLoader(tmp_path, labels_source="interpolation")
    # _build_index_if_needed walks scenes via the .scenes property.
    scenes = loader.scenes
    assert len(scenes) == 1
    assert len(scenes[0].timestamps_ms) == 5


def test_loader_picks_merged_pcd_folder_when_requested(tmp_path):
    """With labels_source='merged_pcd', the loader's scene index should
    pick up only the 1 Hz hand-label ts."""
    from lidar_anchored_depth.data import RoadsideV2XLoader

    _make_scene_tree(
        tmp_path,
        interp_ts=[1000, 1100, 1200, 1300, 1400],
        merged_ts=[1000, 1400],
    )
    loader = RoadsideV2XLoader(tmp_path, labels_source="merged_pcd")
    scenes = loader.scenes
    assert len(scenes) == 1
    # Only the hand-labelled ts are exposed.
    assert sorted(scenes[0].timestamps_ms) == [1000, 1400]


def test_loader_skips_scene_with_no_labels(tmp_path):
    """A scene with empty ``merged_pcd_all/`` should not appear when
    ``labels_source='merged_pcd'``."""
    from lidar_anchored_depth.data import RoadsideV2XLoader

    _make_scene_tree(
        tmp_path,
        interp_ts=[1000, 1100],
        merged_ts=[],  # no hand-labels at all
    )
    loader = RoadsideV2XLoader(tmp_path, labels_source="merged_pcd")
    assert loader.scenes == []  # scene dropped


# ---- SceneConfig defaults -----------------------------------------------

def test_scene_config_defaults_split_static_and_dynamic():
    """The whole point of this feature: by default the static cloud
    is built from hand-labels (more accurate), while dynamic-object
    handling uses interpolation (smoother across video frames)."""
    from lidar_anchored_depth.configs import SceneConfig

    cfg = SceneConfig(scene="008")
    assert cfg.static_labels_source == "merged_pcd"
    assert cfg.dynamic_labels_source == "interpolation"


def test_scene_config_accepts_override():
    """Some scenes won't have hand-labels — user must be able to fall
    back to interpolation for static too."""
    from lidar_anchored_depth.configs import SceneConfig

    cfg = SceneConfig(scene="008", static_labels_source="interpolation")
    assert cfg.static_labels_source == "interpolation"
    assert cfg.dynamic_labels_source == "interpolation"  # default unchanged
