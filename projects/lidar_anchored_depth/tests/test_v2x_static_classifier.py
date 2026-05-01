"""Tests for the V2X static/dynamic classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from lidar_anchored_depth.pipeline.v2x_static_classifier import (
    ObjectMotionClass,
    classify_v2x_objects,
    split_static_dynamic_ids,
)


@dataclass
class _FakeObj:
    id: int
    label: str
    xyz: np.ndarray


@dataclass
class _FakeFrame:
    dynamic_objects: list


@dataclass
class _FakeScene:
    scene_id: str
    timestamps_ms: list


class _FakeLoader:
    """Minimal loader stub: maps (scene_id, ts, cam) -> frame index, and
    indices to frames. Tests only exercise dynamic_objects."""

    def __init__(self, scenes, frames_by_idx, idx_lookup):
        self.scenes = scenes
        self._frames = frames_by_idx
        self._idx_lookup = idx_lookup

    def find_frame_idx(self, scene_id, ts, cam_id):
        key = (scene_id, ts, cam_id)
        if key not in self._idx_lookup:
            raise ValueError(f"no frame for {key}")
        return self._idx_lookup[key]

    def get_frame(self, idx):
        return self._frames[idx]


def _build_loader(scene_id: str, ts_to_objects: dict) -> _FakeLoader:
    timestamps = sorted(ts_to_objects.keys())
    scenes = [_FakeScene(scene_id=scene_id, timestamps_ms=timestamps)]
    frames = {}
    idx_lookup = {}
    for i, ts in enumerate(timestamps):
        frames[i] = _FakeFrame(dynamic_objects=ts_to_objects[ts])
        idx_lookup[(scene_id, ts, "0")] = i
    return _FakeLoader(scenes, frames, idx_lookup)


def test_classify_class_whitelist_static():
    """Bollards / Crash_bucket / Cone are static regardless of drift."""
    loader = _build_loader("s", {
        100: [_FakeObj(1, "Bollards", np.array([0.0, 0.0, 0.0]))],
        200: [_FakeObj(1, "Bollards", np.array([10.0, 10.0, 10.0]))],
    })
    out = classify_v2x_objects(loader, "s")
    assert out[1].is_static
    assert out[1].reason == "class-whitelist"


def test_classify_low_drift_static():
    """A 'Car' that barely moves (within drift threshold) is classified
    static."""
    loader = _build_loader("s", {
        100: [_FakeObj(2, "Car", np.array([0.0, 0.0, 0.0]))],
        200: [_FakeObj(2, "Car", np.array([0.1, 0.1, 0.0]))],
        300: [_FakeObj(2, "Car", np.array([0.0, 0.05, 0.0]))],
    })
    out = classify_v2x_objects(loader, "s", drift_threshold_m=0.5)
    assert out[2].is_static
    assert out[2].reason == "low-drift"


def test_classify_high_drift_dynamic():
    """A 'Car' driving across the intersection is dynamic."""
    loader = _build_loader("s", {
        100: [_FakeObj(3, "Car", np.array([0.0, 0.0, 0.0]))],
        200: [_FakeObj(3, "Car", np.array([5.0, 0.0, 0.0]))],
        300: [_FakeObj(3, "Car", np.array([10.0, 0.0, 0.0]))],
    })
    out = classify_v2x_objects(loader, "s", drift_threshold_m=0.5)
    assert not out[3].is_static
    assert out[3].reason == "high-drift"
    assert out[3].max_drift_m > 5.0


def test_classify_single_ts_treated_dynamic():
    """An obj seen at only one ts can't have its motion measured;
    default to dynamic so it goes through the snapshot path."""
    loader = _build_loader("s", {
        100: [_FakeObj(4, "Car", np.array([0.0, 0.0, 0.0]))],
    })
    out = classify_v2x_objects(loader, "s")
    assert not out[4].is_static
    assert out[4].reason == "single-ts"


def test_split_static_dynamic_ids():
    cls = {
        1: ObjectMotionClass(1, "Bollards", True, 5, 0.0, "class-whitelist"),
        2: ObjectMotionClass(2, "Car", False, 5, 8.0, "high-drift"),
        3: ObjectMotionClass(3, "Car", True, 3, 0.1, "low-drift"),
    }
    static, dyn = split_static_dynamic_ids(cls)
    assert static == {1, 3}
    assert dyn == {2}


def test_classify_unknown_scene_returns_empty():
    loader = _build_loader("s", {100: []})
    out = classify_v2x_objects(loader, "other-scene")
    assert out == {}
