"""Tests for per-object snapshot discovery and injection."""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.pipeline.object_snapshot import (
    _pick_object_ts,
    discover_object_clouds,
)
from lidar_anchored_depth.reconstruction.io import write_ply_xyz


def test_discover_picks_icp_when_present(tmp_path):
    pts_aa = np.array([[1.0, 1.0, 1.0]])
    pts_icp = np.array([[2.0, 2.0, 2.0]])
    pts_lid = np.array([[3.0, 3.0, 3.0]])
    rgb = np.array([[1, 1, 1]], dtype=np.uint8)
    write_ply_xyz(tmp_path / "008_obj7_aa_had.ply", pts_aa, rgb)
    write_ply_xyz(tmp_path / "008_obj7_aa_had_icp.ply", pts_icp, rgb)
    write_ply_xyz(tmp_path / "008_obj7_lidar.ply", pts_lid)

    clouds = discover_object_clouds(tmp_path, prefer_icp=True)
    assert set(clouds.keys()) == {7}
    obj = clouds[7]
    np.testing.assert_allclose(obj.aa_had_local, pts_icp)
    np.testing.assert_allclose(obj.lidar_local, pts_lid)
    assert obj.aa_had_path.name.endswith("aa_had_icp.ply")


def test_discover_falls_back_to_plain_aa_had_when_no_icp(tmp_path):
    pts = np.array([[1.0, 1.0, 1.0]])
    rgb = np.array([[1, 1, 1]], dtype=np.uint8)
    write_ply_xyz(tmp_path / "008_obj3_aa_had.ply", pts, rgb)
    write_ply_xyz(tmp_path / "008_obj3_lidar.ply", pts)

    clouds = discover_object_clouds(tmp_path, prefer_icp=True)
    assert clouds[3].aa_had_path.name.endswith("aa_had.ply")


def test_discover_handles_lidar_only(tmp_path):
    pts = np.array([[5.0, 5.0, 5.0]])
    write_ply_xyz(tmp_path / "008_obj1_lidar.ply", pts)

    clouds = discover_object_clouds(tmp_path)
    assert 1 in clouds
    assert clouds[1].aa_had_local is None
    np.testing.assert_allclose(clouds[1].lidar_local, pts)


def test_discover_ignores_unrelated_files(tmp_path):
    pts = np.array([[1.0, 1.0, 1.0]])
    write_ply_xyz(tmp_path / "008_obj1_aa_had.ply", pts)
    (tmp_path / "008_recon_summary.json").write_text("{}")
    (tmp_path / "008_aahad_static.ply").write_text("ply\n")

    clouds = discover_object_clouds(tmp_path)
    assert set(clouds.keys()) == {1}


def test_discover_prefer_icp_false(tmp_path):
    pts_aa = np.array([[1.0, 1.0, 1.0]])
    pts_icp = np.array([[2.0, 2.0, 2.0]])
    rgb = np.array([[1, 1, 1]], dtype=np.uint8)
    write_ply_xyz(tmp_path / "008_obj7_aa_had.ply", pts_aa, rgb)
    write_ply_xyz(tmp_path / "008_obj7_aa_had_icp.ply", pts_icp, rgb)

    clouds = discover_object_clouds(tmp_path, prefer_icp=False)
    np.testing.assert_allclose(clouds[7].aa_had_local, pts_aa)


# --- _pick_object_ts_video (interpolation + window cull) ----------------

def _make_obj(obj_id, label, xyz, yaw=0.0, pitch=0.0, roll=0.0):
    """Construct a minimal DynamicObject with the fields the picker uses."""
    from lidar_anchored_depth.data.base import DynamicObject

    return DynamicObject(
        id=obj_id, label=label,
        xyz=np.asarray(xyz, dtype=np.float64),
        lwh=np.array([4.0, 2.0, 1.5], dtype=np.float64),
        yaw=float(yaw), pitch=float(pitch), roll=float(roll),
        num_points=10,
    )


def test_pick_video_drops_outside_window():
    from lidar_anchored_depth.pipeline.object_snapshot import _pick_object_ts_video

    cache = {7: [
        (1000, _make_obj(7, "Car", [0, 0, 0])),
        (1100, _make_obj(7, "Car", [10, 0, 0])),
    ]}
    # Way before window
    assert _pick_object_ts_video(cache, 7, 500, max_extrap_ms=100) is None
    # Way after window
    assert _pick_object_ts_video(cache, 7, 2000, max_extrap_ms=100) is None
    # Just inside window (within 100 ms of last ts)
    out = _pick_object_ts_video(cache, 7, 1150, max_extrap_ms=100)
    assert out is not None
    assert out[0] == 1150


def test_pick_video_interpolates_xyz_linearly():
    from lidar_anchored_depth.pipeline.object_snapshot import _pick_object_ts_video

    cache = {7: [
        (1000, _make_obj(7, "Car", [0, 0, 0])),
        (1100, _make_obj(7, "Car", [10, 4, 0])),
    ]}
    # Half-way between the two annotated ts
    pick = _pick_object_ts_video(cache, 7, 1050, max_extrap_ms=0)
    assert pick is not None
    ts_out, obj_out = pick
    assert ts_out == 1050
    np.testing.assert_allclose(obj_out.xyz, [5.0, 2.0, 0.0], atol=1e-6)


def test_pick_video_interpolates_yaw_with_wraparound():
    from lidar_anchored_depth.pipeline.object_snapshot import _pick_object_ts_video

    # yaw goes from -3 rad to +3 rad — the short path is across the
    # +/-pi seam (~0.28 rad), not the long path (~6 rad).
    cache = {7: [
        (1000, _make_obj(7, "Car", [0, 0, 0], yaw=-3.0)),
        (1100, _make_obj(7, "Car", [0, 0, 0], yaw=+3.0)),
    ]}
    pick = _pick_object_ts_video(cache, 7, 1050, max_extrap_ms=0)
    assert pick is not None
    _, obj = pick
    # Half-way along short path -> approx +/-pi
    assert abs(abs(obj.yaw) - 3.1416) < 0.05


def test_pick_video_returns_exact_match():
    from lidar_anchored_depth.pipeline.object_snapshot import _pick_object_ts_video

    cache = {7: [
        (1000, _make_obj(7, "Car", [0, 0, 0])),
        (1100, _make_obj(7, "Car", [10, 0, 0])),
    ]}
    pick = _pick_object_ts_video(cache, 7, 1100, max_extrap_ms=0)
    assert pick is not None
    assert pick[0] == 1100
    np.testing.assert_allclose(pick[1].xyz, [10, 0, 0])


def test_pick_video_returns_none_for_unknown_obj():
    from lidar_anchored_depth.pipeline.object_snapshot import _pick_object_ts_video

    assert _pick_object_ts_video({}, 7, 1000, max_extrap_ms=100) is None


def test_pick_video_drops_inside_huge_gap():
    """User-reported bug: V2X tracking lost between t=1100 and t=2100
    (1 sec gap, 10x larger than normal 100 ms ts spacing). Linearly
    interpolating across the gap teleported the object across the
    scene. Now we drop it."""
    from lidar_anchored_depth.pipeline.object_snapshot import _pick_object_ts_video

    cache = {7: [
        (1000, _make_obj(7, "Car", [50, 0, 0])),
        (1100, _make_obj(7, "Car", [55, 0, 0])),  # smooth motion
        (2100, _make_obj(7, "Car", [0, 0, 0])),   # 1 sec later, jump
        (2200, _make_obj(7, "Car", [0, 0, 0])),
    ]}
    # Gap window 1100-2100 (1000 ms) > max_gap_ms 500 → cull
    assert _pick_object_ts_video(cache, 7, 1500, max_extrap_ms=200, max_gap_ms=500) is None
    assert _pick_object_ts_video(cache, 7, 1700, max_extrap_ms=200, max_gap_ms=500) is None
    # Inside normal-spacing windows: still rendered
    out = _pick_object_ts_video(cache, 7, 1050, max_extrap_ms=200, max_gap_ms=500)
    assert out is not None
    out = _pick_object_ts_video(cache, 7, 2150, max_extrap_ms=200, max_gap_ms=500)
    assert out is not None


def test_pick_video_max_gap_disabled_passes_through():
    """With max_gap_ms = inf, even huge gaps interpolate (old behaviour)."""
    from lidar_anchored_depth.pipeline.object_snapshot import _pick_object_ts_video

    cache = {7: [
        (1000, _make_obj(7, "Car", [0, 0, 0])),
        (5000, _make_obj(7, "Car", [100, 0, 0])),
    ]}
    out = _pick_object_ts_video(cache, 7, 3000, max_extrap_ms=0, max_gap_ms=10**9)
    assert out is not None
    assert abs(out[1].xyz[0] - 50.0) < 1e-3
