"""Tests for per-object snapshot discovery and injection."""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.pipeline.object_snapshot import (
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
