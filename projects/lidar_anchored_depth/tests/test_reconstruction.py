"""Tests for ``reconstruction.{io, chamfer, object_local, unproject}``."""

from __future__ import annotations

import numpy as np
import pytest

from lidar_anchored_depth.data.base import DynamicObject
from lidar_anchored_depth.reconstruction.chamfer import chamfer_distance, chamfer_l2
from lidar_anchored_depth.reconstruction.io import read_ply_xyz, write_ply_xyz
from lidar_anchored_depth.reconstruction.object_local import (
    object_local_to_world,
    voxel_downsample,
    world_to_object_local,
)
from lidar_anchored_depth.reconstruction.unproject import depth_to_world_points


# --------------------------------------------------------------------- #
# PLY round-trip
# --------------------------------------------------------------------- #
def test_ply_ascii_xyz_roundtrip(tmp_path):
    pts = np.array([[1.0, 2.0, 3.0], [-4.5, 6.7, -8.9], [0, 0, 0]], dtype=np.float32)
    p = tmp_path / "a.ply"
    write_ply_xyz(p, pts)
    out = read_ply_xyz(p)
    np.testing.assert_allclose(out, pts, atol=1e-5)


def test_ply_ascii_with_rgb(tmp_path):
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    rgb = np.array([[255, 128, 0], [10, 20, 30]], dtype=np.uint8)
    p = tmp_path / "b.ply"
    write_ply_xyz(p, pts, rgb)
    out = read_ply_xyz(p)
    np.testing.assert_allclose(out, pts, atol=1e-5)
    # Header must mention RGB props
    text = p.read_text()
    assert "property uchar red" in text
    assert "property uchar green" in text
    assert "property uchar blue" in text


def test_ply_binary_xyz_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-100, 100, size=(500, 3)).astype(np.float32)
    p = tmp_path / "bin.ply"
    write_ply_xyz(p, pts, binary=True)
    out = read_ply_xyz(p)
    np.testing.assert_allclose(out, pts, atol=1e-5)


def test_ply_binary_with_rgb_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 10, size=(50, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(50, 3), dtype=np.uint8)
    p = tmp_path / "bin_rgb.ply"
    write_ply_xyz(p, pts, rgb, binary=True)
    out = read_ply_xyz(p)
    np.testing.assert_allclose(out, pts, atol=1e-5)


def test_ply_color_shape_mismatch_raises(tmp_path):
    pts = np.zeros((3, 3), dtype=np.float32)
    bad = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="align"):
        write_ply_xyz(tmp_path / "x.ply", pts, bad)


# --------------------------------------------------------------------- #
# Chamfer
# --------------------------------------------------------------------- #
def test_chamfer_zero_for_identical_clouds():
    pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    cd = chamfer_l2(pts, pts)
    assert cd == pytest.approx(0.0, abs=1e-12)


def test_chamfer_positive_for_translated_clouds():
    A = np.random.default_rng(0).uniform(0, 10, size=(100, 3))
    B = A + np.array([0.5, 0.0, 0.0])
    cd = chamfer_l2(A, B)
    # Each A→nearest-B is at most 0.5; each B→nearest-A is at most 0.5
    # Symmetric chamfer = sum of two means → ≈ 1.0
    assert cd == pytest.approx(1.0, abs=0.05)


def test_chamfer_stats_have_expected_keys():
    A = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    B = np.array([[0, 0, 0]], dtype=np.float64)
    cd, stats = chamfer_distance(A, B)
    expected = {
        "a_to_b_mean", "a_to_b_median", "a_to_b_p95",
        "b_to_a_mean", "b_to_a_median", "b_to_a_p95",
        "n_a", "n_b",
    }
    assert expected <= set(stats.keys())
    assert stats["n_a"] == 2 and stats["n_b"] == 1


def test_chamfer_empty_cloud_raises():
    A = np.zeros((0, 3))
    B = np.array([[1, 2, 3]], dtype=np.float64)
    with pytest.raises(ValueError, match="non-empty"):
        chamfer_distance(A, B)


# --------------------------------------------------------------------- #
# Object-local transforms
# --------------------------------------------------------------------- #
def _make_obj(yaw=0.0, xyz=(10.0, 20.0, 1.0)):
    return DynamicObject(
        id=1, label="Car",
        xyz=np.array(xyz, dtype=np.float64),
        lwh=np.array([4.0, 2.0, 1.5]),
        yaw=yaw,
    )


def test_world_to_object_local_round_trip():
    obj = _make_obj(yaw=0.7, xyz=(5.0, -3.0, 0.5))
    rng = np.random.default_rng(0)
    P_world = rng.uniform(-50, 50, size=(20, 3))
    P_local = world_to_object_local(P_world, obj)
    P_back = object_local_to_world(P_local, obj)
    np.testing.assert_allclose(P_back, P_world, atol=1e-9)


def test_world_to_object_local_centers_centroid():
    """Object centre in world → origin in object-local."""
    obj = _make_obj(yaw=1.2, xyz=(7.0, 8.0, 0.3))
    centred = world_to_object_local(obj.xyz.reshape(1, 3), obj)
    np.testing.assert_allclose(centred[0], [0, 0, 0], atol=1e-12)


def test_world_to_object_local_yaw_undoes_rotation():
    """A point on world +X relative to bbox at yaw=π/2 → +Y in local."""
    obj = _make_obj(yaw=np.pi / 2, xyz=(0.0, 0.0, 0.0))
    p = np.array([[0.0, 1.0, 0.0]])  # in world, this is "ahead of bbox"
    p_local = world_to_object_local(p, obj)
    # bbox local +X (length axis) was rotated to world +Y; so world +Y → local +X
    np.testing.assert_allclose(p_local[0], [1.0, 0.0, 0.0], atol=1e-9)


def test_world_to_object_local_empty():
    obj = _make_obj()
    out = world_to_object_local(np.zeros((0, 3)), obj)
    assert out.shape == (0, 3)


# --------------------------------------------------------------------- #
# Voxel downsampling
# --------------------------------------------------------------------- #
def test_voxel_downsample_dedupes_redundant_points():
    # Three points all in voxel (0, 0, 0) of a 0.05m grid (i.e. all in
    # [0, 0.05)^3), and one far away. Negative coords would cross the
    # voxel boundary at 0 (floor(-0.001/0.05) = -1), so we keep this
    # test strictly positive.
    pts = np.array(
        [[0.001, 0.002, 0.003],
         [0.04, 0.04, 0.04],
         [0.0005, 0.001, 0.0],
         [10, 10, 10]],
        dtype=np.float64,
    )
    out, _ = voxel_downsample(pts, voxel_size=0.05)
    # First three share voxel (0,0,0); last is in (200,200,200) → 2 voxels
    assert out.shape[0] == 2


def test_voxel_downsample_returns_centroids():
    pts = np.array(
        [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]], dtype=np.float64
    )
    out, _ = voxel_downsample(pts, voxel_size=0.05)
    np.testing.assert_allclose(out[0], pts.mean(axis=0))


def test_voxel_downsample_color_averaging():
    pts = np.array([[0, 0, 0], [0.01, 0.01, 0.01]], dtype=np.float64)
    cols = np.array([[100, 0, 0], [0, 100, 0]], dtype=np.uint8)
    _, out_c = voxel_downsample(pts, voxel_size=0.05, colors=cols)
    np.testing.assert_array_equal(out_c[0], [50, 50, 0])


def test_voxel_downsample_empty():
    out, _ = voxel_downsample(np.zeros((0, 3)), 0.05)
    assert out.shape == (0, 3)


def test_voxel_downsample_invalid_size():
    with pytest.raises(ValueError, match="voxel_size"):
        voxel_downsample(np.zeros((1, 3)), -0.1)


# --------------------------------------------------------------------- #
# depth_to_world_points
# --------------------------------------------------------------------- #
def test_depth_to_world_points_basic(synthetic_camera):
    K, T_wc = synthetic_camera
    H, W = 100, 100
    depth = np.full((H, W), 30.0, dtype=np.float32)
    image = np.full((H, W, 3), 200, dtype=np.uint8)
    P_world, colors = depth_to_world_points(
        depth, image, K, T_wc, mask=None, z_min=0.5, z_max=250.0,
    )
    # Every pixel should yield a point
    assert P_world.shape == (H * W, 3)
    assert colors.shape == (H * W, 3)
    np.testing.assert_array_equal(colors[0], [200, 200, 200])


def test_depth_to_world_points_drops_invalid(synthetic_camera):
    K, T_wc = synthetic_camera
    depth = np.array([[10.0, np.nan, -5.0, 300.0, 50.0]], dtype=np.float32)
    image = np.zeros((1, 5, 3), dtype=np.uint8)
    P, _ = depth_to_world_points(
        depth, image, K, T_wc, z_min=1.0, z_max=200.0,
    )
    # Only depths 10 and 50 survive
    assert P.shape[0] == 2


def test_depth_to_world_points_mask_restricts(synthetic_camera):
    K, T_wc = synthetic_camera
    H, W = 50, 50
    depth = np.full((H, W), 30.0, dtype=np.float32)
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 10:20] = True  # 100 pixels
    P, _ = depth_to_world_points(depth, image, K, T_wc, mask=mask)
    assert P.shape[0] == 100


def test_depth_to_world_points_shape_mismatch(synthetic_camera):
    K, T_wc = synthetic_camera
    depth = np.zeros((10, 10))
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="match"):
        depth_to_world_points(depth, image, K, T_wc)
