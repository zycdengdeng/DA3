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


# --------------------------------------------------------------------- #
# ICP
# --------------------------------------------------------------------- #
from lidar_anchored_depth.reconstruction.icp import (  # noqa: E402
    apply_transform,
    icp_point_to_point,
)


def test_icp_identity_when_source_equals_target():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 10, size=(200, 3))
    T, info = icp_point_to_point(pts, pts, max_iterations=10)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-9)
    assert info["final_residual"] < 1e-9


def test_icp_recovers_translation():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(500, 3))
    t_true = np.array([1.5, -2.3, 0.7])
    target = pts + t_true
    # trim_percentile=100 disables outlier rejection; the source is
    # cleanly translated so every correspondence should converge.
    T, info = icp_point_to_point(
        pts, target, max_iterations=50, trim_percentile=100.0,
    )
    np.testing.assert_allclose(T[:3, 3], t_true, atol=1e-2)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-2)


def test_icp_recovers_rotation():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(500, 3))
    # 30° rotation about Z
    theta = np.deg2rad(30.0)
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    target = (R_true @ pts.T).T
    # ICP needs a decent init for rotations; use identity (works because
    # the cloud is well-spread in 3D and 30° is recoverable).
    T, info = icp_point_to_point(pts, target, max_iterations=50, trim_percentile=100.0)
    # Compare the rotation extracted from T to R_true
    np.testing.assert_allclose(T[:3, :3], R_true, atol=1e-2)
    assert info["final_residual"] < 0.5


def test_icp_recovers_full_se3():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(500, 3))
    theta = np.deg2rad(15.0)
    R_true = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    t_true = np.array([0.5, -1.2, 2.0])
    target = (R_true @ pts.T).T + t_true
    T, info = icp_point_to_point(pts, target, max_iterations=50, trim_percentile=100.0)
    # Apply T to source, check it matches target
    src_aligned = apply_transform(pts, T)
    err = np.linalg.norm(src_aligned - target, axis=1).mean()
    assert err < 0.05


def test_icp_robust_to_outliers():
    """20% of source points are far from the target → ICP should still
    align the inliers (trim_percentile=80 drops the worst 20%)."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(500, 3))
    t_true = np.array([1.0, 2.0, 0.5])
    target = pts + t_true
    # Inject 100 outliers
    pts_outlier = np.concatenate([pts, rng.uniform(-50, 50, size=(100, 3))])
    T, info = icp_point_to_point(
        pts_outlier, target, max_iterations=30, trim_percentile=80.0,
    )
    # Translation should still recover
    np.testing.assert_allclose(T[:3, 3], t_true, atol=0.1)


def test_icp_too_few_points_raises():
    with pytest.raises(ValueError, match="need >="):
        icp_point_to_point(np.zeros((2, 3)), np.zeros((10, 3)))


def test_apply_transform_round_trip():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    T = np.eye(4)
    T[:3, 3] = [10, 20, 30]
    out = apply_transform(pts, T)
    np.testing.assert_allclose(out, pts + [10, 20, 30])


def test_apply_transform_empty():
    out = apply_transform(np.zeros((0, 3)), np.eye(4))
    assert out.shape == (0, 3)


# --------------------------------------------------------------------- #
# RGB-aware PLY round-trip
# --------------------------------------------------------------------- #
from lidar_anchored_depth.reconstruction.io import read_ply_xyz_rgb  # noqa: E402


def test_ply_xyz_rgb_round_trip_ascii(tmp_path):
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    rgb = np.array([[255, 128, 0], [10, 20, 30]], dtype=np.uint8)
    p = tmp_path / "x.ply"
    write_ply_xyz(p, pts, rgb)
    pts_back, rgb_back = read_ply_xyz_rgb(p)
    np.testing.assert_allclose(pts_back, pts, atol=1e-5)
    np.testing.assert_array_equal(rgb_back, rgb)


def test_ply_xyz_rgb_returns_none_when_no_rgb(tmp_path):
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    p = tmp_path / "x.ply"
    write_ply_xyz(p, pts)
    pts_back, rgb_back = read_ply_xyz_rgb(p)
    assert rgb_back is None


def test_ply_xyz_rgb_round_trip_binary(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(50, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(50, 3), dtype=np.uint8)
    p = tmp_path / "x.ply"
    write_ply_xyz(p, pts, rgb, binary=True)
    pts_back, rgb_back = read_ply_xyz_rgb(p)
    np.testing.assert_allclose(pts_back, pts, atol=1e-5)
    np.testing.assert_array_equal(rgb_back, rgb)


# --- voxel_downsample_robust ---------------------------------------------

def test_voxel_robust_median_resists_outlier():
    """One voxel with three samples: two grey [120 120 120], one red
    [255 0 0] (the "moving car" outlier). Mean would give pinkish;
    median should return grey."""
    from lidar_anchored_depth.reconstruction import voxel_downsample_robust

    pts = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],
        [0.03, 0.03, 0.03],
    ])
    rgb = np.array([
        [120, 120, 120],
        [120, 120, 120],
        [255, 0, 0],
    ], dtype=np.uint8)
    _, rgb_out = voxel_downsample_robust(
        pts, voxel_size=1.0, colors=rgb, color_method="median",
    )
    assert rgb_out.shape == (1, 3)
    assert tuple(int(c) for c in rgb_out[0]) == (120, 120, 120)


def test_voxel_robust_median_matches_mean_when_uniform():
    from lidar_anchored_depth.reconstruction import voxel_downsample_robust

    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    rgb = np.array([[100, 50, 200], [100, 50, 200]], dtype=np.uint8)
    _, mean_rgb = voxel_downsample_robust(
        pts, 1.0, colors=rgb, color_method="mean",
    )
    _, med_rgb = voxel_downsample_robust(
        pts, 1.0, colors=rgb, color_method="median",
    )
    assert np.array_equal(mean_rgb, med_rgb)


def test_voxel_robust_madtrim_falls_back_to_mean_for_small_voxels():
    from lidar_anchored_depth.reconstruction import voxel_downsample_robust

    # 2 samples per voxel — too few for MAD; fall back to mean.
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    rgb = np.array([[100, 100, 100], [200, 200, 200]], dtype=np.uint8)
    _, rgb_out = voxel_downsample_robust(
        pts, 1.0, colors=rgb, color_method="mad-trim",
    )
    assert tuple(int(c) for c in rgb_out[0]) == (150, 150, 150)


def test_voxel_robust_separates_voxels_by_key():
    from lidar_anchored_depth.reconstruction import voxel_downsample_robust

    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.01, 0.01, 0.01],
        [10.0, 10.0, 10.0],
    ])
    rgb = np.array([
        [120, 120, 120],
        [120, 120, 120],
        [50, 50, 50],
    ], dtype=np.uint8)
    xyz_out, rgb_out = voxel_downsample_robust(
        pts, voxel_size=1.0, colors=rgb, color_method="median",
    )
    assert xyz_out.shape == (2, 3)
    sorted_rgb = sorted(tuple(int(c) for c in row) for row in rgb_out)
    assert sorted_rgb == [(50, 50, 50), (120, 120, 120)]


def test_voxel_robust_unknown_method_raises():
    import pytest
    from lidar_anchored_depth.reconstruction import voxel_downsample_robust

    pts = np.array([[0.0, 0.0, 0.0]])
    rgb = np.array([[100, 100, 100]], dtype=np.uint8)
    with pytest.raises(ValueError):
        voxel_downsample_robust(pts, 1.0, colors=rgb, color_method="bogus")
