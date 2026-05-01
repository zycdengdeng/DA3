"""Tests for object-local LiDAR-priority fusion + LR symmetry mirror."""

from __future__ import annotations

import numpy as np

from lidar_anchored_depth.pipeline.object_snapshot import (
    _DEFAULT_MIRROR_CLASSES,
    _mirror_object_local,
    fuse_object_clouds,
)


def test_mirror_object_local_negates_axis():
    pts = np.array([[1.0, 2.0, 3.0], [-0.5, 4.0, -1.0]])
    out_x = _mirror_object_local(pts, "x")
    np.testing.assert_allclose(out_x, [[-1.0, 2.0, 3.0], [0.5, 4.0, -1.0]])
    out_y = _mirror_object_local(pts, "y")
    np.testing.assert_allclose(out_y, [[1.0, -2.0, 3.0], [-0.5, -4.0, -1.0]])
    out_z = _mirror_object_local(pts, "z")
    np.testing.assert_allclose(out_z, [[1.0, 2.0, -3.0], [-0.5, 4.0, 1.0]])


def test_mirror_object_local_preserves_empty():
    out = _mirror_object_local(np.zeros((0, 3)), "y")
    assert out.shape == (0, 3)


def test_default_mirror_classes_are_four_vehicles():
    assert "Car" in _DEFAULT_MIRROR_CLASSES
    assert "Bus" in _DEFAULT_MIRROR_CLASSES
    assert "Pedestrian" not in _DEFAULT_MIRROR_CLASSES
    assert "Non_motor_rider" not in _DEFAULT_MIRROR_CLASSES


def test_fuse_object_lidar_only_when_aa_had_empty():
    lidar = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    xyz, rgb, src = fuse_object_clouds(
        aa_had_local=None, aa_had_colors=None,
        lidar_local=lidar, voxel_size=0.05, max_dist_to_lidar=None,
    )
    assert xyz.shape[0] == 2
    assert (src == 0).all()


def test_fuse_object_aa_had_only_when_lidar_empty():
    aa = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    aa_rgb = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    xyz, rgb, src = fuse_object_clouds(
        aa_had_local=aa, aa_had_colors=aa_rgb,
        lidar_local=None, voxel_size=0.05, max_dist_to_lidar=None,
    )
    assert xyz.shape[0] == 2
    assert (src == 1).all()


def test_fuse_object_lidar_wins_voxel():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aa = np.array([[0.02, 0.02, 0.02]])  # same voxel at 0.05m
    aa_rgb = np.array([[100, 100, 100]], dtype=np.uint8)
    xyz, rgb, src = fuse_object_clouds(
        aa_had_local=aa, aa_had_colors=aa_rgb,
        lidar_local=lidar, voxel_size=0.05, max_dist_to_lidar=None,
    )
    assert xyz.shape[0] == 1
    assert src[0] == 0  # only LiDAR survives


def test_fuse_object_aa_had_fills_empty_voxel():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aa = np.array([[2.0, 0.0, 0.0]])  # different voxel
    aa_rgb = np.array([[100, 100, 100]], dtype=np.uint8)
    xyz, rgb, src = fuse_object_clouds(
        aa_had_local=aa, aa_had_colors=aa_rgb,
        lidar_local=lidar, voxel_size=0.05, max_dist_to_lidar=None,
    )
    assert xyz.shape[0] == 2
    assert sorted(src.tolist()) == [0, 1]


def test_fuse_object_with_mirror_doubles_coverage():
    """A LiDAR scan with returns only on +y side should, after y-mirror,
    have returns on both +y and -y."""
    lidar = np.array([
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ])  # all +y
    xyz_no_mirror, _, _ = fuse_object_clouds(
        aa_had_local=None, aa_had_colors=None,
        lidar_local=lidar, voxel_size=0.05, max_dist_to_lidar=None,
        mirror_axis=None,
    )
    xyz_mirror, _, _ = fuse_object_clouds(
        aa_had_local=None, aa_had_colors=None,
        lidar_local=lidar, voxel_size=0.05, max_dist_to_lidar=None,
        mirror_axis="y",
    )
    # After mirroring, we should have at least one point with y < 0
    assert (xyz_mirror[:, 1] < -0.4).any()
    assert xyz_mirror.shape[0] > xyz_no_mirror.shape[0]


def test_fuse_object_max_dist_filters_aa_had_outliers():
    lidar = np.array([[0.0, 0.0, 0.0]])
    aa = np.array([
        [0.5, 0.0, 0.0],   # 0.5 m -> keep with max_dist=1
        [10.0, 0.0, 0.0],  # 10 m -> drop with max_dist=1
    ])
    aa_rgb = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    xyz, rgb, src = fuse_object_clouds(
        aa_had_local=aa, aa_had_colors=aa_rgb,
        lidar_local=lidar, voxel_size=0.05, max_dist_to_lidar=1.0,
    )
    n_aa = int((src == 1).sum())
    assert n_aa == 1  # only the close one survives
