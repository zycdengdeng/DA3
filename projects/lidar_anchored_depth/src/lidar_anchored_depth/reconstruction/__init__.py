"""Object-centric reconstruction utilities.

This subpackage holds the pieces that turn per-frame metric depth
(courtesy of AA-HAD) into per-object dense point clouds and
multi-camera multi-frame world-frame reconstructions.

Modules
-------
io
    PLY writer (XYZ + optional RGB), no open3d dependency.
chamfer
    Chamfer distance between two point clouds (KD-tree based).
object_local
    Transform points between world frame and an object's local frame
    using its V2X bbox pose, plus voxel-grid downsampling.
unproject
    Backproject a 2D depth map to a world-frame colored point cloud.
"""

from lidar_anchored_depth.reconstruction.chamfer import (
    chamfer_distance,
    chamfer_l2,
)
from lidar_anchored_depth.reconstruction.icp import (
    apply_transform,
    icp_point_to_point,
)
from lidar_anchored_depth.reconstruction.io import (
    read_ply_xyz,
    read_ply_xyz_rgb,
    write_ply_xyz,
)
from lidar_anchored_depth.reconstruction.object_local import (
    object_local_to_world,
    voxel_downsample,
    world_to_object_local,
)
from lidar_anchored_depth.reconstruction.unproject import (
    depth_to_world_points,
)

__all__ = [
    "apply_transform",
    "chamfer_distance",
    "chamfer_l2",
    "depth_to_world_points",
    "icp_point_to_point",
    "object_local_to_world",
    "read_ply_xyz",
    "read_ply_xyz_rgb",
    "voxel_downsample",
    "world_to_object_local",
    "write_ply_xyz",
]
