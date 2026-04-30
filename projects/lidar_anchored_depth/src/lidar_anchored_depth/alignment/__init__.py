"""Depth-alignment algorithms.

Modules
-------
projection
    LiDAR ↔ image projection utilities (NumPy core).
height_anchor
    HAD: per-instance height-anchored depth solve (M1, mask-derived).
bbox_anchor
    AA-HAD: V2X bbox-anchored solve with motion compensation (M2/M3).
ground_plane
    RANSAC ground-plane fitting and per-pixel ground-depth from plane
    intersection (B5).
global_scale
    Global median / LSQ / RANSAC affine baselines (B0, B1, B2, B3).
region_affine
    Per-mask affine baselines on (d̃, z_lidar) pairs (B4).
adaptive
    Density-aware fusion at instance / ground seams (M4 component).
"""

from lidar_anchored_depth.alignment.bbox_anchor import (
    BboxMatch,
    aa_had,
    match_bboxes_to_masks,
    points_in_oriented_bbox,
)
from lidar_anchored_depth.alignment.global_scale import (
    AffineFit,
    b0_identity,
    b1_median_scale,
    b2_lsq_affine,
    b3_ransac_affine,
)
from lidar_anchored_depth.alignment.ground_plane import (
    GroundPlane,
    fit_ground_plane_ransac,
    ground_depth_map,
    ray_plane_intersection,
)
from lidar_anchored_depth.alignment.height_anchor import (
    HADResult,
    HeightInterval,
    InstanceAnchor,
    had_mask,
    height_interval_from_lidar_in_mask,
    mask_top_bottom_pixels,
    solve_affine_dense_lsq,
    solve_affine_dense_lsq_multi,
    solve_affine_dense_lsq_multi_with_lidar,
    solve_affine_from_height_anchors,
)
from lidar_anchored_depth.alignment.ray_obb import ray_obb_z_target
from lidar_anchored_depth.alignment.projection import (
    BBOX_EDGES,
    bbox_3d_corners,
    bbox_corners_from_object,
    bbox_uv_aabb,
    camera_to_world,
    invert_T,
    pixel_ray_to_world,
    pixel_to_camera_ray,
    project_3d_bbox,
    project_camera_to_image,
    world_to_camera,
    world_to_image,
    world_z_at_pixel_unit_depth,
)
from lidar_anchored_depth.alignment.region_affine import (
    RegionAffineResult,
    apply_region_fits,
    region_affine_align,
)

__all__ = [
    # projection
    "BBOX_EDGES",
    "bbox_3d_corners",
    "bbox_corners_from_object",
    "bbox_uv_aabb",
    "camera_to_world",
    "invert_T",
    "pixel_ray_to_world",
    "pixel_to_camera_ray",
    "project_3d_bbox",
    "project_camera_to_image",
    "world_to_camera",
    "world_to_image",
    "world_z_at_pixel_unit_depth",
    # global scale (B0–B3)
    "AffineFit",
    "b0_identity",
    "b1_median_scale",
    "b2_lsq_affine",
    "b3_ransac_affine",
    # ground plane (B5)
    "GroundPlane",
    "fit_ground_plane_ransac",
    "ground_depth_map",
    "ray_plane_intersection",
    # region affine (B4)
    "RegionAffineResult",
    "apply_region_fits",
    "region_affine_align",
    # height anchor (M1) + shared solver
    "HADResult",
    "HeightInterval",
    "InstanceAnchor",
    "had_mask",
    "height_interval_from_lidar_in_mask",
    "mask_top_bottom_pixels",
    "ray_obb_z_target",
    "solve_affine_dense_lsq",
    "solve_affine_dense_lsq_multi",
    "solve_affine_dense_lsq_multi_with_lidar",
    "solve_affine_from_height_anchors",
    # bbox anchor (M2/M3 AA-HAD)
    "BboxMatch",
    "aa_had",
    "match_bboxes_to_masks",
    "points_in_oriented_bbox",
]
