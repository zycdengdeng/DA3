# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Semantic-Guided Depth Completion Module.

Uses 3D bounding box annotations to separate dynamic objects from static background,
then performs depth completion independently for each region to avoid depth bleeding
across object boundaries.

Key idea:
    1. 3D bbox → filter LiDAR points belonging to each object
    2. 3D bbox → project to 2D convex hull mask
    3. Complete depth separately for:
       - Each dynamic object (using only its own LiDAR points)
       - Static background (using all non-object LiDAR points)
    4. Composite final depth map
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

# Import SAM segmentation module (optional)
try:
    from depth_anything_3.utils.sam_segmentation import (
        SAMGuidedMaskGenerator,
        get_available_sam_backend,
    )
    SAM_INTEGRATION_AVAILABLE = True
except ImportError:
    SAM_INTEGRATION_AVAILABLE = False


# Dynamic object categories (objects that move)
DYNAMIC_CATEGORIES = {
    "Car", "Suv", "Truck", "Bus", "Huge_vehicle", "Vehicle_else",
    "Pedestrian", "Pedestrian_else",
    "Non_motor_rider", "Motor_rider", "Other_rider",
    "Tricycle", "Motorcycle", "Bicycle",
    "Animal_small",
}

# Static object categories (don't move but still need separate handling)
STATIC_OBJECT_CATEGORIES = {
    "Bollards", "Crash_bucket", "Cone", "Vehicle_door",
}

# All annotated categories
ALL_OBJECT_CATEGORIES = DYNAMIC_CATEGORIES | STATIC_OBJECT_CATEGORIES


@dataclass
class BBox3D:
    """3D Bounding Box representation."""
    id: int
    label: str
    center: np.ndarray  # (3,) - x, y, z in world coordinates
    size: np.ndarray    # (3,) - length, width, height
    rotation: np.ndarray  # (3,) - roll, pitch, yaw in radians
    occlusion: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "BBox3D":
        """Create BBox3D from annotation dict."""
        return cls(
            id=d["id"],
            label=d["label"],
            center=np.array([d["x"], d["y"], d["z"]]),
            size=np.array([d["length"], d["width"], d["height"]]),
            rotation=np.array([d["roll"], d["pitch"], d["yaw"]]),
            occlusion=d.get("occlusion", 0),
        )

    def get_corners(self) -> np.ndarray:
        """
        Get 8 corner points of the 3D bounding box in world coordinates.

        Returns:
            corners: (8, 3) array of corner points
        """
        l, w, h = self.size

        # 8 corners in local coordinate (centered at origin)
        # Order: bottom 4, then top 4 (counterclockwise from top view)
        corners_local = np.array([
            [-l/2, -w/2, -h/2],  # 0: back-left-bottom
            [ l/2, -w/2, -h/2],  # 1: front-left-bottom
            [ l/2,  w/2, -h/2],  # 2: front-right-bottom
            [-l/2,  w/2, -h/2],  # 3: back-right-bottom
            [-l/2, -w/2,  h/2],  # 4: back-left-top
            [ l/2, -w/2,  h/2],  # 5: front-left-top
            [ l/2,  w/2,  h/2],  # 6: front-right-top
            [-l/2,  w/2,  h/2],  # 7: back-right-top
        ])

        # Rotation matrix from roll, pitch, yaw
        R = self._euler_to_rotation_matrix(self.rotation)

        # Transform to world coordinates
        corners_world = (R @ corners_local.T).T + self.center

        return corners_world

    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert roll, pitch, yaw to rotation matrix."""
        roll, pitch, yaw = euler

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation: Rz * Ry * Rx (yaw-pitch-roll order)
        return Rz @ Ry @ Rx

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """
        Check which points are inside this 3D bounding box.

        Args:
            points: (N, 3) array of points in world coordinates

        Returns:
            mask: (N,) boolean array, True if point is inside
        """
        # Transform points to local coordinate system
        R = self._euler_to_rotation_matrix(self.rotation)
        points_local = (R.T @ (points - self.center).T).T

        # Check bounds
        l, w, h = self.size
        inside = (
            (np.abs(points_local[:, 0]) <= l/2) &
            (np.abs(points_local[:, 1]) <= w/2) &
            (np.abs(points_local[:, 2]) <= h/2)
        )

        return inside


def load_annotations(json_path: Union[str, Path]) -> List[BBox3D]:
    """
    Load 3D bounding box annotations from JSON file.

    Args:
        json_path: Path to annotation JSON file

    Returns:
        List of BBox3D objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    bboxes = []
    for obj in data.get("object", []):
        try:
            bbox = BBox3D.from_dict(obj)
            bboxes.append(bbox)
        except (KeyError, ValueError) as e:
            print(f"Warning: Failed to parse object {obj.get('id', '?')}: {e}")

    return bboxes


def filter_lidar_by_bbox(
    lidar_points: np.ndarray,
    bbox: BBox3D,
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter LiDAR points that belong to a 3D bounding box.

    Args:
        lidar_points: (N, 3) or (N, 4) LiDAR points
        bbox: 3D bounding box
        margin: Extra margin around bbox (meters)

    Returns:
        inside_points: Points inside the bbox
        inside_mask: Boolean mask
    """
    points_xyz = lidar_points[:, :3]

    if margin > 0:
        # Create expanded bbox
        expanded_bbox = BBox3D(
            id=bbox.id,
            label=bbox.label,
            center=bbox.center.copy(),
            size=bbox.size + 2 * margin,
            rotation=bbox.rotation.copy(),
        )
        inside_mask = expanded_bbox.contains_points(points_xyz)
    else:
        inside_mask = bbox.contains_points(points_xyz)

    return lidar_points[inside_mask], inside_mask


def separate_dynamic_static_points(
    lidar_points: np.ndarray,
    bboxes: List[BBox3D],
    dynamic_only: bool = True,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Separate LiDAR points into dynamic objects and static background.

    Args:
        lidar_points: (N, 3) or (N, 4) LiDAR points
        bboxes: List of 3D bounding boxes
        dynamic_only: If True, only consider DYNAMIC_CATEGORIES as objects

    Returns:
        object_points: Dict of {bbox_id: points}
        static_points: Points not in any bbox
        static_mask: Boolean mask for static points
    """
    N = len(lidar_points)
    object_mask = np.zeros(N, dtype=bool)
    object_points = {}

    categories = DYNAMIC_CATEGORIES if dynamic_only else ALL_OBJECT_CATEGORIES

    for bbox in bboxes:
        if bbox.label not in categories:
            continue

        _, inside_mask = filter_lidar_by_bbox(lidar_points, bbox)
        object_points[bbox.id] = lidar_points[inside_mask]
        object_mask |= inside_mask

    static_mask = ~object_mask
    static_points = lidar_points[static_mask]

    return object_points, static_points, static_mask


def project_bbox_to_2d(
    bbox: BBox3D,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Project 3D bounding box corners to 2D image and create convex hull mask.

    Args:
        bbox: 3D bounding box
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transform
        image_hw: (H, W) image size

    Returns:
        mask: (H, W) boolean mask of the projected bbox region, or None if not visible
    """
    H, W = image_hw

    # Get 8 corners in world coordinates
    corners_world = bbox.get_corners()  # (8, 3)

    # Transform to camera coordinates
    corners_homo = np.hstack([corners_world, np.ones((8, 1))])  # (8, 4)
    corners_cam = (extrinsics @ corners_homo.T).T[:, :3]  # (8, 3)

    # Filter corners behind camera
    valid_depth = corners_cam[:, 2] > 0.1
    if not np.any(valid_depth):
        return None

    # Project to image
    corners_img = (intrinsics @ corners_cam.T).T  # (8, 3)
    corners_2d = corners_img[:, :2] / corners_img[:, 2:3]  # (8, 2)

    # Clip to image bounds (with margin)
    corners_2d[:, 0] = np.clip(corners_2d[:, 0], -W, 2*W)
    corners_2d[:, 1] = np.clip(corners_2d[:, 1], -H, 2*H)

    # Only use corners in front of camera for hull
    corners_2d_valid = corners_2d[valid_depth]

    if len(corners_2d_valid) < 3:
        return None

    try:
        # Compute 2D convex hull
        hull = ConvexHull(corners_2d_valid)
        hull_points = corners_2d_valid[hull.vertices].astype(np.int32)

        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_points, 1)

        return mask.astype(bool)
    except Exception:
        # Degenerate case (collinear points, etc.)
        return None


def interpolate_depth_region(
    sparse_depth: np.ndarray,
    mask: np.ndarray,
    region_mask: Optional[np.ndarray] = None,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate sparse depth within a region.

    Args:
        sparse_depth: (H, W) sparse depth map
        mask: (H, W) boolean mask of valid sparse depth
        region_mask: (H, W) boolean mask of region to fill (None = entire image)
        method: Interpolation method ("linear" or "nearest")

    Returns:
        dense_depth: (H, W) interpolated depth
    """
    H, W = sparse_depth.shape

    # Get valid points within region
    if region_mask is not None:
        valid_mask = mask & region_mask
    else:
        valid_mask = mask

    if valid_mask.sum() < 3:
        # Not enough points for interpolation
        return np.zeros((H, W), dtype=np.float32)

    # Get coordinates and values
    v_valid, u_valid = np.where(valid_mask)
    depths = sparse_depth[valid_mask]

    # Target region
    if region_mask is not None:
        v_target, u_target = np.where(region_mask)
    else:
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        u_target, v_target = u_grid.ravel(), v_grid.ravel()

    target_points = np.stack([u_target, v_target], axis=1)
    valid_coords = np.stack([u_valid, v_valid], axis=1)

    # Interpolate
    interpolated = griddata(
        valid_coords, depths, target_points,
        method=method, fill_value=0
    )

    # Create output
    dense_depth = np.zeros((H, W), dtype=np.float32)
    dense_depth[v_target, u_target] = interpolated

    return dense_depth


class SemanticDepthCompleter:
    """
    Semantic-guided depth completion using 3D bbox annotations.

    Separates dynamic objects from static background and completes
    depth independently for each region.

    Supports SAM (Segment Anything Model) for precise object masks,
    falling back to convex hull projection if SAM is unavailable.
    """

    def __init__(
        self,
        max_depth: float = 300.0,
        min_object_points: int = 5,
        use_da3_fallback: bool = True,
        bilateral_filter: bool = True,
        use_sam: bool = True,
        sam_backend: str = "auto",
        sam_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize semantic depth completer.

        Args:
            max_depth: Maximum valid depth (meters)
            min_object_points: Minimum LiDAR points required per object
            use_da3_fallback: Use DA3 depth for objects with few LiDAR points
            bilateral_filter: Apply bilateral filter for smoothing
            use_sam: Use SAM for precise object masks (recommended)
            sam_backend: SAM backend ("auto", "sam", "sam2", "mobile_sam")
            sam_model_path: Path to SAM model checkpoint
            device: Device for SAM ("cuda" or "cpu")
        """
        self.max_depth = max_depth
        self.min_object_points = min_object_points
        self.use_da3_fallback = use_da3_fallback
        self.bilateral_filter = bilateral_filter

        # Initialize SAM mask generator
        self.sam_generator = None
        self.use_sam = use_sam

        if use_sam and SAM_INTEGRATION_AVAILABLE:
            backend = get_available_sam_backend()
            if backend is not None:
                print(f"  Initializing SAM ({backend}) for precise object masks...")
                self.sam_generator = SAMGuidedMaskGenerator(
                    sam_backend=sam_backend,
                    sam_model_path=sam_model_path,
                    device=device,
                    use_point_prompts=True,
                )
                if self.sam_generator.sam_available:
                    print(f"  SAM ready: precise object segmentation enabled")
                else:
                    print(f"  SAM not available, using convex hull fallback")
                    self.sam_generator = None
            else:
                print("  SAM not installed, using convex hull for object masks")
                print("  Install SAM for better results: pip install sam2")
        elif use_sam and not SAM_INTEGRATION_AVAILABLE:
            print("  SAM integration not available, using convex hull")

    def complete(
        self,
        rgb: np.ndarray,
        lidar_points: np.ndarray,
        bboxes: List[BBox3D],
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        da3_depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete depth using semantic guidance.

        Args:
            rgb: (H, W, 3) RGB image
            lidar_points: (N, 3) or (N, 4) LiDAR points in world coordinates
            bboxes: List of 3D bounding boxes
            intrinsics: (3, 3) camera intrinsics
            extrinsics: (4, 4) world-to-camera transform
            da3_depth: (H, W) optional DA3 relative depth for fallback

        Returns:
            dense_depth: (H, W) completed depth map
            info: Dict with debug info
        """
        H, W = rgb.shape[:2]

        # Step 1: Separate dynamic/static LiDAR points
        object_points, static_points, _ = separate_dynamic_static_points(
            lidar_points, bboxes, dynamic_only=True
        )

        print(f"  Separated: {len(object_points)} objects, {len(static_points)} static points")

        # Step 2: Create sparse depth maps
        # Static background sparse depth
        static_sparse, static_mask = self._project_points_to_depth(
            static_points, intrinsics, extrinsics, (H, W)
        )

        # Step 3: Complete static background first
        dense_depth = self._interpolate_with_filter(
            static_sparse, static_mask, rgb
        )

        # Step 4: Set up SAM if available
        if self.sam_generator is not None and self.sam_generator.sam_available:
            self.sam_generator.set_image(rgb)
            using_sam = True
            print("  Using SAM for precise object masks")
        else:
            using_sam = False
            print("  Using convex hull for object masks")

        # Step 5: Process each dynamic object
        object_info = {}
        for bbox in bboxes:
            if bbox.label not in DYNAMIC_CATEGORIES:
                continue

            # Get LiDAR points for this object (needed for SAM prompts)
            obj_points = object_points.get(bbox.id, np.array([]))

            # Get 2D projected points for SAM prompts
            obj_points_2d = None
            if len(obj_points) > 0:
                obj_sparse_temp, obj_valid_temp = self._project_points_to_depth(
                    obj_points, intrinsics, extrinsics, (H, W)
                )
                if obj_valid_temp.sum() > 0:
                    v_coords, u_coords = np.where(obj_valid_temp)
                    obj_points_2d = np.column_stack([u_coords, v_coords])

            # Get 2D mask for this object
            if using_sam:
                # Use SAM with 3D bbox corners as guidance
                corners_3d = bbox.get_corners()
                obj_mask_2d = self.sam_generator.generate_mask(
                    bbox_3d_corners=corners_3d,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_hw=(H, W),
                    lidar_points_2d=obj_points_2d,
                )
            else:
                # Fallback: convex hull projection
                obj_mask_2d = project_bbox_to_2d(
                    bbox, intrinsics, extrinsics, (H, W)
                )

            if obj_mask_2d is None or obj_mask_2d.sum() < 10:
                continue

            if len(obj_points) >= self.min_object_points:
                # Enough LiDAR points: interpolate within object region
                obj_sparse, obj_valid = self._project_points_to_depth(
                    obj_points, intrinsics, extrinsics, (H, W)
                )

                # Interpolate only within object mask
                obj_depth = interpolate_depth_region(
                    obj_sparse, obj_valid, obj_mask_2d, method="linear"
                )

                # Fill holes with nearest
                holes = obj_mask_2d & (obj_depth <= 0)
                if holes.any() and obj_valid.sum() > 0:
                    obj_depth_nearest = interpolate_depth_region(
                        obj_sparse, obj_valid, obj_mask_2d, method="nearest"
                    )
                    obj_depth[holes] = obj_depth_nearest[holes]

                object_info[bbox.id] = {
                    "label": bbox.label,
                    "n_points": len(obj_points),
                    "depth_method": "lidar_interp",
                    "mask_method": "sam" if using_sam else "convex_hull",
                    "mask_pixels": int(obj_mask_2d.sum()),
                }

            elif da3_depth is not None and self.use_da3_fallback:
                # Few LiDAR points: use DA3 with scale alignment
                obj_depth = self._da3_fallback(
                    da3_depth, obj_points, obj_mask_2d,
                    intrinsics, extrinsics, (H, W)
                )

                object_info[bbox.id] = {
                    "label": bbox.label,
                    "n_points": len(obj_points),
                    "depth_method": "da3_fallback",
                    "mask_method": "sam" if using_sam else "convex_hull",
                    "mask_pixels": int(obj_mask_2d.sum()),
                }
            else:
                # No points and no DA3: skip
                continue

            # Composite: object overwrites background
            valid_obj = obj_mask_2d & (obj_depth > 0)
            dense_depth[valid_obj] = obj_depth[valid_obj]

        # Step 6: Clamp to valid range
        dense_depth = np.clip(dense_depth, 0, self.max_depth)

        info = {
            "n_objects": len(object_info),
            "n_static_points": len(static_points),
            "mask_method": "sam" if using_sam else "convex_hull",
            "objects": object_info,
        }

        return dense_depth.astype(np.float32), info

    def _project_points_to_depth(
        self,
        points: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project LiDAR points to sparse depth map."""
        H, W = image_hw
        sparse_depth = np.zeros((H, W), dtype=np.float32)
        valid_mask = np.zeros((H, W), dtype=bool)

        if len(points) == 0:
            return sparse_depth, valid_mask

        # Get XYZ
        points_xyz = points[:, :3]

        # Transform to camera coordinates
        points_homo = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])
        points_cam = (extrinsics @ points_homo.T).T[:, :3]

        # Filter points behind camera
        depths = points_cam[:, 2]
        valid_depth = (depths > 0.1) & (depths < self.max_depth)

        # Project to image
        points_img = (intrinsics @ points_cam.T).T
        points_2d = points_img[:, :2] / points_img[:, 2:3]

        u = points_2d[:, 0].astype(int)
        v = points_2d[:, 1].astype(int)

        # Filter points inside image
        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = valid_depth & valid_uv

        # Write to depth map (use closest point per pixel)
        for i in np.where(valid)[0]:
            ui, vi, di = u[i], v[i], depths[i]
            if sparse_depth[vi, ui] == 0 or di < sparse_depth[vi, ui]:
                sparse_depth[vi, ui] = di
                valid_mask[vi, ui] = True

        return sparse_depth, valid_mask

    def _interpolate_with_filter(
        self,
        sparse_depth: np.ndarray,
        mask: np.ndarray,
        rgb: np.ndarray,
    ) -> np.ndarray:
        """Interpolate and optionally apply bilateral filter."""
        H, W = sparse_depth.shape

        if mask.sum() < 10:
            return np.zeros((H, W), dtype=np.float32)

        # Linear interpolation
        v_valid, u_valid = np.where(mask)
        depths = sparse_depth[mask]

        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        grid_points = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        valid_coords = np.stack([u_valid, v_valid], axis=1)

        dense = griddata(
            valid_coords, depths, grid_points,
            method='linear', fill_value=0
        ).reshape(H, W)

        # Fill holes with nearest
        holes = dense <= 0
        if holes.any():
            dense_nearest = griddata(
                valid_coords, depths, grid_points,
                method='nearest'
            ).reshape(H, W)
            dense[holes] = dense_nearest[holes]

        # Bilateral filter
        if self.bilateral_filter:
            dense = dense.astype(np.float32)
            depth_norm = dense / (self.max_depth + 1e-6)
            depth_norm = np.clip(depth_norm, 0, 1).astype(np.float32)

            depth_filtered = cv2.bilateralFilter(
                depth_norm, d=9, sigmaColor=0.1, sigmaSpace=75
            )
            dense = depth_filtered * self.max_depth

            # Preserve original sparse values
            dense[mask] = sparse_depth[mask]

        return dense.astype(np.float32)

    def _da3_fallback(
        self,
        da3_depth: np.ndarray,
        obj_points: np.ndarray,
        obj_mask: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> np.ndarray:
        """Use DA3 depth with scale alignment for objects with few LiDAR points."""
        H, W = image_hw

        if len(obj_points) == 0:
            # No points at all: use DA3 directly (relative depth)
            # This is risky but better than nothing
            obj_depth = da3_depth.copy()
            obj_depth[~obj_mask] = 0
            return obj_depth

        # Project object points
        obj_sparse, obj_valid = self._project_points_to_depth(
            obj_points, intrinsics, extrinsics, image_hw
        )

        # Get DA3 values at LiDAR locations
        valid_in_mask = obj_valid & obj_mask
        if valid_in_mask.sum() < 1:
            return da3_depth * obj_mask

        lidar_depths = obj_sparse[valid_in_mask]
        da3_values = da3_depth[valid_in_mask]

        # Compute scale (least squares: lidar = scale * da3)
        da3_values = np.maximum(da3_values, 1e-6)  # Avoid division by zero
        scale = np.median(lidar_depths / da3_values)

        # Apply scale to DA3 depth within object mask
        obj_depth = da3_depth * scale
        obj_depth[~obj_mask] = 0

        # Preserve original LiDAR values
        obj_depth[obj_valid] = obj_sparse[obj_valid]

        return obj_depth.astype(np.float32)


def complete_depth_semantic(
    rgb: np.ndarray,
    lidar_points: np.ndarray,
    annotation_path: Union[str, Path],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    da3_depth: Optional[np.ndarray] = None,
    max_depth: float = 300.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for semantic-guided depth completion.

    Args:
        rgb: (H, W, 3) RGB image
        lidar_points: (N, 3) or (N, 4) LiDAR points
        annotation_path: Path to annotation JSON file
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transform
        da3_depth: Optional DA3 relative depth for fallback
        max_depth: Maximum valid depth

    Returns:
        dense_depth: (H, W) completed depth
        info: Dict with debug info
    """
    # Load annotations
    bboxes = load_annotations(annotation_path)
    print(f"  Loaded {len(bboxes)} bounding boxes")

    # Complete depth
    completer = SemanticDepthCompleter(
        max_depth=max_depth,
        use_da3_fallback=(da3_depth is not None),
    )

    return completer.complete(
        rgb, lidar_points, bboxes, intrinsics, extrinsics, da3_depth
    )
