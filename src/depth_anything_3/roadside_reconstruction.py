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
Roadside Multi-Camera Reconstruction Pipeline.

This module provides a specialized pipeline for reconstructing roadside scenes
(e.g., traffic intersections) using multiple fixed cameras with LiDAR-guided
metric depth alignment.

Key Features:
- Multi-camera (typically 4 cameras at intersection corners) reconstruction
- LiDAR sparse depth alignment for accurate metric scale
- Point cloud fusion with confidence-based filtering
- Support for large scenes (200m x 200m+)

Usage:
    from depth_anything_3.roadside_reconstruction import RoadsideReconstructor

    reconstructor = RoadsideReconstructor(model_name="da3-giant")
    result = reconstructor.reconstruct(
        images=[img1, img2, img3, img4],
        intrinsics=intrinsics_array,
        extrinsics=extrinsics_array,
        lidar_points=[lidar1, lidar2, lidar3, lidar4],
        lidar_to_camera_mapping=[0, 1, 2, 3],  # which lidar for which camera
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import unproject_depth
from depth_anything_3.utils.lidar_alignment import (
    LiDARAlignmentResult,
    align_depth_with_lidar,
    create_sparse_depth_map,
)
from depth_anything_3.utils.logger import logger


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    camera_id: str
    intrinsics: np.ndarray  # (3, 3)
    extrinsics: np.ndarray  # (4, 4) world-to-camera
    image_hw: Tuple[int, int]  # (H, W)
    lidar_id: Optional[str] = None  # Associated LiDAR sensor


@dataclass
class ReconstructionResult:
    """Result of roadside reconstruction."""

    # Per-camera results
    depth_maps: Dict[str, np.ndarray]  # camera_id -> (H, W) metric depth
    confidence_maps: Dict[str, np.ndarray]  # camera_id -> (H, W) confidence
    point_clouds: Dict[str, np.ndarray]  # camera_id -> (N, 3) world points
    colors: Dict[str, np.ndarray]  # camera_id -> (N, 3) RGB colors

    # LiDAR alignment results
    alignment_results: Dict[str, LiDARAlignmentResult]

    # Fused result
    fused_points: np.ndarray  # (M, 3) world coordinates
    fused_colors: np.ndarray  # (M, 3) RGB colors
    fused_confidence: np.ndarray  # (M,) confidence values

    # Metadata
    scale_factors: Dict[str, float]  # camera_id -> scale factor used
    num_points_per_camera: Dict[str, int]


class RoadsideReconstructor:
    """
    Roadside scene reconstructor using DA3 with LiDAR alignment.

    This class handles multi-camera reconstruction for roadside scenarios
    where monocular metric depth estimation is unreliable due to:
    - Non-standard viewpoints (elevated, downward-looking)
    - Large scene scale (200m+ intersections)
    - Limited number of views (typically 4 fixed cameras)

    The solution uses LiDAR sparse depth to provide accurate metric scale,
    while DA3 provides dense relative depth estimation.
    """

    def __init__(
        self,
        model_name: str = "da3-giant",  # Use base model, not nested
        device: str = "cuda",
        process_res: int = 518,
    ):
        """
        Initialize the roadside reconstructor.

        Args:
            model_name: DA3 model name. Recommended: "da3-giant" or "da3-large"
                        Do NOT use "da3nested-*" as the metric branch is trained
                        for front-view driving scenarios.
            device: Device to run inference on
            process_res: Processing resolution for DA3
        """
        self.model_name = model_name
        self.device = device
        self.process_res = process_res

        # Warn about nested models
        if "nested" in model_name.lower():
            logger.warning(
                f"Model '{model_name}' uses monocular metric depth which may not "
                "work well for roadside scenarios. Consider using 'da3-giant' instead "
                "and relying on LiDAR for metric scale."
            )

        # Map model name to HuggingFace repo
        model_repo_map = {
            "da3-giant": "depth-anything/DA3-GIANT",
            "da3-large": "depth-anything/DA3-LARGE",
            "da3-base": "depth-anything/DA3-BASE",
            "da3-small": "depth-anything/DA3-SMALL",
            "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
        }
        repo_id = model_repo_map.get(model_name.lower(), model_name)
        logger.info(f"Loading DA3 model from: {repo_id}")
        self.model = DepthAnything3.from_pretrained(repo_id)
        self.model.to(device)
        self.model.eval()

    def reconstruct(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        intrinsics: np.ndarray,  # (N, 3, 3)
        extrinsics: np.ndarray,  # (N, 4, 4) world-to-camera
        lidar_points: Optional[List[np.ndarray]] = None,  # List of (M, 3) arrays
        lidar_to_camera_mapping: Optional[List[int]] = None,  # LiDAR index for each camera
        conf_threshold_percentile: float = 30.0,
        use_ransac: bool = True,
        ransac_threshold: float = 0.15,
        export_dir: Optional[str] = None,
        max_points_per_camera: int = 500_000,
        max_view_angle: float = 70.0,  # Filter points outside this angle from optical axis
        max_depth: float = 150.0,  # Maximum depth to keep (meters)
    ) -> ReconstructionResult:
        """
        Reconstruct roadside scene from multi-camera images with LiDAR alignment.

        Args:
            images: List of N images (one per camera)
            intrinsics: Camera intrinsics (N, 3, 3)
            extrinsics: World-to-camera transforms (N, 4, 4)
            lidar_points: Optional list of LiDAR point clouds for each camera/LiDAR
            lidar_to_camera_mapping: Mapping from camera index to LiDAR index.
                                     If None, assumes 1:1 mapping.
            conf_threshold_percentile: Percentile for confidence thresholding
            use_ransac: Use RANSAC for robust LiDAR alignment
            ransac_threshold: RANSAC inlier threshold (relative to depth)
            export_dir: Optional directory to export results
            max_points_per_camera: Maximum points to keep per camera
            max_view_angle: Maximum angle (degrees) from camera optical axis.
                           Points outside this cone are filtered. This removes
                           noise from peripheral regions and opposite-side cameras.
                           Default: 70 degrees (keeps central ~140° FOV)
            max_depth: Maximum depth in meters. Points beyond this are filtered.
                       Default: 150m (appropriate for roadside scenarios)

        Returns:
            ReconstructionResult with depth maps, point clouds, and fusion
        """
        N = len(images)
        assert intrinsics.shape == (N, 3, 3), f"Expected intrinsics (N, 3, 3), got {intrinsics.shape}"
        assert extrinsics.shape == (N, 4, 4), f"Expected extrinsics (N, 4, 4), got {extrinsics.shape}"

        if lidar_to_camera_mapping is None and lidar_points is not None:
            lidar_to_camera_mapping = list(range(len(lidar_points)))

        logger.info(f"Reconstructing scene with {N} cameras")

        # Step 1: Run DA3 inference for each camera independently
        # (We process independently since cameras may have very different viewpoints)
        depth_maps = {}
        confidence_maps = {}
        processed_images = {}
        scaled_intrinsics = {}  # Store intrinsics scaled to processing resolution

        for cam_idx in range(N):
            cam_id = f"cam_{cam_idx}"
            logger.info(f"Processing camera {cam_idx + 1}/{N}")

            # Load image to get original size
            if isinstance(images[cam_idx], str):
                img = Image.open(images[cam_idx])
                orig_w, orig_h = img.size
            elif isinstance(images[cam_idx], np.ndarray):
                orig_h, orig_w = images[cam_idx].shape[:2]
            else:
                orig_w, orig_h = images[cam_idx].size

            # Run DA3 inference
            prediction = self.model.inference(
                image=[images[cam_idx]],
                process_res=self.process_res,
            )

            depth_maps[cam_id] = prediction.depth[0]  # (H, W)
            confidence_maps[cam_id] = prediction.conf[0]  # (H, W)
            processed_images[cam_id] = prediction.processed_images[0]  # (H, W, 3)

            # Compute scaled intrinsics for processing resolution
            proc_h, proc_w = depth_maps[cam_id].shape[:2]
            scale_x = proc_w / orig_w
            scale_y = proc_h / orig_h
            K_scaled = intrinsics[cam_idx].copy()
            K_scaled[0, :] *= scale_x  # fx, cx
            K_scaled[1, :] *= scale_y  # fy, cy
            scaled_intrinsics[cam_id] = K_scaled
            logger.info(f"  Original: {orig_w}x{orig_h} -> Processing: {proc_w}x{proc_h}, scale: ({scale_x:.3f}, {scale_y:.3f})")

        # Step 2: LiDAR alignment for metric scale
        alignment_results = {}
        scale_factors = {}

        if lidar_points is not None:
            for cam_idx in range(N):
                cam_id = f"cam_{cam_idx}"
                lidar_idx = lidar_to_camera_mapping[cam_idx]

                if lidar_idx is None or lidar_idx >= len(lidar_points):
                    logger.warning(f"No LiDAR data for camera {cam_idx}, using scale=1.0")
                    scale_factors[cam_id] = 1.0
                    continue

                logger.info(f"Aligning camera {cam_idx} with LiDAR {lidar_idx}")

                try:
                    # Use scaled intrinsics that match the processing resolution
                    result = align_depth_with_lidar(
                        predicted_depth=depth_maps[cam_id],
                        lidar_points=lidar_points[lidar_idx],
                        intrinsics=scaled_intrinsics[cam_id],  # Use scaled intrinsics!
                        extrinsics=extrinsics[cam_idx],
                        confidence=confidence_maps[cam_id],
                        use_ransac=use_ransac,
                        ransac_threshold=ransac_threshold,
                    )

                    depth_maps[cam_id] = result.aligned_depth
                    alignment_results[cam_id] = result
                    scale_factors[cam_id] = result.scale_factor

                    logger.info(
                        f"  Scale: {result.scale_factor:.4f}, "
                        f"Valid points: {result.num_valid_points}, "
                        f"Inlier ratio: {result.inlier_ratio:.2%}, "
                        f"Residual: {result.residual_mean:.3f}m +/- {result.residual_std:.3f}m"
                    )

                except ValueError as e:
                    logger.warning(f"LiDAR alignment failed for camera {cam_idx}: {e}")
                    scale_factors[cam_id] = 1.0
        else:
            logger.warning("No LiDAR data provided, depth will be in relative scale")
            for cam_idx in range(N):
                scale_factors[f"cam_{cam_idx}"] = 1.0

        # Step 3: Generate point clouds
        point_clouds = {}
        colors = {}
        num_points_per_camera = {}

        for cam_idx in range(N):
            cam_id = f"cam_{cam_idx}"

            depth = depth_maps[cam_id]
            conf = confidence_maps[cam_id]
            rgb = processed_images[cam_id]

            H, W = depth.shape

            # Compute confidence threshold
            conf_thresh = np.percentile(conf[conf > 0], conf_threshold_percentile)

            # Create point cloud using scaled intrinsics
            # Filter by view angle to remove noise from peripheral/opposite-side regions
            points, point_colors = self._depth_to_pointcloud(
                depth=depth,
                confidence=conf,
                rgb=rgb,
                intrinsics=scaled_intrinsics[cam_id],  # Use scaled intrinsics!
                extrinsics=extrinsics[cam_idx],
                conf_threshold=conf_thresh,
                max_points=max_points_per_camera,
                max_depth=max_depth,
                max_view_angle=max_view_angle,
            )

            point_clouds[cam_id] = points
            colors[cam_id] = point_colors
            num_points_per_camera[cam_id] = len(points)

            logger.info(f"Camera {cam_idx}: {len(points):,} points")

        # Step 4: Fuse point clouds
        fused_points, fused_colors, fused_confidence = self._fuse_pointclouds(
            point_clouds, colors, confidence_maps, depth_maps, intrinsics, extrinsics
        )

        logger.info(f"Total fused points: {len(fused_points):,}")

        # Step 5: Export if requested
        if export_dir is not None:
            self._export_results(
                export_dir,
                depth_maps,
                confidence_maps,
                point_clouds,
                colors,
                fused_points,
                fused_colors,
                alignment_results,
            )

        return ReconstructionResult(
            depth_maps=depth_maps,
            confidence_maps=confidence_maps,
            point_clouds=point_clouds,
            colors=colors,
            alignment_results=alignment_results,
            fused_points=fused_points,
            fused_colors=fused_colors,
            fused_confidence=fused_confidence,
            scale_factors=scale_factors,
            num_points_per_camera=num_points_per_camera,
        )

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,  # (H, W)
        confidence: np.ndarray,  # (H, W)
        rgb: np.ndarray,  # (H, W, 3)
        intrinsics: np.ndarray,  # (3, 3)
        extrinsics: np.ndarray,  # (4, 4) w2c
        conf_threshold: float = 0.5,
        max_points: int = 500_000,
        max_depth: float = 250.0,
        max_view_angle: float = 70.0,  # Maximum angle from camera optical axis (degrees)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert depth map to world-space point cloud."""

        H, W = depth.shape

        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten().astype(np.float32)
        v = v.flatten().astype(np.float32)
        z = depth.flatten()
        conf = confidence.flatten()

        # Filter by confidence and valid depth
        valid = (conf >= conf_threshold) & (z > 0.1) & (z < max_depth) & np.isfinite(z)
        u, v, z, conf = u[valid], v[valid], z[valid], conf[valid]

        # Subsample if too many points
        if len(z) > max_points:
            # Prefer high-confidence points
            indices = np.argsort(-conf)[:max_points]
            u, v, z, conf = u[indices], v[indices], z[indices], conf[indices]

        # Unproject to camera space
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z

        # Filter by view angle: only keep points within max_view_angle of optical axis
        # This removes noise from peripheral regions and opposite-side cameras
        if max_view_angle < 90.0:
            # Compute angle from optical axis (z-axis in camera space)
            # angle = arctan(sqrt(x^2 + y^2) / z)
            lateral_dist = np.sqrt(x_cam**2 + y_cam**2)
            view_angle = np.degrees(np.arctan2(lateral_dist, z_cam))
            angle_valid = view_angle <= max_view_angle

            x_cam, y_cam, z_cam = x_cam[angle_valid], y_cam[angle_valid], z_cam[angle_valid]
            u, v, conf = u[angle_valid], v[angle_valid], conf[angle_valid]

        points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)  # (N, 4)

        # Transform to world space
        c2w = np.linalg.inv(extrinsics)
        points_world = (c2w @ points_cam.T).T[:, :3]  # (N, 3)

        # Get colors
        u_int, v_int = u.astype(int), v.astype(int)
        point_colors = rgb[v_int, u_int]  # (N, 3)

        return points_world, point_colors

    def _fuse_pointclouds(
        self,
        point_clouds: Dict[str, np.ndarray],
        colors: Dict[str, np.ndarray],
        confidence_maps: Dict[str, np.ndarray],
        depth_maps: Dict[str, np.ndarray],
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fuse point clouds from multiple cameras."""

        all_points = []
        all_colors = []
        all_conf = []

        for cam_id in point_clouds:
            points = point_clouds[cam_id]
            point_colors = colors[cam_id]

            # Get per-point confidence (approximate from depth map)
            cam_idx = int(cam_id.split('_')[1])
            conf_map = confidence_maps[cam_id]

            # Simple confidence assignment (could be improved with reprojection)
            n_points = len(points)
            conf_values = np.ones(n_points) * np.median(conf_map[conf_map > 0])

            all_points.append(points)
            all_colors.append(point_colors)
            all_conf.append(conf_values)

        fused_points = np.concatenate(all_points, axis=0)
        fused_colors = np.concatenate(all_colors, axis=0)
        fused_confidence = np.concatenate(all_conf, axis=0)

        return fused_points, fused_colors, fused_confidence

    def _export_results(
        self,
        export_dir: str,
        depth_maps: Dict[str, np.ndarray],
        confidence_maps: Dict[str, np.ndarray],
        point_clouds: Dict[str, np.ndarray],
        colors: Dict[str, np.ndarray],
        fused_points: np.ndarray,
        fused_colors: np.ndarray,
        alignment_results: Dict[str, LiDARAlignmentResult],
    ):
        """Export reconstruction results."""
        os.makedirs(export_dir, exist_ok=True)

        # Export per-camera results
        for cam_id in depth_maps:
            cam_dir = os.path.join(export_dir, cam_id)
            os.makedirs(cam_dir, exist_ok=True)

            # Save depth and confidence as npz
            np.savez(
                os.path.join(cam_dir, "depth_conf.npz"),
                depth=depth_maps[cam_id],
                confidence=confidence_maps[cam_id],
            )

            # Save point cloud as PLY
            self._save_ply(
                os.path.join(cam_dir, "pointcloud.ply"),
                point_clouds[cam_id],
                colors[cam_id],
            )

        # Save fused point cloud
        self._save_ply(
            os.path.join(export_dir, "fused_pointcloud.ply"),
            fused_points,
            fused_colors,
        )

        # Save alignment statistics
        stats_path = os.path.join(export_dir, "alignment_stats.txt")
        with open(stats_path, "w") as f:
            f.write("LiDAR Alignment Statistics\n")
            f.write("=" * 50 + "\n\n")
            for cam_id, result in alignment_results.items():
                f.write(f"{cam_id}:\n")
                f.write(f"  Scale factor: {result.scale_factor:.6f}\n")
                f.write(f"  Valid points: {result.num_valid_points}\n")
                f.write(f"  Inlier ratio: {result.inlier_ratio:.2%}\n")
                f.write(f"  Mean residual: {result.residual_mean:.4f}m\n")
                f.write(f"  Std residual: {result.residual_std:.4f}m\n\n")

        logger.info(f"Results exported to {export_dir}")

    def _save_ply(
        self,
        filepath: str,
        points: np.ndarray,
        colors: np.ndarray,
    ):
        """Save point cloud as PLY file."""
        with open(filepath, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i].astype(int)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def reconstruct_intersection(
    images: List[Union[np.ndarray, str]],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    lidar_points: Optional[List[np.ndarray]] = None,
    model_name: str = "da3-giant",
    export_dir: Optional[str] = None,
    **kwargs,
) -> ReconstructionResult:
    """
    Convenience function for intersection reconstruction.

    Args:
        images: List of 4 images from intersection cameras
        intrinsics: (4, 3, 3) camera intrinsics
        extrinsics: (4, 4, 4) world-to-camera transforms
        lidar_points: Optional list of 4 LiDAR point clouds
        model_name: DA3 model to use
        export_dir: Optional export directory
        **kwargs: Additional arguments for RoadsideReconstructor.reconstruct()

    Returns:
        ReconstructionResult
    """
    reconstructor = RoadsideReconstructor(model_name=model_name)
    return reconstructor.reconstruct(
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        lidar_points=lidar_points,
        export_dir=export_dir,
        **kwargs,
    )
