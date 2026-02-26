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
LiDAR Sparse Depth Alignment Module for Roadside Reconstruction.

This module provides functions to align DA3 predicted depth maps with
sparse LiDAR point clouds, enabling metric depth recovery in roadside
scenarios where monocular metric depth estimation is unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class LiDARAlignmentResult:
    """Result of LiDAR-guided depth alignment."""

    aligned_depth: np.ndarray  # (H, W) metric depth
    scale_factor: float  # scale applied to relative depth
    num_valid_points: int  # number of LiDAR points used
    inlier_ratio: float  # ratio of inliers after RANSAC
    residual_mean: float  # mean absolute error on valid points
    residual_std: float  # std of absolute error


def project_lidar_to_image(
    lidar_points: np.ndarray,  # (N, 3) or (N, 4) world coordinates
    intrinsics: np.ndarray,  # (3, 3) camera intrinsics
    extrinsics: np.ndarray,  # (4, 4) or (3, 4) world-to-camera
    image_hw: Tuple[int, int],  # (H, W)
    min_depth: float = 0.1,
    max_depth: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project LiDAR points to image plane.

    Args:
        lidar_points: LiDAR points in world coordinates (N, 3) or (N, 4) with intensity
        intrinsics: Camera intrinsic matrix (3, 3)
        extrinsics: World-to-camera transformation (4, 4) or (3, 4)
        image_hw: Image size (height, width)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        pixel_coords: Valid pixel coordinates (M, 2) as (u, v)
        depths: Corresponding depths (M,)
        valid_indices: Indices of valid points in original array (M,)
    """
    # Handle (N, 4) format with intensity
    if lidar_points.shape[1] == 4:
        lidar_points = lidar_points[:, :3]

    # Convert to homogeneous coordinates
    N = lidar_points.shape[0]
    points_homo = np.concatenate([lidar_points, np.ones((N, 1))], axis=1)  # (N, 4)

    # Ensure extrinsics is 4x4
    if extrinsics.shape == (3, 4):
        ext_4x4 = np.eye(4)
        ext_4x4[:3, :] = extrinsics
        extrinsics = ext_4x4

    # Transform to camera coordinates
    points_cam = (extrinsics @ points_homo.T).T  # (N, 4)
    points_cam = points_cam[:, :3]  # (N, 3)

    # Filter points behind camera
    depths = points_cam[:, 2]
    valid_depth = (depths > min_depth) & (depths < max_depth)

    # Project to image plane
    points_img = (intrinsics @ points_cam.T).T  # (N, 3)
    points_img = points_img[:, :2] / points_img[:, 2:3]  # (N, 2)

    # Filter points outside image
    H, W = image_hw
    u, v = points_img[:, 0], points_img[:, 1]
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # Combine validity masks
    valid = valid_depth & valid_uv
    valid_indices = np.where(valid)[0]

    return points_img[valid], depths[valid], valid_indices


def align_depth_with_lidar(
    predicted_depth: np.ndarray,  # (H, W) relative depth
    lidar_points: np.ndarray,  # (N, 3) world coordinates
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) or (3, 4) w2c
    confidence: Optional[np.ndarray] = None,  # (H, W) confidence map
    use_ransac: bool = True,
    ransac_threshold: float = 0.1,  # relative threshold
    ransac_iterations: int = 1000,
    min_valid_points: int = 50,
) -> LiDARAlignmentResult:
    """
    Align predicted depth with LiDAR sparse depth using least-squares with optional RANSAC.

    The alignment finds scale factor s such that: lidar_depth ≈ s * predicted_depth

    Args:
        predicted_depth: DA3 predicted relative depth (H, W)
        lidar_points: LiDAR point cloud in world coordinates (N, 3)
        intrinsics: Camera intrinsic matrix (3, 3)
        extrinsics: World-to-camera transformation (4, 4) w2c
        confidence: Optional confidence map from DA3 (H, W)
        use_ransac: Whether to use RANSAC for robust estimation
        ransac_threshold: Inlier threshold as fraction of depth
        ransac_iterations: Number of RANSAC iterations
        min_valid_points: Minimum number of valid correspondences required

    Returns:
        LiDARAlignmentResult containing aligned depth and statistics
    """
    H, W = predicted_depth.shape

    # Project LiDAR to image
    pixel_coords, lidar_depths, _ = project_lidar_to_image(
        lidar_points, intrinsics, extrinsics, (H, W)
    )

    if len(lidar_depths) < min_valid_points:
        raise ValueError(
            f"Only {len(lidar_depths)} valid LiDAR points projected to image, "
            f"need at least {min_valid_points}"
        )

    # Sample predicted depth at LiDAR projection locations
    u = pixel_coords[:, 0].astype(int)
    v = pixel_coords[:, 1].astype(int)
    pred_depths = predicted_depth[v, u]

    # Get confidence weights if available
    if confidence is not None:
        weights = confidence[v, u]
        weights = np.clip(weights, 0.1, None)  # Minimum weight
    else:
        weights = np.ones_like(pred_depths)

    # Filter out invalid predictions
    valid = (pred_depths > 1e-6) & (lidar_depths > 1e-6) & np.isfinite(pred_depths)
    pred_depths = pred_depths[valid]
    lidar_depths = lidar_depths[valid]
    weights = weights[valid]

    if len(pred_depths) < min_valid_points:
        raise ValueError(
            f"Only {len(pred_depths)} valid depth correspondences, "
            f"need at least {min_valid_points}"
        )

    if use_ransac:
        scale, inlier_mask = _ransac_scale_estimation(
            pred_depths, lidar_depths, weights,
            threshold=ransac_threshold,
            n_iterations=ransac_iterations
        )
        inlier_ratio = inlier_mask.sum() / len(inlier_mask)

        # Refine with inliers only
        scale = _weighted_least_squares_scale(
            pred_depths[inlier_mask],
            lidar_depths[inlier_mask],
            weights[inlier_mask]
        )
    else:
        scale = _weighted_least_squares_scale(pred_depths, lidar_depths, weights)
        inlier_ratio = 1.0

    # Apply scale to depth map
    aligned_depth = predicted_depth * scale

    # Compute residuals
    aligned_sampled = aligned_depth[v[valid], u[valid]]
    if use_ransac:
        residuals = np.abs(aligned_sampled[inlier_mask] - lidar_depths[inlier_mask])
    else:
        residuals = np.abs(aligned_sampled - lidar_depths)

    return LiDARAlignmentResult(
        aligned_depth=aligned_depth,
        scale_factor=scale,
        num_valid_points=len(pred_depths),
        inlier_ratio=inlier_ratio,
        residual_mean=float(np.mean(residuals)),
        residual_std=float(np.std(residuals))
    )


def _weighted_least_squares_scale(
    pred_depths: np.ndarray,
    target_depths: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Compute weighted least-squares scale factor.

    Minimizes: sum(w * (target - s * pred)^2)
    Solution: s = sum(w * target * pred) / sum(w * pred * pred)
    """
    numerator = np.sum(weights * target_depths * pred_depths)
    denominator = np.sum(weights * pred_depths * pred_depths)
    return numerator / max(denominator, 1e-12)


def _ransac_scale_estimation(
    pred_depths: np.ndarray,
    target_depths: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.1,
    n_iterations: int = 1000,
    min_samples: int = 3
) -> Tuple[float, np.ndarray]:
    """
    RANSAC-based robust scale estimation.

    Args:
        pred_depths: Predicted depth values
        target_depths: LiDAR depth values
        weights: Confidence weights
        threshold: Inlier threshold as fraction of target depth
        n_iterations: Number of RANSAC iterations
        min_samples: Minimum samples per iteration

    Returns:
        best_scale: Estimated scale factor
        best_inliers: Boolean mask of inliers
    """
    n_points = len(pred_depths)
    best_scale = 1.0
    best_inliers = np.zeros(n_points, dtype=bool)
    best_score = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        # Random sample
        indices = rng.choice(n_points, size=min(min_samples, n_points), replace=False)

        # Estimate scale from sample
        scale = _weighted_least_squares_scale(
            pred_depths[indices],
            target_depths[indices],
            weights[indices]
        )

        if scale <= 0:
            continue

        # Compute residuals
        aligned = pred_depths * scale
        relative_errors = np.abs(aligned - target_depths) / target_depths

        # Count inliers
        inliers = relative_errors < threshold
        score = np.sum(inliers * weights)

        if score > best_score:
            best_score = score
            best_scale = scale
            best_inliers = inliers

    return best_scale, best_inliers


def align_depth_with_lidar_torch(
    predicted_depth: torch.Tensor,  # (H, W) or (B, H, W)
    lidar_points: torch.Tensor,  # (N, 3)
    intrinsics: torch.Tensor,  # (3, 3) or (B, 3, 3)
    extrinsics: torch.Tensor,  # (4, 4) or (B, 4, 4) w2c
    confidence: Optional[torch.Tensor] = None,
    min_depth: float = 0.1,
    max_depth: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version of LiDAR depth alignment (differentiable).

    Args:
        predicted_depth: Predicted depth (H, W) or (B, H, W)
        lidar_points: LiDAR points in world coordinates (N, 3)
        intrinsics: Camera intrinsics (3, 3) or (B, 3, 3)
        extrinsics: World-to-camera transform (4, 4) or (B, 4, 4)
        confidence: Optional confidence map
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        aligned_depth: Scaled depth map
        scale_factor: Applied scale factor
    """
    device = predicted_depth.device
    dtype = predicted_depth.dtype

    # Handle batch dimension
    if predicted_depth.dim() == 2:
        predicted_depth = predicted_depth.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0) if intrinsics.dim() == 2 else intrinsics
        extrinsics = extrinsics.unsqueeze(0) if extrinsics.dim() == 2 else extrinsics
        squeeze_output = True
    else:
        squeeze_output = False

    B, H, W = predicted_depth.shape
    N = lidar_points.shape[0]

    # Transform LiDAR to camera coordinates
    points_homo = torch.cat([lidar_points, torch.ones(N, 1, device=device, dtype=dtype)], dim=1)

    # Ensure extrinsics is 4x4
    if extrinsics.shape[-2:] == (3, 4):
        ext_4x4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        ext_4x4[:, :3, :] = extrinsics
        extrinsics = ext_4x4

    # Project to camera space: (B, N, 3)
    points_cam = torch.einsum('bij,nj->bni', extrinsics, points_homo)[..., :3]

    # Get depths
    depths = points_cam[..., 2]  # (B, N)

    # Project to image plane
    points_img = torch.einsum('bij,bnj->bni', intrinsics, points_cam)  # (B, N, 3)
    points_2d = points_img[..., :2] / points_img[..., 2:3].clamp(min=1e-6)  # (B, N, 2)

    # Create validity mask
    valid = (
        (depths > min_depth) & (depths < max_depth) &
        (points_2d[..., 0] >= 0) & (points_2d[..., 0] < W) &
        (points_2d[..., 1] >= 0) & (points_2d[..., 1] < H)
    )  # (B, N)

    # Sample predicted depth at LiDAR locations using grid_sample
    # Normalize coordinates to [-1, 1]
    grid_x = 2.0 * points_2d[..., 0] / (W - 1) - 1.0
    grid_y = 2.0 * points_2d[..., 1] / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, N, 2)

    # grid_sample expects (B, C, H, W) input and (B, H_out, W_out, 2) grid
    pred_depth_4d = predicted_depth.unsqueeze(1)  # (B, 1, H, W)
    grid_4d = grid.unsqueeze(1)  # (B, 1, N, 2)

    sampled_pred = torch.nn.functional.grid_sample(
        pred_depth_4d, grid_4d, mode='bilinear', padding_mode='zeros', align_corners=True
    )  # (B, 1, 1, N)
    sampled_pred = sampled_pred.squeeze(1).squeeze(1)  # (B, N)

    # Get confidence weights
    if confidence is not None:
        if confidence.dim() == 2:
            confidence = confidence.unsqueeze(0)
        conf_4d = confidence.unsqueeze(1)
        weights = torch.nn.functional.grid_sample(
            conf_4d, grid_4d, mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(1).squeeze(1)
        weights = weights.clamp(min=0.1)
    else:
        weights = torch.ones_like(sampled_pred)

    # Mask invalid points
    valid = valid & (sampled_pred > 1e-6)
    weights = weights * valid.float()

    # Weighted least-squares scale estimation
    # scale = sum(w * lidar * pred) / sum(w * pred * pred)
    numerator = (weights * depths * sampled_pred).sum(dim=-1)
    denominator = (weights * sampled_pred * sampled_pred).sum(dim=-1).clamp(min=1e-12)
    scale_factor = numerator / denominator  # (B,)

    # Apply scale
    aligned_depth = predicted_depth * scale_factor.view(B, 1, 1)

    if squeeze_output:
        aligned_depth = aligned_depth.squeeze(0)
        scale_factor = scale_factor.squeeze(0)

    return aligned_depth, scale_factor


def create_sparse_depth_map(
    lidar_points: np.ndarray,  # (N, 3) world coordinates
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) w2c
    image_hw: Tuple[int, int],
    dilation_kernel: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a sparse depth map from LiDAR projection.

    Args:
        lidar_points: LiDAR points in world coordinates
        intrinsics: Camera intrinsics
        extrinsics: World-to-camera transformation
        image_hw: Image size (H, W)
        dilation_kernel: Optional dilation kernel size for sparse depth

    Returns:
        sparse_depth: Sparse depth map (H, W), 0 where no LiDAR
        valid_mask: Boolean mask of valid pixels (H, W)
    """
    H, W = image_hw
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    pixel_coords, depths, _ = project_lidar_to_image(
        lidar_points, intrinsics, extrinsics, image_hw
    )

    if len(depths) == 0:
        return sparse_depth, valid_mask

    u = pixel_coords[:, 0].astype(int)
    v = pixel_coords[:, 1].astype(int)

    # Handle multiple points projecting to same pixel (use closest)
    for i in range(len(depths)):
        if sparse_depth[v[i], u[i]] == 0 or depths[i] < sparse_depth[v[i], u[i]]:
            sparse_depth[v[i], u[i]] = depths[i]
            valid_mask[v[i], u[i]] = True

    # Optional dilation
    if dilation_kernel > 0:
        from scipy.ndimage import maximum_filter, binary_dilation
        struct = np.ones((dilation_kernel, dilation_kernel))
        valid_mask = binary_dilation(valid_mask, struct)
        sparse_depth = maximum_filter(sparse_depth, size=dilation_kernel) * valid_mask

    return sparse_depth, valid_mask


def fit_ground_plane_ransac(
    lidar_points: np.ndarray,  # (N, 3) world coordinates
    height_threshold: float = 0.5,  # Only consider points near ground
    n_iterations: int = 1000,
    distance_threshold: float = 0.1,  # Inlier distance threshold (meters)
    min_inlier_ratio: float = 0.3,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit ground plane from LiDAR points using RANSAC.

    Assumes ground is roughly horizontal (normal pointing up in Z).

    Args:
        lidar_points: LiDAR points in world coordinates (N, 3)
        height_threshold: Only use points within this height range for fitting
        n_iterations: Number of RANSAC iterations
        distance_threshold: Distance threshold for inliers (meters)
        min_inlier_ratio: Minimum ratio of inliers to accept fit

    Returns:
        plane_normal: (3,) unit normal vector of ground plane (pointing up)
        plane_d: plane equation: normal.dot(point) + d = 0
        inlier_mask: (N,) boolean mask of inlier points
    """
    # Filter points by height - assume ground is near the minimum Z
    z_values = lidar_points[:, 2]
    z_min = np.percentile(z_values, 5)  # 5th percentile as approximate ground
    z_max = z_min + height_threshold

    ground_candidates = (z_values >= z_min) & (z_values <= z_max)
    candidate_points = lidar_points[ground_candidates]

    if len(candidate_points) < 100:
        # Not enough points, return default horizontal plane
        print(f"  Warning: Only {len(candidate_points)} ground candidates, using default plane")
        return np.array([0, 0, 1]), -z_min, np.zeros(len(lidar_points), dtype=bool)

    print(f"  Ground candidates: {len(candidate_points)} points in Z=[{z_min:.2f}, {z_max:.2f}]")

    rng = np.random.default_rng(42)
    best_normal = np.array([0, 0, 1])
    best_d = -z_min
    best_inliers = np.zeros(len(candidate_points), dtype=bool)
    best_score = 0

    for _ in range(n_iterations):
        # Sample 3 random points
        indices = rng.choice(len(candidate_points), size=3, replace=False)
        p1, p2, p3 = candidate_points[indices]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # Ensure normal points up (positive Z)
        if normal[2] < 0:
            normal = -normal

        # Check if plane is roughly horizontal (normal mostly in Z direction)
        if abs(normal[2]) < 0.9:  # Allow ~25 degree tilt max
            continue

        # Compute d: normal.dot(p) + d = 0 => d = -normal.dot(p)
        d = -np.dot(normal, p1)

        # Count inliers
        distances = np.abs(np.dot(candidate_points, normal) + d)
        inliers = distances < distance_threshold
        score = np.sum(inliers)

        if score > best_score:
            best_score = score
            best_normal = normal
            best_d = d
            best_inliers = inliers

    inlier_ratio = best_score / len(candidate_points)
    print(f"  Ground plane: normal={best_normal}, d={best_d:.3f}")
    print(f"  Inliers: {best_score}/{len(candidate_points)} ({inlier_ratio:.1%})")

    # Convert inlier mask back to full point cloud
    full_inlier_mask = np.zeros(len(lidar_points), dtype=bool)
    full_inlier_mask[ground_candidates] = best_inliers

    return best_normal, best_d, full_inlier_mask


def compute_ground_depth_from_plane(
    plane_normal: np.ndarray,  # (3,) ground plane normal
    plane_d: float,  # plane equation constant
    intrinsics: np.ndarray,  # (3, 3) camera intrinsics
    extrinsics: np.ndarray,  # (4, 4) world-to-camera transform
    image_hw: Tuple[int, int],
    ground_mask: Optional[np.ndarray] = None,  # (H, W) optional mask of ground pixels
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute depth map for ground pixels by ray-plane intersection.

    For each pixel, shoot a ray from camera and find intersection with ground plane.

    Args:
        plane_normal: Ground plane normal (pointing up)
        plane_d: Plane equation constant (normal.dot(p) + d = 0)
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: World-to-camera transformation (4, 4)
        image_hw: Image size (H, W)
        ground_mask: Optional mask indicating ground pixels

    Returns:
        ground_depth: (H, W) depth at ground plane (0 where not ground or invalid)
        valid_mask: (H, W) boolean mask of valid ground depth pixels
    """
    H, W = image_hw

    # Camera position in world coordinates
    # extrinsics: p_cam = R @ p_world + t
    # So: p_world = R^T @ (p_cam - t) = R^T @ p_cam - R^T @ t
    # Camera center: p_cam = 0 => p_world = -R^T @ t
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    R_inv = R.T
    cam_center = -R_inv @ t  # Camera position in world coords

    # Intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Ray direction in camera coordinates (normalized)
    ray_cam_x = (u - cx) / fx
    ray_cam_y = (v - cy) / fy
    ray_cam_z = np.ones_like(u)

    # Stack and normalize
    ray_cam = np.stack([ray_cam_x, ray_cam_y, ray_cam_z], axis=-1)  # (H, W, 3)
    ray_cam = ray_cam / np.linalg.norm(ray_cam, axis=-1, keepdims=True)

    # Transform ray direction to world coordinates
    # ray_world = R^T @ ray_cam
    ray_world = np.einsum('ij,hwj->hwi', R_inv, ray_cam)  # (H, W, 3)

    # Ray-plane intersection
    # Ray: p = cam_center + t * ray_dir
    # Plane: normal.dot(p) + d = 0
    # => normal.dot(cam_center + t * ray_dir) + d = 0
    # => t = -(normal.dot(cam_center) + d) / normal.dot(ray_dir)

    numerator = -(np.dot(plane_normal, cam_center) + plane_d)
    denominator = np.einsum('i,hwi->hw', plane_normal, ray_world)  # (H, W)

    # Avoid division by zero (ray parallel to plane)
    valid = np.abs(denominator) > 1e-6

    # Compute intersection distance (in world coordinates along ray)
    t_intersect = np.zeros((H, W), dtype=np.float32)
    t_intersect[valid] = numerator / denominator[valid]

    # Convert to camera depth (Z in camera coordinates)
    # Intersection point in world: p_world = cam_center + t * ray_world
    # In camera coords: p_cam = R @ p_world + t_ext
    # We want depth = p_cam[2]

    # Simpler: depth = t_intersect * cos(angle between ray and optical axis)
    # For normalized ray_cam: depth = t_intersect * ray_cam_z / ||ray_cam||
    # Since ray_cam_z = 1 before normalization:
    ray_cam_unnorm = np.stack([ray_cam_x, ray_cam_y, ray_cam_z], axis=-1)
    ray_length = np.linalg.norm(ray_cam_unnorm, axis=-1)

    ground_depth = np.zeros((H, W), dtype=np.float32)
    ground_depth[valid] = t_intersect[valid] / ray_length[valid]

    # Filter invalid depths (behind camera or too far)
    valid = valid & (ground_depth > 0.1) & (ground_depth < 500)

    # Apply ground mask if provided
    if ground_mask is not None:
        valid = valid & ground_mask

    ground_depth = np.where(valid, ground_depth, 0)

    return ground_depth, valid


def ground_constrained_depth_fusion(
    camera_depth: np.ndarray,  # (H, W) camera depth (aligned)
    lidar_points: np.ndarray,  # (N, 3) world coordinates
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) w2c
    ground_mask: Optional[np.ndarray] = None,  # (H, W) semantic ground mask
    estimate_ground_from_image: bool = True,  # Estimate ground from image position
    ground_height_tolerance: float = 0.3,  # Tolerance for ground plane (meters)
    blend_width: float = 5.0,  # Pixels to blend at ground boundary
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Fuse camera depth with ground plane constraint.

    Key insight: Ground is flat, so we can use LiDAR to fit a ground plane
    and then compute exact depth for ground pixels.

    Args:
        camera_depth: Camera aligned depth (H, W)
        lidar_points: LiDAR point cloud in world coordinates
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: World-to-camera transformation (4, 4)
        ground_mask: Optional semantic segmentation mask for ground
        estimate_ground_from_image: If True and no ground_mask, estimate ground
                                    from image position (lower part = ground)
        ground_height_tolerance: Height tolerance for ground plane fitting
        blend_width: Width of blending region at ground/non-ground boundary

    Returns:
        fused_depth: (H, W) fused depth with ground constraint
        ground_mask_used: (H, W) actual ground mask used
        info: Dict with fitting info (plane_normal, plane_d, etc.)
    """
    from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

    H, W = camera_depth.shape

    # Step 1: Fit ground plane from LiDAR
    print("Fitting ground plane from LiDAR...")
    plane_normal, plane_d, lidar_ground_mask = fit_ground_plane_ransac(
        lidar_points,
        height_threshold=ground_height_tolerance * 3,
        distance_threshold=ground_height_tolerance,
    )

    # Step 2: Compute ground depth from plane
    print("Computing ground depth from plane intersection...")
    ground_depth, valid_ground = compute_ground_depth_from_plane(
        plane_normal, plane_d, intrinsics, extrinsics, (H, W)
    )

    # Step 3: Determine ground mask
    if ground_mask is not None:
        # Use provided semantic mask
        ground_mask_used = ground_mask.astype(bool)
        print(f"Using provided ground mask: {ground_mask_used.sum()} pixels")
    elif estimate_ground_from_image:
        # Estimate ground from image position + depth consistency
        # Ground is typically in lower part of image and has consistent depth with plane

        # Heuristic: lower 60% of image is potential ground
        v_grid = np.arange(H).reshape(-1, 1)
        position_ground = v_grid > (H * 0.4)  # Lower 60%
        position_ground = np.broadcast_to(position_ground, (H, W))

        # Check depth consistency with ground plane
        depth_diff = np.abs(camera_depth - ground_depth)
        depth_consistent = (depth_diff < ground_height_tolerance * 2) & (ground_depth > 0)

        # Combine: position + depth consistency
        ground_mask_used = position_ground & depth_consistent & valid_ground

        # Clean up mask
        ground_mask_used = binary_erosion(ground_mask_used, iterations=2)
        ground_mask_used = binary_dilation(ground_mask_used, iterations=2)

        print(f"Estimated ground mask: {ground_mask_used.sum()} pixels ({ground_mask_used.mean():.1%})")
    else:
        # No ground mask, use all valid ground depth
        ground_mask_used = valid_ground
        print(f"Using all valid ground: {ground_mask_used.sum()} pixels")

    # Step 4: Create blending weights
    # Smooth transition at ground boundary
    ground_weight = ground_mask_used.astype(np.float32)
    if blend_width > 0:
        ground_weight = gaussian_filter(ground_weight, sigma=blend_width)

    # Step 5: Fuse depths
    # Where we have valid ground depth and ground mask, blend towards ground depth
    fused_depth = camera_depth.copy()

    valid_fusion = (ground_depth > 0.1) & (camera_depth > 0.1)
    fused_depth = np.where(
        valid_fusion,
        ground_weight * ground_depth + (1 - ground_weight) * camera_depth,
        camera_depth
    )

    # Compute stats
    if valid_fusion.any():
        depth_correction = ground_depth[valid_fusion] - camera_depth[valid_fusion]
        print(f"Ground depth correction: mean={depth_correction.mean():.3f}m, "
              f"std={depth_correction.std():.3f}m")

    info = {
        'plane_normal': plane_normal,
        'plane_d': plane_d,
        'n_ground_pixels': int(ground_mask_used.sum()),
        'ground_coverage': float(ground_mask_used.mean()),
    }

    return fused_depth, ground_mask_used.astype(np.float32), info


def lidar_anchored_depth(
    camera_depth: np.ndarray,  # (H, W) raw camera relative depth (NOT aligned)
    lidar_points: np.ndarray,  # (N, 3) world coordinates - THE GROUND TRUTH
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) w2c
    use_ground_plane: bool = True,  # Use ground plane for large-area anchoring
    ground_height_tolerance: float = 0.3,
    local_scale_radius: int = 50,  # Radius for local scale estimation
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    LiDAR-anchored depth estimation.

    Philosophy: LiDAR points are GROUND TRUTH. Use them as anchors, fill gaps with camera.

    Priority:
    1. LiDAR projection points → Direct LiDAR depth (absolute truth)
    2. Ground pixels → Depth from LiDAR-fitted ground plane (geometric truth)
    3. Other pixels → Camera depth with LOCAL scale correction (based on nearby LiDAR)

    Args:
        camera_depth: Raw camera depth (relative, not yet aligned)
        lidar_points: LiDAR point cloud in world coordinates (THE ANCHOR)
        intrinsics: Camera intrinsics
        extrinsics: World-to-camera transformation
        use_ground_plane: Whether to use ground plane constraint
        ground_height_tolerance: Tolerance for ground plane fitting
        local_scale_radius: Radius (pixels) for local scale estimation

    Returns:
        anchored_depth: (H, W) depth anchored to LiDAR
        source_map: (H, W) 0=camera, 1=lidar, 2=ground_plane
        info: Dict with statistics
    """
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    H, W = camera_depth.shape
    anchored_depth = np.zeros((H, W), dtype=np.float32)
    source_map = np.zeros((H, W), dtype=np.uint8)  # 0=camera, 1=lidar, 2=ground

    # ========== Step 1: Project LiDAR to image ==========
    print("Step 1: Projecting LiDAR points...")
    sparse_lidar, lidar_mask = create_sparse_depth_map(
        lidar_points, intrinsics, extrinsics, (H, W)
    )
    n_lidar_pixels = lidar_mask.sum()
    print(f"  LiDAR coverage: {n_lidar_pixels} pixels ({n_lidar_pixels/(H*W)*100:.2f}%)")

    # Use LiDAR directly where available
    anchored_depth[lidar_mask] = sparse_lidar[lidar_mask]
    source_map[lidar_mask] = 1

    # ========== Step 2: Ground plane (optional) ==========
    ground_depth = None
    ground_mask = None
    plane_info = {}

    if use_ground_plane:
        print("Step 2: Fitting ground plane...")
        plane_normal, plane_d, _ = fit_ground_plane_ransac(
            lidar_points,
            height_threshold=ground_height_tolerance * 3,
            distance_threshold=ground_height_tolerance,
        )
        plane_info = {'plane_normal': plane_normal, 'plane_d': plane_d}

        # Compute ground depth for all pixels
        ground_depth, valid_ground = compute_ground_depth_from_plane(
            plane_normal, plane_d, intrinsics, extrinsics, (H, W)
        )

        # Estimate ground region: lower part of image where camera depth matches ground
        # First, compute global scale from LiDAR
        if lidar_mask.sum() > 50:
            lidar_depths = sparse_lidar[lidar_mask]
            camera_at_lidar = camera_depth[lidar_mask]
            valid = camera_at_lidar > 0.01
            if valid.sum() > 20:
                global_scale = np.median(lidar_depths[valid] / camera_at_lidar[valid])
                camera_scaled = camera_depth * global_scale

                # Ground = where scaled camera depth matches ground plane depth
                v_grid = np.arange(H).reshape(-1, 1)
                lower_half = v_grid > (H * 0.35)
                lower_half = np.broadcast_to(lower_half, (H, W))

                depth_match = np.abs(camera_scaled - ground_depth) < ground_height_tolerance
                ground_mask = lower_half & depth_match & valid_ground & (~lidar_mask)

                # Use ground plane depth
                anchored_depth[ground_mask] = ground_depth[ground_mask]
                source_map[ground_mask] = 2

                n_ground = ground_mask.sum()
                print(f"  Ground plane coverage: {n_ground} pixels ({n_ground/(H*W)*100:.2f}%)")

    # ========== Step 3: Fill remaining with locally-scaled camera depth ==========
    print("Step 3: Filling gaps with locally-scaled camera depth...")
    remaining = (source_map == 0) & (camera_depth > 0.01)

    if remaining.any():
        # Compute distance to nearest LiDAR point
        anchor_mask = (source_map > 0)  # LiDAR or ground
        dist_to_anchor = distance_transform_edt(~anchor_mask)

        # For each remaining pixel, estimate local scale from nearby anchors
        # Use a smoothed scale field for efficiency

        # Create scale field: at anchor pixels, scale = anchor_depth / camera_depth
        scale_field = np.ones((H, W), dtype=np.float32)
        anchor_with_camera = anchor_mask & (camera_depth > 0.01)
        if anchor_with_camera.sum() > 10:
            scale_field[anchor_with_camera] = (
                anchored_depth[anchor_with_camera] / camera_depth[anchor_with_camera]
            )

            # Smooth and propagate scale
            # Weight by inverse distance to anchors
            weight_field = np.exp(-dist_to_anchor / local_scale_radius)
            weight_field[anchor_with_camera] = 1.0

            # Weighted smoothing of scale
            scale_sum = gaussian_filter(scale_field * weight_field, sigma=local_scale_radius)
            weight_sum = gaussian_filter(weight_field, sigma=local_scale_radius)
            smoothed_scale = scale_sum / (weight_sum + 1e-6)

            # Apply local scale to remaining pixels
            anchored_depth[remaining] = camera_depth[remaining] * smoothed_scale[remaining]
        else:
            # Fallback: global scale
            if lidar_mask.sum() > 10:
                global_scale = np.median(
                    sparse_lidar[lidar_mask] / camera_depth[lidar_mask]
                )
            else:
                global_scale = 1.0
            anchored_depth[remaining] = camera_depth[remaining] * global_scale

    n_camera = (source_map == 0).sum()
    print(f"  Camera-filled: {n_camera} pixels ({n_camera/(H*W)*100:.2f}%)")

    info = {
        'n_lidar': int(n_lidar_pixels),
        'n_ground': int(ground_mask.sum()) if ground_mask is not None else 0,
        'n_camera': int(n_camera),
        **plane_info,
    }

    return anchored_depth, source_map, info


def compute_lidar_density_map(
    lidar_points: np.ndarray,  # (N, 3) world coordinates
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) w2c
    image_hw: Tuple[int, int],
    kernel_size: int = 31,  # Size of local window for density estimation
    sigma: float = 10.0,  # Gaussian blur sigma
) -> np.ndarray:
    """
    Compute local LiDAR point density map.

    This creates a smooth density map indicating how many LiDAR points
    are available in local neighborhoods. High density regions should
    rely more on LiDAR, low density regions on camera depth.

    Args:
        lidar_points: LiDAR points in world coordinates (N, 3)
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: World-to-camera transformation (4, 4)
        image_hw: Image size (H, W)
        kernel_size: Size of kernel for density smoothing
        sigma: Gaussian blur sigma for smooth density

    Returns:
        density_map: Normalized density map (H, W), values in [0, 1]
    """
    from scipy.ndimage import gaussian_filter

    H, W = image_hw
    count_map = np.zeros((H, W), dtype=np.float32)

    # Project LiDAR points to image
    pixel_coords, depths, _ = project_lidar_to_image(
        lidar_points, intrinsics, extrinsics, image_hw
    )

    if len(depths) == 0:
        return count_map

    # Count points per pixel
    u = pixel_coords[:, 0].astype(int)
    v = pixel_coords[:, 1].astype(int)
    np.add.at(count_map, (v, u), 1)

    # Smooth with Gaussian to get local density
    density_map = gaussian_filter(count_map, sigma=sigma)

    # Normalize to [0, 1] using a soft threshold
    # This determines how many points per area = "dense"
    # Tune this based on typical LiDAR density
    density_threshold = 0.1  # Points per pixel after smoothing
    density_map = np.clip(density_map / density_threshold, 0, 1)

    return density_map


def adaptive_depth_fusion(
    camera_depth: np.ndarray,  # (H, W) aligned camera depth
    lidar_points: np.ndarray,  # (N, 3) world coordinates
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) w2c
    camera_confidence: Optional[np.ndarray] = None,  # (H, W) confidence
    density_kernel: int = 31,
    density_sigma: float = 15.0,
    interpolation_kernel: int = 15,
    min_lidar_weight: float = 0.0,  # Minimum weight for LiDAR
    max_lidar_weight: float = 0.7,  # Maximum weight for LiDAR (reduced from 0.95)
    distance_boost: bool = False,  # Disabled by default - causes uplift issues
    distance_threshold: float = 30.0,  # Distance (m) at which to start boosting LiDAR
    use_dilation_only: bool = True,  # Use dilation instead of interpolation
    dilation_radius: int = 5,  # Radius for LiDAR point dilation
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive fusion of camera depth and LiDAR depth based on local LiDAR density.

    Key insight:
    - Near regions (roadside): Camera pixels dense, LiDAR sparse → Trust camera
    - Far regions (intersection center): Camera pixels sparse, LiDAR dense → Trust LiDAR

    IMPORTANT: This function avoids full-image interpolation which causes ground
    "uplift" artifacts. Instead, it only uses LiDAR values at actual LiDAR point
    locations (with optional small dilation).

    Args:
        camera_depth: Camera aligned depth map (H, W), already scale-corrected
        lidar_points: LiDAR point cloud in world coordinates (N, 3)
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: World-to-camera transformation (4, 4)
        camera_confidence: Optional confidence map from DA3
        density_kernel: Kernel size for density estimation
        density_sigma: Gaussian sigma for density smoothing
        interpolation_kernel: Kernel size for LiDAR depth interpolation (if not using dilation)
        min_lidar_weight: Minimum LiDAR weight (for sparse regions)
        max_lidar_weight: Maximum LiDAR weight (for dense regions)
        distance_boost: Whether to increase LiDAR weight at larger distances
        distance_threshold: Distance at which to start boosting LiDAR weight
        use_dilation_only: If True, only use LiDAR at actual points + small dilation
                          If False, interpolate LiDAR across whole image (NOT recommended)
        dilation_radius: Pixel radius to dilate LiDAR points (only if use_dilation_only=True)

    Returns:
        fused_depth: Fused depth map (H, W)
        lidar_weight_map: Weight map showing LiDAR contribution (H, W)
        lidar_depth_map: LiDAR depth (sparse or interpolated) for debugging (H, W)
    """
    from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt

    H, W = camera_depth.shape

    # Step 1: Create sparse depth map from LiDAR
    sparse_depth, valid_mask = create_sparse_depth_map(
        lidar_points, intrinsics, extrinsics, (H, W)
    )

    n_lidar_pixels = valid_mask.sum()
    if n_lidar_pixels < 10:
        # Not enough LiDAR points, return camera depth unchanged
        return camera_depth.copy(), np.zeros((H, W), dtype=np.float32), sparse_depth

    if use_dilation_only:
        # Conservative approach: only use LiDAR at/near actual LiDAR points
        # Create dilated mask for LiDAR influence region
        if dilation_radius > 0:
            struct = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1))
            dilated_mask = binary_dilation(valid_mask, struct)
        else:
            dilated_mask = valid_mask.copy()

        # For dilated pixels, use nearest LiDAR depth (no interpolation!)
        # Use distance transform to find nearest LiDAR point
        dist_to_lidar = distance_transform_edt(~valid_mask)

        # Create lidar depth map: at each dilated pixel, use nearest lidar value
        lidar_depth_map = np.zeros_like(camera_depth)
        if valid_mask.sum() > 0:
            # For pixels with LiDAR, use LiDAR directly
            lidar_depth_map[valid_mask] = sparse_depth[valid_mask]

            # For dilated pixels without direct LiDAR, find nearest
            dilated_no_direct = dilated_mask & (~valid_mask)
            if dilated_no_direct.any():
                from scipy.ndimage import grey_dilation

                # Use grey dilation to propagate nearest lidar values
                for _ in range(dilation_radius):
                    kernel = np.ones((3, 3))
                    dilated_depth = grey_dilation(lidar_depth_map, footprint=kernel)
                    # Only update where we don't have values yet but are in dilated region
                    update_mask = (lidar_depth_map == 0) & dilated_mask
                    lidar_depth_map[update_mask] = dilated_depth[update_mask]

        # LiDAR weight: only within dilated region, smooth falloff
        lidar_weight = np.zeros((H, W), dtype=np.float32)

        # Weight based on distance to nearest LiDAR point
        # At LiDAR point: weight = max_lidar_weight
        # At edge of dilation: weight = 0
        weight_at_lidar = np.exp(-dist_to_lidar / (dilation_radius / 2 + 1))
        lidar_weight = weight_at_lidar * max_lidar_weight
        lidar_weight[~dilated_mask] = 0  # Zero outside dilated region

    else:
        # Original interpolation approach (NOT recommended - causes uplift)
        from scipy.interpolate import griddata

        # Compute LiDAR density map
        density_map = compute_lidar_density_map(
            lidar_points, intrinsics, extrinsics, (H, W),
            kernel_size=density_kernel, sigma=density_sigma
        )

        # Get coordinates of valid LiDAR points
        v_valid, u_valid = np.where(valid_mask)
        lidar_depths = sparse_depth[valid_mask]

        # Create grid for interpolation
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        grid_points = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        lidar_coords = np.stack([u_valid, v_valid], axis=1)

        # Interpolate (linear with nearest neighbor fallback)
        try:
            lidar_depth_map = griddata(
                lidar_coords, lidar_depths, grid_points,
                method='linear', fill_value=0
            ).reshape(H, W)

            # Fill remaining holes with nearest neighbor
            missing = lidar_depth_map == 0
            if missing.any() and (~missing).any():
                interp_nearest = griddata(
                    lidar_coords, lidar_depths, grid_points,
                    method='nearest'
                ).reshape(H, W)
                lidar_depth_map[missing] = interp_nearest[missing]
        except Exception:
            lidar_depth_map = camera_depth.copy()

        # LiDAR weight based on density
        lidar_weight = density_map * (max_lidar_weight - min_lidar_weight) + min_lidar_weight

        # Reduce weight where interpolation is far from actual LiDAR points
        interpolation_confidence = 1.0 - np.clip(
            gaussian_filter((~valid_mask).astype(float), sigma=density_sigma) * 2, 0, 1
        )
        lidar_weight = lidar_weight * interpolation_confidence

    # Optional: Boost weight at larger distances (DISABLED by default)
    if distance_boost and distance_threshold > 0:
        distance_factor = np.clip(camera_depth / distance_threshold, 0, 2) - 1
        distance_factor = np.clip(distance_factor, 0, 1)  # 0 at near, 1 at far
        lidar_weight = lidar_weight + distance_factor * (1 - lidar_weight) * 0.3

    # Clip to valid range
    lidar_weight = np.clip(lidar_weight, min_lidar_weight, max_lidar_weight)

    # Fuse depths
    fused_depth = lidar_weight * lidar_depth_map + (1 - lidar_weight) * camera_depth

    # Where LiDAR depth is 0, use camera depth
    fused_depth = np.where(lidar_depth_map > 0.1, fused_depth, camera_depth)

    return fused_depth, lidar_weight.astype(np.float32), lidar_depth_map.astype(np.float32)
