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
    max_lidar_weight: float = 0.95,  # Maximum weight for LiDAR (never fully 1.0)
    distance_boost: bool = True,  # Boost LiDAR weight at larger distances
    distance_threshold: float = 50.0,  # Distance (m) at which to start boosting LiDAR
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive fusion of camera depth and LiDAR depth based on local LiDAR density.

    Key insight:
    - Near regions (roadside): Camera pixels dense, LiDAR sparse → Trust camera
    - Far regions (intersection center): Camera pixels sparse, LiDAR dense → Trust LiDAR

    The fusion weight is computed as:
        w_lidar = f(local_lidar_density, distance)

    Args:
        camera_depth: Camera aligned depth map (H, W), already scale-corrected
        lidar_points: LiDAR point cloud in world coordinates (N, 3)
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: World-to-camera transformation (4, 4)
        camera_confidence: Optional confidence map from DA3
        density_kernel: Kernel size for density estimation
        density_sigma: Gaussian sigma for density smoothing
        interpolation_kernel: Kernel size for LiDAR depth interpolation
        min_lidar_weight: Minimum LiDAR weight (for sparse regions)
        max_lidar_weight: Maximum LiDAR weight (for dense regions)
        distance_boost: Whether to increase LiDAR weight at larger distances
        distance_threshold: Distance at which to start boosting LiDAR weight

    Returns:
        fused_depth: Fused depth map (H, W)
        lidar_weight_map: Weight map showing LiDAR contribution (H, W)
        interpolated_lidar: Interpolated LiDAR depth for debugging (H, W)
    """
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import griddata

    H, W = camera_depth.shape

    # Step 1: Compute LiDAR density map
    density_map = compute_lidar_density_map(
        lidar_points, intrinsics, extrinsics, (H, W),
        kernel_size=density_kernel, sigma=density_sigma
    )

    # Step 2: Create sparse depth map from LiDAR
    sparse_depth, valid_mask = create_sparse_depth_map(
        lidar_points, intrinsics, extrinsics, (H, W)
    )

    # Step 3: Interpolate LiDAR depth where we have nearby points
    # Use natural neighbor or linear interpolation
    if valid_mask.sum() > 10:
        # Get coordinates of valid LiDAR points
        v_valid, u_valid = np.where(valid_mask)
        lidar_depths = sparse_depth[valid_mask]

        # Create grid for interpolation
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        grid_points = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        lidar_coords = np.stack([u_valid, v_valid], axis=1)

        # Interpolate (linear with nearest neighbor fallback)
        try:
            interpolated_lidar = griddata(
                lidar_coords, lidar_depths, grid_points,
                method='linear', fill_value=0
            ).reshape(H, W)

            # Fill remaining holes with nearest neighbor
            missing = interpolated_lidar == 0
            if missing.any() and (~missing).any():
                interp_nearest = griddata(
                    lidar_coords, lidar_depths, grid_points,
                    method='nearest'
                ).reshape(H, W)
                interpolated_lidar[missing] = interp_nearest[missing]
        except Exception:
            # Fallback: just use sparse depth with some dilation
            interpolated_lidar = gaussian_filter(sparse_depth, sigma=interpolation_kernel)
            interpolated_lidar = np.where(interpolated_lidar > 0, interpolated_lidar, camera_depth)
    else:
        # Not enough LiDAR points, use camera depth
        interpolated_lidar = camera_depth.copy()

    # Step 4: Compute adaptive weight for LiDAR
    # Base weight from density
    lidar_weight = density_map * (max_lidar_weight - min_lidar_weight) + min_lidar_weight

    # Boost weight at larger distances (where camera is less reliable)
    if distance_boost and distance_threshold > 0:
        distance_factor = np.clip(camera_depth / distance_threshold, 0, 2) - 1
        distance_factor = np.clip(distance_factor, 0, 1)  # 0 at near, 1 at far
        # Blend in more LiDAR weight at distance
        lidar_weight = lidar_weight + distance_factor * (1 - lidar_weight) * 0.5

    # Reduce LiDAR weight where interpolation is far from actual LiDAR points
    interpolation_confidence = 1.0 - np.clip(
        gaussian_filter((~valid_mask).astype(float), sigma=density_sigma) * 2, 0, 1
    )
    lidar_weight = lidar_weight * interpolation_confidence

    # Clip to valid range
    lidar_weight = np.clip(lidar_weight, min_lidar_weight, max_lidar_weight)

    # Step 5: Fuse depths
    fused_depth = lidar_weight * interpolated_lidar + (1 - lidar_weight) * camera_depth

    # Handle edge cases
    fused_depth = np.where(
        (interpolated_lidar > 0.1) & (camera_depth > 0.1),
        fused_depth,
        np.maximum(interpolated_lidar, camera_depth)  # Use whichever is valid
    )

    return fused_depth, lidar_weight.astype(np.float32), interpolated_lidar.astype(np.float32)
