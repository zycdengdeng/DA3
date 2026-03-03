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
Depth Completion Integration Module.

Integrates external depth completion networks (e.g., CompletionFormer) with
our roadside reconstruction pipeline. These networks take sparse LiDAR depth
+ RGB image as input and output dense metric depth.

This approach is superior to scale alignment because:
1. The network learns to propagate sparse depth respecting image edges
2. Output is already metric depth (no scale ambiguity)
3. Handles occlusions and complex geometry better

Usage:
    from depth_anything_3.utils.depth_completion import DepthCompleter

    completer = DepthCompleter(method="completionformer")
    dense_depth = completer.complete(rgb_image, sparse_lidar_depth)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

# Lazy imports for optional dependencies
_completionformer_available = None
_torch_available = None


def _check_torch():
    global _torch_available
    if _torch_available is None:
        try:
            import torch
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def _check_completionformer():
    global _completionformer_available
    if _completionformer_available is None:
        try:
            # Check if CompletionFormer is in path
            from model.completionformer import CompletionFormer
            _completionformer_available = True
        except ImportError:
            _completionformer_available = False
    return _completionformer_available


class DepthCompleter:
    """
    Unified interface for depth completion methods.

    Supports:
    - sam: SAM-guided region-wise completion (best quality, requires SAM)
    - completionformer: CompletionFormer (CVPR 2023)
    - nlspn: NLSPN (ECCV 2020)
    - simple: Simple interpolation fallback (no external deps)
    """

    SUPPORTED_METHODS = ["sam", "completionformer", "nlspn", "simple"]

    def __init__(
        self,
        method: str = "sam",
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_depth: float = 100.0,
        sam_points_per_side: int = 32,
    ):
        """
        Initialize depth completer.

        Args:
            method: Completion method ("sam", "completionformer", "nlspn", "simple")
            model_path: Path to pretrained weights (SAM or CompletionFormer)
            device: Device to run on ("cuda" or "cpu")
            max_depth: Maximum depth value (meters)
            sam_points_per_side: SAM grid density (higher = finer segmentation)
        """
        self.method = method.lower()
        self.model_path = model_path
        self.device = device
        self.max_depth = max_depth
        self.model = None
        self.sam_segmenter = None
        self.sam_points_per_side = sam_points_per_side

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Supported: {self.SUPPORTED_METHODS}")

        if self.method == "sam":
            self._init_sam()
        elif self.method == "completionformer":
            self._init_completionformer()
        elif self.method == "nlspn":
            self._init_nlspn()
        # "simple" doesn't need initialization

    def _init_sam(self):
        """Initialize SAM for region-wise completion."""
        from depth_anything_3.utils.region_depth_alignment import SAMAutoSegmenter

        self.sam_segmenter = SAMAutoSegmenter(
            checkpoint_path=self.model_path,
            device=self.device,
            points_per_side=self.sam_points_per_side,
            min_mask_region_area=50,
        )

        if not self.sam_segmenter.is_available:
            print("WARNING: SAM not available, falling back to 'simple' method")
            self.method = "simple"

    def _init_completionformer(self):
        """Initialize CompletionFormer model."""
        if not _check_torch():
            raise ImportError("PyTorch is required for CompletionFormer")

        import torch

        # Try to find CompletionFormer
        completionformer_path = os.environ.get(
            "COMPLETIONFORMER_PATH",
            str(Path(__file__).parent.parent.parent.parent.parent / "CompletionFormer" / "src")
        )

        if completionformer_path not in sys.path:
            sys.path.insert(0, completionformer_path)

        try:
            from model.completionformer import CompletionFormer
            from config import args as cf_args
        except ImportError as e:
            raise ImportError(
                f"CompletionFormer not found. Please:\n"
                f"1. Clone: git clone https://github.com/youmi-zym/CompletionFormer\n"
                f"2. Set COMPLETIONFORMER_PATH env var to the 'src' directory\n"
                f"   OR place it at: {completionformer_path}\n"
                f"Original error: {e}"
            )

        # Configure args for inference
        cf_args.prop_time = 6
        cf_args.prop_kernel = 3
        cf_args.preserve_input = True
        cf_args.affinity = "TGASS"
        cf_args.affinity_gamma = 0.5
        cf_args.conf_prop = True
        cf_args.no_pos = False

        # Create model
        self.model = CompletionFormer(cf_args)

        # Load weights
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading CompletionFormer weights from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'net' in checkpoint:
                self.model.load_state_dict(checkpoint['net'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("Warning: No CompletionFormer weights provided, using random initialization")

        self.model = self.model.to(self.device)
        self.model.eval()

    def _init_nlspn(self):
        """Initialize NLSPN model."""
        raise NotImplementedError("NLSPN integration not yet implemented")

    def complete(
        self,
        rgb: np.ndarray,  # (H, W, 3) uint8 or float32
        sparse_depth: np.ndarray,  # (H, W) sparse depth map
        mask: Optional[np.ndarray] = None,  # (H, W) valid depth mask
        da3_depth: Optional[np.ndarray] = None,  # (H, W) DA3 relative depth
    ) -> np.ndarray:
        """
        Complete sparse depth to dense depth.

        Args:
            rgb: RGB image (H, W, 3), uint8 [0-255] or float32 [0-1]
            sparse_depth: Sparse depth map (H, W), 0 where no depth
            mask: Optional validity mask (H, W), True where depth is valid
            da3_depth: Optional DA3 relative depth (H, W), for SAM method

        Returns:
            dense_depth: Dense depth map (H, W) in meters
        """
        if self.method == "sam":
            return self._complete_sam(rgb, sparse_depth, mask, da3_depth=da3_depth)
        elif self.method == "simple":
            return self._complete_simple(rgb, sparse_depth, mask)
        elif self.method == "completionformer":
            return self._complete_completionformer(rgb, sparse_depth, mask)
        elif self.method == "nlspn":
            return self._complete_nlspn(rgb, sparse_depth, mask)

    def _complete_sam(
        self,
        rgb: np.ndarray,
        sparse_depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
        da3_depth: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ) -> np.ndarray:
        """
        SAM-guided region-wise depth completion using DA3 relative depth.

        Key idea:
        1. SAM segments the image into regions (cars, poles, buildings, road, etc.)
        2. DA3 provides complete relative depth structure (no holes)
        3. For each region with LiDAR points: compute scale/shift to align DA3 to LiDAR
        4. Result: complete depth with clean object boundaries

        Args:
            rgb: RGB image
            sparse_depth: Sparse LiDAR depth
            mask: Valid depth mask
            da3_depth: DA3 relative depth (required for best results)
            return_debug: Return debug info
        """
        import cv2

        H, W = sparse_depth.shape
        debug_info = {}

        # Get valid depth points
        if mask is None:
            mask = sparse_depth > 0.1

        if mask.sum() < 10:
            print("Warning: Too few valid depth points for completion")
            if return_debug:
                return sparse_depth.copy(), {}
            return sparse_depth.copy()

        # Check DA3 depth
        if da3_depth is None:
            print("    WARNING: DA3 depth not provided, using interpolation fallback")
            return self._complete_sam_interpolation(rgb, sparse_depth, mask, return_debug)

        # Step 1: Segment image with SAM
        print(f"    SAM segmenting image...")
        region_labels, masks_info = self.sam_segmenter.segment(rgb)
        n_regions = len(masks_info)
        print(f"    SAM found {n_regions} regions")

        # Save debug info
        debug_info['region_labels'] = region_labels.copy()
        debug_info['n_regions'] = n_regions
        debug_info['masks_info'] = masks_info
        debug_info['da3_depth'] = da3_depth.copy()

        # Create colorful region visualization
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(n_regions + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background is black
        region_vis = colors[region_labels]
        debug_info['region_vis'] = region_vis

        # Step 2: Compute global scale/shift as fallback
        da3_at_lidar = da3_depth[mask]
        lidar_depths = sparse_depth[mask]

        # Robust global alignment (RANSAC-like: use median)
        valid_da3 = da3_at_lidar > 1e-6
        if valid_da3.sum() > 10:
            ratios = lidar_depths[valid_da3] / da3_at_lidar[valid_da3]
            global_scale = np.median(ratios)
            global_shift = 0.0  # Simple scale model
        else:
            global_scale = 1.0
            global_shift = 0.0

        print(f"    Global scale: {global_scale:.3f}")

        # Step 3: Initialize output with globally scaled DA3
        dense_depth = da3_depth * global_scale + global_shift
        dense_depth = np.clip(dense_depth, 0, self.max_depth)

        # Track which regions get local alignment
        region_point_counts = {}
        region_scales = {}
        completed_mask = np.zeros((H, W), dtype=bool)

        # Step 4: Per-region local alignment (two-pass)
        min_points_for_local = 5  # Need at least 5 points for reliable local scale

        # Store region info for neighbor-based scale propagation
        region_info = {}  # region_id -> {mask, center, da3_mean, n_points}
        regions_needing_neighbor_scale = []  # regions with < 5 LiDAR points

        # First pass: compute local scales for regions with enough LiDAR points
        for region_id in range(1, n_regions + 1):
            region_mask = region_labels == region_id

            if region_mask.sum() < 10:
                continue

            # Compute region center for neighbor finding
            ys, xs = np.where(region_mask)
            center_y, center_x = np.mean(ys), np.mean(xs)
            da3_mean = np.mean(da3_depth[region_mask])

            # Get LiDAR points within this region
            region_sparse_mask = mask & region_mask
            n_points = region_sparse_mask.sum()
            region_point_counts[region_id] = int(n_points)

            region_info[region_id] = {
                'mask': region_mask,
                'center': (center_y, center_x),
                'da3_mean': da3_mean,
                'n_points': n_points,
            }

            if n_points < min_points_for_local:
                # Mark for second pass (neighbor-based scale)
                regions_needing_neighbor_scale.append(region_id)
                continue

            # Compute local scale for this region
            da3_region = da3_depth[region_sparse_mask]
            lidar_region = sparse_depth[region_sparse_mask]

            valid_da3 = da3_region > 1e-6
            if valid_da3.sum() < 3:
                regions_needing_neighbor_scale.append(region_id)
                continue

            # Local scale (median for robustness)
            local_ratios = lidar_region[valid_da3] / da3_region[valid_da3]
            local_scale = np.median(local_ratios)

            # Sanity check: scale shouldn't be too different from global
            if local_scale < global_scale * 0.2 or local_scale > global_scale * 5:
                # Suspicious, use global
                local_scale = global_scale

            region_scales[region_id] = float(local_scale)
            region_info[region_id]['scale'] = local_scale

            # Apply local scale to this region
            region_depth = da3_depth[region_mask] * local_scale
            region_depth = np.clip(region_depth, 0, self.max_depth)
            dense_depth[region_mask] = region_depth
            completed_mask[region_mask] = True

        # Second pass: assign scales to regions without enough LiDAR points
        # Use weighted average of nearby regions with known scales
        n_neighbor_scale = 0
        for region_id in regions_needing_neighbor_scale:
            info = region_info[region_id]
            region_mask = info['mask']
            center = info['center']
            da3_mean = info['da3_mean']

            # Find nearby regions with known scales
            neighbor_scales = []
            neighbor_weights = []

            for other_id, other_info in region_info.items():
                if other_id == region_id:
                    continue
                if 'scale' not in other_info:
                    continue

                # Compute distance (spatial + depth similarity)
                other_center = other_info['center']
                spatial_dist = np.sqrt(
                    (center[0] - other_center[0])**2 +
                    (center[1] - other_center[1])**2
                )

                # Depth similarity: prefer regions with similar DA3 depth
                depth_diff = abs(da3_mean - other_info['da3_mean'])
                depth_weight = np.exp(-depth_diff / max(da3_mean, 0.1))

                # Spatial weight: closer is better
                spatial_weight = np.exp(-spatial_dist / 200)  # 200 pixels decay

                weight = spatial_weight * depth_weight
                if weight > 0.01:  # Only consider reasonably close/similar regions
                    neighbor_scales.append(other_info['scale'])
                    neighbor_weights.append(weight)

            if neighbor_scales:
                # Weighted average of neighbor scales
                neighbor_weights = np.array(neighbor_weights)
                neighbor_weights /= neighbor_weights.sum()
                neighbor_scale = np.sum(np.array(neighbor_scales) * neighbor_weights)
                region_scales[region_id] = float(neighbor_scale)
                n_neighbor_scale += 1
            else:
                # No neighbors found, use global scale
                neighbor_scale = global_scale

            # Apply scale to this region
            region_depth = da3_depth[region_mask] * neighbor_scale
            region_depth = np.clip(region_depth, 0, self.max_depth)
            dense_depth[region_mask] = region_depth
            completed_mask[region_mask] = True

        debug_info['region_point_counts'] = region_point_counts
        debug_info['region_scales'] = region_scales
        debug_info['completed_mask_before_fill'] = completed_mask.copy()
        debug_info['dense_before_smooth'] = dense_depth.copy()

        # Stats
        n_local = len([rid for rid, info in region_info.items()
                      if 'scale' in info and rid not in regions_needing_neighbor_scale])
        n_global = len(regions_needing_neighbor_scale) - n_neighbor_scale
        print(f"    Regions with local scale: {n_local}")
        print(f"    Regions with neighbor scale: {n_neighbor_scale}")
        print(f"    Regions with global scale: {n_global}")

        # Step 5: Light smoothing within regions (optional)
        depth_smoothed = dense_depth.copy()

        # Bilateral filter per region to smooth while preserving edges
        for region_id in range(1, min(n_regions + 1, 200)):
            region_mask = region_labels == region_id

            if region_mask.sum() < 100:
                continue

            # Get bounding box
            ys, xs = np.where(region_mask)
            y_min, y_max = ys.min(), ys.max() + 1
            x_min, x_max = xs.min(), xs.max() + 1

            # Extract patch
            patch = dense_depth[y_min:y_max, x_min:x_max].copy()
            patch_mask = region_mask[y_min:y_max, x_min:x_max]

            if patch.max() <= 0:
                continue

            # Normalize and filter
            patch_norm = patch / (self.max_depth + 1e-6)
            patch_norm = np.clip(patch_norm, 0, 1).astype(np.float32)

            # Light bilateral filter
            patch_filtered = cv2.bilateralFilter(
                patch_norm, d=5, sigmaColor=0.02, sigmaSpace=10
            )
            patch_filtered = patch_filtered * self.max_depth

            # Write back
            depth_smoothed[y_min:y_max, x_min:x_max][patch_mask] = \
                patch_filtered[patch_mask]

        # Preserve original LiDAR values
        depth_smoothed[mask] = sparse_depth[mask]

        print(f"    Final completion: {(depth_smoothed > 0).sum()/(H*W)*100:.1f}%")

        if return_debug:
            return depth_smoothed.astype(np.float32), debug_info
        return depth_smoothed.astype(np.float32)

    def _complete_sam_interpolation(
        self,
        rgb: np.ndarray,
        sparse_depth: np.ndarray,
        mask: np.ndarray,
        return_debug: bool = False,
    ) -> np.ndarray:
        """Fallback: SAM + interpolation when DA3 is not available."""
        from scipy.interpolate import griddata

        H, W = sparse_depth.shape
        debug_info = {}

        # Segment with SAM
        region_labels, masks_info = self.sam_segmenter.segment(rgb)
        n_regions = len(masks_info)

        debug_info['region_labels'] = region_labels.copy()
        debug_info['n_regions'] = n_regions

        # Initialize
        dense_depth = np.zeros((H, W), dtype=np.float32)

        # Per-region interpolation
        for region_id in range(1, n_regions + 1):
            region_mask = region_labels == region_id
            region_sparse_mask = mask & region_mask

            if region_sparse_mask.sum() < 3:
                continue

            v_valid, u_valid = np.where(region_sparse_mask)
            depths = sparse_depth[region_sparse_mask]
            v_target, u_target = np.where(region_mask)

            target_points = np.stack([u_target, v_target], axis=1)
            valid_coords = np.stack([u_valid, v_valid], axis=1)

            try:
                region_depth = griddata(
                    valid_coords, depths, target_points,
                    method='nearest'
                )
                dense_depth[v_target, u_target] = region_depth
            except:
                continue

        # Fill remaining with global nearest
        unfilled = dense_depth <= 0
        if unfilled.any() and mask.sum() > 0:
            v_valid, u_valid = np.where(mask)
            depths = sparse_depth[mask]
            v_target, u_target = np.where(unfilled)

            target_points = np.stack([u_target, v_target], axis=1)
            valid_coords = np.stack([u_valid, v_valid], axis=1)

            fill_depth = griddata(valid_coords, depths, target_points, method='nearest')
            dense_depth[v_target, u_target] = fill_depth

        if return_debug:
            return dense_depth.astype(np.float32), debug_info
        return dense_depth.astype(np.float32)

    def complete_with_debug(
        self,
        rgb: np.ndarray,
        sparse_depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
        da3_depth: Optional[np.ndarray] = None,
    ) -> tuple:
        """Complete depth and return debug information."""
        if self.method == "sam":
            return self._complete_sam(rgb, sparse_depth, mask, da3_depth=da3_depth, return_debug=True)
        else:
            return self.complete(rgb, sparse_depth, mask), {}

    def _complete_simple(
        self,
        rgb: np.ndarray,
        sparse_depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Simple depth completion using bilateral filtering and interpolation.

        This is a fallback when neural network methods are not available.
        """
        import cv2
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter

        H, W = sparse_depth.shape

        # Get valid depth points
        if mask is None:
            mask = sparse_depth > 0.1

        if mask.sum() < 10:
            print("Warning: Too few valid depth points for completion")
            return sparse_depth.copy()

        # Get coordinates and values
        v_valid, u_valid = np.where(mask)
        depths = sparse_depth[mask]

        # Create grid for interpolation
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        grid_points = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        valid_coords = np.stack([u_valid, v_valid], axis=1)

        # Interpolate using linear method
        dense_depth = griddata(
            valid_coords, depths, grid_points,
            method='linear', fill_value=0
        ).reshape(H, W)

        # Fill remaining holes with nearest neighbor
        holes = dense_depth <= 0
        if holes.any():
            dense_nearest = griddata(
                valid_coords, depths, grid_points,
                method='nearest'
            ).reshape(H, W)
            dense_depth[holes] = dense_nearest[holes]

        # Edge-aware smoothing using bilateral filter
        dense_depth = dense_depth.astype(np.float32)

        # Normalize for bilateral filter (needs 8u or 32f)
        depth_norm = dense_depth / (self.max_depth + 1e-6)
        depth_norm = np.clip(depth_norm, 0, 1).astype(np.float32)

        # Apply bilateral filter (edge-preserving smoothing) - use 32f format
        depth_filtered = cv2.bilateralFilter(
            depth_norm,
            d=9, sigmaColor=0.1, sigmaSpace=75
        )

        # Scale back to metric depth
        depth_filtered = depth_filtered * self.max_depth

        # Preserve original sparse depth values
        depth_filtered[mask] = sparse_depth[mask]

        return depth_filtered.astype(np.float32)

    def _complete_completionformer(
        self,
        rgb: np.ndarray,
        sparse_depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Complete depth using CompletionFormer."""
        import torch
        import torch.nn.functional as F

        H, W = sparse_depth.shape

        # Prepare RGB tensor
        if rgb.dtype == np.uint8:
            rgb_float = rgb.astype(np.float32) / 255.0
        else:
            rgb_float = rgb.astype(np.float32)

        # Normalize RGB (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        rgb_norm = (rgb_float - mean) / std

        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float()
        dep_tensor = torch.from_numpy(sparse_depth).unsqueeze(0).unsqueeze(0).float()

        # Pad to multiple of 32 (model requirement)
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if pad_h > 0 or pad_w > 0:
            rgb_tensor = F.pad(rgb_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            dep_tensor = F.pad(dep_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # Move to device
        rgb_tensor = rgb_tensor.to(self.device)
        dep_tensor = dep_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            sample = {'rgb': rgb_tensor, 'dep': dep_tensor}
            output = self.model(sample)
            pred = output['pred']

        # Remove padding and convert to numpy
        if pad_h > 0 or pad_w > 0:
            pred = pred[:, :, :H, :W]

        dense_depth = pred.squeeze().cpu().numpy()

        # Clamp to valid range
        dense_depth = np.clip(dense_depth, 0, self.max_depth)

        return dense_depth

    def _complete_nlspn(
        self,
        rgb: np.ndarray,
        sparse_depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Complete depth using NLSPN."""
        raise NotImplementedError("NLSPN integration not yet implemented")


def create_sparse_depth_from_lidar(
    lidar_points: np.ndarray,  # (N, 3) world coords
    intrinsics: np.ndarray,  # (3, 3)
    extrinsics: np.ndarray,  # (4, 4) w2c
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sparse depth map from LiDAR projection.

    Args:
        lidar_points: LiDAR points in world coordinates (N, 3)
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: World-to-camera transformation (4, 4)
        image_hw: Image size (H, W)

    Returns:
        sparse_depth: Sparse depth map (H, W)
        valid_mask: Boolean mask of valid pixels (H, W)
    """
    from .lidar_alignment import project_lidar_to_image

    H, W = image_hw
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    # Project LiDAR to image
    pixel_coords, depths, _ = project_lidar_to_image(
        lidar_points, intrinsics, extrinsics, image_hw
    )

    if len(depths) == 0:
        return sparse_depth, valid_mask

    u = pixel_coords[:, 0].astype(int)
    v = pixel_coords[:, 1].astype(int)

    # Handle multiple points per pixel (use closest)
    for i in range(len(depths)):
        if sparse_depth[v[i], u[i]] == 0 or depths[i] < sparse_depth[v[i], u[i]]:
            sparse_depth[v[i], u[i]] = depths[i]
            valid_mask[v[i], u[i]] = True

    return sparse_depth, valid_mask


def complete_depth_for_multiview(
    images: dict,  # {cam_id: np.ndarray (H, W, 3)}
    lidar_points: np.ndarray,  # (N, 3) world coords
    intrinsics: dict,  # {cam_id: (3, 3)}
    extrinsics: dict,  # {cam_id: (4, 4)}
    method: str = "completionformer",
    model_path: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """
    Complete depth for all cameras in a multi-view setup.

    Args:
        images: Dict of camera_id -> RGB image
        lidar_points: LiDAR point cloud in world coordinates
        intrinsics: Dict of camera_id -> intrinsic matrix
        extrinsics: Dict of camera_id -> extrinsic matrix (w2c)
        method: Depth completion method
        model_path: Path to pretrained model weights
        device: Device to run on

    Returns:
        depths: Dict of camera_id -> dense depth map
    """
    # Initialize completer once
    completer = DepthCompleter(method=method, model_path=model_path, device=device)

    depths = {}
    for cam_id in images:
        print(f"Completing depth for camera {cam_id}...")

        rgb = images[cam_id]
        K = intrinsics[cam_id]
        E = extrinsics[cam_id]
        H, W = rgb.shape[:2]

        # Create sparse depth from LiDAR
        sparse_depth, valid_mask = create_sparse_depth_from_lidar(
            lidar_points, K, E, (H, W)
        )

        n_valid = valid_mask.sum()
        print(f"  Sparse depth: {n_valid} valid pixels ({n_valid/(H*W)*100:.2f}%)")

        # Complete depth
        dense_depth = completer.complete(rgb, sparse_depth, valid_mask)

        depths[cam_id] = dense_depth
        print(f"  Dense depth range: [{dense_depth.min():.2f}, {dense_depth.max():.2f}]m")

    return depths
