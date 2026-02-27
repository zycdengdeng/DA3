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
Region-wise Depth Alignment using SAM segmentation and LiDAR anchors.

Key idea:
    1. DA3 outputs relative depth (good structure, wrong scale)
    2. SAM segments the entire image into regions (roads, vehicles, poles, etc.)
    3. LiDAR points are projected as sparse anchor points
    4. For each SAM region: compute local affine transform (ax + b) using anchors
    5. SAM boundaries act as hard constraints (no cross-region blending)

This approach preserves DA3's structural accuracy while using LiDAR for
per-region metric scale alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

# Try import SAM for automatic segmentation
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RegionAlignmentResult:
    """Result of region-wise depth alignment."""
    aligned_depth: np.ndarray  # (H, W) metric depth
    region_masks: np.ndarray  # (H, W) region labels (0 = background)
    region_params: Dict[int, Tuple[float, float]]  # region_id -> (scale, shift)
    region_stats: Dict[int, Dict]  # region_id -> stats
    num_regions: int
    coverage: float  # fraction of pixels with valid alignment


class SAMAutoSegmenter:
    """
    SAM-based automatic image segmentation.

    Uses SAM's "Segment Everything" mode to partition the entire image
    into semantic regions without any prompts.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ):
        """
        Initialize SAM auto segmenter.

        Args:
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Path to SAM checkpoint
            device: Device to run on
            points_per_side: Grid density for automatic mask generation
            pred_iou_thresh: IoU threshold for mask filtering
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area in pixels
        """
        self.device = device
        self.generator = None

        if not SAM_AVAILABLE:
            print("WARNING: SAM not available. Install with:")
            print("  pip install segment-anything")
            print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return

        # Find checkpoint
        if checkpoint_path is None:
            import os
            for path in [
                "sam_vit_h_4b8939.pth",
                os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth"),
                "/tmp/sam_vit_h_4b8939.pth",
            ]:
                if os.path.exists(path):
                    checkpoint_path = path
                    break

        if checkpoint_path is None:
            print("SAM checkpoint not found. Download from:")
            print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return

        try:
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)

            self.generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
            print(f"SAM auto segmenter initialized: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to initialize SAM: {e}")

    @property
    def is_available(self) -> bool:
        return self.generator is not None

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Segment image into regions.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            labels: (H, W) integer labels, 0 = background
            masks_info: List of mask info dicts with 'area', 'bbox', etc.
        """
        if not self.is_available:
            # Fallback: return single region
            H, W = image.shape[:2]
            return np.ones((H, W), dtype=np.int32), [{'area': H * W}]

        # Generate masks
        masks = self.generator.generate(image)

        if len(masks) == 0:
            H, W = image.shape[:2]
            return np.ones((H, W), dtype=np.int32), [{'area': H * W}]

        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Create label map (handle overlapping masks by priority)
        H, W = image.shape[:2]
        labels = np.zeros((H, W), dtype=np.int32)

        # Assign labels in reverse order (smaller masks override larger)
        for i, mask_info in enumerate(reversed(masks)):
            mask = mask_info['segmentation']
            labels[mask] = len(masks) - i

        return labels, masks


def project_lidar_to_image(
    lidar_points: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project LiDAR points to image plane.

    Args:
        lidar_points: (N, 3) points in world coordinates
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transform
        image_hw: (H, W) image size

    Returns:
        uv: (M, 2) valid pixel coordinates
        depths: (M,) depth values
        valid_mask: (N,) boolean mask
    """
    H, W = image_hw
    N = len(lidar_points)

    # Transform to camera coordinates
    pts_homo = np.hstack([lidar_points[:, :3], np.ones((N, 1))])
    pts_cam = (extrinsics @ pts_homo.T).T[:, :3]

    # Filter behind camera
    valid = pts_cam[:, 2] > 0.1

    # Project to image
    pts_img = (intrinsics @ pts_cam.T).T
    uv = pts_img[:, :2] / (pts_img[:, 2:3] + 1e-8)
    depths = pts_cam[:, 2]

    # Filter outside image
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < W)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < H)
    valid &= np.isfinite(depths) & (depths < 300)

    return uv[valid], depths[valid], valid


def create_sparse_depth_map(
    uv: np.ndarray,
    depths: np.ndarray,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sparse depth map from projected points.

    Args:
        uv: (M, 2) pixel coordinates
        depths: (M,) depth values
        image_hw: (H, W)

    Returns:
        sparse_depth: (H, W) sparse depth map
        valid_mask: (H, W) boolean mask of valid pixels
    """
    H, W = image_hw
    sparse_depth = np.zeros((H, W), dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    if len(uv) == 0:
        return sparse_depth, valid_mask

    u = uv[:, 0].astype(int)
    v = uv[:, 1].astype(int)

    # Use minimum depth per pixel (closest point)
    for i in range(len(u)):
        ui, vi, di = u[i], v[i], depths[i]
        if sparse_depth[vi, ui] == 0 or di < sparse_depth[vi, ui]:
            sparse_depth[vi, ui] = di
            valid_mask[vi, ui] = True

    return sparse_depth, valid_mask


class DepthFitModel:
    """
    Different models for mapping relative depth to metric depth.

    DA3 outputs disparity-like values, so the relationship to real depth
    may be non-linear. We try multiple models and pick the best.
    """

    @staticmethod
    def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[callable, np.ndarray, Dict]:
        """Linear: depth = a*rel + b"""
        A = np.column_stack([x, np.ones_like(x)])
        params, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = params

        def predict(rel):
            return a * rel + b

        return predict, params, {'model': 'affine', 'a': float(a), 'b': float(b)}

    @staticmethod
    def fit_inverse(x: np.ndarray, y: np.ndarray) -> Tuple[callable, np.ndarray, Dict]:
        """Inverse (disparity): depth = a/rel + b"""
        # Avoid division by zero
        x_safe = np.clip(x, 1e-6, None)
        A = np.column_stack([1.0 / x_safe, np.ones_like(x)])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = params

        def predict(rel):
            return a / np.clip(rel, 1e-6, None) + b

        return predict, params, {'model': 'inverse', 'a': float(a), 'b': float(b)}

    @staticmethod
    def fit_quadratic(x: np.ndarray, y: np.ndarray) -> Tuple[callable, np.ndarray, Dict]:
        """Quadratic: depth = a*rel² + b*rel + c"""
        A = np.column_stack([x**2, x, np.ones_like(x)])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c = params

        def predict(rel):
            return a * rel**2 + b * rel + c

        return predict, params, {'model': 'quadratic', 'a': float(a), 'b': float(b), 'c': float(c)}

    @staticmethod
    def fit_log(x: np.ndarray, y: np.ndarray) -> Tuple[callable, np.ndarray, Dict]:
        """Logarithmic: depth = a*log(rel) + b"""
        x_safe = np.clip(x, 1e-6, None)
        A = np.column_stack([np.log(x_safe), np.ones_like(x)])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = params

        def predict(rel):
            return a * np.log(np.clip(rel, 1e-6, None)) + b

        return predict, params, {'model': 'log', 'a': float(a), 'b': float(b)}

    @staticmethod
    def fit_power(x: np.ndarray, y: np.ndarray) -> Tuple[callable, np.ndarray, Dict]:
        """Power law: depth = a * rel^b (linearized via log-log)"""
        x_safe = np.clip(x, 1e-6, None)
        y_safe = np.clip(y, 1e-6, None)

        # log(y) = log(a) + b*log(x)
        A = np.column_stack([np.log(x_safe), np.ones_like(x)])
        params, _, _, _ = np.linalg.lstsq(A, np.log(y_safe), rcond=None)
        b, log_a = params
        a = np.exp(log_a)

        def predict(rel):
            return a * np.power(np.clip(rel, 1e-6, None), b)

        return predict, np.array([a, b]), {'model': 'power', 'a': float(a), 'b': float(b)}


def compute_region_alignment(
    relative_depth: np.ndarray,
    lidar_depth: np.ndarray,
    lidar_mask: np.ndarray,
    region_mask: np.ndarray,
    min_points: int = 3,
    use_ransac: bool = True,
    fit_model: str = "auto",
) -> Optional[Tuple[float, float, Dict]]:
    """
    Compute depth alignment for a single region.

    Args:
        relative_depth: (H, W) DA3 relative depth
        lidar_depth: (H, W) sparse LiDAR depth
        lidar_mask: (H, W) valid LiDAR pixels
        region_mask: (H, W) boolean mask for this region
        min_points: minimum LiDAR points required
        use_ransac: use RANSAC for robust fitting (handles outliers)
        fit_model: fitting model - "affine", "inverse", "quadratic", "log", "power", or "auto"

    Returns:
        (scale, shift, stats) or None if not enough points
        Note: For non-affine models, scale/shift are dummy values; use stats['predict_fn']
    """
    # Get LiDAR anchors in this region
    anchors = region_mask & lidar_mask
    n_anchors = anchors.sum()

    if n_anchors < min_points:
        return None

    # Get values at anchor points
    rel_values = relative_depth[anchors]
    lid_values = lidar_depth[anchors]

    # Filter invalid values
    valid = (rel_values > 0) & (lid_values > 0) & np.isfinite(rel_values) & np.isfinite(lid_values)
    if valid.sum() < min_points:
        return None

    rel_values = rel_values[valid]
    lid_values = lid_values[valid]

    # Choose fitting model
    if fit_model == "auto":
        # Try multiple models and pick best by RMSE
        best_rmse = float('inf')
        best_result = None

        for model_name in ["affine", "inverse", "quadratic"]:
            try:
                result = _fit_model(rel_values, lid_values, model_name, use_ransac)
                if result is not None and result[2]['rmse'] < best_rmse:
                    best_rmse = result[2]['rmse']
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            return None

        scale, shift, stats = best_result
    else:
        result = _fit_model(rel_values, lid_values, fit_model, use_ransac)
        if result is None:
            return None
        scale, shift, stats = result

    stats['n_anchors'] = int(valid.sum())
    stats['rel_range'] = (float(rel_values.min()), float(rel_values.max()))
    stats['lid_range'] = (float(lid_values.min()), float(lid_values.max()))

    return scale, shift, stats


def _fit_model(
    rel_values: np.ndarray,
    lid_values: np.ndarray,
    model_name: str,
    use_ransac: bool,
) -> Optional[Tuple[float, float, Dict]]:
    """Fit a specific model and return results."""

    # Get fitting function
    fit_funcs = {
        'affine': DepthFitModel.fit_affine,
        'inverse': DepthFitModel.fit_inverse,
        'quadratic': DepthFitModel.fit_quadratic,
        'log': DepthFitModel.fit_log,
        'power': DepthFitModel.fit_power,
    }

    if model_name not in fit_funcs:
        model_name = 'affine'

    fit_func = fit_funcs[model_name]

    # Use RANSAC for affine model
    if use_ransac and model_name == 'affine' and len(rel_values) >= 10:
        scale, shift, inlier_mask = _ransac_affine_fit(rel_values, lid_values)
        n_inliers = inlier_mask.sum()

        # Compute residual
        predicted = scale * rel_values + shift
        residual = lid_values - predicted
        rmse = np.sqrt(np.mean(residual ** 2))

        stats = {
            'model': 'affine',
            'n_inliers': int(n_inliers),
            'scale': float(scale),
            'shift': float(shift),
            'rmse': float(rmse),
            'a': float(scale),
            'b': float(shift),
        }

        # Create predict function
        def predict(rel):
            return scale * rel + shift
        stats['predict_fn'] = predict

        return scale, shift, stats

    # Standard fitting
    try:
        predict_fn, params, model_info = fit_func(rel_values, lid_values)
    except Exception:
        return None

    # Compute residual
    predicted = predict_fn(rel_values)
    residual = lid_values - predicted
    rmse = np.sqrt(np.mean(residual ** 2))

    # For compatibility, extract scale/shift for affine
    if model_name == 'affine':
        scale, shift = params[0], params[1]
    else:
        # Use dummy values; actual prediction uses predict_fn
        scale, shift = 1.0, 0.0

    stats = {
        **model_info,
        'rmse': float(rmse),
        'n_inliers': len(rel_values),
        'predict_fn': predict_fn,
    }

    return scale, shift, stats


def _ransac_affine_fit(
    x: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 100,
    threshold: float = 2.0,
) -> Tuple[float, float, np.ndarray]:
    """RANSAC-based affine fitting for robust scale/shift estimation."""
    best_scale, best_shift = 1.0, 0.0
    best_inliers = np.zeros(len(x), dtype=bool)

    for _ in range(n_iterations):
        # Random sample 2 points
        idx = np.random.choice(len(x), 2, replace=False)
        x_sample, y_sample = x[idx], y[idx]

        # Fit line
        if abs(x_sample[1] - x_sample[0]) < 1e-6:
            continue
        scale = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0])
        shift = y_sample[0] - scale * x_sample[0]

        # Count inliers
        pred = scale * x + shift
        errors = np.abs(y - pred)
        inliers = errors < threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_scale, best_shift = scale, shift

    # Refit with all inliers
    if best_inliers.sum() >= 2:
        A = np.column_stack([x[best_inliers], np.ones(best_inliers.sum())])
        result = np.linalg.lstsq(A, y[best_inliers], rcond=None)
        best_scale, best_shift = result[0]

    return best_scale, best_shift, best_inliers


def align_depth_by_regions(
    relative_depth: np.ndarray,
    region_labels: np.ndarray,
    lidar_depth: np.ndarray,
    lidar_mask: np.ndarray,
    min_anchors_per_region: int = 3,
    fallback_to_global: bool = True,
    propagate_to_empty: bool = True,
    depth_stratified: bool = True,
    depth_bins: List[float] = None,
    fit_model: str = "auto",
) -> RegionAlignmentResult:
    """
    Align relative depth to metric depth using per-region transforms.

    Args:
        relative_depth: (H, W) DA3 relative depth
        region_labels: (H, W) integer region labels (0 = unlabeled)
        lidar_depth: (H, W) sparse LiDAR depth
        lidar_mask: (H, W) valid LiDAR pixels
        min_anchors_per_region: minimum LiDAR points per region
        fallback_to_global: use global alignment for regions without anchors
        propagate_to_empty: propagate alignment from neighbors to empty regions
        depth_stratified: split large regions by depth range for better near-field alignment
        depth_bins: depth boundaries for stratification (default: [0, 10, 25, 50, 100, 200])
        fit_model: fitting model - "affine", "inverse", "quadratic", "auto" (pick best)

    Returns:
        RegionAlignmentResult
    """
    H, W = relative_depth.shape
    aligned_depth = np.zeros((H, W), dtype=np.float32)
    region_params = {}
    region_stats = {}

    if depth_bins is None:
        depth_bins = [0, 10, 25, 50, 100, 200]

    # Get unique regions
    unique_regions = np.unique(region_labels)
    unique_regions = unique_regions[unique_regions > 0]  # Exclude background

    print(f"  Aligning {len(unique_regions)} regions...")

    # Compute global alignment as fallback
    global_scale, global_shift = 1.0, 0.0
    global_predict_fn = None
    if fallback_to_global:
        global_result = compute_region_alignment(
            relative_depth, lidar_depth, lidar_mask,
            np.ones((H, W), dtype=bool), min_points=10, fit_model=fit_model
        )
        if global_result is not None:
            global_scale, global_shift, global_stats = global_result
            global_predict_fn = global_stats.get('predict_fn')
            model_name = global_stats.get('model', 'affine')
            print(f"    Global fallback ({model_name}): scale={global_scale:.4f}, shift={global_shift:.4f}")

    # Process each region
    regions_aligned = 0
    regions_fallback = 0
    regions_stratified = 0

    for region_id in unique_regions:
        region_mask = region_labels == region_id
        region_area = region_mask.sum()

        # Check if depth stratification should be applied
        # Apply to large regions (likely ground) with sufficient LiDAR coverage
        lidar_in_region = (region_mask & lidar_mask).sum()
        use_stratified = (
            depth_stratified and
            region_area > 5000 and  # Large region
            lidar_in_region > 50    # Enough LiDAR points
        )

        if use_stratified:
            # Depth-stratified alignment for large regions (e.g., ground plane)
            aligned_region, stats = _stratified_region_alignment(
                relative_depth, lidar_depth, lidar_mask,
                region_mask, depth_bins, min_anchors_per_region,
                global_scale, global_shift
            )
            aligned_depth[region_mask] = aligned_region[region_mask]
            region_params[region_id] = (stats.get('avg_scale', global_scale),
                                        stats.get('avg_shift', global_shift))
            region_stats[region_id] = stats
            regions_stratified += 1
            regions_aligned += 1
        else:
            # Standard single-scale alignment
            result = compute_region_alignment(
                relative_depth, lidar_depth, lidar_mask,
                region_mask, min_points=min_anchors_per_region,
                fit_model=fit_model
            )

            if result is not None:
                scale, shift, stats = result
                regions_aligned += 1

                # Use predict_fn if available (for non-linear models)
                if 'predict_fn' in stats:
                    aligned_depth[region_mask] = stats['predict_fn'](relative_depth[region_mask])
                else:
                    aligned_depth[region_mask] = scale * relative_depth[region_mask] + shift
            else:
                # Fallback to global
                scale, shift = global_scale, global_shift
                stats = {'n_anchors': 0, 'fallback': 'global'}
                regions_fallback += 1
                if global_predict_fn is not None:
                    aligned_depth[region_mask] = global_predict_fn(relative_depth[region_mask])
                else:
                    aligned_depth[region_mask] = scale * relative_depth[region_mask] + shift

            region_params[region_id] = (scale, shift)
            # Remove predict_fn before storing (not serializable)
            stats_clean = {k: v for k, v in stats.items() if k != 'predict_fn'}
            region_stats[region_id] = stats_clean

    # Handle unlabeled pixels (region_labels == 0)
    unlabeled = region_labels == 0
    if unlabeled.any():
        if global_predict_fn is not None:
            aligned_depth[unlabeled] = global_predict_fn(relative_depth[unlabeled])
        else:
            aligned_depth[unlabeled] = global_scale * relative_depth[unlabeled] + global_shift

    # Propagate to regions without anchors from neighbors
    if propagate_to_empty and regions_fallback > 0:
        aligned_depth = _propagate_from_neighbors(
            aligned_depth, region_labels, region_params, relative_depth
        )

    # Compute coverage
    valid_aligned = aligned_depth > 0
    coverage = valid_aligned.sum() / (H * W)

    print(f"    Aligned: {regions_aligned} (stratified: {regions_stratified}), Fallback: {regions_fallback}, Coverage: {coverage:.1%}")

    return RegionAlignmentResult(
        aligned_depth=aligned_depth,
        region_masks=region_labels,
        region_params=region_params,
        region_stats=region_stats,
        num_regions=len(unique_regions),
        coverage=coverage,
    )


def _stratified_region_alignment(
    relative_depth: np.ndarray,
    lidar_depth: np.ndarray,
    lidar_mask: np.ndarray,
    region_mask: np.ndarray,
    depth_bins: List[float],
    min_points: int,
    global_scale: float,
    global_shift: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Depth-stratified alignment for large regions (e.g., ground plane).

    Splits region by LiDAR depth ranges and computes separate scale/shift
    for each depth stratum. This fixes the near-field ground depression issue
    caused by DA3's non-linear relative depth.

    Args:
        relative_depth: (H, W) DA3 relative depth
        lidar_depth: (H, W) sparse LiDAR depth
        lidar_mask: (H, W) valid LiDAR pixels
        region_mask: (H, W) mask for this region
        depth_bins: depth boundaries [0, 10, 25, 50, 100, ...]
        min_points: minimum points per stratum
        global_scale, global_shift: fallback parameters

    Returns:
        aligned_region: (H, W) aligned depth (only region_mask pixels are valid)
        stats: dict with per-stratum statistics
    """
    H, W = relative_depth.shape
    aligned_region = np.zeros((H, W), dtype=np.float32)

    # Get LiDAR anchors in this region
    anchors = region_mask & lidar_mask
    if anchors.sum() < min_points:
        # Fallback to global
        aligned_region[region_mask] = global_scale * relative_depth[region_mask] + global_shift
        return aligned_region, {'fallback': 'global', 'n_strata': 0}

    # Compute stratum params using LiDAR depth to determine boundaries
    stratum_params = {}  # depth_bin_idx -> (scale, shift)

    for i in range(len(depth_bins) - 1):
        d_min, d_max = depth_bins[i], depth_bins[i + 1]

        # Get anchors in this depth range
        stratum_anchors = anchors & (lidar_depth >= d_min) & (lidar_depth < d_max)
        n_anchors = stratum_anchors.sum()

        if n_anchors >= min_points:
            rel_vals = relative_depth[stratum_anchors]
            lid_vals = lidar_depth[stratum_anchors]

            valid = (rel_vals > 0) & np.isfinite(rel_vals)
            if valid.sum() >= min_points:
                rel_vals, lid_vals = rel_vals[valid], lid_vals[valid]

                # RANSAC fit for robustness
                if len(rel_vals) >= 10:
                    scale, shift, _ = _ransac_affine_fit(rel_vals, lid_vals)
                else:
                    A = np.column_stack([rel_vals, np.ones_like(rel_vals)])
                    result = np.linalg.lstsq(A, lid_vals, rcond=None)
                    scale, shift = result[0]

                stratum_params[i] = (scale, shift, n_anchors)

    # If no strata computed, use global
    if len(stratum_params) == 0:
        aligned_region[region_mask] = global_scale * relative_depth[region_mask] + global_shift
        return aligned_region, {'fallback': 'global', 'n_strata': 0}

    # Apply per-stratum alignment to all pixels in region
    # Use the relative depth to determine which stratum each pixel belongs to
    # by finding the stratum whose fitted depth is closest

    region_rel = relative_depth[region_mask]
    region_aligned = np.zeros_like(region_rel)

    # For each pixel, find best stratum based on relative depth interpolation
    # Strategy: compute predicted depth for each stratum and use weighted blend

    for i, (scale, shift, n_pts) in stratum_params.items():
        d_min, d_max = depth_bins[i], depth_bins[i + 1]
        pred = scale * region_rel + shift

        # Pixels whose predicted depth falls in this stratum
        in_stratum = (pred >= d_min) & (pred < d_max)
        region_aligned[in_stratum] = pred[in_stratum]

    # Handle pixels not covered by any stratum (use nearest or global)
    uncovered = region_aligned == 0
    if uncovered.any():
        # Use global fallback for uncovered pixels
        region_aligned[uncovered] = global_scale * region_rel[uncovered] + global_shift

    aligned_region[region_mask] = region_aligned

    # Compute stats
    scales = [p[0] for p in stratum_params.values()]
    shifts = [p[1] for p in stratum_params.values()]
    stats = {
        'n_strata': len(stratum_params),
        'strata_depths': [(depth_bins[i], depth_bins[i+1]) for i in stratum_params.keys()],
        'avg_scale': float(np.mean(scales)),
        'avg_shift': float(np.mean(shifts)),
        'stratified': True,
    }

    return aligned_region, stats


def _propagate_from_neighbors(
    aligned_depth: np.ndarray,
    region_labels: np.ndarray,
    region_params: Dict[int, Tuple[float, float]],
    relative_depth: np.ndarray,
) -> np.ndarray:
    """
    Propagate alignment parameters from neighboring regions.

    For regions that fell back to global alignment, try to use
    parameters from adjacent regions that have proper anchors.
    """
    # Find regions that need propagation (those with n_anchors == 0)
    # This is a simple heuristic - could be improved

    # For now, just return as-is
    # TODO: implement neighbor-based propagation
    return aligned_depth


class RegionWiseDepthAligner:
    """
    Main class for region-wise depth alignment.

    Workflow:
        1. DA3 inference -> relative depth
        2. SAM segment everything -> region masks
        3. LiDAR projection -> sparse anchors
        4. Per-region ax+b alignment
        5. Compose final metric depth
    """

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        device: str = "cuda",
        min_anchors_per_region: int = 3,
        sam_points_per_side: int = 32,
        fit_model: str = "auto",
    ):
        """
        Initialize region-wise depth aligner.

        Args:
            sam_checkpoint: Path to SAM checkpoint
            device: Device for SAM
            min_anchors_per_region: Minimum LiDAR anchors per region
            sam_points_per_side: SAM grid density
            fit_model: Fitting model - "affine", "inverse", "quadratic", "auto"
        """
        self.device = device
        self.min_anchors = min_anchors_per_region
        self.fit_model = fit_model

        # Initialize SAM auto segmenter
        self.segmenter = SAMAutoSegmenter(
            checkpoint_path=sam_checkpoint,
            device=device,
            points_per_side=sam_points_per_side,
        )

    @property
    def sam_available(self) -> bool:
        return self.segmenter.is_available

    def align(
        self,
        image: np.ndarray,
        relative_depth: np.ndarray,
        lidar_points: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> RegionAlignmentResult:
        """
        Perform region-wise depth alignment.

        Args:
            image: RGB image (H, W, 3)
            relative_depth: DA3 relative depth (H, W)
            lidar_points: LiDAR points (N, 3) in world coordinates
            intrinsics: Camera intrinsics (3, 3)
            extrinsics: World-to-camera transform (4, 4)

        Returns:
            RegionAlignmentResult with aligned metric depth
        """
        H, W = relative_depth.shape

        # Step 1: SAM segmentation
        print("  [1/3] SAM segmentation...")
        if image.shape[:2] != (H, W):
            image_resized = cv2.resize(image, (W, H))
        else:
            image_resized = image

        region_labels, masks_info = self.segmenter.segment(image_resized)
        print(f"    Found {len(masks_info)} regions")

        # Step 2: Project LiDAR to image
        print("  [2/3] LiDAR projection...")
        uv, depths, _ = project_lidar_to_image(
            lidar_points, intrinsics, extrinsics, (H, W)
        )
        lidar_depth, lidar_mask = create_sparse_depth_map(uv, depths, (H, W))
        print(f"    {lidar_mask.sum():,} LiDAR anchors")

        # Step 3: Per-region alignment
        print("  [3/3] Region alignment...")
        result = align_depth_by_regions(
            relative_depth=relative_depth,
            region_labels=region_labels,
            lidar_depth=lidar_depth,
            lidar_mask=lidar_mask,
            min_anchors_per_region=self.min_anchors,
            fit_model=self.fit_model,
        )

        return result

    def visualize(
        self,
        image: np.ndarray,
        result: RegionAlignmentResult,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize region alignment results.

        Args:
            image: RGB image
            result: Alignment result
            output_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt

        H, W = result.aligned_depth.shape

        # Resize image if needed
        if image.shape[:2] != (H, W):
            image = cv2.resize(image, (W, H))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RGB
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("RGB Image")
        axes[0, 0].axis('off')

        # Region labels
        region_vis = plt.cm.tab20(result.region_masks % 20)[:, :, :3]
        axes[0, 1].imshow(region_vis)
        axes[0, 1].set_title(f"SAM Regions ({result.num_regions})")
        axes[0, 1].axis('off')

        # Aligned depth
        depth_vis = result.aligned_depth.copy()
        vmax = np.percentile(depth_vis[depth_vis > 0], 95) if (depth_vis > 0).any() else 100
        axes[1, 0].imshow(depth_vis, cmap='turbo', vmin=0, vmax=vmax)
        axes[1, 0].set_title(f"Aligned Depth (coverage: {result.coverage:.1%})")
        axes[1, 0].axis('off')

        # Per-region scale visualization
        scale_map = np.zeros((H, W), dtype=np.float32)
        for region_id, (scale, shift) in result.region_params.items():
            mask = result.region_masks == region_id
            scale_map[mask] = scale

        axes[1, 1].imshow(scale_map, cmap='coolwarm', vmin=0.5, vmax=2.0)
        axes[1, 1].set_title("Per-Region Scale")
        axes[1, 1].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization: {output_path}")

        # Convert to image array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return vis_image


def region_wise_depth_alignment(
    image: np.ndarray,
    relative_depth: np.ndarray,
    lidar_points: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    sam_checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> RegionAlignmentResult:
    """
    Convenience function for region-wise depth alignment.

    Args:
        image: RGB image (H, W, 3)
        relative_depth: DA3 relative depth (H, W)
        lidar_points: LiDAR points (N, 3)
        intrinsics: Camera intrinsics (3, 3)
        extrinsics: World-to-camera (4, 4)
        sam_checkpoint: Path to SAM checkpoint
        device: Device for SAM

    Returns:
        RegionAlignmentResult
    """
    aligner = RegionWiseDepthAligner(
        sam_checkpoint=sam_checkpoint,
        device=device,
    )

    return aligner.align(
        image=image,
        relative_depth=relative_depth,
        lidar_points=lidar_points,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
