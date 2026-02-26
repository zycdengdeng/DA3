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
SAM (Segment Anything Model) integration for precise object segmentation.

Uses 3D bbox projections as prompts to get pixel-accurate masks instead of
coarse convex hull masks.

Supports:
- SAM (original)
- SAM2 (faster, better quality)
- Mobile SAM (lightweight)

Falls back to convex hull if SAM is not available.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

# Try to import SAM variants
SAM_AVAILABLE = False
SAM2_AVAILABLE = False
MOBILE_SAM_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    pass

try:
    from mobile_sam import sam_model_registry as mobile_sam_registry
    from mobile_sam import SamPredictor as MobileSamPredictor
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    pass


def get_available_sam_backend() -> Optional[str]:
    """Get the best available SAM backend."""
    if SAM2_AVAILABLE:
        return "sam2"
    elif SAM_AVAILABLE:
        return "sam"
    elif MOBILE_SAM_AVAILABLE:
        return "mobile_sam"
    return None


class SAMSegmenter:
    """
    SAM-based object segmentation using bbox prompts.

    Given a 2D bounding box (from 3D bbox projection), uses SAM to get
    a precise pixel-accurate mask of the object.
    """

    def __init__(
        self,
        backend: str = "auto",
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize SAM segmenter.

        Args:
            backend: "sam", "sam2", "mobile_sam", or "auto" (picks best available)
            model_path: Path to model checkpoint (optional, will download if needed)
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.predictor = None
        self.backend = None
        self._image_set = False

        if backend == "auto":
            backend = get_available_sam_backend()
            if backend is None:
                print("WARNING: No SAM backend available. Install with:")
                print("  pip install segment-anything  # SAM original")
                print("  pip install sam2              # SAM2 (recommended)")
                print("  pip install mobile-sam        # Mobile SAM (lightweight)")
                return

        self.backend = backend
        self._init_predictor(model_path)

    def _init_predictor(self, model_path: Optional[str]):
        """Initialize the SAM predictor."""
        if self.backend == "sam2" and SAM2_AVAILABLE:
            if model_path is None:
                # Default SAM2 model
                model_path = "facebook/sam2-hiera-large"
            try:
                model = build_sam2(model_path)
                self.predictor = SAM2ImagePredictor(model)
                self.predictor.model.to(self.device)
                print(f"Loaded SAM2 model: {model_path}")
            except Exception as e:
                print(f"Failed to load SAM2: {e}")
                self.backend = None

        elif self.backend == "sam" and SAM_AVAILABLE:
            if model_path is None:
                model_type = "vit_h"
                # Try common paths
                import os
                for path in [
                    "sam_vit_h_4b8939.pth",
                    os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth"),
                    "/tmp/sam_vit_h_4b8939.pth",
                ]:
                    if os.path.exists(path):
                        model_path = path
                        break
                if model_path is None:
                    print("SAM checkpoint not found. Download from:")
                    print("https://github.com/facebookresearch/segment-anything")
                    self.backend = None
                    return
            else:
                model_type = "vit_h"  # Assume vit_h if custom path

            try:
                sam = sam_model_registry[model_type](checkpoint=model_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                print(f"Loaded SAM model: {model_path}")
            except Exception as e:
                print(f"Failed to load SAM: {e}")
                self.backend = None

        elif self.backend == "mobile_sam" and MOBILE_SAM_AVAILABLE:
            if model_path is None:
                model_type = "vit_t"
                import os
                for path in [
                    "mobile_sam.pt",
                    os.path.expanduser("~/.cache/mobile_sam/mobile_sam.pt"),
                ]:
                    if os.path.exists(path):
                        model_path = path
                        break
                if model_path is None:
                    print("Mobile SAM checkpoint not found")
                    self.backend = None
                    return
            else:
                model_type = "vit_t"

            try:
                sam = mobile_sam_registry[model_type](checkpoint=model_path)
                sam.to(device=self.device)
                self.predictor = MobileSamPredictor(sam)
                print(f"Loaded Mobile SAM: {model_path}")
            except Exception as e:
                print(f"Failed to load Mobile SAM: {e}")
                self.backend = None

    @property
    def is_available(self) -> bool:
        """Check if SAM is available and loaded."""
        return self.predictor is not None

    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.

        Args:
            image: RGB image (H, W, 3)
        """
        if not self.is_available:
            return

        self.predictor.set_image(image)
        self._image_set = True

    def segment_bbox(
        self,
        bbox_2d: np.ndarray,
        multimask_output: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Segment object given 2D bounding box.

        Args:
            bbox_2d: (4,) array [x1, y1, x2, y2] bounding box
            multimask_output: Return multiple masks (for ambiguous cases)

        Returns:
            mask: (H, W) boolean mask, or None if failed
        """
        if not self.is_available or not self._image_set:
            return None

        try:
            masks, scores, _ = self.predictor.predict(
                box=bbox_2d,
                multimask_output=multimask_output,
            )

            if multimask_output:
                # Return the mask with highest score
                best_idx = np.argmax(scores)
                return masks[best_idx].astype(bool)
            else:
                return masks[0].astype(bool)

        except Exception as e:
            print(f"SAM segmentation failed: {e}")
            return None

    def segment_with_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        bbox_2d: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Segment object using point prompts (positive/negative points).

        Args:
            point_coords: (N, 2) array of point coordinates
            point_labels: (N,) array of labels (1=foreground, 0=background)
            bbox_2d: Optional bounding box to constrain segmentation

        Returns:
            mask: (H, W) boolean mask
        """
        if not self.is_available or not self._image_set:
            return None

        try:
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox_2d,
                multimask_output=True,
            )

            best_idx = np.argmax(scores)
            return masks[best_idx].astype(bool)

        except Exception as e:
            print(f"SAM point segmentation failed: {e}")
            return None


def project_bbox_to_2d_rect(
    corners_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Project 3D bbox corners to 2D and return axis-aligned bounding box.

    Args:
        corners_3d: (8, 3) 3D bbox corners in world coordinates
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transform
        image_hw: (H, W) image size

    Returns:
        bbox_2d: (4,) [x1, y1, x2, y2] axis-aligned bounding box, or None
    """
    H, W = image_hw

    # Transform to camera coordinates
    corners_homo = np.hstack([corners_3d, np.ones((8, 1))])
    corners_cam = (extrinsics @ corners_homo.T).T[:, :3]

    # Filter corners behind camera
    valid_depth = corners_cam[:, 2] > 0.1
    if not np.any(valid_depth):
        return None

    # Project to image
    corners_img = (intrinsics @ corners_cam.T).T
    corners_2d = corners_img[:, :2] / corners_img[:, 2:3]

    # Only use corners in front of camera
    corners_2d_valid = corners_2d[valid_depth]

    if len(corners_2d_valid) < 2:
        return None

    # Get axis-aligned bounding box
    x1 = max(0, int(corners_2d_valid[:, 0].min()))
    y1 = max(0, int(corners_2d_valid[:, 1].min()))
    x2 = min(W - 1, int(corners_2d_valid[:, 0].max()))
    y2 = min(H - 1, int(corners_2d_valid[:, 1].max()))

    if x2 <= x1 or y2 <= y1:
        return None

    return np.array([x1, y1, x2, y2])


def get_object_center_point(
    corners_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Get the projected center point of a 3D bbox.

    Args:
        corners_3d: (8, 3) 3D bbox corners
        intrinsics: Camera intrinsics
        extrinsics: World-to-camera transform

    Returns:
        center_2d: (2,) projected center point [x, y]
    """
    # Compute 3D center
    center_3d = corners_3d.mean(axis=0)

    # Transform to camera coordinates
    center_homo = np.array([*center_3d, 1.0])
    center_cam = (extrinsics @ center_homo)[:3]

    if center_cam[2] <= 0.1:
        return None

    # Project to image
    center_img = intrinsics @ center_cam
    center_2d = center_img[:2] / center_img[2]

    return center_2d


class SAMGuidedMaskGenerator:
    """
    Generate object masks using SAM with 3D bbox guidance.

    Combines 3D bbox projection (for prompt) with SAM (for precise mask).
    Falls back to convex hull if SAM is unavailable.
    """

    def __init__(
        self,
        sam_backend: str = "auto",
        sam_model_path: Optional[str] = None,
        device: str = "cuda",
        use_point_prompts: bool = True,
    ):
        """
        Args:
            sam_backend: SAM backend to use
            sam_model_path: Path to SAM model
            device: Device for SAM
            use_point_prompts: Use projected LiDAR points as additional prompts
        """
        self.segmenter = SAMSegmenter(
            backend=sam_backend,
            model_path=sam_model_path,
            device=device,
        )
        self.use_point_prompts = use_point_prompts
        self._current_image = None

    @property
    def sam_available(self) -> bool:
        return self.segmenter.is_available

    def set_image(self, image: np.ndarray):
        """Set image for mask generation."""
        self._current_image = image
        if self.sam_available:
            self.segmenter.set_image(image)

    def generate_mask(
        self,
        bbox_3d_corners: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_hw: Tuple[int, int],
        lidar_points_2d: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate object mask from 3D bbox.

        Args:
            bbox_3d_corners: (8, 3) 3D bbox corners
            intrinsics: Camera intrinsics
            extrinsics: World-to-camera transform
            image_hw: (H, W) image size
            lidar_points_2d: Optional (N, 2) projected LiDAR points on object

        Returns:
            mask: (H, W) boolean mask
        """
        H, W = image_hw

        if self.sam_available:
            # Use SAM for precise mask
            bbox_2d = project_bbox_to_2d_rect(
                bbox_3d_corners, intrinsics, extrinsics, image_hw
            )
            if bbox_2d is None:
                return None

            if self.use_point_prompts and lidar_points_2d is not None and len(lidar_points_2d) > 0:
                # Use LiDAR points as positive prompts
                # Add center point
                center = get_object_center_point(bbox_3d_corners, intrinsics, extrinsics)
                if center is not None:
                    point_coords = np.vstack([center, lidar_points_2d[:5]])  # Limit points
                else:
                    point_coords = lidar_points_2d[:5]

                point_labels = np.ones(len(point_coords))  # All foreground

                mask = self.segmenter.segment_with_points(
                    point_coords=point_coords.astype(np.float32),
                    point_labels=point_labels.astype(np.int32),
                    bbox_2d=bbox_2d.astype(np.float32),
                )
            else:
                # Use bbox prompt only
                mask = self.segmenter.segment_bbox(bbox_2d.astype(np.float32))

            if mask is not None:
                return mask

        # Fallback to convex hull
        return self._generate_convex_hull_mask(
            bbox_3d_corners, intrinsics, extrinsics, image_hw
        )

    def _generate_convex_hull_mask(
        self,
        corners_3d: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Fallback: generate convex hull mask from 3D bbox corners."""
        from scipy.spatial import ConvexHull

        H, W = image_hw

        # Transform to camera coordinates
        corners_homo = np.hstack([corners_3d, np.ones((8, 1))])
        corners_cam = (extrinsics @ corners_homo.T).T[:, :3]

        # Filter corners behind camera
        valid_depth = corners_cam[:, 2] > 0.1
        if not np.any(valid_depth):
            return None

        # Project to image
        corners_img = (intrinsics @ corners_cam.T).T
        corners_2d = corners_img[:, :2] / corners_img[:, 2:3]

        # Clip to image bounds
        corners_2d[:, 0] = np.clip(corners_2d[:, 0], -W, 2*W)
        corners_2d[:, 1] = np.clip(corners_2d[:, 1], -H, 2*H)

        corners_2d_valid = corners_2d[valid_depth]
        if len(corners_2d_valid) < 3:
            return None

        try:
            hull = ConvexHull(corners_2d_valid)
            hull_points = corners_2d_valid[hull.vertices].astype(np.int32)

            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_points, 1)
            return mask.astype(bool)
        except Exception:
            return None


def check_sam_installation():
    """Print SAM installation status and instructions."""
    print("=" * 60)
    print("SAM Installation Status")
    print("=" * 60)

    print(f"\nSAM (original):  {'✓ Available' if SAM_AVAILABLE else '✗ Not installed'}")
    print(f"SAM2:            {'✓ Available' if SAM2_AVAILABLE else '✗ Not installed'}")
    print(f"Mobile SAM:      {'✓ Available' if MOBILE_SAM_AVAILABLE else '✗ Not installed'}")

    best = get_available_sam_backend()
    if best:
        print(f"\nBest available backend: {best}")
    else:
        print("\n" + "-" * 60)
        print("Installation instructions:")
        print("-" * 60)
        print("\n# Option 1: SAM2 (recommended - best quality)")
        print("pip install sam2")
        print("\n# Option 2: Original SAM")
        print("pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("# Then download checkpoint:")
        print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("\n# Option 3: Mobile SAM (lightweight, fast)")
        print("pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

    print("=" * 60)


if __name__ == "__main__":
    check_sam_installation()
