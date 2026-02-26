#!/usr/bin/env python3
"""
Multi-view DA3 depth estimation and PLY point cloud fusion.
Uses calibrated camera poses for accurate point cloud fusion.
Supports LiDAR alignment to convert relative depth to metric depth.

Usage:
    # Relative depth with LiDAR alignment (recommended)
    python examples/da3_multiview_pointcloud.py \
        --data_root /mnt/car_road_data_TianJin \
        --scene 001_car0325_road0327_t1 \
        --timestamp 1742877031036 \
        --lidar_align \
        --output_dir ./output_da3_ply

    # Relative depth only (no alignment)
    python examples/da3_multiview_pointcloud.py \
        --scene 001_car0325_road0327_t1 \
        --timestamp 1742877031036 \
        --output_dir ./output_da3_ply
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.datasets.car_road_dataset import CarRoadDatasetLoader
from depth_anything_3.utils.lidar_alignment import (
    adaptive_depth_fusion,
    lidar_anchored_depth,
)
from depth_anything_3.utils.depth_completion import (
    DepthCompleter,
    create_sparse_depth_from_lidar,
)
from depth_anything_3.utils.semantic_depth_completion import (
    SemanticDepthCompleter,
    load_annotations,
)


def project_lidar_to_camera(
    lidar_points: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    img_width: int,
    img_height: int,
) -> tuple:
    """
    Project LiDAR points to camera image plane.

    Args:
        lidar_points: (N, 3) LiDAR points in world coordinates
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) world-to-camera transformation
        img_width: image width
        img_height: image height

    Returns:
        uv: (M, 2) pixel coordinates of valid projections
        depths: (M,) depth values at those pixels
    """
    # Transform to camera coordinates
    N = lidar_points.shape[0]
    pts_homo = np.hstack([lidar_points, np.ones((N, 1))])  # (N, 4)
    pts_cam = (extrinsics @ pts_homo.T).T[:, :3]  # (N, 3)

    # Filter points behind the camera
    valid_depth = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid_depth]

    if len(pts_cam) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    # Project to image plane
    pts_2d = (intrinsics @ pts_cam.T).T  # (M, 3)
    uv = pts_2d[:, :2] / pts_2d[:, 2:3]  # (M, 2)
    depths = pts_cam[:, 2]  # (M,)

    # Filter points outside image
    valid_uv = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_height)
    )
    uv = uv[valid_uv]
    depths = depths[valid_uv]

    return uv, depths


def align_depth_with_lidar(
    pred_depth: np.ndarray,
    lidar_uv: np.ndarray,
    lidar_depths: np.ndarray,
    method: str = "scale",
    depth_range: tuple = None,
) -> tuple:
    """
    Align predicted depth with LiDAR ground truth.

    Args:
        pred_depth: (H, W) predicted relative depth
        lidar_uv: (M, 2) pixel coordinates of LiDAR projections
        lidar_depths: (M,) LiDAR depth values
        method:
            - "scale": scale only (recommended for DA3)
            - "scale_shift": affine alignment (may cause issues)
            - "ransac": RANSAC robust fitting
            - "near_priority": prioritize near points for fitting
        depth_range: (min, max) only use LiDAR points in this depth range

    Returns:
        aligned_depth: (H, W) aligned depth map
        scale: fitted scale factor
        shift: fitted shift (0 if method="scale")
    """
    if len(lidar_uv) < 10:
        print(f"  Warning: Only {len(lidar_uv)} LiDAR points, using default scale")
        return pred_depth, 1.0, 0.0

    # Sample predicted depth at LiDAR locations
    u = lidar_uv[:, 0].astype(int)
    v = lidar_uv[:, 1].astype(int)
    pred_at_lidar = pred_depth[v, u]

    # Filter out invalid predictions
    valid = pred_at_lidar > 0

    # Apply depth range filter if specified
    if depth_range is not None:
        valid = valid & (lidar_depths >= depth_range[0]) & (lidar_depths <= depth_range[1])
        print(f"  Using depth range [{depth_range[0]:.1f}, {depth_range[1]:.1f}]m")

    pred_at_lidar = pred_at_lidar[valid]
    lidar_d = lidar_depths[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    if len(pred_at_lidar) < 10:
        print(f"  Warning: Only {len(pred_at_lidar)} valid matches, using default scale")
        return pred_depth, 1.0, 0.0

    print(f"  Using {len(pred_at_lidar)} LiDAR points for alignment")
    print(f"  LiDAR depth range: [{lidar_d.min():.2f}, {lidar_d.max():.2f}]m")

    if method == "scale_shift":
        # Solve: lidar_d = scale * pred_at_lidar + shift
        A = np.vstack([pred_at_lidar, np.ones_like(pred_at_lidar)]).T
        result, _, _, _ = np.linalg.lstsq(A, lidar_d, rcond=None)
        scale, shift = result

    elif method == "scale":
        # Scale only (recommended for DA3 relative depth)
        scale = np.median(lidar_d / pred_at_lidar)
        shift = 0.0

    elif method == "ransac":
        # RANSAC robust fitting (scale only)
        n_iters = 100
        best_scale = 1.0
        best_inliers = 0
        threshold = 0.1  # 10% relative error

        for _ in range(n_iters):
            # Random sample
            idx = np.random.randint(0, len(pred_at_lidar))
            scale_sample = lidar_d[idx] / pred_at_lidar[idx]

            # Count inliers
            pred_scaled = pred_at_lidar * scale_sample
            rel_error = np.abs(pred_scaled - lidar_d) / lidar_d
            inliers = np.sum(rel_error < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_scale = scale_sample

        # Refine with all inliers
        pred_scaled = pred_at_lidar * best_scale
        rel_error = np.abs(pred_scaled - lidar_d) / lidar_d
        inlier_mask = rel_error < threshold
        if np.sum(inlier_mask) > 10:
            scale = np.median(lidar_d[inlier_mask] / pred_at_lidar[inlier_mask])
        else:
            scale = best_scale
        shift = 0.0
        print(f"  RANSAC: {best_inliers}/{len(pred_at_lidar)} inliers")

    elif method == "near_priority":
        # Weight near points more heavily (roadside scenario)
        # Near points are more reliable for DA3, far points more reliable for LiDAR
        # Use median of near+mid range points
        mid_depth = np.median(lidar_d)
        near_mask = lidar_d < mid_depth
        if np.sum(near_mask) > 20:
            scale = np.median(lidar_d[near_mask] / pred_at_lidar[near_mask])
            print(f"  Near priority: using {np.sum(near_mask)} near points (< {mid_depth:.1f}m)")
        else:
            scale = np.median(lidar_d / pred_at_lidar)
        shift = 0.0

    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply alignment
    aligned_depth = pred_depth * scale + shift
    aligned_depth = np.maximum(aligned_depth, 0)  # No negative depths

    # Calculate alignment error
    aligned_at_lidar = aligned_depth[v_valid, u_valid]
    mae = np.mean(np.abs(aligned_at_lidar - lidar_d))
    rel_mae = np.mean(np.abs(aligned_at_lidar - lidar_d) / lidar_d) * 100
    print(f"  LiDAR alignment: scale={scale:.4f}, shift={shift:.4f}")
    print(f"  MAE={mae:.4f}m, RelMAE={rel_mae:.2f}%")

    return aligned_depth, scale, shift


def depth_to_pointcloud(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    conf: np.ndarray = None,
    conf_threshold: float = 0.3,
    max_depth: float = 100.0,
    max_view_angle: float = 90.0,
    downsample: int = 1,
) -> tuple:
    """
    Convert depth map to colored point cloud in world coordinates.

    Args:
        depth: (H, W) depth map
        image: (H, W, 3) RGB image
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) world-to-camera transformation
        conf: (H, W) confidence map (optional)
        conf_threshold: Minimum confidence to keep points
        max_depth: Maximum depth to keep
        max_view_angle: Maximum angle (degrees) from optical axis.
                       Points outside this cone are filtered.
                       Set to 90 to disable. Try 50-70 for roadside.
        downsample: Downsample factor for points

    Returns:
        points: (N, 3) world coordinates
        colors: (N, 3) RGB colors (0-255)
    """
    h, w = depth.shape

    # Resize image to match depth if needed
    if image.shape[:2] != depth.shape:
        image = cv2.resize(image, (w, h))

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Downsample
    if downsample > 1:
        u = u[::downsample, ::downsample]
        v = v[::downsample, ::downsample]
        depth = depth[::downsample, ::downsample]
        image = image[::downsample, ::downsample]
        if conf is not None:
            conf = conf[::downsample, ::downsample]

    # Flatten
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    colors = image.reshape(-1, 3)

    # Create mask for valid points
    valid = (z > 0) & (z < max_depth) & np.isfinite(z)
    if conf is not None:
        conf_flat = conf.flatten()
        valid &= conf_flat > conf_threshold

    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors = colors[valid]

    # Unproject to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Filter by view angle: only keep points within max_view_angle of optical axis
    # This removes noise from peripheral regions and opposite-side cameras
    if max_view_angle < 90.0:
        lateral_dist = np.sqrt(x**2 + y**2)
        view_angle = np.degrees(np.arctan2(lateral_dist, z))
        angle_valid = view_angle <= max_view_angle

        x = x[angle_valid]
        y = y[angle_valid]
        z = z[angle_valid]
        colors = colors[angle_valid]

    # Stack to (N, 3) camera coordinates
    points_cam = np.stack([x, y, z], axis=1)

    # Transform to world coordinates
    # extrinsics is world-to-camera: p_cam = R @ p_world + t
    # So p_world = R^T @ (p_cam - t)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    points_world = (R_inv @ points_cam.T).T + t_inv

    return points_world, colors


def save_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    """Save point cloud as PLY file."""
    n_points = len(points)

    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filepath, 'w') as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"Saved PLY with {n_points:,} points to {filepath}")


def visualize_depth(depth: np.ndarray, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Visualize depth map."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d_min, d_max = np.percentile(valid, [2, 98])
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), colormap)
    return depth_color


def main():
    parser = argparse.ArgumentParser(description="Multi-view DA3 point cloud fusion")
    parser.add_argument("--data_root", type=str, default="/mnt/car_road_data_TianJin")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_da3_ply")
    parser.add_argument("--model", type=str, default="da3-giant",
                        help="Model: da3-giant, da3-large, da3nested-giant-large")
    parser.add_argument("--lidar_align", action="store_true",
                        help="Align depth with LiDAR (recommended)")
    parser.add_argument("--align_method", type=str, default="scale",
                        choices=["scale", "scale_shift", "ransac", "near_priority"],
                        help="Alignment method: scale (default, recommended), "
                             "scale_shift (may cause curvature issues), "
                             "ransac (robust), near_priority (roadside scenario)")
    parser.add_argument("--align_depth_min", type=float, default=None,
                        help="Min depth for alignment (meters)")
    parser.add_argument("--align_depth_max", type=float, default=None,
                        help="Max depth for alignment (meters)")
    parser.add_argument("--process_res", type=int, default=518)
    parser.add_argument("--downsample", type=int, default=2, help="Downsample factor")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--max_depth", type=float, default=100.0, help="Max depth (meters)")
    parser.add_argument("--max_view_angle", type=float, default=90.0,
                        help="Max view angle from optical axis (degrees). "
                             "Filters peripheral noise. Try 50-70 for roadside. Default: 90 (no filter)")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Manual depth scale")
    parser.add_argument("--adaptive_fusion", action="store_true",
                        help="Enable adaptive camera/LiDAR fusion. Uses LiDAR where it's dense, "
                             "camera where LiDAR is sparse. Improves intersection center quality.")
    parser.add_argument("--lidar_anchored", action="store_true",
                        help="Use LiDAR-anchored depth: LiDAR as ground truth, ground plane "
                             "constraint, camera only fills gaps. RECOMMENDED for best quality.")
    parser.add_argument("--use_ground_plane", action="store_true", default=True,
                        help="Enable ground plane constraint (default: True)")
    parser.add_argument("--depth_completion", type=str, default=None,
                        choices=["completionformer", "simple", "semantic"],
                        help="Use depth completion network instead of DA3. "
                             "Input: sparse LiDAR + RGB -> Output: dense metric depth. "
                             "'simple': basic interpolation, 'semantic': use 3D bbox annotations "
                             "to separate dynamic/static objects (best for roadside).")
    parser.add_argument("--completion_model", type=str, default=None,
                        help="Path to depth completion model weights")
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="Path to 3D bbox annotation JSON file (for semantic depth completion)")
    parser.add_argument("--density_sigma", type=float, default=15.0,
                        help="Density smoothing sigma for adaptive fusion (default: 15)")
    parser.add_argument("--distance_threshold", type=float, default=30.0,
                        help="Distance (m) to start boosting LiDAR weight (default: 30)")
    args = parser.parse_args()

    if args.lidar_align:
        print("LiDAR alignment enabled - will convert relative depth to metric depth")

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.scene, args.timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    loader = CarRoadDatasetLoader(args.data_root)

    # Initialize depth estimation model
    model = None
    depth_completer = None
    semantic_completer = None
    annotation_bboxes = None
    is_metric = False

    if args.depth_completion:
        # Use depth completion network (sparse LiDAR + RGB -> dense depth)
        print(f"\nUsing depth completion: {args.depth_completion}")

        if args.depth_completion == "semantic":
            # Semantic-guided depth completion using 3D bbox annotations
            if not args.annotation_path:
                # Try to find annotation automatically
                annotation_path = os.path.join(
                    args.data_root, args.scene, "road_labels",
                    "interpolation_labels", f"{args.timestamp}.json"
                )
                if not os.path.exists(annotation_path):
                    annotation_path = os.path.join(
                        args.data_root, args.scene, "road_labels",
                        "ori_labels", f"{args.timestamp}.json"
                    )
            else:
                annotation_path = args.annotation_path

            if os.path.exists(annotation_path):
                print(f"Loading annotations: {annotation_path}")
                annotation_bboxes = load_annotations(annotation_path)
                print(f"  Loaded {len(annotation_bboxes)} bounding boxes")
            else:
                print(f"WARNING: Annotation file not found: {annotation_path}")
                print("  Falling back to 'simple' depth completion")
                args.depth_completion = "simple"

            semantic_completer = SemanticDepthCompleter(
                max_depth=args.max_depth,
                min_object_points=5,
                use_da3_fallback=False,  # No DA3 in this mode
                bilateral_filter=True,
            )
        else:
            depth_completer = DepthCompleter(
                method=args.depth_completion,
                model_path=args.completion_model,
                device="cuda",
                max_depth=args.max_depth,
            )
        is_metric = True  # Depth completion outputs metric depth
        print("Depth completion outputs METRIC depth (no alignment needed)")
    else:
        # Load DA3 model
        print(f"\nLoading DA3 model: {args.model}")
        model_repo_map = {
            "da3-giant": "depth-anything/DA3-GIANT",
            "da3-large": "depth-anything/DA3-LARGE",
            "da3-base": "depth-anything/DA3-BASE",
            "da3-small": "depth-anything/DA3-SMALL",
            "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
            "da3nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "da3metric-large": "depth-anything/DA3METRIC-LARGE",
        }
        repo_id = model_repo_map.get(args.model.lower(), args.model)
        print(f"Loading from: {repo_id}")
        model = DepthAnything3.from_pretrained(repo_id)
        model.to("cuda")
        model.eval()

        is_metric = "nested" in args.model.lower() or "metric" in args.model.lower()
        print(f"Metric depth: {is_metric}")

    # Get pinhole camera IDs
    pinhole_camera_ids = ["0", "3", "6", "9"]

    # Load scene data (with LiDAR if alignment enabled)
    print(f"\nLoading scene: {args.scene} @ {args.timestamp}")
    scene_data = loader.load_scene_data(
        scene_name=args.scene,
        timestamp=args.timestamp,
        camera_ids=pinhole_camera_ids,
        load_merged_lidar=args.lidar_align,
        load_individual_lidars=False,
    )

    # Get merged LiDAR points if available
    lidar_points = None
    if args.lidar_align and scene_data.merged_points is not None:
        lidar_points = scene_data.merged_points[:, :3]  # (N, 3) xyz
        print(f"Loaded merged LiDAR: {len(lidar_points):,} points")

    available_cameras = sorted(scene_data.images.keys())
    print(f"Available cameras: {available_cameras}")

    if not available_cameras:
        print("ERROR: No images found!")
        return

    # Get calibration (using our accurate calibrated poses)
    intrinsics_all, extrinsics_all = loader.get_camera_arrays(available_cameras)

    # Collect all points and colors
    all_points = []
    all_colors = []

    # Process each camera
    for i, cam_id in enumerate(available_cameras):
        print(f"\n{'='*50}")
        print(f"Processing camera {cam_id} ({i+1}/{len(available_cameras)})")
        print(f"{'='*50}")

        # Get image path
        img_path = scene_data.image_paths[cam_id]
        print(f"Image: {img_path}")

        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")

        # Get calibration for this camera
        K = intrinsics_all[i]  # (3, 3)
        E = extrinsics_all[i]  # (4, 4)

        conf = None
        conf_resized = None

        if semantic_completer is not None and annotation_bboxes is not None:
            # Semantic-guided depth completion mode
            print(f"Running semantic depth completion...")

            if lidar_points is None:
                print("ERROR: Semantic depth completion requires LiDAR! Use --lidar_align")
                return

            # Complete depth using 3D bbox separation
            depth_resized, info = semantic_completer.complete(
                rgb=img_rgb,
                lidar_points=lidar_points,
                bboxes=annotation_bboxes,
                intrinsics=K,
                extrinsics=E,
                da3_depth=None,  # No DA3 fallback
            )
            print(f"  Objects: {info['n_objects']}, Static points: {info['n_static_points']}")
            print(f"  Completed depth range: [{depth_resized.min():.2f}, {depth_resized.max():.2f}]m")

            # Skip further LiDAR processing (already metric)
            lidar_points_for_camera = None

        elif depth_completer is not None:
            # Depth completion mode: sparse LiDAR + RGB -> dense depth
            print(f"Running depth completion ({args.depth_completion})...")

            if lidar_points is None:
                print("ERROR: Depth completion requires LiDAR! Use --lidar_align")
                return

            # Create sparse depth from LiDAR projection
            sparse_depth, valid_mask = create_sparse_depth_from_lidar(
                lidar_points, K, E, (img.shape[0], img.shape[1])
            )
            n_valid = valid_mask.sum()
            print(f"  Sparse LiDAR: {n_valid} pixels ({n_valid/(img.shape[0]*img.shape[1])*100:.2f}%)")

            # Complete depth
            depth_resized = depth_completer.complete(img_rgb, sparse_depth, valid_mask)
            print(f"  Completed depth range: [{depth_resized.min():.2f}, {depth_resized.max():.2f}]m")

            # Skip further LiDAR processing (already metric)
            lidar_points_for_camera = None

        else:
            # DA3 mode: run depth estimation
            print(f"Running DA3 inference (process_res={args.process_res})...")
            prediction = model.inference(
                image=[img_path],
                process_res=args.process_res,
            )

            depth = prediction.depth[0]  # (H, W)
            conf = prediction.conf[0] if prediction.conf is not None else None

            print(f"Depth shape: {depth.shape}")
            print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")

            # Apply manual scale
            depth = depth * args.depth_scale

            # Resize depth to original image size
            depth_resized = cv2.resize(depth, (img.shape[1], img.shape[0]))
            if conf is not None:
                conf_resized = cv2.resize(conf, (img.shape[1], img.shape[0]))

            # LiDAR-based depth processing (only for DA3 mode)
            if lidar_points is not None:
                if args.lidar_anchored:
                    # NEW: LiDAR-anchored depth (recommended)
                    # LiDAR = ground truth, ground plane = geometric truth, camera fills gaps
                    print(f"Using LiDAR-anchored depth (ground_plane={args.use_ground_plane})...")
                    depth_resized, source_map, info = lidar_anchored_depth(
                        camera_depth=depth_resized,  # Raw relative depth
                        lidar_points=lidar_points,
                        intrinsics=K,
                        extrinsics=E,
                        use_ground_plane=args.use_ground_plane,
                    )
                    print(f"  Source: LiDAR={info['n_lidar']}, Ground={info['n_ground']}, Camera={info['n_camera']}")
                    print(f"  Anchored depth range: [{depth_resized.min():.4f}, {depth_resized.max():.4f}]")

                else:
                    # Original: scale alignment + optional adaptive fusion
                    print(f"Aligning depth with LiDAR (method={args.align_method})...")
                    lidar_uv, lidar_depths = project_lidar_to_camera(
                        lidar_points, K, E, img.shape[1], img.shape[0]
                    )
                    print(f"  Projected {len(lidar_uv)} LiDAR points to camera")

                    # Build depth range if specified
                    depth_range = None
                    if args.align_depth_min is not None or args.align_depth_max is not None:
                        depth_range = (
                            args.align_depth_min if args.align_depth_min else 0.0,
                            args.align_depth_max if args.align_depth_max else 1000.0,
                        )

                    depth_resized, scale, shift = align_depth_with_lidar(
                        depth_resized, lidar_uv, lidar_depths,
                        method=args.align_method,
                        depth_range=depth_range,
                    )
                    print(f"  Aligned depth range: [{depth_resized.min():.4f}, {depth_resized.max():.4f}]")

                    # Adaptive fusion: use LiDAR where dense, camera where sparse
                    if args.adaptive_fusion:
                        print(f"Applying adaptive camera/LiDAR fusion...")
                        fused_depth, lidar_weight, _ = adaptive_depth_fusion(
                            camera_depth=depth_resized,
                            lidar_points=lidar_points,
                            intrinsics=K,
                            extrinsics=E,
                            camera_confidence=conf_resized,
                            density_sigma=args.density_sigma,
                            distance_threshold=args.distance_threshold,
                        )
                        depth_resized = fused_depth
                        print(f"  LiDAR weight: mean={lidar_weight.mean():.2%}, max={lidar_weight.max():.2%}")
                        print(f"  Fused depth range: [{depth_resized.min():.4f}, {depth_resized.max():.4f}]")

        print(f"Using calibrated intrinsics:\n{K}")
        print(f"Using calibrated extrinsics (w2c):\n{E}")

        # Convert to point cloud
        print(f"Converting to point cloud (max_view_angle={args.max_view_angle}°)...")
        points, colors = depth_to_pointcloud(
            depth=depth_resized,
            image=img_rgb,
            intrinsics=K,
            extrinsics=E,
            conf=conf_resized,
            conf_threshold=args.conf_threshold,
            max_depth=args.max_depth,
            max_view_angle=args.max_view_angle,
            downsample=args.downsample,
        )

        print(f"Generated {len(points):,} points")

        all_points.append(points)
        all_colors.append(colors)

        # Save individual camera results
        cam_output_dir = os.path.join(output_dir, f"cam_{cam_id}")
        os.makedirs(cam_output_dir, exist_ok=True)

        # Save depth visualization
        depth_vis = visualize_depth(depth_resized)
        cv2.imwrite(os.path.join(cam_output_dir, "depth.jpg"), depth_vis)

        # Save individual PLY
        save_ply(
            os.path.join(cam_output_dir, "pointcloud.ply"),
            points, colors
        )

        # Save side-by-side visualization
        h, w = img.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
        vis[:, :w] = img
        vis[:, w:] = depth_vis
        cv2.imwrite(os.path.join(cam_output_dir, "comparison.jpg"), vis)

    # Fuse all point clouds
    print(f"\n{'='*50}")
    print("Fusing all point clouds...")
    print(f"{'='*50}")

    fused_points = np.vstack(all_points)
    fused_colors = np.vstack(all_colors)

    print(f"Total fused points: {len(fused_points):,}")

    # Save fused PLY
    fused_ply_path = os.path.join(output_dir, "fused_pointcloud.ply")
    save_ply(fused_ply_path, fused_points, fused_colors)

    # Print summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for i, cam_id in enumerate(available_cameras):
        print(f"  Camera {cam_id}: {len(all_points[i]):,} points")
    print(f"  Total: {len(fused_points):,} points")
    print(f"\nResults saved to: {output_dir}")
    print(f"Fused PLY: {fused_ply_path}")


if __name__ == "__main__":
    main()
