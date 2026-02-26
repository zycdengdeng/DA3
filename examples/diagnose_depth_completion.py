#!/usr/bin/env python3
"""
Diagnostic tool for depth completion and SAM segmentation.

Visualizes:
1. DA3 raw depth (scaled by LiDAR, but no vehicle completion)
2. Depth after vehicle completion
3. SAM segmentation masks for each vehicle

This helps diagnose issues with:
- DA3 relative depth quality
- LiDAR scale alignment
- SAM segmentation accuracy
- Vehicle depth completion
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.lidar_alignment import align_depth_with_lidar
from depth_anything_3.utils.semantic_depth_completion import (
    BBox3D,
    SemanticDepthCompleter,
    load_annotations,
    project_bbox_to_2d,
    DYNAMIC_CATEGORIES,
)

# Try import SAM
try:
    from depth_anything_3.utils.sam_segmentation import (
        SAMGuidedMaskGenerator,
        SAMSegmenter,
        get_available_sam_backend,
        check_sam_installation,
    )
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


def colorize_depth(depth, vmin=None, vmax=None, cmap='turbo'):
    """Convert depth to colored image."""
    if vmin is None:
        vmin = np.percentile(depth[depth > 0], 1) if (depth > 0).any() else 0
    if vmax is None:
        vmax = np.percentile(depth[depth > 0], 99) if (depth > 0).any() else 1

    depth_norm = (depth - vmin) / (vmax - vmin + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    colormap = plt.get_cmap(cmap)
    depth_color = colormap(depth_norm)[:, :, :3]
    depth_color = (depth_color * 255).astype(np.uint8)

    # Mark invalid as black
    depth_color[depth <= 0] = 0

    return depth_color


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay mask on image with transparency."""
    result = image.copy()
    overlay = np.zeros_like(image)
    overlay[mask] = color
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    return result


def draw_bbox_2d(image, corners_2d, color=(255, 0, 0), thickness=2):
    """Draw projected 3D bbox edges on image."""
    # 3D bbox edge connections
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7),  # Verticals
    ]

    result = image.copy()
    for i, j in edges:
        pt1 = tuple(corners_2d[i].astype(int))
        pt2 = tuple(corners_2d[j].astype(int))
        cv2.line(result, pt1, pt2, color, thickness)

    return result


def diagnose_single_camera(
    image_path: str,
    lidar_path: str,
    annotation_path: str,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    output_dir: str,
    model_name: str = "da3-giant",
    device: str = "cuda",
):
    """
    Run diagnosis for a single camera.

    Outputs:
        - rgb.png: Original image
        - depth_raw.png: DA3 raw depth (before any completion)
        - depth_scaled.png: DA3 depth after LiDAR scale alignment
        - depth_completed.png: Depth after vehicle completion
        - depth_diff.png: Difference between scaled and completed
        - sam_masks.png: SAM segmentation masks for all vehicles
        - sam_mask_N.png: Individual SAM mask for each vehicle
        - convex_hull_masks.png: Convex hull masks (fallback)
        - diagnosis_summary.png: Combined visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Diagnosing: {image_path}")
    print(f"{'='*60}")

    # Load data
    print("\n[1/6] Loading data...")
    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]
    print(f"  Image: {W}x{H}")

    # Load LiDAR
    lidar_path = Path(lidar_path)
    if lidar_path.suffix == '.bin':
        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    elif lidar_path.suffix == '.npy':
        lidar_points = np.load(lidar_path)
        if lidar_points.shape[1] > 3:
            lidar_points = lidar_points[:, :3]
    else:
        lidar_points = np.loadtxt(lidar_path)[:, :3]
    print(f"  LiDAR: {len(lidar_points):,} points")

    # Load annotations
    bboxes = load_annotations(annotation_path)
    dynamic_bboxes = [b for b in bboxes if b.label in DYNAMIC_CATEGORIES]
    print(f"  Annotations: {len(bboxes)} total, {len(dynamic_bboxes)} dynamic objects")

    # Save RGB
    Image.fromarray(image).save(output_dir / "rgb.png")

    # Step 2: Run DA3 inference
    print("\n[2/6] Running DA3 inference...")
    model = DepthAnything3.from_pretrained(
        {"da3-giant": "depth-anything/DA3-GIANT",
         "da3-large": "depth-anything/DA3-LARGE",
         "da3-base": "depth-anything/DA3-BASE"}.get(model_name, model_name)
    )
    model.to(device)
    model.eval()

    prediction = model.inference(image=[image], process_res=518)
    da3_depth_raw = prediction.depth[0]  # This is RELATIVE depth
    da3_conf = prediction.conf[0]
    proc_image = prediction.processed_images[0]

    proc_H, proc_W = da3_depth_raw.shape
    print(f"  DA3 output: {proc_W}x{proc_H}")
    print(f"  Depth range (relative): {da3_depth_raw.min():.2f} - {da3_depth_raw.max():.2f}")

    # Scale intrinsics
    scale_x = proc_W / W
    scale_y = proc_H / H
    K_scaled = intrinsics.copy()
    K_scaled[0, :] *= scale_x
    K_scaled[1, :] *= scale_y

    # Save raw DA3 depth
    depth_raw_color = colorize_depth(da3_depth_raw)
    Image.fromarray(depth_raw_color).save(output_dir / "depth_raw_relative.png")

    # Step 3: LiDAR scale alignment
    print("\n[3/6] LiDAR scale alignment...")
    try:
        align_result = align_depth_with_lidar(
            predicted_depth=da3_depth_raw,
            lidar_points=lidar_points,
            intrinsics=K_scaled,
            extrinsics=extrinsics,
            confidence=da3_conf,
            use_ransac=True,
        )
        da3_depth_scaled = align_result.aligned_depth
        print(f"  Scale factor: {align_result.scale_factor:.4f}")
        print(f"  Valid points: {align_result.num_valid_points}")
        print(f"  Inlier ratio: {align_result.inlier_ratio:.1%}")
        print(f"  Residual: {align_result.residual_mean:.2f}m +/- {align_result.residual_std:.2f}m")
        print(f"  Depth range (metric): {da3_depth_scaled.min():.2f}m - {da3_depth_scaled.max():.2f}m")
    except Exception as e:
        print(f"  WARNING: Alignment failed: {e}")
        da3_depth_scaled = da3_depth_raw.copy()

    # Save scaled depth
    depth_scaled_color = colorize_depth(da3_depth_scaled, vmin=0, vmax=150)
    Image.fromarray(depth_scaled_color).save(output_dir / "depth_scaled.png")

    # Step 4: Check SAM
    print("\n[4/6] Checking SAM...")
    if SAM_AVAILABLE:
        check_sam_installation()
        sam_backend = get_available_sam_backend()
        print(f"  SAM backend: {sam_backend or 'NOT AVAILABLE'}")
    else:
        print("  SAM module not importable")
        sam_backend = None

    # Step 5: Generate masks and visualize
    print("\n[5/6] Generating object masks...")

    # Resize image to processing resolution for visualization
    proc_image_resized = cv2.resize(image, (proc_W, proc_H))

    # Visualize each vehicle
    all_sam_masks = np.zeros((proc_H, proc_W), dtype=np.uint8)
    all_convex_masks = np.zeros((proc_H, proc_W), dtype=np.uint8)
    mask_overlay = proc_image_resized.copy()
    convex_overlay = proc_image_resized.copy()

    # Initialize SAM if available
    sam_generator = None
    if sam_backend is not None:
        sam_generator = SAMGuidedMaskGenerator(
            sam_backend="auto",
            device=device,
        )
        if sam_generator.sam_available:
            sam_generator.set_image(proc_image_resized)
            print(f"  SAM initialized: YES")
        else:
            print(f"  SAM initialized: NO (model not loaded)")
            sam_generator = None

    # Color palette for different objects
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]

    for i, bbox in enumerate(dynamic_bboxes):
        color = colors[i % len(colors)]
        print(f"\n  Object {i}: {bbox.label} (id={bbox.id})")

        corners_3d = bbox.get_corners()

        # Project to 2D for visualization
        corners_homo = np.hstack([corners_3d, np.ones((8, 1))])
        corners_cam = (extrinsics @ corners_homo.T).T[:, :3]
        valid_depth = corners_cam[:, 2] > 0.1

        if not valid_depth.any():
            print(f"    Behind camera, skipping")
            continue

        corners_img = (K_scaled @ corners_cam.T).T
        corners_2d = corners_img[:, :2] / corners_img[:, 2:3]

        # Draw 3D bbox projection
        mask_overlay = draw_bbox_2d(mask_overlay, corners_2d, color=color)
        convex_overlay = draw_bbox_2d(convex_overlay, corners_2d, color=color)

        # Convex hull mask (always available)
        convex_mask = project_bbox_to_2d(bbox, K_scaled, extrinsics, (proc_H, proc_W))
        if convex_mask is not None:
            all_convex_masks[convex_mask] = i + 1
            convex_overlay = overlay_mask(convex_overlay, convex_mask, color=color, alpha=0.3)
            print(f"    Convex hull: {convex_mask.sum():,} pixels")

        # SAM mask
        if sam_generator is not None:
            sam_mask = sam_generator.generate_mask(
                bbox_3d_corners=corners_3d,
                intrinsics=K_scaled,
                extrinsics=extrinsics,
                image_hw=(proc_H, proc_W),
            )
            if sam_mask is not None:
                all_sam_masks[sam_mask] = i + 1
                mask_overlay = overlay_mask(mask_overlay, sam_mask, color=color, alpha=0.4)
                print(f"    SAM mask: {sam_mask.sum():,} pixels")

                # Save individual mask
                mask_vis = np.zeros((proc_H, proc_W, 3), dtype=np.uint8)
                mask_vis[sam_mask] = color
                Image.fromarray(mask_vis).save(output_dir / f"sam_mask_{i}_{bbox.label}.png")
            else:
                print(f"    SAM mask: FAILED")
        else:
            print(f"    SAM mask: NOT AVAILABLE")

    # Save mask visualizations
    Image.fromarray(mask_overlay).save(output_dir / "sam_masks_overlay.png")
    Image.fromarray(convex_overlay).save(output_dir / "convex_hull_overlay.png")

    # Step 6: Run depth completion
    print("\n[6/6] Running depth completion...")
    completer = SemanticDepthCompleter(
        max_depth=300.0,
        min_object_points=5,
        use_da3_fallback=True,
        use_sam=(sam_generator is not None),
        device=device,
    )

    depth_completed, completion_info = completer.complete(
        rgb=proc_image_resized,
        lidar_points=lidar_points,
        bboxes=bboxes,
        intrinsics=K_scaled,
        extrinsics=extrinsics,
        da3_depth=da3_depth_scaled,  # Use scaled DA3 as fallback
    )

    print(f"\n  Completion info:")
    print(f"    Objects processed: {completion_info['n_objects']}")
    print(f"    Static points: {completion_info['n_static_points']}")
    print(f"    Mask method: {completion_info['mask_method']}")

    for obj_id, obj_info in completion_info.get('objects', {}).items():
        print(f"    Object {obj_id}: {obj_info['label']}, "
              f"points={obj_info['n_points']}, "
              f"method={obj_info['depth_method']}, "
              f"mask={obj_info['mask_method']}")

    # Save completed depth
    depth_completed_color = colorize_depth(depth_completed, vmin=0, vmax=150)
    Image.fromarray(depth_completed_color).save(output_dir / "depth_completed.png")

    # Compute and save difference
    depth_diff = depth_completed - da3_depth_scaled
    depth_diff_abs = np.abs(depth_diff)

    # Colorize difference
    diff_max = np.percentile(depth_diff_abs[depth_diff_abs > 0], 95) if (depth_diff_abs > 0).any() else 10
    diff_norm = np.clip(depth_diff_abs / diff_max, 0, 1)
    diff_color = (plt.get_cmap('hot')(diff_norm)[:, :, :3] * 255).astype(np.uint8)
    diff_color[depth_diff_abs < 0.1] = 0  # Mark small differences as black
    Image.fromarray(diff_color).save(output_dir / "depth_diff.png")

    # Create summary figure
    print("\n[Summary] Creating visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(proc_image_resized)
    axes[0, 0].set_title("RGB Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(depth_raw_color)
    axes[0, 1].set_title("DA3 Raw (Relative)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(depth_scaled_color)
    axes[0, 2].set_title(f"DA3 Scaled (scale={align_result.scale_factor:.3f})")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(depth_completed_color)
    axes[0, 3].set_title("Depth Completed")
    axes[0, 3].axis('off')

    axes[1, 0].imshow(convex_overlay)
    axes[1, 0].set_title("Convex Hull Masks")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_overlay)
    axes[1, 1].set_title(f"SAM Masks ({sam_backend or 'N/A'})")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(diff_color)
    axes[1, 2].set_title("Depth Difference (abs)")
    axes[1, 2].axis('off')

    # Confidence map
    conf_color = colorize_depth(da3_conf, cmap='viridis')
    axes[1, 3].imshow(conf_color)
    axes[1, 3].set_title("DA3 Confidence")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"Diagnosis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nKey files to check:")
    print(f"  - diagnosis_summary.png: Overview of all steps")
    print(f"  - depth_scaled.png: DA3 depth BEFORE vehicle completion")
    print(f"  - depth_completed.png: Depth AFTER vehicle completion")
    print(f"  - depth_diff.png: Difference (shows what completion changed)")
    print(f"  - sam_masks_overlay.png: SAM segmentation results")
    print(f"  - convex_hull_overlay.png: Convex hull fallback")

    return {
        'scale_factor': align_result.scale_factor,
        'completion_info': completion_info,
        'sam_available': sam_generator is not None,
    }


def diagnose_car_road_dataset(
    data_root: str,
    scene: str,
    timestamp: str,
    camera_id: str,
    output_dir: str,
    model_name: str = "da3-giant",
    device: str = "cuda",
):
    """
    Diagnose using car_road dataset structure.

    Example:
        diagnose_car_road_dataset(
            data_root="/mnt/car_road_data_fix",
            scene="001_car0325_road0327_t1",
            timestamp="1742877031036",
            camera_id="0",
            output_dir="./diagnosis_output",
        )
    """
    from depth_anything_3.datasets.car_road_dataset import CarRoadDatasetLoader

    print(f"\n{'='*60}")
    print(f"Car-Road Dataset Diagnosis")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Scene: {scene}")
    print(f"Timestamp: {timestamp}")
    print(f"Camera: {camera_id}")

    # Load dataset
    loader = CarRoadDatasetLoader(data_root=data_root)

    # Get camera calibration
    cam = loader.cameras[camera_id]
    intrinsics = cam.intrinsics
    extrinsics = cam.extrinsics_w2c

    # Load image
    image, image_path = loader.load_image(scene, camera_id, timestamp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image: {image_path}")

    # Load merged LiDAR
    lidar_points = loader.load_merged_points(scene, timestamp)
    print(f"LiDAR: {len(lidar_points):,} points")

    # Find annotation
    annotation_path = os.path.join(
        data_root, scene, "road_labels",
        "interpolation_labels", f"{timestamp}.json"
    )
    if not os.path.exists(annotation_path):
        annotation_path = os.path.join(
            data_root, scene, "road_labels",
            "ori_labels", f"{timestamp}.json"
        )
    print(f"Annotation: {annotation_path}")

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")

    # Load annotations
    bboxes = load_annotations(annotation_path)
    dynamic_bboxes = [b for b in bboxes if b.label in DYNAMIC_CATEGORIES]
    print(f"Bboxes: {len(bboxes)} total, {len(dynamic_bboxes)} dynamic")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save RGB
    Image.fromarray(image).save(output_dir / "rgb.png")

    H, W = image.shape[:2]

    # Run DA3
    print(f"\nRunning DA3 ({model_name})...")
    model = DepthAnything3.from_pretrained(
        {"da3-giant": "depth-anything/DA3-GIANT",
         "da3-large": "depth-anything/DA3-LARGE"}.get(model_name, model_name)
    )
    model.to(device)
    model.eval()

    prediction = model.inference(image=[image], process_res=518)
    da3_depth_raw = prediction.depth[0]
    da3_conf = prediction.conf[0]
    proc_image = prediction.processed_images[0]

    proc_H, proc_W = da3_depth_raw.shape
    print(f"DA3 output: {proc_W}x{proc_H}")

    # Scale intrinsics
    scale_x = proc_W / W
    scale_y = proc_H / H
    K_scaled = intrinsics.copy()
    K_scaled[0, :] *= scale_x
    K_scaled[1, :] *= scale_y

    # Save raw depth
    depth_raw_color = colorize_depth(da3_depth_raw)
    Image.fromarray(depth_raw_color).save(output_dir / "depth_raw_relative.png")

    # LiDAR alignment
    print("\nLiDAR alignment...")
    from depth_anything_3.utils.lidar_alignment import align_depth_with_lidar as align_fn
    align_result = align_fn(
        predicted_depth=da3_depth_raw,
        lidar_points=lidar_points,
        intrinsics=K_scaled,
        extrinsics=extrinsics,
        confidence=da3_conf,
        use_ransac=True,
    )
    da3_depth_scaled = align_result.aligned_depth
    print(f"Scale: {align_result.scale_factor:.4f}")
    print(f"Inliers: {align_result.inlier_ratio:.1%}")

    depth_scaled_color = colorize_depth(da3_depth_scaled, vmin=0, vmax=150)
    Image.fromarray(depth_scaled_color).save(output_dir / "depth_scaled.png")

    # Resize image
    proc_image_resized = cv2.resize(image, (proc_W, proc_H))

    # SAM masks
    print("\nGenerating SAM masks...")
    sam_backend = get_available_sam_backend() if SAM_AVAILABLE else None
    print(f"SAM backend: {sam_backend or 'NOT AVAILABLE'}")

    sam_generator = None
    if sam_backend:
        sam_generator = SAMGuidedMaskGenerator(sam_backend="auto", device=device)
        if sam_generator.sam_available:
            sam_generator.set_image(proc_image_resized)

    # Visualize masks
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    mask_overlay = proc_image_resized.copy()
    convex_overlay = proc_image_resized.copy()

    for i, bbox in enumerate(dynamic_bboxes):
        color = colors[i % len(colors)]
        corners_3d = bbox.get_corners()

        # Convex hull
        convex_mask = project_bbox_to_2d(bbox, K_scaled, extrinsics, (proc_H, proc_W))
        if convex_mask is not None:
            convex_overlay = overlay_mask(convex_overlay, convex_mask, color, 0.3)

        # SAM mask
        if sam_generator and sam_generator.sam_available:
            sam_mask = sam_generator.generate_mask(
                bbox_3d_corners=corners_3d,
                intrinsics=K_scaled,
                extrinsics=extrinsics,
                image_hw=(proc_H, proc_W),
            )
            if sam_mask is not None:
                mask_overlay = overlay_mask(mask_overlay, sam_mask, color, 0.4)
                mask_vis = np.zeros((proc_H, proc_W, 3), dtype=np.uint8)
                mask_vis[sam_mask] = color
                Image.fromarray(mask_vis).save(output_dir / f"sam_mask_{i}_{bbox.label}.png")
                print(f"  {bbox.label}: SAM={sam_mask.sum():,}px, convex={convex_mask.sum() if convex_mask is not None else 0:,}px")

    Image.fromarray(mask_overlay).save(output_dir / "sam_masks_overlay.png")
    Image.fromarray(convex_overlay).save(output_dir / "convex_hull_overlay.png")

    # Depth completion
    print("\nDepth completion...")
    completer = SemanticDepthCompleter(
        max_depth=300.0,
        use_sam=(sam_generator is not None and sam_generator.sam_available),
        device=device,
    )

    depth_completed, info = completer.complete(
        rgb=proc_image_resized,
        lidar_points=lidar_points,
        bboxes=bboxes,
        intrinsics=K_scaled,
        extrinsics=extrinsics,
        da3_depth=da3_depth_scaled,
    )

    depth_completed_color = colorize_depth(depth_completed, vmin=0, vmax=150)
    Image.fromarray(depth_completed_color).save(output_dir / "depth_completed.png")

    # Difference
    depth_diff = np.abs(depth_completed - da3_depth_scaled)
    diff_max = np.percentile(depth_diff[depth_diff > 0], 95) if (depth_diff > 0).any() else 10
    diff_norm = np.clip(depth_diff / diff_max, 0, 1)
    diff_color = (plt.get_cmap('hot')(diff_norm)[:, :, :3] * 255).astype(np.uint8)
    diff_color[depth_diff < 0.1] = 0
    Image.fromarray(diff_color).save(output_dir / "depth_diff.png")

    # Summary figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(proc_image_resized)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(depth_raw_color)
    axes[0, 1].set_title("DA3 Raw (Relative)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(depth_scaled_color)
    axes[0, 2].set_title(f"Scaled (s={align_result.scale_factor:.3f})")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(depth_completed_color)
    axes[0, 3].set_title("Completed")
    axes[0, 3].axis('off')

    axes[1, 0].imshow(convex_overlay)
    axes[1, 0].set_title("Convex Hull")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_overlay)
    axes[1, 1].set_title(f"SAM ({sam_backend or 'N/A'})")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(diff_color)
    axes[1, 2].set_title("Diff (completion)")
    axes[1, 2].axis('off')

    conf_color = colorize_depth(da3_conf, cmap='viridis')
    axes[1, 3].imshow(conf_color)
    axes[1, 3].set_title("Confidence")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis_summary.png", dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print(f"Done! Results: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose depth completion and SAM segmentation")

    # Mode 1: Car-road dataset (simpler)
    parser.add_argument("--data_root", type=str, default="/mnt/car_road_data_fix",
                        help="Car-road dataset root")
    parser.add_argument("--scene", type=str, help="Scene name (e.g., 001_car0325_road0327_t1)")
    parser.add_argument("--timestamp", type=str, help="Timestamp (e.g., 1742877031036)")
    parser.add_argument("--camera_id", type=str, default="0", help="Camera ID (0, 3, 6, or 9)")

    # Mode 2: Manual paths
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--lidar", type=str, help="Path to LiDAR points")
    parser.add_argument("--annotation", type=str, help="Path to annotation JSON")
    parser.add_argument("--intrinsics", type=str, help="Path to intrinsics (txt or npy)")
    parser.add_argument("--extrinsics", type=str, help="Path to extrinsics (txt or npy)")

    # Common
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="da3-giant", help="DA3 model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Mode 1: Car-road dataset
    if args.scene and args.timestamp:
        diagnose_car_road_dataset(
            data_root=args.data_root,
            scene=args.scene,
            timestamp=args.timestamp,
            camera_id=args.camera_id,
            output_dir=args.output,
            model_name=args.model,
            device=args.device,
        )
        return

    # Mode 2: Manual paths
    if not all([args.image, args.lidar, args.annotation, args.intrinsics, args.extrinsics]):
        parser.error("Either provide --scene/--timestamp OR all manual paths")

    # Load calibration
    if args.intrinsics.endswith('.npy'):
        intrinsics = np.load(args.intrinsics)
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
    else:
        intrinsics = np.loadtxt(args.intrinsics).reshape(3, 3)

    if args.extrinsics.endswith('.npy'):
        extrinsics = np.load(args.extrinsics)
        if extrinsics.ndim == 3:
            extrinsics = extrinsics[0]
    else:
        extrinsics = np.loadtxt(args.extrinsics).reshape(4, 4)

    diagnose_single_camera(
        image_path=args.image,
        lidar_path=args.lidar,
        annotation_path=args.annotation,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        output_dir=args.output,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
