#!/usr/bin/env python3
"""
Simple DA3 depth inference without LiDAR alignment.
Just visualize the raw relative depth output.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from depth_anything_3.api import DepthAnything3


def visualize_depth(depth, colormap=cv2.COLORMAP_INFERNO):
    """Visualize depth map with colormap."""
    # Normalize to 0-1
    d_min, d_max = np.percentile(depth[depth > 0], [2, 98])
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), colormap)
    return depth_color


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default="da3-giant", help="Model name")
    parser.add_argument("--process_res", type=int, default=518, help="Processing resolution")
    parser.add_argument("--output", type=str, default="./depth_output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = DepthAnything3(model_name=args.model)
    model.to("cuda")
    model.eval()

    # Load image
    print(f"Loading image: {args.image}")
    img = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    print(f"Running inference with process_res={args.process_res}")
    prediction = model.inference(
        image=[args.image],
        process_res=args.process_res,
    )

    depth = prediction.depth[0]  # (H, W)
    conf = prediction.conf[0] if prediction.conf is not None else None

    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Depth mean: {depth.mean():.4f}")

    # Resize depth to original image size for overlay
    depth_resized = cv2.resize(depth, (img.shape[1], img.shape[0]))

    # Visualize
    depth_vis = visualize_depth(depth_resized)

    # Create side-by-side visualization
    h, w = img.shape[:2]
    vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
    vis[:, :w] = img
    vis[:, w:] = depth_vis

    # Add labels
    cv2.putText(vis, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "DA3 Depth (relative)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save
    basename = Path(args.image).stem
    vis_path = os.path.join(args.output, f"{basename}_depth.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"Saved: {vis_path}")

    # Also save depth at processing resolution
    depth_proc_vis = visualize_depth(depth)
    depth_proc_path = os.path.join(args.output, f"{basename}_depth_proc_res.jpg")
    cv2.imwrite(depth_proc_path, depth_proc_vis)
    print(f"Saved: {depth_proc_path}")

    # Save raw depth as numpy
    np.save(os.path.join(args.output, f"{basename}_depth.npy"), depth)
    print(f"Saved: {basename}_depth.npy")

    if conf is not None:
        conf_vis = visualize_depth(conf, cv2.COLORMAP_VIRIDIS)
        conf_path = os.path.join(args.output, f"{basename}_conf.jpg")
        cv2.imwrite(conf_path, conf_vis)
        print(f"Saved: {conf_path}")


if __name__ == "__main__":
    main()
