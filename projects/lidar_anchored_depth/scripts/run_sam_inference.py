"""Run SAM with V2X-bbox prompts (HuggingFace transformers backend).

For every V2X-annotated dynamic object in a frame we project its 3D
bbox to image space, take the AABB, and use that AABB as a **box
prompt** to SAM. We use the HuggingFace transformers ``SamModel`` so
the checkpoint is auto-downloaded from HuggingFace Hub — set
``HF_ENDPOINT=https://hf-mirror.com`` first to pull from the China
mirror at native bandwidth.

This is much faster than SAM's auto-mask-generation mode for our use
case: we already know what we care about, we just want pixel-precise
masks for those instances.

Output (`preview/sam/<scene>_ts<ts>_cam<id>_sam.npz`):
    mask_ids : (K,) int32      — V2X bbox.id corresponding to each mask
    masks    : (K, H, W) bool  — segmentation masks at image resolution
    scores   : (K,) float32    — SAM's predicted mask quality (IoU) score
    boxes    : (K, 4) float64  — the box prompts used (u_min,v_min,u_max,v_max)
    image_hw : (2,) int32      — original image dimensions
    scene_id, ts_ms, cam_id, sam_model

Setup
-----
    conda activate lad
    pip install transformers

    # First-run: weights auto-download from HF mirror
    export HF_ENDPOINT=https://hf-mirror.com

Usage
-----
    PYTHONPATH=src python scripts/run_sam_inference.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 --cam all \\
        --timestamps 1742879639783 1742879642908 \\
        --output preview/sam/

    # smaller / faster ViT-B / ViT-L variants:
    ... --model-id facebook/sam-vit-base
    ... --model-id facebook/sam-vit-large
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.alignment.projection import (
    bbox_uv_aabb,
    project_3d_bbox,
)
from lidar_anchored_depth.data import (
    PINHOLE_CAMERA_IDS,
    RoadsideV2XLoader,
)


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--timestamps", nargs="+", default=None,
        help="ms timestamps; default = all frames in the scene",
    )
    parser.add_argument(
        "--cam", default="all",
        help="pinhole id (0/3/6/9) or 'all'",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-id", default="facebook/sam-vit-huge",
        help="HuggingFace model id. Try sam-vit-base / sam-vit-large for "
        "smaller / faster variants.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bbox-pad-px", type=int, default=8,
        help="pad each projected V2X bbox AABB by this many pixels before "
        "feeding it to SAM (helps SAM see the full object silhouette).",
    )
    parser.add_argument(
        "--multimask", action="store_true",
        help="ask SAM for 3 candidate masks per box and keep the highest-"
        "score one. Slightly slower, slightly better.",
    )
    parser.add_argument(
        "--max-boxes-per-batch", type=int, default=64,
        help="chunk size for SAM batch prediction (default 64). Lower if "
        "you hit VRAM limits with many V2X bboxes per frame.",
    )
    parser.add_argument(
        "--loader-min-points", type=int, default=5,
        help="V2X annotation filter: drop bboxes with num_points below "
        "this. Set to 0 for interpolated frames where num_points is 0.",
    )
    args = parser.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print(
            "[note] HF_ENDPOINT not set. If you are on a China mainland "
            "host, run `export HF_ENDPOINT=https://hf-mirror.com` first to "
            "pull weights from the local mirror at native bandwidth.",
            file=sys.stderr,
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports here so --help is fast
    import torch
    from transformers import SamModel, SamProcessor

    print(f"[load] {args.model_id}")
    t0 = time.time()
    processor = SamProcessor.from_pretrained(args.model_id)
    model = SamModel.from_pretrained(args.model_id).to(args.device).eval()
    print(f"  ↳ loaded in {time.time() - t0:.1f}s on {args.device}")

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)

    if args.timestamps:
        ts_list = [int(t) for t in args.timestamps]
    else:
        ts_list = list(scene.timestamps_ms)

    if args.cam == "all":
        cams = list(PINHOLE_CAMERA_IDS)
    else:
        if args.cam not in PINHOLE_CAMERA_IDS:
            raise SystemExit(f"--cam={args.cam!r} not pinhole")
        cams = [args.cam]

    print(f"[scene] {scene_id}  ts_count={len(ts_list)}  cams={cams}")
    print()

    for ts_ms in ts_list:
        for cam_id in cams:
            try:
                idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
            except ValueError as e:
                print(f"  skip ts={ts_ms} cam{cam_id}: {e}")
                continue
            frame = loader.get_frame(idx)
            H, W = frame.image.shape[:2]
            dist = frame.meta.get("distortion")

            # Collect all V2X bbox AABBs to pass as a batch to SAM
            boxes: list[list[float]] = []
            ids: list[int] = []
            for obj in frame.dynamic_objects or []:
                uv8, _ = project_3d_bbox(obj, frame.K, frame.T_wc, dist)
                if uv8 is None:
                    continue
                u_min, v_min, u_max, v_max = bbox_uv_aabb(uv8)
                u_min = max(0, int(u_min) - args.bbox_pad_px)
                v_min = max(0, int(v_min) - args.bbox_pad_px)
                u_max = min(W - 1, int(u_max) + args.bbox_pad_px)
                v_max = min(H - 1, int(v_max) + args.bbox_pad_px)
                if u_max <= u_min or v_max <= v_min:
                    continue
                boxes.append([float(u_min), float(v_min), float(u_max), float(v_max)])
                ids.append(int(obj.id))

            out_path = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam.npz"
            if not boxes:
                np.savez_compressed(
                    out_path,
                    mask_ids=np.zeros(0, dtype=np.int32),
                    masks=np.zeros((0, H, W), dtype=bool),
                    scores=np.zeros(0, dtype=np.float32),
                    boxes=np.zeros((0, 4), dtype=np.float64),
                    image_hw=np.array([H, W], dtype=np.int32),
                    scene_id=scene_id, ts_ms=int(ts_ms), cam_id=cam_id,
                    sam_model=args.model_id,
                )
                print(f"  ts={ts_ms} cam{cam_id}  no V2X bboxes")
                continue

            # Run SAM on the batch (chunked if large)
            t0 = time.time()
            all_masks: list[np.ndarray] = []
            all_scores: list[float] = []
            for chunk_start in range(0, len(boxes), args.max_boxes_per_batch):
                chunk_boxes = boxes[chunk_start : chunk_start + args.max_boxes_per_batch]
                inputs = processor(
                    frame.image,
                    input_boxes=[chunk_boxes],
                    return_tensors="pt",
                ).to(args.device)
                with torch.inference_mode():
                    outputs = model(
                        **inputs, multimask_output=bool(args.multimask)
                    )
                # Post-process to original resolution
                masks_list = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )
                # masks_list[0] shape: (num_boxes_in_chunk, num_masks, H, W)
                masks_tensor = masks_list[0]
                iou_scores = outputs.iou_scores.cpu().numpy()[0]  # (num_boxes, num_masks)

                if args.multimask:
                    best = iou_scores.argmax(axis=1)
                    for i in range(len(chunk_boxes)):
                        all_masks.append(
                            masks_tensor[i, best[i]].numpy().astype(bool)
                        )
                        all_scores.append(float(iou_scores[i, best[i]]))
                else:
                    for i in range(len(chunk_boxes)):
                        all_masks.append(
                            masks_tensor[i, 0].numpy().astype(bool)
                        )
                        all_scores.append(float(iou_scores[i, 0]))
            t_pred = time.time() - t0

            masks_arr = np.stack(all_masks, axis=0)
            scores_arr = np.array(all_scores, dtype=np.float32)
            boxes_arr = np.array(boxes, dtype=np.float64)

            np.savez_compressed(
                out_path,
                mask_ids=np.array(ids, dtype=np.int32),
                masks=masks_arr,
                scores=scores_arr,
                boxes=boxes_arr,
                image_hw=np.array([H, W], dtype=np.int32),
                scene_id=scene_id, ts_ms=int(ts_ms), cam_id=cam_id,
                sam_model=args.model_id,
            )
            avg_area = float(np.mean([m.sum() for m in masks_arr]))
            print(
                f"  ts={ts_ms} cam{cam_id}  K={len(masks_arr):>3}  "
                f"sam {t_pred*1000:.0f}ms  "
                f"avg score {scores_arr.mean():.2f}  avg area {avg_area:.0f}px"
            )

    print()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
