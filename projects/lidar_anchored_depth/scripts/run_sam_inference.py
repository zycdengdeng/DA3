"""Run SAM with V2X-bbox prompts on roadside frames.

For every V2X-annotated dynamic object in a frame we project its 3D
bbox to image space, take the AABB, and use that AABB as a **box
prompt** to SAM. This is dramatically faster and tighter than SAM's
auto-mask-generation mode for our use case — we already know what we
care about, we just want pixel-precise masks for those instances.

Output (`preview/sam/<scene>_ts<ts>_cam<id>_sam.npz`):
    mask_ids : (K,) int32      — V2X bbox.id corresponding to each mask
    masks    : (K, H, W) bool  — segmentation masks, image-resolution
    scores   : (K,) float32    — SAM's predicted mask quality score
    boxes    : (K, 4) float64  — the box prompts used (u_min,v_min,u_max,v_max)
    image_hw : (2,) int32      — original image dimensions
    scene_id : str
    ts_ms    : int
    cam_id   : str
    sam_model: str

Usage
-----
    pip install segment-anything
    # Download checkpoint (~2.6 GB) for ViT-H:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \\
        -O /mnt/zyc_wzh/DA3_lad/checkpoints/sam/sam_vit_h.pth

    PYTHONPATH=src python scripts/run_sam_inference.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 --cam all --timestamps 1742879639783 1742879642908 \\
        --checkpoint /mnt/zyc_wzh/DA3_lad/checkpoints/sam/sam_vit_h.pth \\
        --output preview/sam/

    # smaller / faster ViT-B / ViT-L variants also supported:
    ... --sam-type vit_b --checkpoint .../sam_vit_b_01ec64.pth
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--sam-type", default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model variant matching the checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bbox-pad-px", type=int, default=8,
        help="add this many pixels around the projected bbox AABB before "
        "feeding it as a SAM box prompt (helps SAM see the full object)",
    )
    parser.add_argument(
        "--multimask", action="store_true",
        help="ask SAM for 3 candidate masks per box and keep the highest-"
        "score one. Slightly slower but slightly better.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports here so --help stays fast
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    print(f"[load] SAM {args.sam_type} from {args.checkpoint}")
    t0 = time.time()
    sam = sam_model_registry[args.sam_type](checkpoint=args.checkpoint)
    sam.to(device=args.device).eval()
    predictor = SamPredictor(sam)
    print(f"  ↳ loaded in {time.time() - t0:.1f}s on {args.device}")

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
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

            t0 = time.time()
            predictor.set_image(frame.image)  # RGB uint8
            t_set = time.time() - t0

            mask_ids: list[int] = []
            masks: list[np.ndarray] = []
            scores: list[float] = []
            boxes: list[np.ndarray] = []

            t0 = time.time()
            for obj in frame.dynamic_objects or []:
                uv8, _ = project_3d_bbox(obj, frame.K, frame.T_wc, dist)
                if uv8 is None:
                    continue
                u_min, v_min, u_max, v_max = bbox_uv_aabb(uv8)
                # Pad + clip
                u_min = max(0, int(u_min) - args.bbox_pad_px)
                v_min = max(0, int(v_min) - args.bbox_pad_px)
                u_max = min(W - 1, int(u_max) + args.bbox_pad_px)
                v_max = min(H - 1, int(v_max) + args.bbox_pad_px)
                if u_max <= u_min or v_max <= v_min:
                    continue
                box = np.array([u_min, v_min, u_max, v_max], dtype=np.float64)

                pred_masks, pred_scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=args.multimask,
                )
                # pred_masks shape: (n_candidates, H, W) bool
                if args.multimask:
                    best = int(np.argmax(pred_scores))
                else:
                    best = 0

                mask_ids.append(int(obj.id))
                masks.append(pred_masks[best].astype(bool))
                scores.append(float(pred_scores[best]))
                boxes.append(box)

            t_pred = time.time() - t0

            out_path = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam.npz"
            if not mask_ids:
                # Save an empty record so downstream tooling can still
                # detect "we ran SAM here, found nothing" vs "we never ran".
                np.savez_compressed(
                    out_path,
                    mask_ids=np.zeros(0, dtype=np.int32),
                    masks=np.zeros((0, H, W), dtype=bool),
                    scores=np.zeros(0, dtype=np.float32),
                    boxes=np.zeros((0, 4), dtype=np.float64),
                    image_hw=np.array([H, W], dtype=np.int32),
                    scene_id=scene_id, ts_ms=int(ts_ms), cam_id=cam_id,
                    sam_model=args.sam_type,
                )
                print(
                    f"  ts={ts_ms} cam{cam_id}  set_image {t_set*1000:.0f}ms  "
                    f"no V2X bboxes to prompt"
                )
                continue

            np.savez_compressed(
                out_path,
                mask_ids=np.array(mask_ids, dtype=np.int32),
                masks=np.stack(masks, axis=0).astype(bool),
                scores=np.array(scores, dtype=np.float32),
                boxes=np.stack(boxes, axis=0),
                image_hw=np.array([H, W], dtype=np.int32),
                scene_id=scene_id, ts_ms=int(ts_ms), cam_id=cam_id,
                sam_model=args.sam_type,
            )
            avg_area = float(np.mean([m.sum() for m in masks]))
            print(
                f"  ts={ts_ms} cam{cam_id}  K={len(masks):>3}  "
                f"set_image {t_set*1000:.0f}ms  predict {t_pred*1000:.0f}ms  "
                f"avg score {np.mean(scores):.2f}  avg area {avg_area:.0f}px"
            )

    print()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
