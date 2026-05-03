"""Run SAM Automatic Mask Generation on a frame and save segment IDs.

Unlike ``run_sam_inference.py`` (which prompts SAM with V2X bboxes
and only segments dynamic objects), this script grids the entire
image with point prompts and recovers a *full-image* segmentation —
roads, sidewalks, buildings, poles, trees, signs, plus everything
else SAM can find.

The per-pixel **segment id** is the artefact we care about for the
region-aware refinement (Q1): two pixels in the same segment get
the same id, two pixels in different segments get different ids.
SAM Auto does NOT give class labels (it is class-agnostic), only
the partition of the image into instances/regions.

Output:

    <output>/<scene>_ts<ts>_cam<cam>_sam_auto.npz
        segment_id : (H, W) int32   — 0 = no segment, k>=1 = SAM mask k
        scores     : (K,) float32   — per-mask confidence
        areas      : (K,) int32     — pixel count per mask
        scene_id, ts_ms, cam_id

    <output>/<scene>_ts<ts>_cam<cam>_sam_auto.png  (optional --save-viz)
        coloured overlay for visual inspection.

Usage
-----
    HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src \\
    python scripts/run_sam_auto.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 --ts 1742879650704 --cam 3 \\
        --output preview/sam_auto/ \\
        --save-viz \\
        --model facebook/sam-vit-base
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.data import RoadsideV2XLoader


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


def _color_for_id(mid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(mid) * 2654435761 % (2**32))
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _save_segmap_viz(
    image: np.ndarray,
    segment_id: np.ndarray,
    out_path: Path,
    alpha: float = 0.55,
) -> None:
    from PIL import Image

    H, W = image.shape[:2]
    out = image.astype(np.float32).copy()
    for k in np.unique(segment_id):
        if k <= 0:
            continue
        m = segment_id == k
        col = np.array(_color_for_id(int(k)), dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * col[None]
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(out_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ts", default=None, type=int,
                        help="single ts; default = first in scene")
    parser.add_argument("--all-timestamps", action="store_true")
    parser.add_argument("--ts-shard", default=None,
                        help="k/N for parallel sharding across GPUs")
    parser.add_argument("--cam", default="3",
                        help="single cam id or 'all'")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model", default="facebook/sam-vit-base",
        help="HF model id; sam-vit-base is fast, sam-vit-huge is best",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--points-per-side", type=int, default=32,
        help="grid resolution for SAM Auto. 32 is HF default; 64 for "
        "denser segmentation but ~4x slower",
    )
    parser.add_argument(
        "--pred-iou-thresh", type=float, default=0.86,
        help="HF default 0.88; lower = more masks (more noise)",
    )
    parser.add_argument(
        "--stability-score-thresh", type=float, default=0.92,
    )
    parser.add_argument(
        "--min-mask-area", type=int, default=50,
        help="drop masks smaller than this (pixels)",
    )
    parser.add_argument(
        "--save-viz", action="store_true",
        help="also save a coloured overlay PNG next to the npz",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
    )
    parser.add_argument("--loader-min-points", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports here so --help is fast.
    print(f"[load] {args.model}")
    t0 = time.time()
    from PIL import Image
    from transformers import pipeline

    generator = pipeline(
        "mask-generation",
        model=args.model,
        device=args.device,
        points_per_batch=64,
    )
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

    # Build ts list.
    if args.all_timestamps:
        ts_list = list(scene.timestamps_ms)
    elif args.ts is not None:
        ts_list = [int(args.ts)]
    else:
        ts_list = [scene.timestamps_ms[0]]

    if args.ts_shard:
        try:
            shard_k_str, shard_n_str = args.ts_shard.split("/")
            shard_k, shard_n = int(shard_k_str), int(shard_n_str)
        except ValueError as e:
            raise SystemExit(f"--ts-shard must be 'k/N': {args.ts_shard!r}") from e
        ts_list = [ts for i, ts in enumerate(ts_list) if i % shard_n == shard_k]
        print(f"[shard] {shard_k}/{shard_n} -> {len(ts_list)} ts on this worker")

    cams = (
        ["0", "3", "6", "9"] if args.cam == "all" else [args.cam]
    )

    print(f"[scene] {scene_id}  ts_count={len(ts_list)}  cams={cams}")
    print()

    n_done = 0
    n_skipped = 0
    t_total = time.time()
    for ts_ms in ts_list:
      for cam_id in cams:
        out_npz = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam_auto.npz"
        if args.skip_existing and out_npz.is_file():
            n_skipped += 1
            continue
        try:
            idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
        except ValueError:
            print(f"  ts={ts_ms} cam{cam_id}: no frame")
            continue
        frame = loader.get_frame(idx)
        H, W = frame.image.shape[:2]

        t_inf = time.time()
        # HF mask-generation pipeline doesn't accept numpy arrays;
        # wrap the (H, W, 3) uint8 RGB into a PIL Image first.
        pil_image = Image.fromarray(frame.image)
        out = generator(
            pil_image,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_area,
        )
        # HF output: dict with 'masks' (list of HxW bool), 'scores' (list)
        masks = out["masks"]
        scores = out.get("scores", [1.0] * len(masks))

        # Build a single segment_id map; later masks overwrite earlier
        # ones in case of overlap. Sort by area descending so the
        # smallest (most specific) wins on top.
        areas = np.array([int(m.sum()) for m in masks], dtype=np.int32)
        order = np.argsort(-areas)
        segment_id = np.zeros((H, W), dtype=np.int32)
        for new_id, k in enumerate(order, start=1):
            segment_id[masks[k]] = new_id
        # Reorder scores / areas to match the new ids.
        scores_out = np.array([float(scores[k]) for k in order], dtype=np.float32)
        areas_out = areas[order]

        np.savez_compressed(
            out_npz,
            segment_id=segment_id,
            scores=scores_out,
            areas=areas_out,
            scene_id=scene_id,
            ts_ms=int(ts_ms),
            cam_id=cam_id,
            model=args.model,
            image_hw=np.array([H, W], dtype=np.int32),
        )
        if args.save_viz:
            viz_path = out_npz.with_suffix(".png")
            _save_segmap_viz(frame.image, segment_id, viz_path)

        elapsed = time.time() - t_inf
        print(
            f"  ts={ts_ms} cam{cam_id}  K={len(masks)}  "
            f"{elapsed:.1f}s  ->  {out_npz.name}"
            + ("   + .png" if args.save_viz else "")
        )
        n_done += 1

    print()
    print(
        f"done.  wrote {n_done}  skipped {n_skipped}  "
        f"total {time.time() - t_total:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
