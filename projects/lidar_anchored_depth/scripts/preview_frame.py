"""Visualize one (scene, timestamp, camera) frame from THICV-R2A.

Loads the calibrated frame (image + merged LiDAR + V2X annotations),
projects the LiDAR points and the 3D bboxes onto the pinhole image, and
saves an annotated PNG. This is the Stage 2A smoke test for the data
pipeline — once it produces a sensible-looking PNG on real data, we know
the loader, calibration, and projection geometry agree end-to-end.

Usage
-----
    python scripts/preview_frame.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 \\
        --timestamp 1742879642908 \\
        --cam 3 \\
        --out preview/

    # all 4 pinholes for the same timestamp:
    python scripts/preview_frame.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 \\
        --timestamp 1742879642908 \\
        --cam all \\
        --out preview/

    # default: pick the first valid timestamp of the scene
    python scripts/preview_frame.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 \\
        --out preview/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from lidar_anchored_depth.data import (
    PINHOLE_CAMERA_IDS,
    RoadsideV2XLoader,
)
from lidar_anchored_depth.viz.overlay import (
    color_for_label,
    draw_bbox_3d_overlay,
    draw_lidar_overlay,
    world_to_image,
)


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    """Accept either a full scene name or a numeric prefix like ``008``."""
    candidates = [s for s in loader.scenes if s.scene_id == scene_arg]
    if candidates:
        return candidates[0].scene_id
    candidates = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if candidates:
        return candidates[0].scene_id
    raise SystemExit(
        f"no scene matched {scene_arg!r} in {loader.data_root}"
    )


def _find_idx(
    loader: RoadsideV2XLoader, scene_id: str, ts_ms: int, cam_id: str
) -> int:
    if not loader._index_built:
        loader._build_index()
    n_cams = len(loader.cameras)
    running = 0
    for s in loader.scenes:
        if s.scene_id != scene_id:
            running += len(s) * n_cams
            continue
        try:
            ts_i = s.timestamps_ms.index(ts_ms)
        except ValueError as exc:
            raise SystemExit(
                f"timestamp {ts_ms} not present in {scene_id} "
                f"(first 3 ts: {s.timestamps_ms[:3]})"
            ) from exc
        try:
            cam_i = loader.cameras.index(cam_id)
        except ValueError as exc:
            raise SystemExit(f"camera {cam_id!r} not in loader.cameras") from exc
        return running + ts_i * n_cams + cam_i
    raise SystemExit(f"scene {scene_id} not in loader index")


def render_frame(
    loader: RoadsideV2XLoader,
    scene_id: str,
    ts_ms: int,
    cam_id: str,
    *,
    point_radius: int = 1,
    bbox_thickness: int = 2,
) -> tuple[bytes, dict]:
    """Build the overlay image as PNG bytes plus a small stats dict.

    Returns
    -------
    png_bytes, stats
    """
    idx = _find_idx(loader, scene_id, ts_ms, cam_id)
    frame = loader.get_frame(idx)

    bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    dist = frame.meta.get("distortion")

    uv, z, in_front = world_to_image(
        frame.lidar_world, frame.K, frame.T_wc, dist
    )
    out = draw_lidar_overlay(bgr, uv, z, radius=point_radius)

    n_drawn = 0
    for obj in frame.dynamic_objects or []:
        out, drew = draw_bbox_3d_overlay(
            out, obj, frame.K, frame.T_wc, dist,
            color=color_for_label(obj.label),
            thickness=bbox_thickness,
        )
        if drew:
            n_drawn += 1

    ok, encoded = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    stats = {
        "frame_id": frame.frame_id,
        "scene_id": scene_id,
        "ts_ms": ts_ms,
        "cam_id": cam_id,
        "image_hw": frame.image.shape[:2],
        "lidar_total": int(frame.lidar_world.shape[0]),
        "lidar_in_front": int(in_front.sum()),
        "lidar_drawn_in_image": int(uv.shape[0]),
        "dynamic_objects": len(frame.dynamic_objects or []),
        "bboxes_drawn": n_drawn,
    }
    return encoded.tobytes(), stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--scene", required=True,
        help="scene name (e.g. '008_car0325_road0327_t8') or numeric prefix '008'",
    )
    parser.add_argument(
        "--timestamp", default=None, type=str,
        help="ms timestamp; default = first available in the scene",
    )
    parser.add_argument(
        "--cam", default="3",
        help="pinhole cam id ({0,3,6,9}) or 'all' for all 4",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--bbox-thickness", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
    )
    if "_" not in args.scene:
        # numeric prefix; restrict scene_filter after _build_index runs
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)

    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    if args.timestamp is None:
        ts_ms = scene.timestamps_ms[0]
    else:
        ts_ms = int(args.timestamp)

    if args.cam == "all":
        cams = list(PINHOLE_CAMERA_IDS)
    else:
        cams = [args.cam]
        if args.cam not in PINHOLE_CAMERA_IDS:
            raise SystemExit(
                f"--cam={args.cam!r} not in pinhole set {PINHOLE_CAMERA_IDS}"
            )

    print(f"[scene] {scene_id}  ts={ts_ms}  cams={cams}")
    print(f"[out]   {out_dir}")
    print()

    for cam_id in cams:
        png_bytes, stats = render_frame(
            loader,
            scene_id,
            ts_ms,
            cam_id,
            point_radius=args.point_radius,
            bbox_thickness=args.bbox_thickness,
        )
        out_path = out_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}.png"
        out_path.write_bytes(png_bytes)
        print(
            f"  cam{cam_id}  ->  {out_path.name}    "
            f"LiDAR {stats['lidar_drawn_in_image']}/{stats['lidar_in_front']}/"
            f"{stats['lidar_total']}  "
            f"bbox {stats['bboxes_drawn']}/{stats['dynamic_objects']}"
        )
    print()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
