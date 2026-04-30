"""Per-object temporal accumulation — Stage 3C headline experiment.

For one tracked V2X object (identified by ``--bbox-id``) across many
frames + cameras we build **two** dense object-local point clouds:

  1. ``<obj>_lidar.ply`` : the LiDAR ground truth — every LiDAR return
     that fell inside the object's 3D bbox at each frame, transformed
     into the object's local frame.
  2. ``<obj>_aa_had.ply``: the AA-HAD prediction — for every (camera,
     frame) where SAM has a mask of this object, solve the per-instance
     ``(a_i, b_i)``, unproject every mask pixel via the camera ray, and
     transform into the object's local frame.

Then compute the symmetric Chamfer distance between the two clouds. This
is the central per-object validation metric for the project: if our
metric depth is right, the AA-HAD cloud should hug the LiDAR cloud at
the centimeter scale.

Usage
-----
    PYTHONPATH=src python scripts/run_object_accumulation.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 \\
        --bbox-id 1 \\
        --d-paths-glob 'preview/da3/008_*_d.npz' \\
        --sam-mask-dir preview/sam/ \\
        --output preview/recon/

    # Full sweep across all bbox ids in the scene:
    PYTHONPATH=src python scripts/run_object_accumulation.py ... --all-bboxes

The script prints one summary line per object:

    bbox  1  Bus     N_frames=8   N_lidar=15234  N_aa_had=78912  chamfer=0.42 m

and writes ``recon/<scene>_obj<id>_<lidar|aa_had>.ply`` plus a JSON
summary ``recon/<scene>_obj<id>_recon.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from lidar_anchored_depth.alignment.bbox_anchor import (
    points_in_oriented_bbox,
)
from lidar_anchored_depth.alignment.height_anchor import (
    solve_affine_decoupled_mask_slope_lidar_offset,
    solve_affine_dense_lsq,
    solve_affine_dense_lsq_multi,
    solve_affine_dense_lsq_multi_with_lidar,
)
from lidar_anchored_depth.data import RoadsideV2XLoader
from lidar_anchored_depth.reconstruction import (
    chamfer_distance,
    depth_to_world_points,
    voxel_downsample,
    world_to_object_local,
    write_ply_xyz,
)


def _colorize_by_z(points: np.ndarray, base_color: tuple[int, int, int] | None = None) -> np.ndarray:
    """Color a (N, 3) cloud by its Z coordinate (jet-like ramp).

    If ``base_color`` is given, paint uniformly with that color instead.
    Useful for the LiDAR PLY so it's visually distinguishable from the
    AA-HAD prediction in viewers that don't differentiate by file.
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if base_color is not None:
        out = np.empty((n, 3), dtype=np.uint8)
        out[:] = np.asarray(base_color, dtype=np.uint8).reshape(1, 3)
        return out
    z = points[:, 2].astype(np.float64)
    z_lo, z_hi = float(np.percentile(z, 5)), float(np.percentile(z, 95))
    if z_hi - z_lo < 1e-6:
        z_hi = z_lo + 1.0
    t = np.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0)
    # Jet-like ramp: blue → cyan → green → yellow → red
    r = np.clip(1.5 - np.abs(4 * t - 3), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0.0, 1.0)
    return np.stack(
        [(r * 255).astype(np.uint8),
         (g * 255).astype(np.uint8),
         (b * 255).astype(np.uint8)],
        axis=1,
    )


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _load_sam_masks(
    sam_dir: Path | None,
    scene_id: str, ts_ms: int, cam_id: str,
) -> dict[int, np.ndarray]:
    if sam_dir is None:
        return {}
    p = sam_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam.npz"
    if not p.is_file():
        return {}
    data = np.load(p, allow_pickle=True)
    ids = list(data["mask_ids"])
    masks = data["masks"]
    return {int(i): masks[k].astype(bool) for k, i in enumerate(ids)}


def _frames_with_bbox(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cams: list[str],
    bbox_id: int,
) -> list[tuple[int, str]]:
    """Find every (ts_ms, cam_id) where this V2X bbox.id is annotated.

    The cam_id loop is purely for accumulation breadth; the V2X bbox
    annotation is camera-independent (world-frame), but each camera's
    frame independently provides image / SAM mask.
    """
    out: list[tuple[int, str]] = []
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    for ts in scene.timestamps_ms:
        # Use cam0 as the "discovery" camera since the dynamic_objects
        # list is shared across cameras for the same timestamp.
        try:
            idx = loader.find_frame_idx(scene_id, ts, cams[0])
        except ValueError:
            continue
        frame = loader.get_frame(idx)
        ids_here = {int(o.id) for o in frame.dynamic_objects or []}
        if bbox_id not in ids_here:
            continue
        for c in cams:
            out.append((ts, c))
    return out


def _all_bbox_ids_in_scene(
    loader: RoadsideV2XLoader,
    scene_id: str,
    cam_for_discovery: str,
) -> dict[int, str]:
    """Return ``{bbox_id: label}`` for every dynamic object ever seen
    in the scene from ``cam_for_discovery``."""
    out: dict[int, str] = {}
    scene = next(s for s in loader.scenes if s.scene_id == scene_id)
    for ts in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts, cam_for_discovery)
        except ValueError:
            continue
        frame = loader.get_frame(idx)
        for obj in frame.dynamic_objects or []:
            out.setdefault(int(obj.id), obj.label)
    return out


def _resolve_scene_id(loader: RoadsideV2XLoader, scene_arg: str) -> str:
    cands = [s for s in loader.scenes if s.scene_id == scene_arg]
    if cands:
        return cands[0].scene_id
    cands = [s for s in loader.scenes if s.scene_id.startswith(scene_arg + "_")]
    if not cands:
        raise SystemExit(f"no scene matched {scene_arg!r}")
    return cands[0].scene_id


# --------------------------------------------------------------------- #
# Main per-object work
# --------------------------------------------------------------------- #
def accumulate_one_object(
    loader: RoadsideV2XLoader,
    scene_id: str,
    bbox_id: int,
    *,
    cams: list[str],
    d_paths_index: dict[tuple[str, int, str], Path],
    sam_dir: Path | None,
    bbox_expand: float,
    bbox_clip_expand: float,
    ground_cut_m: float,
    voxel_size: float,
    z_min: float,
    z_max: float,
    dense_min_pixels: int,
    solver_mode: str = "per-camera",  # 'per-frame' | 'per-camera' | 'per-camera-lidar'
    z_target: str = "row-linear",      # 'row-linear' | 'ray-obb'
    lidar_weight: float = 100.0,
) -> dict | None:
    """Accumulate LiDAR + AA-HAD points for one bbox.id across all frames it appears in.

    Returns a dict with the per-object summary, or ``None`` if no frame
    yielded usable points.
    """
    # Walk every (ts, cam) where this bbox.id is annotated.
    candidates = _frames_with_bbox(loader, scene_id, cams, bbox_id)
    if not candidates:
        return None

    label = "Unknown"
    lidar_local: list[np.ndarray] = []
    aa_had_local: list[np.ndarray] = []
    aa_had_colors: list[np.ndarray] = []
    n_frames_lidar = 0
    n_frames_aa_had = 0

    # ---------------- pass 1: collect per-(cam, ts) frame data ----------
    # For solver_mode='per-camera' we need every (mask, d̃, K, T_wc, obj)
    # for one camera before we can solve a single (a, b). Even for
    # solver_mode='per-frame' the structure is uniform, so we use the
    # same collection pass for both modes.
    by_cam: dict[str, list[dict]] = {}
    lidar_data: list[tuple[np.ndarray, "DynamicObject"]] = []  # (P_world, obj)

    for ts_ms, cam_id in candidates:
        try:
            idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
        except ValueError:
            continue
        frame = loader.get_frame(idx)
        obj = next(
            (o for o in (frame.dynamic_objects or []) if int(o.id) == bbox_id),
            None,
        )
        if obj is None:
            continue
        label = obj.label

        # LiDAR side accumulates immediately (no solver involved).
        # Two independent slices of the LiDAR cloud:
        #   in_bbox_for_gt : ground-cut applied — used for the LiDAR
        #                    PLY and chamfer GT.
        #   in_bbox_for_solve : NO ground cut — used as anchor rows in
        #                       the per-camera-lidar solver. We want
        #                       every measurement we can get for the
        #                       solver, including wheel-contact returns.
        in_bbox_for_gt = points_in_oriented_bbox(
            frame.lidar_world, obj,
            expand=bbox_expand,
            z_local_min_offset=ground_cut_m,
        )
        if int(in_bbox_for_gt.sum()) > 0:
            P_world = frame.lidar_world[in_bbox_for_gt]
            lidar_local.append(world_to_object_local(P_world, obj))
            n_frames_lidar += 1

        in_bbox_for_solve = points_in_oriented_bbox(
            frame.lidar_world, obj, expand=bbox_expand,
        )
        lidar_world_in_bbox = (
            frame.lidar_world[in_bbox_for_solve]
            if int(in_bbox_for_solve.sum()) > 0
            else None
        )

        # AA-HAD side: stash for solver pass 2.
        sam_masks = _load_sam_masks(sam_dir, scene_id, ts_ms, cam_id)
        mask = sam_masks.get(bbox_id)
        if mask is None or not mask.any():
            continue
        d_path = d_paths_index.get((scene_id, ts_ms, cam_id))
        if d_path is None or not d_path.is_file():
            continue
        d_data = np.load(d_path, allow_pickle=True)
        d_image = d_data["depth"]
        if d_image.shape != frame.image.shape[:2]:
            continue

        by_cam.setdefault(cam_id, []).append({
            "ts_ms": ts_ms,
            "K": frame.K, "T_wc": frame.T_wc,
            "mask": mask, "d_pred_image": d_image,
            "Z_max": obj.Z_max, "Z_min": obj.Z_min,
            "obj": obj,
            "image_rgb": frame.image,
            "bbox_expand": bbox_expand,
            "lidar_world_in_bbox": lidar_world_in_bbox,
            "dist": frame.meta.get("distortion"),
        })

    # ---------------- pass 2: solve (a, b) per cam or per frame ---------
    # In per-camera mode we get ONE (a_c, b_c) per camera by stacking all
    # of that camera's frames into one LSQ. In per-frame mode we keep
    # solving per frame (the original Stage 3C behavior, kept for
    # ablation). The dict ``ab_by_frame[(cam, ts)] = (a, b)`` is the
    # cached result either way.
    ab_by_frame: dict[tuple[str, int], tuple[float, float]] = {}
    ab_by_cam: dict[str, tuple[float, float]] = {}

    if solver_mode == "per-camera":
        for cam_id, entries in by_cam.items():
            sol = solve_affine_dense_lsq_multi(
                entries,
                z_target=z_target,
                min_pixels_per_frame=dense_min_pixels,
            )
            if sol is None:
                continue
            ab_by_cam[cam_id] = sol
            for e in entries:
                ab_by_frame[(cam_id, e["ts_ms"])] = sol
    elif solver_mode == "per-camera-lidar":
        # M5 — per-camera joint LSQ that anchors (a, b) to the actual
        # in-bbox LiDAR points. Implies ray-OBB z_target on the mask side.
        for cam_id, entries in by_cam.items():
            sol = solve_affine_dense_lsq_multi_with_lidar(
                entries,
                lidar_weight=lidar_weight,
                min_pixels_per_frame=dense_min_pixels,
            )
            if sol is None:
                continue
            ab_by_cam[cam_id] = sol
            for e in entries:
                ab_by_frame[(cam_id, e["ts_ms"])] = sol
    elif solver_mode == "per-camera-lidar-offset":
        # M5b — slope from mask (ray-OBB), offset from LiDAR median.
        # Avoids the depth-compression artefact of joint-weighted M5
        # that arises when LiDAR's narrow d̃ range pollutes the slope.
        for cam_id, entries in by_cam.items():
            sol = solve_affine_decoupled_mask_slope_lidar_offset(
                entries,
                min_pixels_per_frame=dense_min_pixels,
            )
            if sol is None:
                continue
            ab_by_cam[cam_id] = sol
            for e in entries:
                ab_by_frame[(cam_id, e["ts_ms"])] = sol
    elif solver_mode == "per-frame":
        for cam_id, entries in by_cam.items():
            for e in entries:
                if z_target == "ray-obb":
                    sol = solve_affine_dense_lsq_multi(
                        [e], z_target="ray-obb",
                        min_pixels_per_frame=dense_min_pixels,
                    )
                else:
                    sol = solve_affine_dense_lsq(
                        K=e["K"], T_wc=e["T_wc"],
                        mask=e["mask"], d_pred_image=e["d_pred_image"],
                        Z_max=e["Z_max"], Z_min=e["Z_min"],
                        min_pixels=dense_min_pixels,
                    )
                if sol is None:
                    continue
                ab_by_frame[(cam_id, e["ts_ms"])] = sol
    else:
        raise ValueError(f"unknown solver_mode={solver_mode!r}")

    # ---------------- pass 3: apply (a, b) to each frame's mask ---------
    for cam_id, entries in by_cam.items():
        for e in entries:
            ab = ab_by_frame.get((cam_id, e["ts_ms"]))
            if ab is None:
                continue
            a_i, b_i = ab
            obj = e["obj"]
            mask = e["mask"]
            d_image = e["d_pred_image"]

            z_cam = (a_i * d_image + b_i).astype(np.float32)
            P_world_pred, colors = depth_to_world_points(
                depth=z_cam, image_rgb=e["image_rgb"],
                K=e["K"], T_wc=e["T_wc"],
                mask=mask, z_min=z_min, z_max=z_max,
            )
            if P_world_pred.shape[0] == 0:
                continue

            # Bbox volume clip (catches SAM-leak + LSQ elongation outliers).
            keep_in_bbox = points_in_oriented_bbox(
                P_world_pred, obj, expand=bbox_clip_expand,
            )
            if not keep_in_bbox.any():
                continue
            P_world_pred = P_world_pred[keep_in_bbox]
            colors = colors[keep_in_bbox]

            P_local_pred = world_to_object_local(P_world_pred, obj)
            if ground_cut_m > 0.0:
                keep = P_local_pred[:, 2] >= -obj.lwh[2] / 2.0 + ground_cut_m
                if not keep.any():
                    continue
                P_local_pred = P_local_pred[keep]
                colors = colors[keep]
            aa_had_local.append(P_local_pred)
            aa_had_colors.append(colors)
            n_frames_aa_had += 1

    if not lidar_local and not aa_had_local:
        return None

    summary: dict = {
        "scene_id": scene_id,
        "bbox_id": bbox_id,
        "label": label,
        "n_frames_lidar": n_frames_lidar,
        "n_frames_aa_had": n_frames_aa_had,
        "voxel_size_m": voxel_size,
    }

    # Aggregate + voxel-downsample
    if lidar_local:
        L = np.concatenate(lidar_local, axis=0)
        L_v, _ = voxel_downsample(L, voxel_size)
        summary["n_lidar_raw"] = int(L.shape[0])
        summary["n_lidar_voxel"] = int(L_v.shape[0])
        summary["lidar_local_points"] = L_v
    else:
        L_v = np.zeros((0, 3), dtype=np.float64)
        summary["n_lidar_raw"] = 0
        summary["n_lidar_voxel"] = 0
        summary["lidar_local_points"] = L_v

    if aa_had_local:
        A = np.concatenate(aa_had_local, axis=0)
        C = np.concatenate(aa_had_colors, axis=0)
        A_v, C_v = voxel_downsample(A, voxel_size, colors=C)
        summary["n_aa_had_raw"] = int(A.shape[0])
        summary["n_aa_had_voxel"] = int(A_v.shape[0])
        summary["aa_had_local_points"] = A_v
        summary["aa_had_local_colors"] = C_v
    else:
        A_v = np.zeros((0, 3), dtype=np.float64)
        C_v = np.zeros((0, 3), dtype=np.uint8)
        summary["n_aa_had_raw"] = 0
        summary["n_aa_had_voxel"] = 0
        summary["aa_had_local_points"] = A_v
        summary["aa_had_local_colors"] = C_v

    # Chamfer (only if both clouds non-empty)
    if L_v.shape[0] > 0 and A_v.shape[0] > 0:
        cd, stats = chamfer_distance(A_v, L_v)
        summary["chamfer_m"] = float(cd)
        summary["chamfer_stats"] = stats
    else:
        summary["chamfer_m"] = None
        summary["chamfer_stats"] = None

    return summary


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--bbox-id", type=int, default=None,
        help="V2X bbox.id to accumulate. Use --all-bboxes for a sweep.",
    )
    parser.add_argument(
        "--all-bboxes", action="store_true",
        help="run on every dynamic-object id seen in the scene",
    )
    parser.add_argument(
        "--cams", nargs="+", default=["0", "3", "6", "9"],
        help="pinhole cam ids to accumulate over",
    )
    parser.add_argument(
        "--d-paths-glob", required=True,
        help="single-quoted glob for the .npz files written by "
        "run_da3_inference.py, e.g. 'preview/da3/008_*_d.npz'",
    )
    parser.add_argument(
        "--sam-mask-dir", default=None,
        help="directory with SAM .npz files (if absent, only the LiDAR "
        "side is accumulated; no AA-HAD prediction)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.05,
        help="voxel size (m) for downsampling. Default 5 cm.",
    )
    parser.add_argument(
        "--bbox-expand", type=float, default=0.10,
        help="expand bbox half-extents by this many meters when picking "
        "LiDAR points inside the bbox",
    )
    parser.add_argument(
        "--ground-cut-m", type=float, default=0.10,
        help="drop the bottom this-many meters of the bbox volume from "
        "BOTH the LiDAR and AA-HAD clouds. V2X bbox annotations usually "
        "wrap a thin road-surface band beneath the vehicle; cutting it "
        "out yields cleaner per-object reconstruction. Default 0.10 m.",
    )
    parser.add_argument(
        "--lidar-color", default="height",
        choices=["height", "red", "white", "none"],
        help="how to color the LiDAR PLY (which has no native RGB): "
        "'height' = jet by Z, 'red' = uniform red, 'white' = uniform "
        "white, 'none' = XYZ-only (default 'height')",
    )
    parser.add_argument(
        "--bbox-clip-expand", type=float, default=0.30,
        help="meters of slack when clipping AA-HAD predicted points to "
        "lie inside the V2X 3D bbox. Removes (a) SAM-mask leakage to "
        "background and (b) dense-LSQ elongation along the camera ray. "
        "Default 0.30 m — tighten to 0.10 for cleaner volumes, or set "
        "to a very large number (e.g. 1000) to effectively disable.",
    )
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=250.0)
    parser.add_argument("--dense-min-pixels", type=int, default=50)
    parser.add_argument(
        "--loader-min-points", type=int, default=0,
        help="V2X annotation filter: drop bboxes with num_points below "
        "this. Default 0 to be permissive on interpolated frames.",
    )
    parser.add_argument(
        "--ply-binary", action="store_true",
        help="write PLY in binary (faster + smaller for large clouds)",
    )
    parser.add_argument(
        "--solver-mode", default="per-camera",
        choices=[
            "per-frame", "per-camera",
            "per-camera-lidar", "per-camera-lidar-offset",
        ],
        help="how to solve (a, b). "
        "'per-frame' = one fit per (cam, ts) (Stage 3C, sensitive to "
        "per-frame DA3 / SAM noise). "
        "'per-camera' = stack all of one camera's frames into one LSQ "
        "→ one (a, b) per camera per object (Stage 3D-A; eliminates "
        "frame-jitter shells). "
        "'per-camera-lidar' = M5 — per-camera joint LSQ + LiDAR points "
        "as additional anchor rows. Honors LiDAR but tends to compress "
        "the prediction along the camera ray when --lidar-weight is "
        "large (LiDAR's narrow d̃ range pollutes slope). "
        "'per-camera-lidar-offset' = M5b — slope from mask ray-OBB, "
        "offset from LiDAR median residual. Decoupled, no compression "
        "artefact. Recommended for the headline reconstruction. "
        "Default 'per-camera'.",
    )
    parser.add_argument(
        "--lidar-weight", type=float, default=100.0,
        help="when --solver-mode=per-camera-lidar: how many mask pixels "
        "one LiDAR row counts as in the LSQ. 100 means a single LiDAR "
        "anchor pulls (a, b) as hard as 100 mask pixels (LiDAR is "
        "centimeter-precise; mask pixels carry ~1%% d̃ noise). Default 100.",
    )
    parser.add_argument(
        "--z-target", default="row-linear",
        choices=["row-linear", "ray-obb"],
        help="how to assign per-pixel target depth in the LSQ. "
        "'row-linear' = linear interp from Z_max at mask top row to "
        "Z_min at mask bottom row (Stage 3C; biased for non-frontal "
        "cameras). 'ray-obb' = exact camera-axis depth from "
        "ray ↔ V2X 3D bbox surface intersection (Stage 3D-B; "
        "geometrically correct for any view direction). "
        "Default 'row-linear'; pair with --solver-mode per-camera and "
        "--z-target ray-obb for the headline reconstruction.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build d_path index keyed by (scene_id, ts_ms, cam_id)
    d_paths_index: dict[tuple[str, int, str], Path] = {}
    for p in sorted(glob.glob(args.d_paths_glob)):
        try:
            data = np.load(p, allow_pickle=True)
        except Exception:
            continue
        try:
            key = (str(data["scene_id"]), int(data["ts_ms"]), str(data["cam_id"]))
        except KeyError:
            continue
        d_paths_index[key] = Path(p)
    print(f"[load] {len(d_paths_index)} d̃ frames indexed")

    sam_dir = Path(args.sam_mask_dir) if args.sam_mask_dir else None
    if sam_dir is not None and not sam_dir.is_dir():
        raise SystemExit(f"--sam-mask-dir {sam_dir} not a directory")

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
        min_num_points=args.loader_min_points,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    scene_id = _resolve_scene_id(loader, args.scene)

    if args.all_bboxes:
        ids_labels = _all_bbox_ids_in_scene(loader, scene_id, args.cams[0])
        target_ids = sorted(ids_labels.keys())
        print(f"[scene] {scene_id}  found {len(target_ids)} dynamic bbox ids")
    else:
        if args.bbox_id is None:
            raise SystemExit("specify --bbox-id N or --all-bboxes")
        target_ids = [args.bbox_id]

    print(f"[run] cams={args.cams}  voxel={args.voxel_size}m")
    print()

    fmt = "  bbox {:>4}  {:<22} N_frames(L,A)=({:>3},{:>3})  N_pts(L,A)=({:>6},{:>6})  chamfer={:>8}"
    aggregated: list[dict] = []
    for bbox_id in target_ids:
        t0 = time.time()
        summary = accumulate_one_object(
            loader=loader, scene_id=scene_id, bbox_id=bbox_id,
            cams=args.cams,
            d_paths_index=d_paths_index, sam_dir=sam_dir,
            bbox_expand=args.bbox_expand,
            bbox_clip_expand=args.bbox_clip_expand,
            ground_cut_m=args.ground_cut_m,
            voxel_size=args.voxel_size,
            z_min=args.z_min, z_max=args.z_max,
            dense_min_pixels=args.dense_min_pixels,
            solver_mode=args.solver_mode,
            z_target=args.z_target,
            lidar_weight=args.lidar_weight,
        )
        if summary is None:
            print(f"  bbox {bbox_id:>4}  no usable frames")
            continue

        # Write PLYs
        L_pts = summary.pop("lidar_local_points")
        A_pts = summary.pop("aa_had_local_points")
        A_col = summary.pop("aa_had_local_colors")
        if L_pts.shape[0] > 0:
            lidar_colors: np.ndarray | None
            if args.lidar_color == "none":
                lidar_colors = None
            elif args.lidar_color == "height":
                lidar_colors = _colorize_by_z(L_pts)
            elif args.lidar_color == "red":
                lidar_colors = _colorize_by_z(L_pts, base_color=(220, 30, 30))
            elif args.lidar_color == "white":
                lidar_colors = _colorize_by_z(L_pts, base_color=(240, 240, 240))
            else:
                lidar_colors = None
            write_ply_xyz(
                out_dir / f"{scene_id}_obj{bbox_id}_lidar.ply",
                L_pts, lidar_colors, binary=args.ply_binary,
            )
        if A_pts.shape[0] > 0:
            write_ply_xyz(
                out_dir / f"{scene_id}_obj{bbox_id}_aa_had.ply",
                A_pts, A_col, binary=args.ply_binary,
            )

        chamfer_str = (
            f"{summary['chamfer_m']:.3f} m"
            if summary["chamfer_m"] is not None
            else "—"
        )
        print(fmt.format(
            bbox_id, summary["label"][:22],
            summary["n_frames_lidar"], summary["n_frames_aa_had"],
            summary["n_lidar_voxel"], summary["n_aa_had_voxel"],
            chamfer_str,
        ))

        # JSON-friendly summary (drop the numpy arrays)
        aggregated.append({
            k: v for k, v in summary.items() if k != "lidar_local_points"
            and k != "aa_had_local_points" and k != "aa_had_local_colors"
        })

    # Aggregate report
    chamfers = [
        s["chamfer_m"] for s in aggregated
        if s.get("chamfer_m") is not None and np.isfinite(s["chamfer_m"])
    ]
    print()
    print("=" * 64)
    print(f"AGGREGATE over {len(aggregated)} objects "
          f"({len(chamfers)} with valid chamfer)")
    print("=" * 64)
    if chamfers:
        c = np.array(chamfers)
        print(f"  median chamfer: {float(np.median(c)):.3f} m")
        print(f"  p25 / p75      : {float(np.percentile(c, 25)):.3f} / "
              f"{float(np.percentile(c, 75)):.3f} m")
        print(f"  min  / max     : {float(c.min()):.3f} / {float(c.max()):.3f} m")

        # Per-class breakdown
        by_lab: dict[str, list[float]] = defaultdict(list)
        for s in aggregated:
            if s.get("chamfer_m") is not None:
                by_lab[s["label"]].append(s["chamfer_m"])
        print()
        print("Per-class median chamfer:")
        for lab in sorted(by_lab, key=lambda k: -len(by_lab[k])):
            vs = by_lab[lab]
            print(f"  {lab:<22} N={len(vs):>3}  median {np.median(vs):.3f} m")

    json_path = out_dir / f"{scene_id}_recon_summary.json"
    json_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print()
    print(f"  ↳ json : {json_path}")
    print(f"  ↳ ply  : {out_dir}/{scene_id}_obj*_*.ply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
