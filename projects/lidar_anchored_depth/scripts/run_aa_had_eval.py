"""AA-HAD per-instance vs global affine — the headline experiment.

For every V2X-annotated dynamic object visible in a frame, we:

  1. Solve a per-instance ``(a_i, b_i)`` via AA-HAD using the bbox's
     exact ``(Z_min, Z_max)`` as height anchors (no SAM needed —
     :mod:`alignment.bbox_anchor` projects the bbox AABB onto the image
     and uses that as a stand-in mask).
  2. Find LiDAR points truly inside the 3D bbox (oriented-bbox
     containment in world space, NOT just inside the projected AABB).
  3. Predict their metric depth two ways:
       z_pred_inst = a_i · d̃ + b_i           (AA-HAD per-instance)
       z_pred_glob = a_g · d̃ + b_g           (single global affine fit
                                              on ALL valid frame pairs)
  4. Report per-bbox RMSE for both. The per-instance / global ratio is
     the headline number for the paper's main claim:

         "Per-instance HAD reduces metric depth RMSE on roadside V2X
          objects from N m (single-affine baseline) to M m (per-instance
          height-anchored), an X× improvement."

Usage
-----
    python scripts/run_aa_had_eval.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --d-paths 'preview/da3/008_*_d.npz' \\
        --output preview/eval/

    # filter by class
    ... --classes Car Suv Truck Bus

    # bbox expansion to capture surface returns slightly outside the
    # nominal bbox (V2X annotation is not pixel-perfect)
    ... --bbox-expand 0.15
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from lidar_anchored_depth.alignment.bbox_anchor import (
    points_in_oriented_bbox,
)
from lidar_anchored_depth.alignment.global_scale import (
    b3_ransac_affine,
)
from lidar_anchored_depth.alignment.height_anchor import (
    mask_top_bottom_pixels,
    solve_affine_dense_lsq,
    solve_affine_from_height_anchors,
)
from lidar_anchored_depth.alignment.projection import (
    bbox_uv_aabb,
    project_3d_bbox,
    world_to_image,
)
from lidar_anchored_depth.data import RoadsideV2XLoader


def _aabb_to_mask(
    uv8: np.ndarray, image_hw: tuple[int, int]
) -> np.ndarray:
    """Build a (H, W) bool mask = AABB of the 8 projected bbox corners."""
    H, W = image_hw
    u_min, v_min, u_max, v_max = bbox_uv_aabb(uv8)
    u_lo = max(0, int(np.floor(u_min)))
    u_hi = min(W, int(np.ceil(u_max)) + 1)
    v_lo = max(0, int(np.floor(v_min)))
    v_hi = min(H, int(np.ceil(v_max)) + 1)
    mask = np.zeros((H, W), dtype=bool)
    if u_lo < u_hi and v_lo < v_hi:
        mask[v_lo:v_hi, u_lo:u_hi] = True
    return mask


def _filter_finite_inside_image(
    uv: np.ndarray, z: np.ndarray, hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter NaN uv + clip to image bounds. Returns (uv, z, u_int, v_int)."""
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    z = z[finite]
    if uv.size == 0:
        return uv, z, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    uv_int = np.round(uv).astype(np.int64)
    H, W = hw
    inside = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    return (
        uv[inside],
        z[inside],
        uv_int[inside, 0],
        uv_int[inside, 1],
    )


def _load_sam_masks(
    sam_dir: Path | None,
    scene_id: str, ts_ms: int, cam_id: str,
) -> dict[int, np.ndarray]:
    """Load ``{bbox.id: (H, W) bool mask}`` from a SAM .npz, or {} if absent."""
    if sam_dir is None:
        return {}
    p = sam_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam.npz"
    if not p.is_file():
        return {}
    data = np.load(p, allow_pickle=True)
    ids = list(data["mask_ids"])
    masks = data["masks"]
    return {int(i): masks[k].astype(bool) for k, i in enumerate(ids)}


def _eval_one_frame(
    loader: RoadsideV2XLoader,
    d_path: Path,
    *,
    classes: set[str] | None,
    bbox_expand: float,
    z_min: float,
    z_max: float,
    min_lidar_per_obj: int,
    sam_dir: Path | None = None,
    solver: str = "dense",
    dense_min_pixels: int = 50,
) -> tuple[list[dict], dict]:
    """Run AA-HAD on every dynamic object and compare to global baseline."""
    data = np.load(d_path, allow_pickle=True)
    d_image = data["depth"]
    scene_id = str(data["scene_id"])
    ts_ms = int(data["ts_ms"])
    cam_id = str(data["cam_id"])
    sam_masks = _load_sam_masks(sam_dir, scene_id, ts_ms, cam_id)

    idx = loader.find_frame_idx(scene_id, ts_ms, cam_id)
    frame = loader.get_frame(idx)
    H, W = frame.image.shape[:2]
    if d_image.shape != (H, W):
        raise SystemExit(
            f"depth shape {d_image.shape} != image {H, W}; "
            "regenerate with run_da3_inference.py"
        )

    K, T_wc = frame.K, frame.T_wc
    dist = frame.meta.get("distortion")

    # 1. Project ALL LiDAR (used for global baseline)
    uv_all, z_all, _ = world_to_image(frame.lidar_world, K, T_wc, dist)
    _, z_all_f, u_all, v_all = _filter_finite_inside_image(
        uv_all, z_all, (H, W)
    )
    valid_g = (z_all_f >= z_min) & (z_all_f <= z_max)
    if valid_g.sum() < 100:
        return [], {
            "scene_id": scene_id, "ts_ms": ts_ms, "cam_id": cam_id,
            "n_lidar_in_frame": int(valid_g.sum()),
            "skipped": True, "reason": "too few global samples",
        }

    d_at_all = d_image[v_all[valid_g], u_all[valid_g]].astype(np.float64)
    z_at_all = z_all_f[valid_g].astype(np.float64)
    valid_d = np.isfinite(d_at_all) & (d_at_all > 0)
    fit_global = b3_ransac_affine(
        d_at_all[valid_d], z_at_all[valid_d],
        threshold_rel=0.10, threshold_abs=1.0, n_iterations=300, seed=0,
    )

    # 2. Per-object loop, with dropout counters for debugging
    drops = {
        "total": 0, "class_filter": 0, "behind_camera": 0, "empty_mask": 0,
        "mask_no_topbot": 0, "d_nan": 0, "d_degenerate": 0,
        "solver_singular": 0, "few_lidar_3d": 0, "few_lidar_in_image": 0,
        "few_lidar_valid": 0,
    }
    n_sam_used = 0
    n_aabb_fallback = 0
    rows: list[dict] = []
    for obj in frame.dynamic_objects or []:
        drops["total"] += 1
        if classes is not None and obj.label not in classes:
            drops["class_filter"] += 1
            continue

        # 2a. Mask source: SAM (if loaded for this bbox.id) else AABB.
        if int(obj.id) in sam_masks:
            mask = sam_masks[int(obj.id)]
            if mask.shape != (H, W):
                # Defensive — SAM mask must match the d_image / frame
                # resolution we are about to read d̃ from.
                drops["empty_mask"] += 1
                continue
            n_sam_used += 1
        else:
            uv8, _ = project_3d_bbox(obj, K, T_wc, dist)
            if uv8 is None:
                drops["behind_camera"] += 1
                continue
            mask = _aabb_to_mask(uv8, (H, W))
            n_aabb_fallback += 1
        if not mask.any():
            drops["empty_mask"] += 1
            continue

        # 2b. Solve (a_i, b_i)
        if solver == "dense":
            sol = solve_affine_dense_lsq(
                K=K, T_wc=T_wc, mask=mask, d_pred_image=d_image,
                Z_max=obj.Z_max, Z_min=obj.Z_min,
                min_pixels=dense_min_pixels,
            )
            if sol is None:
                drops["solver_singular"] += 1
                continue
            a_i, b_i = sol
        else:  # solver == "2anchor"
            topbot = mask_top_bottom_pixels(mask)
            if topbot is None:
                drops["mask_no_topbot"] += 1
                continue
            uv_top, uv_bot = topbot
            d_top = float(d_image[int(uv_top[1]), int(uv_top[0])])
            d_bot = float(d_image[int(uv_bot[1]), int(uv_bot[0])])
            if not (np.isfinite(d_top) and np.isfinite(d_bot)):
                drops["d_nan"] += 1
                continue
            if abs(d_top - d_bot) < 1e-9:
                drops["d_degenerate"] += 1
                continue
            sol = solve_affine_from_height_anchors(
                K=K, T_wc=T_wc,
                uv_top=uv_top, uv_bot=uv_bot,
                d_pred_top=d_top, d_pred_bot=d_bot,
                Z_max=obj.Z_max, Z_min=obj.Z_min,
            )
            if sol is None:
                drops["solver_singular"] += 1
                continue
            a_i, b_i = sol

        # 2c. Find LiDAR points actually inside the 3D bbox
        in_bbox_3d = points_in_oriented_bbox(
            frame.lidar_world, obj, expand=bbox_expand,
        )
        if int(in_bbox_3d.sum()) < min_lidar_per_obj:
            drops["few_lidar_3d"] += 1
            continue

        # 2d. Project those points; sample d̃; predict both ways
        pts = frame.lidar_world[in_bbox_3d]
        uv_obj, z_obj, _ = world_to_image(pts, K, T_wc, dist)
        _, z_obj, u_obj, v_obj = _filter_finite_inside_image(
            uv_obj, z_obj, (H, W)
        )
        if u_obj.size < min_lidar_per_obj:
            drops["few_lidar_in_image"] += 1
            continue

        d_obj = d_image[v_obj, u_obj].astype(np.float64)
        valid = (
            np.isfinite(d_obj) & (d_obj > 0)
            & (z_obj >= z_min) & (z_obj <= z_max)
        )
        if int(valid.sum()) < min_lidar_per_obj:
            drops["few_lidar_valid"] += 1
            continue
        d_obj = d_obj[valid]
        z_obj = z_obj[valid].astype(np.float64)

        z_inst = a_i * d_obj + b_i
        z_glob = fit_global.a * d_obj + fit_global.b
        rmse_inst = float(np.sqrt(np.mean((z_inst - z_obj) ** 2)))
        rmse_glob = float(np.sqrt(np.mean((z_glob - z_obj) ** 2)))
        gain = rmse_glob / max(rmse_inst, 1e-6)

        rows.append({
            "scene_id": scene_id, "ts_ms": ts_ms, "cam_id": cam_id,
            "id": int(obj.id), "label": obj.label,
            "n_lidar_in_bbox": int(valid.sum()),
            "z_min": float(z_obj.min()), "z_max": float(z_obj.max()),
            "z_median": float(np.median(z_obj)),
            "a_i": float(a_i), "b_i": float(b_i),
            "rmse_aa_had": rmse_inst,
            "rmse_global": rmse_glob,
            "gain_x": float(gain),
            "occlusion": int(obj.occlusion),
            "num_points_v2x": int(obj.num_points),
            "mask_source": "sam" if int(obj.id) in sam_masks else "aabb",
        })

    summary = {
        "scene_id": scene_id, "ts_ms": ts_ms, "cam_id": cam_id,
        "n_lidar_in_frame": int(valid_g.sum()),
        "n_objects_kept": len(rows),
        "n_objects_total": drops["total"],
        "drops": drops,
        "n_sam_used": n_sam_used,
        "n_aabb_fallback": n_aabb_fallback,
        "n_sam_loaded": len(sam_masks),
        "global_a": float(fit_global.a),
        "global_b": float(fit_global.b),
        "global_rmse_full": float(fit_global.residual_rmse),
    }
    return rows, summary


def _print_per_object_table(rows: list[dict]) -> None:
    if not rows:
        return
    fmt = "  {:<22} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8} {:>6}"
    print(fmt.format(
        "scene/ts/cam", "id", "lab", "n_lidar",
        "z_med", "AA-HAD", "global", "gain×",
    ))
    for r in rows[:40]:
        print(fmt.format(
            f"{r['scene_id'].split('_')[0]}/{r['ts_ms'] % 100000}/c{r['cam_id']}",
            r["id"],
            r["label"][:5],
            r["n_lidar_in_bbox"],
            f"{r['z_median']:.1f}",
            f"{r['rmse_aa_had']:.2f}",
            f"{r['rmse_global']:.2f}",
            f"{r['gain_x']:.1f}",
        ))
    if len(rows) > 40:
        print(f"  ... and {len(rows) - 40} more (see JSON)")


def _print_summary_stats(rows: list[dict]) -> dict:
    if not rows:
        print("\n  no usable rows.")
        return {}

    rmse_inst = np.array([r["rmse_aa_had"] for r in rows])
    rmse_glob = np.array([r["rmse_global"] for r in rows])
    gain = np.array([r["gain_x"] for r in rows])

    print()
    print("=" * 64)
    print(f"AGGREGATE over {len(rows)} V2X objects across all frames")
    print("=" * 64)
    fmt = "  {:<38} {:>10}"
    print(fmt.format("median per-object RMSE  AA-HAD :", f"{np.median(rmse_inst):.3f} m"))
    print(fmt.format("median per-object RMSE  global :", f"{np.median(rmse_glob):.3f} m"))
    print(fmt.format("median improvement (global / AA-HAD):",
                     f"{np.median(gain):.2f}×"))
    print(fmt.format("p25 / p75 RMSE  AA-HAD :",
                     f"{np.percentile(rmse_inst, 25):.2f} / {np.percentile(rmse_inst, 75):.2f} m"))
    print(fmt.format("p25 / p75 RMSE  global :",
                     f"{np.percentile(rmse_glob, 25):.2f} / {np.percentile(rmse_glob, 75):.2f} m"))

    # Per-mask-source split (if SAM was used at all)
    has_sam = any(r.get("mask_source") == "sam" for r in rows)
    if has_sam:
        print()
        print("Split by mask source:")
        fmt3 = "  {:<22} N={:>4}  AA-HAD {:>7}  global {:>7}  gain {:>5}"
        for src in ("sam", "aabb"):
            rs = [r for r in rows if r.get("mask_source") == src]
            if not rs:
                continue
            print(fmt3.format(
                f"mask_source={src}", len(rs),
                f"{median(r['rmse_aa_had'] for r in rs):.2f} m",
                f"{median(r['rmse_global'] for r in rs):.2f} m",
                f"{median(r['gain_x'] for r in rs):.1f}×",
            ))

    print()
    print("Per-class median RMSE:")
    by_lab: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_lab[r["label"]].append(r)
    fmt2 = "  {:<22} N={:>4}  AA-HAD {:>7}  global {:>7}  gain {:>5}"
    for lab in sorted(by_lab, key=lambda k: -len(by_lab[k])):
        rs = by_lab[lab]
        print(fmt2.format(
            lab, len(rs),
            f"{median(r['rmse_aa_had'] for r in rs):.2f} m",
            f"{median(r['rmse_global'] for r in rs):.2f} m",
            f"{median(r['gain_x'] for r in rs):.1f}×",
        ))

    return {
        "n_objects": len(rows),
        "median_rmse_aa_had": float(np.median(rmse_inst)),
        "median_rmse_global": float(np.median(rmse_glob)),
        "median_gain_x": float(np.median(gain)),
        "p25_aa_had": float(np.percentile(rmse_inst, 25)),
        "p75_aa_had": float(np.percentile(rmse_inst, 75)),
        "p25_global": float(np.percentile(rmse_glob, 25)),
        "p75_global": float(np.percentile(rmse_glob, 75)),
    }


def _make_figure(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmse_inst = np.array([r["rmse_aa_had"] for r in rows])
    rmse_glob = np.array([r["rmse_global"] for r in rows])
    z_med = np.array([r["z_median"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    bins = np.linspace(0, max(20.0, np.percentile(rmse_glob, 95) * 1.1), 50)
    ax.hist(rmse_glob, bins=bins, color="#d62728", alpha=0.55,
            label=f"global affine  (median {np.median(rmse_glob):.2f} m)")
    ax.hist(rmse_inst, bins=bins, color="#1f77b4", alpha=0.7,
            label=f"AA-HAD          (median {np.median(rmse_inst):.2f} m)")
    ax.set_xlabel("per-object RMSE  (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-object metric depth RMSE (N={len(rows)} objects)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(z_med, rmse_glob, s=10, alpha=0.4, color="#d62728", label="global affine")
    ax2.scatter(z_med, rmse_inst, s=10, alpha=0.6, color="#1f77b4", label="AA-HAD")
    ax2.set_xlabel("object median z  (m)")
    ax2.set_ylabel("RMSE  (m)")
    ax2.set_title("RMSE vs object distance")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--d-paths", required=True,
        help="single-quoted glob for .npz files written by "
        "run_da3_inference.py. e.g. 'preview/da3/008_*_d.npz'",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="restrict to these labels (e.g. Car Suv Truck Bus)",
    )
    parser.add_argument(
        "--bbox-expand", type=float, default=0.10,
        help="m added to each bbox half-extent for 'point in bbox'",
    )
    parser.add_argument("--z-min", type=float, default=1.0)
    parser.add_argument("--z-max", type=float, default=200.0)
    parser.add_argument("--min-lidar-per-obj", type=int, default=10)
    parser.add_argument(
        "--loader-min-points", type=int, default=5,
        help="V2X annotation filter: drop bboxes with num_points below "
        "this. Set to 0 for interpolated frames where num_points field "
        "is always 0 (e.g. THICV-R2A's interpolation_labels at non-key "
        "frames). Default 5.",
    )
    parser.add_argument(
        "--sam-mask-dir", default=None,
        help="directory containing per-frame SAM .npz files written by "
        "run_sam_inference.py. When given, SAM masks are used; missing "
        "ids fall back to the projected-bbox AABB (and tagged accordingly "
        "in the per-object JSON).",
    )
    parser.add_argument(
        "--solver", default="dense", choices=["dense", "2anchor"],
        help="HAD solver: 'dense' uses every mask pixel via LSQ (robust "
        "to single-pixel d̃ noise; default), '2anchor' uses the original "
        "closed-form on (v_top, v_bot) — only useful for ablation.",
    )
    parser.add_argument(
        "--dense-min-pixels", type=int, default=50,
        help="minimum mask pixel count for the dense solver to attempt "
        "a fit (default 50)",
    )
    args = parser.parse_args()
    sam_dir = Path(args.sam_mask_dir) if args.sam_mask_dir else None

    paths = sorted(glob.glob(args.d_paths))
    if not paths:
        raise SystemExit(f"no file matched --d-paths={args.d_paths!r}")
    print(f"[load] {len(paths)} .npz frames")

    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        min_num_points=args.loader_min_points,
    )
    classes = set(args.classes) if args.classes else None

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_summaries: list[dict] = []
    for p in paths:
        rows, summary = _eval_one_frame(
            loader, Path(p),
            classes=classes,
            bbox_expand=args.bbox_expand,
            z_min=args.z_min, z_max=args.z_max,
            min_lidar_per_obj=args.min_lidar_per_obj,
            sam_dir=sam_dir,
            solver=args.solver,
            dense_min_pixels=args.dense_min_pixels,
        )
        all_rows.extend(rows)
        all_summaries.append(summary)
        kept = summary.get("n_objects_kept", 0)
        total = summary.get("n_objects_total", 0)
        line = (
            f"  {Path(p).name:<60}  "
            f"N_obj={kept}/{total}  "
            f"global_RMSE={summary.get('global_rmse_full', float('nan')):.2f} m"
        )
        if summary.get("n_sam_loaded", 0) > 0:
            line += (
                f"  SAM:{summary['n_sam_used']} "
                f"AABB-fallback:{summary['n_aabb_fallback']}"
            )
        if kept == 0 and total > 0:
            d = summary["drops"]
            top = sorted(
                ((k, v) for k, v in d.items() if k not in ("total",) and v > 0),
                key=lambda x: -x[1],
            )
            top_str = ", ".join(f"{k}={v}" for k, v in top)
            line += f"  ← all dropped: {top_str}"
        elif kept == 0 and total == 0:
            line += "  ← no annotated dynamic objects in this frame"
        print(line)

    print()
    _print_per_object_table(all_rows)
    aggregate = _print_summary_stats(all_rows)

    json_path = out_dir / "aa_had_eval.json"
    json_path.write_text(json.dumps({
        "frames": all_summaries,
        "objects": all_rows,
        "aggregate": aggregate,
    }, indent=2))

    fig_path = out_dir / "aa_had_eval.png"
    _make_figure(all_rows, fig_path)

    print()
    print(f"  ↳ json  : {json_path}")
    print(f"  ↳ figure: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
