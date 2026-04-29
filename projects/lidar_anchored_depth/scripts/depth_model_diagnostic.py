"""Diagnostic: which depth model best maps DA3's ``d̃`` to LiDAR ``z``?

Tests three candidate functional forms over the (d̃, z) pairs sampled at
LiDAR projection pixels, then reports per-range RMSE and saves a scatter
+ residual figure. The output is the *empirical* answer to the
fundamental question:

    "Should HAD's affine be applied in z-space, in 1/z-space, or do we
    need a non-linear correction (g_φ)?"

Models compared:

    Linear (current HAD assumption):    z   = a · d̃ + b
    Inverse (disparity-like):           1/z = a · d̃ + b
    Quadratic (non-linear distortion):  z   = a · d̃² + b · d̃ + c

All RMSEs are computed in z-space (meters) so they are directly
comparable. A verdict line at the bottom of stdout names the model with
the lowest overall RMSE, and a JSON summary is written for downstream
ablation tables.

Usage
-----
    python scripts/depth_model_diagnostic.py \\
        --data-root /mnt/car_road_data_TianJin \\
        --scene 008 --timestamp 1742879642908 --cam 3 \\
        --d-path preview/da3/008_*_ts1742879642908_cam3_d.npz \\
        --output preview/diag/

    # multiple frames at once for a more robust verdict (one PNG + JSON each)
    for cam in 0 3 6 9; do
        python scripts/depth_model_diagnostic.py ... --cam $cam --d-path .../cam${cam}_d.npz ...
    done
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.alignment.projection import world_to_image
from lidar_anchored_depth.data import RoadsideV2XLoader


# --------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------- #
def fit_linear(d: np.ndarray, z: np.ndarray) -> dict:
    A = np.stack([d, np.ones_like(d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    z_pred = sol[0] * d + sol[1]
    return {
        "name": "linear",
        "form": "z = a · d̃ + b",
        "params": {"a": float(sol[0]), "b": float(sol[1])},
        "z_pred": z_pred,
        "rmse": float(np.sqrt(np.mean((z_pred - z) ** 2))),
    }


def fit_inverse(d: np.ndarray, z: np.ndarray) -> dict:
    A = np.stack([d, np.ones_like(d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, 1.0 / z, rcond=None)
    inv_pred = sol[0] * d + sol[1]
    valid = inv_pred > 1e-6
    z_pred = np.full_like(z, np.nan)
    z_pred[valid] = 1.0 / inv_pred[valid]
    rmse = (
        float(np.sqrt(np.mean((z_pred[valid] - z[valid]) ** 2)))
        if valid.any()
        else float("inf")
    )
    return {
        "name": "inverse",
        "form": "1/z = a · d̃ + b",
        "params": {"a": float(sol[0]), "b": float(sol[1])},
        "z_pred": z_pred,
        "rmse": rmse,
        "valid_mask": valid,
    }


def fit_quadratic(d: np.ndarray, z: np.ndarray) -> dict:
    A = np.stack([d ** 2, d, np.ones_like(d)], axis=1)
    sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    z_pred = sol[0] * d ** 2 + sol[1] * d + sol[2]
    return {
        "name": "quadratic",
        "form": "z = a · d̃² + b · d̃ + c",
        "params": {"a": float(sol[0]), "b": float(sol[1]), "c": float(sol[2])},
        "z_pred": z_pred,
        "rmse": float(np.sqrt(np.mean((z_pred - z) ** 2))),
    }


def stratified_rmse(z: np.ndarray, residuals: np.ndarray) -> dict[str, float]:
    """Per-range RMSE on residuals (already pred − true)."""
    bins = [(0, 20), (20, 50), (50, 100), (100, 200)]
    out: dict[str, float] = {}
    for lo, hi in bins:
        mask = (z >= lo) & (z < hi) & np.isfinite(residuals)
        n = int(mask.sum())
        out[f"{lo:>3}-{hi:<3}m  (n={n})"] = (
            float(np.sqrt(np.mean(residuals[mask] ** 2)))
            if n > 0
            else float("nan")
        )
    return out


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #
def make_figure(
    d: np.ndarray, z: np.ndarray,
    fits: list[dict], out_path: Path,
    *, scene_id: str, ts_ms: int, cam_id: str, model_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subsample for plot speed
    n_plot = min(8000, len(d))
    rng = np.random.default_rng(0)
    sub = rng.choice(len(d), n_plot, replace=False)

    # Left: scatter (d, z) with three fits overlaid. Clip y-axis to data
    # range with a small margin so the inverse fit's blow-up doesn't
    # squash the actual point cloud into a bottom strip.
    ax = axes[0]
    ax.scatter(d[sub], z[sub], s=2, alpha=0.25, color="0.5", label=f"data (N={len(d)})")
    d_grid = np.linspace(float(np.percentile(d, 1)), float(np.percentile(d, 99)), 300)
    colors = {"linear": "#d62728", "inverse": "#2ca02c", "quadratic": "#1f77b4"}
    for fit in fits:
        if fit["name"] == "linear":
            zg = fit["params"]["a"] * d_grid + fit["params"]["b"]
        elif fit["name"] == "inverse":
            inv = fit["params"]["a"] * d_grid + fit["params"]["b"]
            zg = np.where(inv > 1e-6, 1.0 / np.clip(inv, 1e-6, None), np.nan)
        else:
            p = fit["params"]
            zg = p["a"] * d_grid ** 2 + p["b"] * d_grid + p["c"]
        ax.plot(
            d_grid, zg, color=colors[fit["name"]], lw=1.5,
            label=f'{fit["name"]} ({fit["form"]})  RMSE={fit["rmse"]:.2f} m',
        )
    ax.set_xlabel("d̃  (DA3 relative depth)")
    ax.set_ylabel("z  (LiDAR camera-axis depth, meters)")
    ax.set_title(
        f"{scene_id}  ts={ts_ms}  cam{cam_id}  model={model_name}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    z_top = float(np.percentile(z, 99.5))
    ax.set_ylim(min(0.0, float(z.min())), z_top * 1.05)

    # Right: residuals vs z, log-y for visibility
    ax2 = axes[1]
    for fit in fits:
        z_pred = fit["z_pred"]
        resid = z_pred - z
        if fit["name"] == "inverse":
            valid = fit["valid_mask"]
            ax2.scatter(
                z[sub & valid[sub] if False else sub][valid[sub]],
                resid[sub][valid[sub]],
                s=2, alpha=0.25, color=colors[fit["name"]],
                label=fit["name"],
            )
        else:
            ax2.scatter(
                z[sub], resid[sub],
                s=2, alpha=0.25, color=colors[fit["name"]],
                label=fit["name"],
            )
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("true z  (meters)")
    ax2.set_ylabel("residual  (z_pred − z) [m]")
    ax2.set_title("Per-sample residual by model")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    # Clip residual y-axis to a meaningful range; the inverse model's
    # outliers near the d̃ → 1/0 region can be many km and squash the
    # informative part of the plot.
    lin_resid = fits[0]["z_pred"] - z
    quad_resid = fits[2]["z_pred"] - z
    ref = np.concatenate([np.abs(lin_resid), np.abs(quad_resid)])
    if np.isfinite(ref).any():
        ymax = float(np.percentile(ref[np.isfinite(ref)], 99.9)) * 1.5
        ymax = max(ymax, 5.0)
        ax2.set_ylim(-ymax, ymax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--timestamp", required=True, type=str)
    parser.add_argument("--cam", required=True)
    parser.add_argument(
        "--d-path", required=True,
        help="path to the .npz produced by run_da3_inference.py "
        "(globs allowed for convenience)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--z-min", type=float, default=1.0,
        help="reject LiDAR samples with z < z_min (meters)",
    )
    parser.add_argument(
        "--z-max", type=float, default=200.0,
        help="reject LiDAR samples with z > z_max (meters)",
    )
    parser.add_argument(
        "--conf-percentile", type=float, default=0.0,
        help="if > 0, drop LiDAR samples whose DA3 confidence falls "
        "below this percentile (e.g. 20.0)",
    )
    args = parser.parse_args()

    # Resolve glob in --d-path
    d_paths = sorted(glob.glob(args.d_path))
    if not d_paths:
        raise SystemExit(f"no file matched --d-path={args.d_path!r}")
    if len(d_paths) > 1:
        raise SystemExit(
            f"--d-path matched {len(d_paths)} files; please be specific"
        )
    d_path = Path(d_paths[0])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DA3 depth
    data = np.load(d_path, allow_pickle=True)
    d_image = data["depth"]
    conf_image = data.get("conf") if hasattr(data, "get") else (
        data["conf"] if "conf" in data.files else None
    )
    model_name = (
        str(data["model"]) if "model" in data.files else "unknown"
    )

    # Load frame
    loader = RoadsideV2XLoader(
        data_root=args.data_root,
        scenes=[args.scene] if "_" in args.scene else None,
    )
    if "_" not in args.scene:
        loader.scene_filter = [args.scene]
    full_scene_id = next(
        (s.scene_id for s in loader.scenes if s.scene_id.startswith(args.scene)),
        None,
    )
    if full_scene_id is None:
        raise SystemExit(f"no scene matched {args.scene!r}")
    idx = loader.find_frame_idx(full_scene_id, int(args.timestamp), args.cam)
    frame = loader.get_frame(idx)

    H, W = frame.image.shape[:2]
    if d_image.shape != (H, W):
        raise SystemExit(
            f"depth image shape {d_image.shape} != frame image {H, W}; "
            "regenerate with run_da3_inference.py"
        )

    # Project LiDAR
    dist = frame.meta.get("distortion")
    uv, z_lidar, in_front = world_to_image(
        frame.lidar_world, frame.K, frame.T_wc, dist,
    )
    if uv.size == 0:
        raise SystemExit("no LiDAR points project in front of the camera")

    # Sample d̃ and conf at LiDAR pixel locations.
    # Filter NaN/Inf from uv first (cv2.projectPoints with strong distortion
    # can yield non-finite values for points near the unprojection boundary).
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    z_lidar = z_lidar[finite]
    if uv.size == 0:
        raise SystemExit("no finite uv after distortion projection")

    uv_int = np.round(uv).astype(np.int64)
    in_image = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W)
        & (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )
    u = uv_int[in_image, 0]
    v = uv_int[in_image, 1]
    d_at = d_image[v, u].astype(np.float64)
    z_at = z_lidar[in_image].astype(np.float64)
    conf_at = conf_image[v, u].astype(np.float64) if conf_image is not None else None

    # Validity filter
    valid = (
        np.isfinite(d_at)
        & np.isfinite(z_at)
        & (z_at >= args.z_min)
        & (z_at <= args.z_max)
    )
    if conf_at is not None and args.conf_percentile > 0:
        thr = np.nanpercentile(conf_at, args.conf_percentile)
        valid &= conf_at >= thr

    d = d_at[valid]
    z = z_at[valid]
    n = d.size
    if n < 100:
        raise SystemExit(
            f"only {n} valid (d̃, z) pairs after filtering; need at least 100"
        )

    print()
    print(
        f"[diag] {full_scene_id}  ts={args.timestamp}  cam{args.cam}  "
        f"model={model_name}"
    )
    print(f"  N pairs (after filter): {n}")
    print(
        f"  d̃ range  : [{d.min():.4f}, {d.max():.4f}]  "
        f"median {np.median(d):.4f}"
    )
    print(
        f"  z range  : [{z.min():.4f}, {z.max():.4f}]  "
        f"median {np.median(z):.4f}"
    )
    print()

    # Fit
    fits = [fit_linear(d, z), fit_inverse(d, z), fit_quadratic(d, z)]

    # Print per-range RMSE
    fmt = "  {:<10}  {:>10}  {:>10}  {:>10}  {:>10}"
    print(fmt.format("range", "n", "linear", "inverse", "quadratic"))
    bins = [(0, 20), (20, 50), (50, 100), (100, 200)]
    summary_strat: dict = {}
    for lo, hi in bins:
        mask = (z >= lo) & (z < hi)
        if not mask.any():
            print(fmt.format(f"[{lo}-{hi}]", 0, "—", "—", "—"))
            continue
        rmses = []
        for fit in fits:
            res = fit["z_pred"][mask] - z[mask]
            res = res[np.isfinite(res)]
            rmse = float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan")
            rmses.append(rmse)
        print(fmt.format(
            f"[{lo}-{hi}]", int(mask.sum()),
            f"{rmses[0]:>9.3f}",
            f"{rmses[1]:>9.3f}",
            f"{rmses[2]:>9.3f}",
        ))
        summary_strat[f"{lo}-{hi}"] = {
            "n": int(mask.sum()),
            "linear": rmses[0], "inverse": rmses[1], "quadratic": rmses[2],
        }
    print()
    print("  overall RMSE (m):")
    for fit in fits:
        print(f"    {fit['name']:<10}  {fit['rmse']:>8.3f}    {fit['form']}")

    # Verdict
    fits_sorted = sorted(fits, key=lambda f: f["rmse"])
    best = fits_sorted[0]
    second = fits_sorted[1]
    margin = (second["rmse"] - best["rmse"]) / max(best["rmse"], 1e-6)
    confidence = "high" if margin > 0.10 else ("medium" if margin > 0.03 else "low")
    print()
    print(f"  VERDICT: {best['name']} ({best['form']}) — "
          f"RMSE={best['rmse']:.3f} m, "
          f"margin over runner-up = {margin*100:.1f}% ({confidence} confidence)")

    # Plot
    fig_path = out_dir / (d_path.stem.replace("_d", "_diag") + ".png")
    make_figure(
        d, z, fits, fig_path,
        scene_id=full_scene_id, ts_ms=int(args.timestamp),
        cam_id=args.cam, model_name=model_name,
    )

    # JSON summary (cannot serialize numpy arrays)
    summary = {
        "scene_id": full_scene_id,
        "ts_ms": int(args.timestamp),
        "cam_id": args.cam,
        "model": model_name,
        "n_pairs": n,
        "d_min": float(d.min()), "d_max": float(d.max()),
        "z_min": float(z.min()), "z_max": float(z.max()),
        "fits": [
            {"name": f["name"], "form": f["form"],
             "params": f["params"], "rmse_m": f["rmse"]}
            for f in fits
        ],
        "stratified": summary_strat,
        "verdict": {
            "best": best["name"],
            "rmse_m": best["rmse"],
            "margin_over_runner_up": margin,
            "confidence": confidence,
        },
    }
    json_path = out_dir / (d_path.stem.replace("_d", "_diag") + ".json")
    json_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"  ↳ figure: {fig_path}")
    print(f"  ↳ json  : {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
