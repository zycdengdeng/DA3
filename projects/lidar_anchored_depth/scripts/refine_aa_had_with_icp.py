"""Refine per-object AA-HAD point clouds onto LiDAR via rigid ICP.

For every ``<scene>_obj{N}_aa_had.ply`` in a recon directory we look up
the matching ``<scene>_obj{N}_lidar.ply`` and run point-to-point rigid
ICP (4×4 SE(3)) that aligns the AA-HAD cloud onto the LiDAR cloud. The
refined AA-HAD cloud is written as ``..._aa_had_icp.ply`` with the
ORIGINAL per-point RGB preserved (only XYZ is transformed).

This is a non-blind refinement: it requires LiDAR ground truth at
inference time. The motivation is the deployment scenario — for a V2X
roadside rig, LiDAR is part of the deployment anyway, so using it to
align the camera-derived per-object clouds is just "use everything
available". The AA-HAD contribution remains the dense COLOR field; ICP
only fixes the residual SE(3) misalignment between AA-HAD's prediction
and LiDAR's measurement.

Usage
-----
    PYTHONPATH=src python scripts/refine_aa_had_with_icp.py \\
        --recon-dir preview/recon_3F_M5b/

Outputs alongside the inputs:
    <scene>_obj{N}_aa_had_icp.ply        — the refined cloud
    <scene>_recon_icp_summary.json       — chamfer before / after,
                                           per-object SE(3), gain×
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from lidar_anchored_depth.reconstruction.chamfer import chamfer_distance
from lidar_anchored_depth.reconstruction.icp import (
    apply_transform,
    icp_point_to_point,
)
from lidar_anchored_depth.reconstruction.io import (
    read_ply_xyz,
    read_ply_xyz_rgb,
    write_ply_xyz,
)


_AAHAD_PATTERN = re.compile(r"^(?P<stem>.+)_obj(?P<id>\d+)_aa_had\.ply$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recon-dir", required=True)
    parser.add_argument(
        "--max-iterations", type=int, default=30,
        help="ICP iteration cap (default 30)",
    )
    parser.add_argument(
        "--trim-percentile", type=float, default=80.0,
        help="keep only the closest k%% of correspondences each "
        "iteration (default 80; set to 100 to disable robustness)",
    )
    parser.add_argument(
        "--ply-binary", action="store_true",
        help="write refined PLYs in binary little-endian",
    )
    args = parser.parse_args()

    recon_dir = Path(args.recon_dir)
    if not recon_dir.is_dir():
        raise SystemExit(f"--recon-dir {recon_dir} not a directory")

    pairs: list[tuple[Path, Path, str]] = []
    for ply in sorted(recon_dir.glob("*_aa_had.ply")):
        m = _AAHAD_PATTERN.match(ply.name)
        if m is None:
            continue
        stem = m.group("stem")
        obj_id = m.group("id")
        lidar = recon_dir / f"{stem}_obj{obj_id}_lidar.ply"
        if not lidar.is_file():
            print(f"  skip {ply.name}: missing LiDAR pair {lidar.name}")
            continue
        pairs.append((ply, lidar, obj_id))

    if not pairs:
        raise SystemExit(f"no aa_had / lidar pairs found under {recon_dir}")
    print(f"[load] {len(pairs)} (aa_had, lidar) pairs in {recon_dir}")
    print()

    fmt = "  obj {:>4}  N(aa,L)=({:>6},{:>6})  init={:>7}  final={:>7}  gain={:>5}"
    print(fmt.format("id", "n_aa", "n_l", "before", "after", "×"))
    rows: list[dict] = []
    for ply_aa, ply_lidar, obj_id in pairs:
        aa_pts, aa_col = read_ply_xyz_rgb(ply_aa)
        lidar_pts = read_ply_xyz(ply_lidar)
        if aa_pts.shape[0] < 3 or lidar_pts.shape[0] < 3:
            print(f"  obj {obj_id}: skipping (too few points)")
            continue

        # Initial chamfer
        cd_before, _ = chamfer_distance(
            aa_pts.astype(np.float64), lidar_pts.astype(np.float64),
        )

        t0 = time.time()
        T, info = icp_point_to_point(
            aa_pts.astype(np.float64),
            lidar_pts.astype(np.float64),
            max_iterations=args.max_iterations,
            trim_percentile=args.trim_percentile,
        )
        t_icp = time.time() - t0

        aa_refined = apply_transform(aa_pts.astype(np.float64), T).astype(np.float32)
        cd_after, _ = chamfer_distance(
            aa_refined.astype(np.float64), lidar_pts.astype(np.float64),
        )

        out_path = ply_aa.with_name(ply_aa.stem + "_icp.ply")
        write_ply_xyz(out_path, aa_refined, aa_col, binary=args.ply_binary)

        gain = cd_before / max(cd_after, 1e-9)
        print(fmt.format(
            obj_id, int(aa_pts.shape[0]), int(lidar_pts.shape[0]),
            f"{cd_before:.3f}m", f"{cd_after:.3f}m", f"{gain:.1f}",
        ))
        rows.append({
            "obj_id": int(obj_id),
            "n_aa_had": int(aa_pts.shape[0]),
            "n_lidar": int(lidar_pts.shape[0]),
            "chamfer_before_m": float(cd_before),
            "chamfer_after_m": float(cd_after),
            "gain_x": float(gain),
            "icp_iterations": int(info["n_iterations"]),
            "icp_converged": bool(info["converged"]),
            "icp_time_ms": float(t_icp * 1000),
            "T_4x4": T.tolist(),
            "out_ply": str(out_path),
        })

    if rows:
        before = np.array([r["chamfer_before_m"] for r in rows])
        after = np.array([r["chamfer_after_m"] for r in rows])
        print()
        print("=" * 64)
        print(f"AGGREGATE over {len(rows)} objects")
        print("=" * 64)
        print(f"  median chamfer before : {float(np.median(before)):.3f} m")
        print(f"  median chamfer after  : {float(np.median(after)):.3f} m")
        print(f"  median gain           : {float(np.median(before / np.maximum(after, 1e-9))):.2f}×")

    out_json = recon_dir / "icp_refine_summary.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print()
    print(f"  ↳ json: {out_json}")
    print(f"  ↳ ply : {recon_dir}/<...>_aa_had_icp.ply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
