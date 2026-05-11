#!/usr/bin/env python3
"""Batch-run ``lad pipeline`` over multiple V2X scenes.

For each scene id given on the CLI (positional args), this script:

  1. Auto-resolves the per-scene calib JSON, DA3 .npz glob, and
     anchor ts (median of the DA3 ts list) — saving you from
     re-typing scene-specific paths for every scene.
  2. Spawns ``lad pipeline`` as a subprocess so a failure in one
     scene does not kill the whole batch.
  3. Tee's the stdout / stderr to ``<output-root>/<scene>/batch.log``.
  4. Prints a one-line status while running and a summary table
     at the end.

Example:

    PYTHONPATH=src python scripts/run_batch.py \\
        002 008 010 \\
        --data-root /mnt/car_road_data_TianJin \\
        --residual-checkpoint preview/point_completion_ckpt_v2/best.pt \\
        --recon-dir preview/recon_3I_full/ \\
        --output-root outputs \\
        --gpu-ids 0,1,2,3 \\
        --sam-mask-dir preview/sam/ \\
        --segformer-dir preview/segformer/

To override the auto-detected anchor ts for a given scene, pass
``--anchor-ts 008:1742879650704`` (repeatable).
"""

from __future__ import annotations

import argparse
import glob
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SceneRun:
    scene: str
    status: str           # "ok", "failed", "skipped"
    wall_s: float
    note: str = ""


def _resolve_calib(scene: str, scan_dir: Path) -> Path | None:
    """Find ``preview/scene_recon/<scene>_*_static_calib.json``."""
    cands = sorted(scan_dir.glob(f"{scene}_*_static_calib.json"))
    return cands[0] if cands else None


def _resolve_anchor_ts(scene: str, d_paths_dir: Path) -> int | None:
    """Median ts of ``preview/da3/<scene>_*_d.npz`` filenames.

    DA3 .npz filenames are ``<scene_id>_ts<ms>_cam<X>_d.npz`` — parse
    the ms substring instead of opening every file.
    """
    ts_set: set[int] = set()
    for p in d_paths_dir.glob(f"{scene}_*_d.npz"):
        # filename: <scene>_ts<ms>_cam<X>_d.npz
        name = p.stem
        try:
            ts_part = next(t for t in name.split("_") if t.startswith("ts"))
            ts_set.add(int(ts_part[2:]))
        except (StopIteration, ValueError):
            continue
    if not ts_set:
        return None
    sorted_ts = sorted(ts_set)
    return sorted_ts[len(sorted_ts) // 2]


def _build_cmd(
    scene: str,
    anchor_ts: int,
    calib_json: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Translate batch CLI args into a ``lad pipeline`` invocation."""
    cmd: list[str] = [
        sys.executable, "-m", "lidar_anchored_depth.cli", "pipeline",
        "--complete.scene.scene", scene,
        "--complete.scene.data-root", args.data_root,
        "--complete.d-paths-glob", f"{args.d_paths_dir}/{scene}_*_d.npz",
        "--complete.calib-json", str(calib_json),
        "--complete.residual-checkpoint", args.residual_checkpoint,
        "--complete.object-anchor-ts-ms", str(anchor_ts),
        "--complete.output.root", args.output_root,
        "--inject.recon-dir", args.recon_dir,
        "--inject.anchor-ts-ms", str(anchor_ts),
        "--render-bev.recon-dir", args.recon_dir,
    ]
    if args.sam_mask_dir:
        cmd += ["--complete.sam-mask-dir", args.sam_mask_dir]
    if args.segformer_dir:
        cmd += ["--complete.segformer-dir", args.segformer_dir]
    if args.gpu_ids:
        gpus = args.gpu_ids.split(",")
        cmd += ["--complete.runtime.gpu-ids", *gpus]
    if args.mirror_axis:
        cmd += [
            "--inject.mirror-axis", args.mirror_axis,
            "--render-bev.mirror-axis", args.mirror_axis,
        ]
    if args.cams:
        cams = args.cams.split(",")
        cmd += ["--complete.scene.cams", *cams]
    cmd += args.passthrough
    return cmd


def _run_one(
    scene: str,
    anchor_ts: int,
    calib_json: Path,
    args: argparse.Namespace,
) -> SceneRun:
    log_dir = Path(args.output_root) / scene
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "batch.log"
    cmd = _build_cmd(scene, anchor_ts, calib_json, args)

    pretty = " ".join(shlex.quote(c) for c in cmd)
    log_path.write_text(f"# cmd\n{pretty}\n\n", encoding="utf-8")
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
        )
    wall = time.time() - t0
    if proc.returncode == 0:
        return SceneRun(scene=scene, status="ok", wall_s=wall)
    return SceneRun(
        scene=scene, status="failed", wall_s=wall,
        note=f"exit {proc.returncode}; see {log_path}",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenes", nargs="+",
                   help="Scene ids to process (e.g. 002 008 010).")
    p.add_argument("--data-root", required=True)
    p.add_argument("--d-paths-dir", default="preview/da3",
                   help="Where the DA3 .npz files live "
                        "(filename pattern: <scene>_ts<ms>_cam<X>_d.npz).")
    p.add_argument("--scene-recon-dir", default="preview/scene_recon",
                   help="Where the static_calib.json files live.")
    p.add_argument("--residual-checkpoint", required=True)
    p.add_argument("--recon-dir", required=True,
                   help="Per-object accumulated PLYs.")
    p.add_argument("--sam-mask-dir", default=None)
    p.add_argument("--segformer-dir", default=None)
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--gpu-ids", default=None,
                   help="Comma-separated GPU ids for multi-GPU sharding "
                        "of the complete stage (e.g. '0,1,2,3').")
    p.add_argument("--cams", default=None,
                   help="Comma-separated camera ids (default: 0,3,6,9).")
    p.add_argument("--mirror-axis", default=None,
                   help="Mirror axis for per-object reconstruction "
                        "('y' for typical roadside).")
    p.add_argument(
        "--anchor-ts", action="append", default=[],
        metavar="SCENE:TS_MS",
        help="Override the auto-detected anchor ts for a given scene. "
             "Repeatable; format e.g. '008:1742879650704'.",
    )
    p.add_argument(
        "passthrough", nargs="*",
        help="Extra args appended to every `lad pipeline` call.",
    )
    args = p.parse_args()

    # Parse the --anchor-ts overrides into a dict.
    anchor_overrides: dict[str, int] = {}
    for raw in args.anchor_ts:
        try:
            scene, ts = raw.split(":")
            anchor_overrides[scene] = int(ts)
        except ValueError:
            sys.exit(f"--anchor-ts expected SCENE:TS_MS, got {raw!r}")

    runs: list[SceneRun] = []
    print(f"[batch] {len(args.scenes)} scenes:  {' '.join(args.scenes)}")
    print(f"[batch] log dir per scene:  {args.output_root}/<scene>/batch.log")
    print()

    for i, scene in enumerate(args.scenes, 1):
        # Resolve scene-specific paths.
        calib = _resolve_calib(scene, Path(args.scene_recon_dir))
        if calib is None:
            runs.append(SceneRun(
                scene=scene, status="skipped", wall_s=0.0,
                note=f"no calib JSON under {args.scene_recon_dir}/{scene}_*",
            ))
            print(f"[{i}/{len(args.scenes)}] {scene}: SKIP (no calib)")
            continue
        anchor = anchor_overrides.get(scene) or _resolve_anchor_ts(
            scene, Path(args.d_paths_dir),
        )
        if anchor is None:
            runs.append(SceneRun(
                scene=scene, status="skipped", wall_s=0.0,
                note=f"no DA3 .npz under {args.d_paths_dir}/{scene}_*",
            ))
            print(f"[{i}/{len(args.scenes)}] {scene}: SKIP (no DA3 .npz)")
            continue
        print(
            f"[{i}/{len(args.scenes)}] {scene}: running  "
            f"calib={calib.name}  anchor_ts={anchor}",
            flush=True,
        )
        run = _run_one(scene, anchor, calib, args)
        runs.append(run)
        print(
            f"           -> {run.status}  ({run.wall_s:.0f}s)"
            + (f"  [{run.note}]" if run.note else ""),
            flush=True,
        )

    # ---- Summary table ----
    print()
    print("=" * 70)
    print(f"{'scene':<20} {'status':<10} {'wall_s':>10}  note")
    print("-" * 70)
    for r in runs:
        print(f"{r.scene:<20} {r.status:<10} {r.wall_s:>10.0f}  {r.note}")
    print("=" * 70)

    n_ok = sum(1 for r in runs if r.status == "ok")
    n_fail = sum(1 for r in runs if r.status == "failed")
    n_skip = sum(1 for r in runs if r.status == "skipped")
    print(f"ok={n_ok}  failed={n_fail}  skipped={n_skip}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
