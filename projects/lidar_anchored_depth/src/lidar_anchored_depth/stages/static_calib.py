"""``calib`` stage: static scene reconstruction + AA-HAD calibration.

Wraps ``scripts/run_static_scene_recon.py``. Single-GPU; the per-camera
LSQ fit doesn't benefit from sharding.
"""

from __future__ import annotations

import time

from lidar_anchored_depth.configs.stages.calib import CalibConfig
from lidar_anchored_depth.engine.discovery import (
    resolve_upstream_dir,
    resolve_upstream_glob,
)
from lidar_anchored_depth.engine.script_runner import run_script_subprocess
from lidar_anchored_depth.stages.base import Stage, StageArtifacts


class StaticCalibStage(Stage[CalibConfig]):
    name = "calib"

    def run(self) -> StageArtifacts:
        cfg = self.cfg
        out = self.output_dir
        t0 = time.time()

        d_paths_glob = cfg.d_paths_glob
        if d_paths_glob is None:
            d_paths_glob = resolve_upstream_glob(
                cfg.output.root, cfg.scene.scene, "depth", "*_d.npz",
                flag_hint="d-paths-glob",
            )
            print(f"[auto] d-paths-glob <- {d_paths_glob}")
        sam_mask_dir = cfg.sam_mask_dir
        if sam_mask_dir is None:
            sam_mask_dir = resolve_upstream_dir(
                cfg.output.root, cfg.scene.scene, "mask",
                required=False, flag_hint="sam-mask-dir",
            )
            if sam_mask_dir is not None:
                print(f"[auto] sam-mask-dir <- {sam_mask_dir}")

        argv = [
            "--data-root", str(cfg.scene.data_root),
            "--scene", cfg.scene.scene,
            "--output", str(out),
            "--d-paths-glob", str(d_paths_glob),
            "--cams", *cfg.scene.cams,
            "--aahad-pixel-stride", str(cfg.aahad_pixel_stride),
            "--voxel-size", str(cfg.voxel_size),
            "--z-min", str(cfg.z_min),
            "--z-max", str(cfg.z_max),
            "--sam-dilate-px", str(cfg.sam_dilate_px),
            "--region-grid", str(cfg.region_grid_rows), str(cfg.region_grid_cols),
        ]
        if sam_mask_dir is not None:
            argv += ["--sam-mask-dir", str(sam_mask_dir)]
        if cfg.ply_binary:
            argv.append("--ply-binary")

        rc = run_script_subprocess(
            "run_static_scene_recon.py", argv, log_path=out / "run.log",
        )
        if rc != 0:
            raise SystemExit(f"`lad calib` script exited {rc}; see logs in {out}")

        calib_files = list(out.glob("*_static_calib.json"))
        return StageArtifacts(
            output_dir=out,
            files={
                "calib_json": calib_files[0] if calib_files else out,
                "static_ply": out,
            },
            summary={
                "scene": cfg.scene.scene,
                "n_calib_json": len(calib_files),
                "wall_time_s": round(time.time() - t0, 2),
            },
        )
