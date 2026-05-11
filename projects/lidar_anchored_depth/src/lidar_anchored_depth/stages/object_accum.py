"""``object-accum`` stage: per-object 3I accumulation.

Wraps ``scripts/run_object_accumulation.py``. Single-process — the per-
object inner loop is the heavy work and the original script doesn't
have a sharding flag.
"""

from __future__ import annotations

import time

from lidar_anchored_depth.configs.stages.object_accum import ObjectAccumConfig
from lidar_anchored_depth.engine.script_runner import run_script_subprocess
from lidar_anchored_depth.stages.base import Stage, StageArtifacts


class ObjectAccumStage(Stage[ObjectAccumConfig]):
    name = "object-accum"

    def run(self) -> StageArtifacts:
        cfg = self.cfg
        out = self.output_dir
        t0 = time.time()

        argv = [
            "--data-root", str(cfg.scene.data_root),
            "--scene", cfg.scene.scene,
            "--output", str(out),
            "--all-bboxes",
            "--cams", *cfg.scene.cams,
            "--d-paths-glob", cfg.d_paths_glob,
            "--voxel-size", str(cfg.voxel_size),
            "--bbox-expand", str(cfg.bbox_expand),
            "--ground-cut-m", str(cfg.ground_cut_m),
            "--bbox-clip-expand", str(cfg.bbox_clip_expand),
            "--z-min", str(cfg.z_min),
            "--z-max", str(cfg.z_max),
            "--dense-min-pixels", str(cfg.dense_min_pixels),
            "--lidar-color", cfg.lidar_color,
            "--solver-mode", cfg.solver_mode,
        ]
        if cfg.sam_mask_dir is not None:
            argv += ["--sam-mask-dir", str(cfg.sam_mask_dir)]
        if cfg.ply_binary:
            argv.append("--ply-binary")

        rc = run_script_subprocess(
            "run_object_accumulation.py", argv, log_path=out / "run.log",
        )
        if rc != 0:
            raise SystemExit(
                f"`lad object-accum` script exited {rc}; see logs in {out}"
            )

        obj_files = list(out.glob("*_obj*.ply"))
        return StageArtifacts(
            output_dir=out,
            files={"recon_dir": out},
            summary={
                "scene": cfg.scene.scene,
                "n_object_ply": len(obj_files),
                "solver_mode": cfg.solver_mode,
                "wall_time_s": round(time.time() - t0, 2),
            },
        )
