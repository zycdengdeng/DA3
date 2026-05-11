"""``depth`` stage: DA3 monocular relative-depth inference.

Wraps ``scripts/run_da3_inference.py``. Multi-GPU via
``--runtime.gpu-ids`` — each GPU gets one ``--ts-shard k/N`` worker.
"""

from __future__ import annotations

import time

from lidar_anchored_depth.configs.stages.depth import DepthConfig
from lidar_anchored_depth.engine.script_runner import (
    run_script_sharded_by_ts,
)
from lidar_anchored_depth.stages.base import Stage, StageArtifacts


class DA3DepthStage(Stage[DepthConfig]):
    name = "depth"

    def run(self) -> StageArtifacts:
        cfg = self.cfg
        out = self.output_dir
        t0 = time.time()

        base_argv = [
            "--data-root", str(cfg.scene.data_root),
            "--scene", cfg.scene.scene,
            "--output", str(out),
            "--model", cfg.model,
            "--process-res", str(cfg.process_res),
            "--cam", "all",
            "--all-timestamps",
        ]
        if cfg.skip_existing:
            base_argv.append("--skip-existing")

        rc = run_script_sharded_by_ts(
            "run_da3_inference.py",
            base_argv,
            gpu_ids=list(cfg.runtime.gpu_ids),
            log_dir=out,
        )
        if rc != 0:
            raise SystemExit(f"`lad depth` script exited {rc}; see logs in {out}")

        npz_files = list(out.glob("*_d.npz"))
        return StageArtifacts(
            output_dir=out,
            files={"depth_glob": out / "*_d.npz"},
            summary={
                "scene": cfg.scene.scene,
                "n_depth_npz": len(npz_files),
                "model": cfg.model,
                "wall_time_s": round(time.time() - t0, 2),
                "gpu_ids": list(cfg.runtime.gpu_ids),
            },
        )
