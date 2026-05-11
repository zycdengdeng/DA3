"""``seg`` stage: SegFormer 19-class semantic segmentation.

Wraps ``scripts/run_segformer_inference.py``. Multi-GPU via
``--ts-shard k/N`` — same pattern as DA3.
"""

from __future__ import annotations

import time

from lidar_anchored_depth.configs.stages.seg import SegConfig
from lidar_anchored_depth.engine.script_runner import (
    run_script_sharded_by_ts,
)
from lidar_anchored_depth.stages.base import Stage, StageArtifacts


class SegFormerStage(Stage[SegConfig]):
    name = "seg"

    def run(self) -> StageArtifacts:
        cfg = self.cfg
        out = self.output_dir
        t0 = time.time()

        base_argv = [
            "--data-root", str(cfg.scene.data_root),
            "--scene", cfg.scene.scene,
            "--output", str(out),
            "--model", cfg.model,
            "--cam", "all",
            "--all-timestamps",
        ]
        if cfg.save_viz:
            base_argv.append("--save-viz")
        if cfg.skip_existing:
            base_argv.append("--skip-existing")

        rc = run_script_sharded_by_ts(
            "run_segformer_inference.py",
            base_argv,
            gpu_ids=list(cfg.runtime.gpu_ids),
            log_dir=out,
        )
        if rc != 0:
            raise SystemExit(f"`lad seg` script exited {rc}; see logs in {out}")

        seg_files = list(out.glob("*_seg.npz"))
        return StageArtifacts(
            output_dir=out,
            files={"seg_dir": out},
            summary={
                "scene": cfg.scene.scene,
                "n_seg_npz": len(seg_files),
                "model": cfg.model,
                "wall_time_s": round(time.time() - t0, 2),
                "gpu_ids": list(cfg.runtime.gpu_ids),
            },
        )
