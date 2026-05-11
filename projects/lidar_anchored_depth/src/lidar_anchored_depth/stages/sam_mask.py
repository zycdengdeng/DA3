"""``mask`` stage: SAM dynamic-object masks.

Wraps ``scripts/run_sam_inference.py``. Multi-GPU shards by camera
since the script doesn't expose ``--ts-shard``.
"""

from __future__ import annotations

import time

from lidar_anchored_depth.configs.stages.mask import MaskConfig
from lidar_anchored_depth.engine.script_runner import (
    run_script_sharded_by_cam,
)
from lidar_anchored_depth.stages.base import Stage, StageArtifacts


class SAMMaskStage(Stage[MaskConfig]):
    name = "mask"

    def run(self) -> StageArtifacts:
        cfg = self.cfg
        out = self.output_dir
        t0 = time.time()

        base_argv = [
            "--data-root", str(cfg.scene.data_root),
            "--scene", cfg.scene.scene,
            "--output", str(out),
            "--model-id", cfg.model_id,
            "--bbox-pad-px", str(cfg.bbox_pad_px),
            "--max-boxes-per-batch", str(cfg.max_boxes_per_batch),
        ]
        if cfg.multimask:
            base_argv.append("--multimask")

        rc = run_script_sharded_by_cam(
            "run_sam_inference.py",
            base_argv,
            cams=list(cfg.scene.cams),
            gpu_ids=list(cfg.runtime.gpu_ids),
            log_dir=out,
        )
        if rc != 0:
            raise SystemExit(f"`lad mask` script exited {rc}; see logs in {out}")

        mask_files = list(out.glob("*.npz"))
        return StageArtifacts(
            output_dir=out,
            files={"mask_dir": out},
            summary={
                "scene": cfg.scene.scene,
                "n_mask_npz": len(mask_files),
                "model_id": cfg.model_id,
                "wall_time_s": round(time.time() - t0, 2),
                "gpu_ids": list(cfg.runtime.gpu_ids),
            },
        )
