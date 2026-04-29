"""End-to-end roadside HAD pipeline.

Wires together: DA3 inference → SAM masks → projection → HAD instance solve
→ ground branch → adaptive fusion → output. Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_anchored_depth.data.base import Frame


@dataclass(slots=True)
class PipelineOutput:
    metric_depth: np.ndarray
    confidence: np.ndarray
    diagnostics: dict


class RoadsidePipeline:
    """High-level orchestrator. Configurable per-stage method via Hydra."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run(self, frame: Frame) -> PipelineOutput:
        raise NotImplementedError("Stage 2")
