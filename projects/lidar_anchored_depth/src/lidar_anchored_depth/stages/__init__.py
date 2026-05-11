"""Pipeline stages. Each stage takes a frozen config and runs to
completion, writing artifacts to a managed output directory.

The base contract is :class:`Stage`. Phase 2+ adds concrete stages
(``DA3Depth``, ``DenseCompletion``, ``BevRender``, ...).
"""

from lidar_anchored_depth.stages.base import Stage, StageArtifacts

__all__ = ["Stage", "StageArtifacts"]
