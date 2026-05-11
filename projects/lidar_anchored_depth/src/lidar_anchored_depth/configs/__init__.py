"""Frozen dataclass configs that drive every stage.

Re-exports the base configs so ``from lidar_anchored_depth.configs import
SceneConfig`` works without going three directories deep.
"""

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)

__all__ = ["OutputConfig", "RuntimeConfig", "SceneConfig"]
