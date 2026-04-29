"""Stage 1 import smoke test: all modules exist and import cleanly."""

from __future__ import annotations


def test_top_level_imports():
    import lidar_anchored_depth  # noqa: F401
    import lidar_anchored_depth.alignment  # noqa: F401
    import lidar_anchored_depth.alignment.adaptive  # noqa: F401
    import lidar_anchored_depth.alignment.global_scale  # noqa: F401
    import lidar_anchored_depth.alignment.ground_plane  # noqa: F401
    import lidar_anchored_depth.alignment.height_anchor  # noqa: F401
    import lidar_anchored_depth.alignment.projection  # noqa: F401
    import lidar_anchored_depth.alignment.region_affine  # noqa: F401
    import lidar_anchored_depth.analysis  # noqa: F401
    import lidar_anchored_depth.analysis.error_propagation  # noqa: F401
    import lidar_anchored_depth.cli  # noqa: F401
    import lidar_anchored_depth.data  # noqa: F401
    import lidar_anchored_depth.data.base  # noqa: F401
    import lidar_anchored_depth.data.roadside_v2x  # noqa: F401
    import lidar_anchored_depth.eval  # noqa: F401
    import lidar_anchored_depth.eval.metrics  # noqa: F401
    import lidar_anchored_depth.pipeline  # noqa: F401
    import lidar_anchored_depth.pipeline.roadside  # noqa: F401
    import lidar_anchored_depth.segmentation  # noqa: F401
    import lidar_anchored_depth.segmentation.sam  # noqa: F401
    import lidar_anchored_depth.viz  # noqa: F401
