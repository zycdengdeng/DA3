"""Config for the ``render-bev`` stage (per-ts BEV video frames).

For each timestamp in the scene (stride configurable), inject the
per-object snapshots with linear pose interpolation between the two
annotated ts that bracket the frame, then rasterise top-down as a PNG.
Output PNGs feed straight into ``ffmpeg`` for video assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)


@dataclass
class BevRenderConfig:
    """Render per-ts BEV PNGs for video assembly.

    Inputs
    ------
    Same as ``inject``: a static cloud + ``recon_dir`` of per-object PLYs.

    Outputs
    -------
    ``bev_frames/bev_<i:04d>.png`` per timestamp, plus
    ``render_summary.json``.  An optional ``frame_plys/`` directory
    is written when ``write_plys`` is on (useful for debugging
    individual frames but disk-hungry).
    """

    scene: SceneConfig

    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ---- inputs ----
    recon_dir: Path = Path("preview/recon_3I_full")
    """Dir of per-object accumulated PLYs."""

    static_ply: Path | None = None
    """Override the auto-discovered static cloud
    (default: ``<output.root>/<scene>/complete/latest/hybrid.ply``)."""

    # ---- per-object injection (subset of InjectConfig) ----
    object_fusion: Literal["lidar-priority", "lidar-only", "aa-had-only"] = (
        "lidar-priority"
    )
    object_voxel_size: float = 0.05
    object_max_dist_to_lidar: float = 0.5
    mirror_axis: Literal["x", "y", "z"] | None = None
    mirror_classes: tuple[str, ...] = ("Car", "Suv", "Bus", "Truck")
    motion_drift_threshold_m: float = 0.5

    # ---- video time axis ----
    ts_stride: int = 1
    """Render every Nth ts. Default 1 = every ts."""

    max_extrap_ms: int = 200
    """Drop objects whose video ts is outside their annotation window
    by more than this many ms (otherwise extrapolation produces
    teleporting objects)."""

    max_gap_ms: int = 500
    """Drop objects whose bracketing annotated ts are farther apart than
    this (a long gap usually means V2X tracking failed across that
    span; interpolating across it teleports the object). Set huge
    to disable."""

    # ---- BEV viewport ----
    voxel_size: float = 0.05
    """Voxel size used when pre-downsampling the static cloud once
    outside the per-ts loop (this avoids re-running ``np.unique``
    on 50 M points per frame)."""

    image_size: tuple[int, int] = (1024, 1024)
    """(H, W) of the rendered BEV PNG."""

    x_range: tuple[float, float] | None = None
    """World X range (metres). Auto-computed from the static cloud if None."""

    y_range: tuple[float, float] | None = None
    """World Y range (metres). Auto-computed if None."""

    pad_m: float = 5.0
    """Padding (m) around the auto-computed BEV bounds."""

    robust_color: Literal["mean", "median", "mad-trim"] = "median"
    """Per-voxel colour aggregation used when pre-downsampling static."""

    # ---- output knobs ----
    write_plys: bool = False
    """Also write per-frame PLY (debug; lots of disk)."""
