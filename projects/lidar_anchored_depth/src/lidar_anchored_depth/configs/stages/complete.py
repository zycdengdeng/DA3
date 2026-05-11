"""Config for the ``complete`` stage (Layer 2 refine + Layer 4 fuse).

Composes the shared ``SceneConfig`` / ``OutputConfig`` / ``RuntimeConfig``
plus stage-specific knobs. Every field maps 1-to-1 onto the legacy
``run_point_completion_inference.py`` argparse args so behaviour stays
identical when the stage is wired up in Phase 2.
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
class CompleteConfig:
    """Run the trained point-cloud completion network + Layer 4 LiDAR fuse.

    Inputs
    ------
    DA3 per-frame depth ``.npz`` (from ``run_da3_inference.py``), SAM
    dynamic masks (from ``run_sam_inference.py``), optional SegFormer
    per-pixel segments (from ``run_segformer_inference.py``), and a
    per-camera ``(a, b)`` AA-HAD calibration JSON.

    Outputs
    -------
    ``refined.ply``, ``baseline.ply`` (no Δxyz), and ``hybrid.ply``
    (LiDAR-priority fused with the refined fill), plus a summary
    ``summary.json``.
    """

    # ---- shared blocks ----
    scene: SceneConfig
    """V2X dataset root + scene id + camera selection."""

    output: OutputConfig = field(default_factory=OutputConfig)
    """Output dir layout (``<root>/<scene>/complete/<run_id>/``)."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    """Multi-GPU sharding via ``runtime.gpu_ids``."""

    # ---- external artifact paths ----
    d_paths_glob: str = "preview/da3/*_d.npz"
    """Glob matching DA3 ``.npz`` files (one per (scene, ts, cam))."""

    sam_mask_dir: Path | None = None
    """Dir with SAM dynamic-mask npz; static prior subtracts these."""

    sam_auto_dir: Path | None = None
    """Optional SAM Auto segmentation (deprecated; prefer SegFormer)."""

    segformer_dir: Path | None = None
    """Dir with SegFormer per-pixel Cityscapes class npz."""

    calib_json: Path = Path("calib.json")
    """Per-camera AA-HAD linear calibration ``(a, b)``."""

    residual_checkpoint: Path = Path("ckpt.pt")
    """Trained point-cloud residual flow head weights."""

    # ---- Stage 2: prior + network refine ----
    aahad_pixel_stride: int = 4
    """Stride on depth-image pixels when building the prior."""

    z_min: float = 0.5
    """Drop prior pixels with z_cam below this (m)."""

    z_max: float = 300.0
    """Drop prior pixels with z_cam above this (m)."""

    sam_dilate_px: int = 12
    """Dilate the SAM dynamic mask before subtracting from prior."""

    bbox_expand: float = 0.10
    """Inflate V2X bboxes by this fraction when culling dyn LiDAR."""

    max_points_per_frame: int = 12000
    """Cap per-frame prior size; KNN cost is O(N²) in the network."""

    require_v2x_frames: bool = True
    """Skip frames with no V2X annotations."""

    residual_device: str = "cuda"
    """Torch device for the predictor; overridden by ``runtime.gpu_ids``."""

    residual_steps: int = 4
    """Number of CFM integration steps for the residual flow head."""

    gate_distance_m: float | None = None
    """If set, force Δxyz = 0 for prior points whose nearest LiDAR > this."""

    post_network_ground_snap: bool = True
    """Re-snap network-displaced ground points to the LiDAR ground grid."""

    # ---- ground grid (used by Stage 2 ground-snap + Stage 4 skip-ground) ----
    ground_cell_size: float = 0.30
    """Ground-grid horizontal cell size (m)."""

    ground_low_quantile: float = 0.10
    """Low-quantile per ground cell becomes the cell's Z (anti-occluder)."""

    ground_snap_max_dz: float = 0.4
    """Max vertical distance (m) a point may be snapped to ground."""

    # ---- Stage 4: LiDAR-priority fuse ----
    include_lidar_priority: bool = True
    """Run Stage 4 LiDAR-priority voxel fusion on top of refined."""

    voxel_size: float = 0.05
    """Voxel size for downsampling and LiDAR-priority fusion (m)."""

    max_dist_to_lidar: float = 2.0
    """Drop refined points farther than this from the LiDAR backbone (m)."""

    lidar_skip_ground: bool = True
    """Drop LiDAR points within +/- band of the ground grid before fuse."""

    lidar_skip_ground_band_m: float = 0.5
    """Half-thickness of the ground-band cull (m)."""

    # ---- colorisation ----
    colorize_lidar: bool = True
    """Re-colour fused-cloud points by projecting into anchor-ts cameras."""

    colorize_only_lidar_points: bool = True
    """When colorising, only touch LiDAR-source points (keep refined RGB)."""

    robust_color: Literal["mean", "median", "mad-trim"] = "median"
    """Per-voxel colour aggregation across multi-frame samples."""

    object_anchor_ts_ms: int | None = None
    """Anchor ts for the colorisation cameras (ms). Required if colorize on."""

    # ---- output format ----
    ply_binary: bool = False
    """Write PLYs as binary (smaller, ~10× faster to load)."""
