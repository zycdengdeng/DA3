"""Depth-alignment algorithms.

Modules
-------
projection
    LiDAR ↔ image projection utilities (NumPy + Torch).
height_anchor
    HAD: per-instance height-anchored depth solve. The paper's main
    contribution.
ground_plane
    RANSAC ground-plane fitting and per-pixel ground-depth from plane
    intersection.
global_scale
    Global LSQ / RANSAC scale and affine baselines (B2, B3).
region_affine
    Per-mask affine baselines on (d̃, z_lidar) pairs (B4).
adaptive
    Density-aware fusion at instance / ground seams (M3 component).
"""
