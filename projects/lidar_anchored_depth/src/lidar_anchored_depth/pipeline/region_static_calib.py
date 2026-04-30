"""Per-image-grid affine calibration for the static branch.

A single global per-camera ``z = a * d_tilde + b`` cannot express the
non-linearity of DA3's depth at the near range — the road surface
near the camera curves upward in the unprojected cloud. The fix is
to fit ``(a, b)`` per image grid cell on the static-LiDAR pairs and
apply per-cell affine at unproject time.

This module is a thin wrapper around
:func:`lidar_anchored_depth.alignment.region_affine.region_affine_align`
that takes care of building the ``(N_rows * N_cols, H, W)`` grid mask
stack and reporting per-cell residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_anchored_depth.alignment.global_scale import AffineFit
from lidar_anchored_depth.alignment.region_affine import (
    RegionAffineResult,
    apply_region_fits,
    region_affine_align,
)


@dataclass(slots=True)
class GridStaticCalib:
    """Per-camera grid affine calibration."""

    n_rows: int
    n_cols: int
    image_hw: tuple[int, int]
    masks: np.ndarray  # (n_rows * n_cols, H, W) bool
    result: RegionAffineResult

    @property
    def n_cells_solved(self) -> int:
        return self.result.n_regions_solved

    @property
    def n_cells_total(self) -> int:
        return self.n_rows * self.n_cols

    def fallback(self) -> AffineFit | None:
        return self.result.fallback_global

    def to_serialisable(self) -> dict:
        cells = []
        for k, fit in self.result.fits.items():
            row = int(k // self.n_cols)
            col = int(k % self.n_cols)
            cells.append({
                "k": int(k),
                "row": row,
                "col": col,
                "a": float(fit.a),
                "b": float(fit.b),
                "rmse_m": float(fit.residual_rmse),
            })
        out: dict = {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "image_hw": list(self.image_hw),
            "n_cells_solved": int(self.n_cells_solved),
            "n_cells_total": int(self.n_cells_total),
            "cells": cells,
        }
        if self.fallback() is not None:
            out["fallback"] = {
                "a": float(self.fallback().a),
                "b": float(self.fallback().b),
                "rmse_m": float(self.fallback().residual_rmse),
            }
        return out

    def apply(self, d_image: np.ndarray) -> np.ndarray:
        """Map a raw d_tilde image to per-pixel metric depth."""
        if d_image.shape != self.image_hw:
            raise ValueError(
                f"d_image shape {d_image.shape} != calibrated {self.image_hw}"
            )
        return apply_region_fits(d_image, self.masks, self.result)


def build_grid_masks(
    image_hw: tuple[int, int], n_rows: int, n_cols: int
) -> np.ndarray:
    """Build a ``(n_rows * n_cols, H, W)`` bool mask stack covering the
    image with ``n_rows × n_cols`` rectangular cells in row-major order
    (cell ``k = row * n_cols + col``)."""
    H, W = image_hw
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")
    row_edges = np.linspace(0, H, n_rows + 1, dtype=np.int64)
    col_edges = np.linspace(0, W, n_cols + 1, dtype=np.int64)
    masks = np.zeros((n_rows * n_cols, H, W), dtype=bool)
    for r in range(n_rows):
        for c in range(n_cols):
            k = r * n_cols + c
            masks[k, row_edges[r]:row_edges[r + 1],
                  col_edges[c]:col_edges[c + 1]] = True
    return masks


def fit_grid_static_calib(
    d_pred: np.ndarray,
    z_lidar: np.ndarray,
    pixel_uv: np.ndarray,
    image_hw: tuple[int, int],
    *,
    n_rows: int,
    n_cols: int,
    min_lidar_per_cell: int = 30,
    use_ransac: bool = True,
    ransac_seed: int = 0,
) -> GridStaticCalib:
    """Fit a per-cell affine over the static-LiDAR pairs.

    Cells with fewer than ``min_lidar_per_cell`` samples fall back to
    the global fit (computed on all samples) at apply time.
    """
    masks = build_grid_masks(image_hw, n_rows, n_cols)
    result = region_affine_align(
        d_pred=d_pred,
        z_lidar=z_lidar,
        pixel_uv=pixel_uv,
        masks=masks,
        min_lidar_per_region=min_lidar_per_cell,
        use_ransac=use_ransac,
        fallback_to_global=True,
        ransac_seed=ransac_seed,
    )
    return GridStaticCalib(
        n_rows=n_rows,
        n_cols=n_cols,
        image_hw=image_hw,
        masks=masks,
        result=result,
    )


__all__ = [
    "GridStaticCalib",
    "build_grid_masks",
    "fit_grid_static_calib",
]
