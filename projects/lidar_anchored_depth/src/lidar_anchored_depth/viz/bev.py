"""Top-down BEV rasterisation for static/dynamic point clouds.

The renderer is a vectorised z-buffer (highest world Z per BEV cell
wins), implemented in numpy. No open3d / OpenGL dependency — output
is a plain ``(H, W, 3)`` uint8 image suitable for ``ffmpeg`` framing.
"""

from __future__ import annotations

import numpy as np


def render_bev(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    image_hw: tuple[int, int] = (1024, 1024),
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Top-down z-buffer BEV renderer (highest Z wins per pixel).

    Coordinate convention: world X grows to the right, world Y grows
    downwards (image row index). World Z determines depth ordering.
    """
    H, W = image_hw
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = background
    if xyz.shape[0] == 0:
        return img

    x_min, x_max = x_range
    y_min, y_max = y_range
    cell_x = (x_max - x_min) / W
    cell_y = (y_max - y_min) / H

    px = np.floor((xyz[:, 0] - x_min) / cell_x).astype(np.int64)
    py = np.floor((xyz[:, 1] - y_min) / cell_y).astype(np.int64)
    py = (H - 1) - py  # flip so +Y in world is up in image
    z = xyz[:, 2].astype(np.float32)

    in_image = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px = px[in_image]; py = py[in_image]; z = z[in_image]
    if rgb is not None:
        rgb_v = rgb[in_image]
    else:
        rgb_v = np.full((px.size, 3), 200, dtype=np.uint8)

    # z-buffer: highest z wins. Sort ascending so high z overwrites.
    order = np.argsort(z, kind="stable")
    px = px[order]; py = py[order]; rgb_v = rgb_v[order]

    img[py, px] = rgb_v
    return img


def auto_bev_range(
    xyz: np.ndarray,
    pad_m: float = 5.0,
    quantile: float = 0.02,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Robust X/Y range = quantile-trimmed bounds + padding.

    The result is a square BEV viewport centred on the trimmed
    centroid; ``quantile`` trims this fraction from each tail to
    discount outliers from the BEV bounds.
    """
    if xyz.shape[0] == 0:
        return (-50.0, 50.0), (-50.0, 50.0)
    x_lo = float(np.quantile(xyz[:, 0], quantile)) - pad_m
    x_hi = float(np.quantile(xyz[:, 0], 1.0 - quantile)) + pad_m
    y_lo = float(np.quantile(xyz[:, 1], quantile)) - pad_m
    y_hi = float(np.quantile(xyz[:, 1], 1.0 - quantile)) + pad_m
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    half = max(x_hi - x_lo, y_hi - y_lo) * 0.5
    return (cx - half, cx + half), (cy - half, cy + half)


__all__ = ["auto_bev_range", "render_bev"]
