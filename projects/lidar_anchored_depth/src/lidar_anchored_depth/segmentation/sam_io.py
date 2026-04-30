"""Helpers for loading SAM .npz outputs (written by ``run_sam_inference.py``)
and turning them into pixel masks.

The static-scene reconstruction and sigma-head training both need a
single ``(H, W)`` boolean "any dynamic object" mask per (scene, ts,
cam). DA3's depth prediction tends to bleed past the SAM mask edge by
a few pixels (the network sees a soft transition where SAM cuts a
hard one), so optionally dilating the union mask by a small kernel
removes the residual ghosts that otherwise leak into the static
branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _dilate_bool_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Binary-dilate a (H, W) bool mask by ``dilate_px`` pixels with a
    square structuring element. Falls back to a NumPy implementation
    when SciPy is unavailable."""
    if dilate_px <= 0 or not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_dilation

        size = 2 * int(dilate_px) + 1
        struct = np.ones((size, size), dtype=bool)
        return binary_dilation(mask, structure=struct).astype(bool)
    except ModuleNotFoundError:
        # Manual square dilation by axis-aligned shifts.
        out = mask.copy()
        for dy in range(-dilate_px, dilate_px + 1):
            for dx in range(-dilate_px, dilate_px + 1):
                if dy == 0 and dx == 0:
                    continue
                out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        return out


def load_sam_masks_dict(
    sam_dir: Path | None,
    scene_id: str,
    ts_ms: int,
    cam_id: str,
) -> dict[int, np.ndarray]:
    """Return ``{bbox_id: (H, W) bool mask}`` for one (scene, ts, cam).

    Returns an empty dict when the .npz is absent (e.g. an interpolated
    frame with no V2X bboxes).
    """
    if sam_dir is None:
        return {}
    p = sam_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam.npz"
    if not p.is_file():
        return {}
    data = np.load(p, allow_pickle=True)
    ids = list(data["mask_ids"])
    masks = data["masks"]
    return {int(i): masks[k].astype(bool) for k, i in enumerate(ids)}


def load_sam_dynamic_mask(
    sam_dir: Path | None,
    scene_id: str,
    ts_ms: int,
    cam_id: str,
    image_hw: tuple[int, int],
    *,
    dilate_px: int = 0,
) -> np.ndarray:
    """Build a ``(H, W)`` bool mask = union of every per-bbox SAM mask
    in the corresponding ``..._sam.npz``, optionally dilated by
    ``dilate_px`` pixels.

    ``True`` = dynamic-object pixel (i.e. should be excluded from any
    static-scene branch).
    """
    H, W = image_hw
    out = np.zeros((H, W), dtype=bool)
    if sam_dir is None:
        return out
    p = sam_dir / f"{scene_id}_ts{ts_ms}_cam{cam_id}_sam.npz"
    if not p.is_file():
        return out
    data = np.load(p, allow_pickle=True)
    masks = data["masks"]
    if masks.shape[0] == 0:
        return out
    union = np.any(masks, axis=0).astype(bool)
    if dilate_px > 0:
        union = _dilate_bool_mask(union, dilate_px)
    return union


__all__ = [
    "load_sam_dynamic_mask",
    "load_sam_masks_dict",
]
