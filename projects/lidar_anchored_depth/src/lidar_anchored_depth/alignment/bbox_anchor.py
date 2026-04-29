"""AA-HAD: V2X bbox-anchored solve with optional motion compensation.

This module implements:

- **M2 HAD-bbox**: per-instance ``(a, b)`` solved from the V2X bbox's
  exact ``(Z_min, Z_max)`` instead of the LiDAR-in-mask percentiles
  used by M1 (:mod:`alignment.height_anchor`).
- **M3 AA-HAD**: M2 plus motion compensation of the bbox centre
  (``Δt · v``) for objects that moved between LiDAR and camera
  timestamps. Height is invariant to ``v_z ≈ 0`` so we only adjust the
  XY of the bbox before bbox→mask matching.

Mask association is done by IoU between the projected bbox AABB and
each candidate SAM mask's bounding box (a stand-in until we have learned
σ-uncertainty in Stage 2C+).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_anchored_depth.alignment.height_anchor import (
    HADResult,
    InstanceAnchor,
    mask_top_bottom_pixels,
    solve_affine_from_height_anchors,
)
from lidar_anchored_depth.alignment.projection import (
    bbox_uv_aabb,
    project_3d_bbox,
)
from lidar_anchored_depth.data.base import DynamicObject


# --------------------------------------------------------------------- #
# Mask AABB + IoU
# --------------------------------------------------------------------- #
def _mask_aabb(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Tight pixel bounding box of a mask, ``(u_min, v_min, u_max, v_max)``."""
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    return (
        float(cols.min()), float(rows.min()),
        float(cols.max()), float(rows.max()),
    )


def _aabb_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU of two ``(u_min, v_min, u_max, v_max)`` rectangles."""
    iu_min = max(a[0], b[0])
    iv_min = max(a[1], b[1])
    iu_max = min(a[2], b[2])
    iv_max = min(a[3], b[3])
    iw = max(0.0, iu_max - iu_min)
    ih = max(0.0, iv_max - iv_min)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _motion_compensate_bbox(
    obj: DynamicObject, dt_seconds: float
) -> DynamicObject:
    """Return a copy of ``obj`` with its centre translated by ``v · Δt``.

    Z is left unchanged (the project assumes ``v_z = 0`` for ground
    targets). Yaw is also held fixed — at the typical Δt < 100 ms we
    care about, yaw drift is well below the V2X annotation noise floor.
    """
    if dt_seconds == 0.0:
        return obj
    new_xyz = obj.xyz.copy()
    new_xyz[0] += float(obj.velocity_xy[0]) * dt_seconds
    new_xyz[1] += float(obj.velocity_xy[1]) * dt_seconds
    return DynamicObject(
        id=obj.id, label=obj.label,
        xyz=new_xyz, lwh=obj.lwh.copy(), yaw=obj.yaw,
        velocity_xy=obj.velocity_xy.copy(),
        roll=obj.roll, pitch=obj.pitch,
        occlusion=obj.occlusion, num_points=obj.num_points,
    )


# --------------------------------------------------------------------- #
# AA-HAD core
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class BboxMatch:
    """One bbox→mask association with diagnostic info."""

    bbox_id: int
    mask_id: int
    iou: float
    bbox_aabb: tuple[float, float, float, float]
    mask_aabb: tuple[float, float, float, float]


def match_bboxes_to_masks(
    objects: list[DynamicObject],
    masks: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    dt_seconds: float = 0.0,
    iou_threshold: float = 0.30,
) -> list[BboxMatch]:
    """For each bbox, find its best-matching SAM mask via AABB IoU.

    Returns a list of accepted matches with ``iou >= iou_threshold``.
    Each mask can match at most one bbox (the first one with the highest
    IoU on it wins).
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be (K, H, W); got {masks.shape}")

    mask_aabbs: dict[int, tuple[float, float, float, float]] = {}
    for k in range(masks.shape[0]):
        ab = _mask_aabb(masks[k])
        if ab is not None:
            mask_aabbs[k] = ab

    # Score every (bbox, mask) pair, then greedy-assign
    scored: list[tuple[float, int, int, tuple, tuple]] = []
    for bbox_idx, obj in enumerate(objects):
        obj_t = _motion_compensate_bbox(obj, dt_seconds)
        uv8, _ = project_3d_bbox(obj_t, K, T_wc, dist)
        if uv8 is None:
            continue
        bbox_box = bbox_uv_aabb(uv8)
        for mask_idx, mask_box in mask_aabbs.items():
            iou = _aabb_iou(bbox_box, mask_box)
            if iou >= iou_threshold:
                scored.append((iou, bbox_idx, mask_idx, bbox_box, mask_box))

    scored.sort(key=lambda x: x[0], reverse=True)
    used_bbox: set[int] = set()
    used_mask: set[int] = set()
    matches: list[BboxMatch] = []
    for iou, bi, mi, bb, mb in scored:
        if bi in used_bbox or mi in used_mask:
            continue
        used_bbox.add(bi)
        used_mask.add(mi)
        matches.append(
            BboxMatch(
                bbox_id=bi, mask_id=mi, iou=iou,
                bbox_aabb=bb, mask_aabb=mb,
            )
        )
    return matches


def aa_had(
    d_pred_image: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    masks: np.ndarray,
    dynamic_objects: list[DynamicObject],
    *,
    dist: np.ndarray | None = None,
    dt_seconds: float = 0.0,
    iou_threshold: float = 0.30,
) -> HADResult:
    """M2/M3: per-instance HAD anchored on V2X bbox heights.

    The bbox provides the **exact** ``(Z_min, Z_max)`` (no LiDAR
    percentiles), and ``(vx, vy) · Δt`` corrects the bbox centre into
    the camera-timestamp frame for bbox→mask matching. The closed-form
    solver itself is identical to M1's.

    Parameters
    ----------
    d_pred_image : (H, W) DA3 relative depth.
    K, T_wc, dist : pinhole calibration.
    masks : (K, H, W) bool SAM masks.
    dynamic_objects : list of V2X bbox annotations (at LiDAR timestamp).
    dt_seconds : ``camera_ts - lidar_ts`` in seconds. Default 0 = no
        compensation (M2). Non-zero invokes the M3 motion-comp path.
    iou_threshold : minimum bbox-to-mask AABB IoU for a match.
    """
    H, W = d_pred_image.shape
    aligned = np.full((H, W), np.nan, dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)

    matches = match_bboxes_to_masks(
        dynamic_objects, masks, K, T_wc, dist,
        dt_seconds=dt_seconds, iou_threshold=iou_threshold,
    )

    anchors: list[InstanceAnchor] = []
    n_attempt = len(matches)
    n_solved = 0

    for m in matches:
        obj = dynamic_objects[m.bbox_id]
        mask_k = masks[m.mask_id]
        topbot = mask_top_bottom_pixels(mask_k)
        if topbot is None:
            continue
        uv_top, uv_bot = topbot

        d_top = float(d_pred_image[int(uv_top[1]), int(uv_top[0])])
        d_bot = float(d_pred_image[int(uv_bot[1]), int(uv_bot[0])])
        if abs(d_top - d_bot) < 1e-9:
            continue

        sol = solve_affine_from_height_anchors(
            K=K, T_wc=T_wc,
            uv_top=uv_top, uv_bot=uv_bot,
            d_pred_top=d_top, d_pred_bot=d_bot,
            Z_max=obj.Z_max, Z_min=obj.Z_min,
        )
        if sol is None:
            continue
        a, b = sol

        aligned[mask_k] = (a * d_pred_image[mask_k] + b).astype(np.float32)
        coverage |= mask_k
        n_solved += 1

        # Confidence scaffold: simple function of IoU + bbox.num_points.
        # The σ-uncertainty MLP (Stage 2C+) will replace this.
        conf = float(np.clip(m.iou, 0.0, 1.0)) * float(
            np.clip(obj.num_points / 200.0, 0.0, 1.0)
        )
        anchors.append(
            InstanceAnchor(
                instance_id=m.mask_id,
                uv_top=uv_top, uv_bot=uv_bot,
                d_pred_top=d_top, d_pred_bot=d_bot,
                Z_min=obj.Z_min, Z_max=obj.Z_max,
                a=a, b=b, confidence=conf,
            )
        )

    return HADResult(
        aligned_depth=aligned,
        coverage_mask=coverage,
        anchors=anchors,
        n_masks_attempted=n_attempt,
        n_masks_solved=n_solved,
    )
