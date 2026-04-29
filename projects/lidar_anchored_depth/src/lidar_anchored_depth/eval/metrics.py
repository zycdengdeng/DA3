"""Per-pixel depth metrics. Stage 1 includes the standard set; Stage 2 adds
range/class-stratified runners.
"""

from __future__ import annotations

import numpy as np


def _valid_mask(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)


def abs_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute relative error: ``mean(|pred - gt| / gt)``."""
    m = _valid_mask(pred, gt)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))


def sq_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    m = _valid_mask(pred, gt)
    if not np.any(m):
        return float("nan")
    return float(np.mean(((pred[m] - gt[m]) ** 2) / gt[m]))


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    m = _valid_mask(pred, gt)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)))


def rmse_log(pred: np.ndarray, gt: np.ndarray) -> float:
    m = _valid_mask(pred, gt)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((np.log(pred[m]) - np.log(gt[m])) ** 2)))


def delta_threshold(pred: np.ndarray, gt: np.ndarray, *, t: float = 1.25) -> float:
    """Fraction of pixels with ``max(pred/gt, gt/pred) < t``."""
    m = _valid_mask(pred, gt)
    if not np.any(m):
        return float("nan")
    ratio = np.maximum(pred[m] / gt[m], gt[m] / pred[m])
    return float(np.mean(ratio < t))


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "AbsRel": abs_rel(pred, gt),
        "SqRel": sq_rel(pred, gt),
        "RMSE": rmse(pred, gt),
        "RMSElog": rmse_log(pred, gt),
        "delta_1.25": delta_threshold(pred, gt, t=1.25),
        "delta_1.25^2": delta_threshold(pred, gt, t=1.25**2),
        "delta_1.25^3": delta_threshold(pred, gt, t=1.25**3),
    }
