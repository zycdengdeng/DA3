"""Chamfer distance between two point clouds (KD-tree based).

Used to compare AA-HAD-recovered points against accumulated LiDAR
"ground truth" in the per-object reconstruction evaluation. Symmetric
formulation:

    chamfer(A, B) = mean_{a∈A} min_{b∈B} ||a − b||
                  + mean_{b∈B} min_{a∈A} ||b − a||

The ``L2`` variant uses Euclidean distance (meters in our world frame);
the ``L2-squared`` variant returns the same with no sqrt. We expose
both.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(
    A: np.ndarray,
    B: np.ndarray,
    *,
    squared: bool = False,
) -> tuple[float, dict]:
    """Symmetric chamfer distance between two ``(N, 3)`` clouds.

    Returns
    -------
    chamfer : float
        ``mean(min_b ||a-b||) + mean(min_a ||b-a||)``. Lower is better.
    stats : dict with the breakdown:
        - ``a_to_b_mean``, ``a_to_b_median``, ``a_to_b_p95``
        - ``b_to_a_mean``, ``b_to_a_median``, ``b_to_a_p95``
        - ``n_a``, ``n_b``
    """
    A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(B, dtype=np.float64).reshape(-1, 3)
    if A.shape[0] == 0 or B.shape[0] == 0:
        raise ValueError(
            f"chamfer_distance: both clouds must be non-empty "
            f"(got |A|={A.shape[0]}, |B|={B.shape[0]})"
        )

    tree_b = cKDTree(B)
    tree_a = cKDTree(A)
    dists_a, _ = tree_b.query(A, k=1)  # for each a, nearest b
    dists_b, _ = tree_a.query(B, k=1)  # for each b, nearest a

    if squared:
        dists_a = dists_a ** 2
        dists_b = dists_b ** 2

    chamfer = float(dists_a.mean() + dists_b.mean())
    stats = {
        "a_to_b_mean": float(dists_a.mean()),
        "a_to_b_median": float(np.median(dists_a)),
        "a_to_b_p95": float(np.percentile(dists_a, 95)),
        "b_to_a_mean": float(dists_b.mean()),
        "b_to_a_median": float(np.median(dists_b)),
        "b_to_a_p95": float(np.percentile(dists_b, 95)),
        "n_a": int(A.shape[0]),
        "n_b": int(B.shape[0]),
    }
    return chamfer, stats


def chamfer_l2(A: np.ndarray, B: np.ndarray) -> float:
    """Just the chamfer L2 distance scalar (no stats)."""
    cd, _ = chamfer_distance(A, B, squared=False)
    return cd
