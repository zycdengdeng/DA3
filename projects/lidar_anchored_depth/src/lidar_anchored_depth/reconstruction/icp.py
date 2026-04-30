"""Point-to-point rigid ICP (Iterative Closest Point) for SE(3) alignment.

Used by ``scripts/refine_aa_had_with_icp.py`` to pull the per-object
AA-HAD predicted cloud onto its accumulated LiDAR ground-truth cloud.

Algorithm (classical):
  1. For each source point, find the nearest target point (KD-tree).
  2. Optionally trim the worst k% of correspondences (robust to noisy
     edges / partial overlap).
  3. Estimate the rigid SE(3) ``T`` that minimizes
     ``Σ ‖T · src_i − tgt_{nn(i)}‖²`` via the SVD-based closed form.
  4. Apply ``T`` to the source, recompose with the running estimate.
  5. Repeat until the mean correspondence distance stops improving.

Returns the cumulative ``T`` (4×4 SE(3)) and an info dict.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 30,
    convergence_tol: float = 1e-6,
    trim_percentile: float = 80.0,
    initial_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Align ``source`` to ``target`` by rigid SE(3) ICP.

    Parameters
    ----------
    source : (N, 3) array — points to be transformed.
    target : (M, 3) array — fixed reference cloud.
    max_iterations : iteration cap (default 30).
    convergence_tol : stop when the mean correspondence distance change
        is below this value across consecutive iterations.
    trim_percentile : keep only the closest ``trim_percentile``%% of
        correspondences each iteration. Default 80 → drop worst 20%
        (robust to noise + partial overlap). Set to 100 to disable.
    initial_transform : optional (4, 4) SE(3) starting guess. Default
        identity.

    Returns
    -------
    T : (4, 4) float64 — the cumulative SE(3) that maps source onto target.
    info : dict with
        - ``n_iterations``  : number of iterations actually run
        - ``final_residual``: mean correspondence distance at the end (m)
        - ``initial_residual`` : same metric at iteration 0
        - ``converged``    : True iff stopped early (vs hit the cap)
        - ``history``      : per-iteration residuals (list of floats)
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        raise ValueError(
            f"icp_point_to_point: need >= 3 points each "
            f"(got |source|={src.shape[0]}, |target|={tgt.shape[0]})"
        )

    T = (
        initial_transform.copy().astype(np.float64)
        if initial_transform is not None
        else np.eye(4, dtype=np.float64)
    )
    src_t = (T[:3, :3] @ src.T).T + T[:3, 3]
    tree = cKDTree(tgt)

    history: list[float] = []
    converged = False

    for it in range(max_iterations):
        dists, idx = tree.query(src_t, k=1)
        if trim_percentile < 100.0:
            thr = float(np.percentile(dists, trim_percentile))
            keep = dists <= thr
        else:
            keep = np.ones(src_t.shape[0], dtype=bool)

        if int(keep.sum()) < 3:
            break

        a = src_t[keep]
        b = tgt[idx[keep]]
        ca = a.mean(axis=0)
        cb = b.mean(axis=0)
        a_c = a - ca
        b_c = b - cb
        H = a_c.T @ b_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cb - R @ ca

        dT = np.eye(4, dtype=np.float64)
        dT[:3, :3] = R
        dT[:3, 3] = t
        T = dT @ T
        src_t = (R @ src_t.T).T + t

        residual = float(np.mean(dists[keep]))
        history.append(residual)

        if it > 0 and abs(history[-2] - history[-1]) < convergence_tol:
            converged = True
            break

    initial_residual: float | None = None
    if history:
        # Recompute initial residual for reporting
        dists0, _ = tree.query(
            (initial_transform[:3, :3] @ src.T).T + initial_transform[:3, 3]
            if initial_transform is not None
            else src,
            k=1,
        )
        initial_residual = float(np.mean(dists0))

    return T, {
        "n_iterations": len(history),
        "final_residual": history[-1] if history else None,
        "initial_residual": initial_residual,
        "converged": converged,
        "history": history,
    }


def apply_transform(
    points: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Apply a 4×4 SE(3) ``T`` to ``(N, 3)`` points: ``T·[p; 1]``."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if P.size == 0:
        return P
    return (T[:3, :3] @ P.T).T + T[:3, 3]
