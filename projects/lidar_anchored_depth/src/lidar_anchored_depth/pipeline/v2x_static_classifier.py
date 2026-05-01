"""V2X-annotation static/dynamic classifier — Stage 6A.

For per-object snapshot injection we want to treat *static* objects
differently from *dynamic* ones:

* **Static objects** (Bollards, Crash_bucket, Cone, plus any
  V2X-annotated object whose centroid does not move across the
  session) belong in the scene at every frame. They should bypass any
  ``--strict-anchor`` ts filter so they always appear in the
  reconstructed snapshot.

* **Dynamic objects** (Cars, Suvs, Buses, Trucks, etc. that actually
  move) should follow the strict-anchor rule — they appear only at
  the chosen anchor ts to avoid temporal smear.

We classify by two signals (logical OR):

1. **Class whitelist** — anything in
   :data:`lidar_anchored_depth.data.roadside_v2x.STATIC_FIXTURE_CLASSES`
   is unconditionally static.
2. **Centroid drift** — for any other class, walk every annotated ts
   and measure the diameter of the bbox-centroid trajectory in world
   metres. If the diameter is below ``drift_threshold_m``, the object
   is static (e.g. a parked car the V2X tracker labelled as ``Car``).

Output: ``{obj_id: ObjectMotionClass}`` with the boolean and the
diagnostic numbers (label, max drift, n ts seen).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lidar_anchored_depth.data.roadside_v2x import (
    STATIC_FIXTURE_CLASSES,
    RoadsideV2XLoader,
)


@dataclass(slots=True)
class ObjectMotionClass:
    """Motion classification for one V2X-annotated object."""

    obj_id: int
    label: str
    is_static: bool
    n_ts_seen: int
    max_drift_m: float
    reason: str  # "class-whitelist" | "low-drift" | "high-drift"


def classify_v2x_objects(
    loader: RoadsideV2XLoader,
    scene_id: str,
    *,
    cam_for_discovery: str = "0",
    drift_threshold_m: float = 0.5,
    static_classes: frozenset[str] = STATIC_FIXTURE_CLASSES,
) -> dict[int, ObjectMotionClass]:
    """Walk all (ts, cam) pairs of one scene, group annotations by
    ``obj_id``, and decide which are static.

    Parameters
    ----------
    cam_for_discovery : V2X annotations are camera-independent, so any
        camera with a frame at a given ts works for discovery.
    drift_threshold_m : maximum centroid trajectory diameter (in
        metres) for a non-fixture class to be classified as static.
        Default 0.5 m — covers V2X tracking jitter (~10–20 cm) + GPS
        wobble for parked vehicles.
    static_classes : frozenset of class labels treated as
        unconditionally static.

    Returns
    -------
    ``{obj_id: ObjectMotionClass}``
    """
    scene = next((s for s in loader.scenes if s.scene_id == scene_id), None)
    if scene is None:
        return {}

    # Collect every (obj_id, ts, label, centroid) tuple.
    by_obj: dict[int, list[tuple[int, str, np.ndarray]]] = {}
    for ts in scene.timestamps_ms:
        try:
            idx = loader.find_frame_idx(scene_id, ts, cam_for_discovery)
        except ValueError:
            continue
        frame = loader.get_frame(idx)
        for obj in frame.dynamic_objects or []:
            by_obj.setdefault(int(obj.id), []).append(
                (int(ts), obj.label, np.asarray(obj.xyz, dtype=np.float64).reshape(3))
            )

    out: dict[int, ObjectMotionClass] = {}
    for obj_id, samples in by_obj.items():
        labels = {s[1] for s in samples}
        # If a single id was annotated under multiple labels (rare),
        # pick the most common one.
        if len(labels) == 1:
            label = next(iter(labels))
        else:
            counts: dict[str, int] = {}
            for _, lbl, _ in samples:
                counts[lbl] = counts.get(lbl, 0) + 1
            label = max(counts, key=counts.get)

        n_ts = len(samples)
        if label in static_classes:
            out[obj_id] = ObjectMotionClass(
                obj_id=obj_id, label=label, is_static=True,
                n_ts_seen=n_ts, max_drift_m=0.0,
                reason="class-whitelist",
            )
            continue

        if n_ts < 2:
            # Single-frame appearance: not enough evidence; treat as
            # dynamic so it goes through the snapshot path.
            out[obj_id] = ObjectMotionClass(
                obj_id=obj_id, label=label, is_static=False,
                n_ts_seen=n_ts, max_drift_m=0.0,
                reason="single-ts",
            )
            continue

        centroids = np.stack([s[2] for s in samples], axis=0)
        diffs = centroids[:, None, :] - centroids[None, :, :]
        drift = float(np.linalg.norm(diffs, axis=2).max())
        is_static = drift < float(drift_threshold_m)
        out[obj_id] = ObjectMotionClass(
            obj_id=obj_id, label=label, is_static=is_static,
            n_ts_seen=n_ts, max_drift_m=drift,
            reason="low-drift" if is_static else "high-drift",
        )

    return out


def split_static_dynamic_ids(
    classification: dict[int, ObjectMotionClass],
) -> tuple[set[int], set[int]]:
    """Convenience: ``({static_ids}, {dynamic_ids})``."""
    static_ids = {oid for oid, c in classification.items() if c.is_static}
    dynamic_ids = {oid for oid, c in classification.items() if not c.is_static}
    return static_ids, dynamic_ids


__all__ = [
    "ObjectMotionClass",
    "classify_v2x_objects",
    "split_static_dynamic_ids",
]
