"""Per-scene ego-vehicle lookup from ``support_info/carid.json``.

Reference: ``docs/dataset_guide.md`` §3.2.

Notes
-----
- The ego car is a vehicle annotated in ``road_labels`` whose ``id``
  matches ``nearest_carid``. Use ``id`` only — the ``nearest_label``
  field can be stale relative to current annotations.
- Roadside-only mode rarely needs ego info, but masking the ego from
  "dynamic objects" is occasionally useful for downstream evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EgoEntry:
    """One scene's ego-vehicle record from ``carid.json``."""

    clip_name: str  # e.g. "008_car0325_road0327_t8"
    nearest_carid: int  # the V2X annotation id that is the ego
    visualize_roadtime: str | None = None  # ms timestamp suggested for vis
    visualize_camera: str | None = None  # cam id suggested for vis


@dataclass(slots=True)
class CaridLookup:
    """Read-only lookup from clip name → ego entry."""

    origin_lat: float
    origin_lon: float
    earth_radius_m: float
    entries: dict[str, EgoEntry]

    def get_ego_id(self, clip_name: str) -> int | None:
        """Return the ego ``id`` for a scene, or ``None`` if unknown."""
        entry = self.entries.get(clip_name)
        return None if entry is None else entry.nearest_carid

    def __contains__(self, clip_name: str) -> bool:
        return clip_name in self.entries

    def __len__(self) -> int:
        return len(self.entries)


def load_carid_lookup(path: str | Path) -> CaridLookup:
    """Parse ``support_info/carid.json`` into a :class:`CaridLookup`."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("metadata", {})
    origin = meta.get("origin_gps", {})

    entries: dict[str, EgoEntry] = {}
    for r in raw.get("results", []):
        clip = r["clip_name"]
        entries[clip] = EgoEntry(
            clip_name=clip,
            nearest_carid=int(r["nearest_carid"]),
            visualize_roadtime=r.get("visualize_roadtime"),
            visualize_camera=str(r["visualize_camera"])
            if r.get("visualize_camera") is not None
            else None,
        )

    return CaridLookup(
        origin_lat=float(origin.get("latitude", 0.0)),
        origin_lon=float(origin.get("longitude", 0.0)),
        earth_radius_m=float(origin.get("R_earth", 6378137.0)),
        entries=entries,
    )
