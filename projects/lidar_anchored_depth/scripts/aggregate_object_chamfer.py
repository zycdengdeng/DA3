"""Cross-scene aggregator over per-object reconstruction summaries.

Reads every ``*_recon_summary.json`` written by
``run_object_accumulation.py`` and produces a population-level view of
the project's per-object validation metric.

Usage
-----
    python scripts/aggregate_object_chamfer.py --recon-dir preview/recon/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recon-dir", required=True)
    args = parser.parse_args()

    recon_dir = Path(args.recon_dir)
    paths = sorted(recon_dir.glob("*_recon_summary.json"))
    if not paths:
        raise SystemExit(f"no *_recon_summary.json under {recon_dir}")

    rows: list[dict] = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"  [skip]  {p.name}: {e}", file=sys.stderr)
            continue
        for obj in data:
            rows.append({
                "scene": obj["scene_id"],
                "id": obj["bbox_id"],
                "label": obj["label"],
                "n_lidar_voxel": obj.get("n_lidar_voxel", 0),
                "n_aa_had_voxel": obj.get("n_aa_had_voxel", 0),
                "n_frames_lidar": obj.get("n_frames_lidar", 0),
                "n_frames_aa_had": obj.get("n_frames_aa_had", 0),
                "chamfer_m": obj.get("chamfer_m"),
            })

    if not rows:
        raise SystemExit("no objects found in those JSONs")

    valid = [r for r in rows if r["chamfer_m"] is not None]
    print()
    print(f"Found {len(rows)} reconstructed objects "
          f"({len(valid)} with valid chamfer)")
    if not valid:
        return 0

    cs = sorted(r["chamfer_m"] for r in valid)
    print()
    print("Population chamfer (m):")
    print(f"  median : {median(cs):.3f}")
    print(f"  p25    : {cs[len(cs) // 4]:.3f}")
    print(f"  p75    : {cs[3 * len(cs) // 4]:.3f}")
    print(f"  min    : {cs[0]:.3f}")
    print(f"  max    : {cs[-1]:.3f}")

    by_lab: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        by_lab[r["label"]].append(r["chamfer_m"])
    print()
    print("Per-class median chamfer:")
    for lab in sorted(by_lab, key=lambda k: -len(by_lab[k])):
        vs = by_lab[lab]
        print(
            f"  {lab:<22} N={len(vs):>3}  "
            f"median {median(vs):.3f} m  "
            f"min {min(vs):.3f}  max {max(vs):.3f}"
        )

    by_n_frames: dict[int, list[float]] = defaultdict(list)
    for r in valid:
        bin_ = min(20, r["n_frames_aa_had"])
        by_n_frames[bin_].append(r["chamfer_m"])
    print()
    print("Chamfer vs N_frames_aa_had (more frames → tighter cloud → ?):")
    for n in sorted(by_n_frames):
        vs = by_n_frames[n]
        print(
            f"  N_frames={n:>2}  N_obj={len(vs):>3}  "
            f"median chamfer {median(vs):.3f} m"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
