"""Aggregate JSON outputs of ``depth_model_diagnostic.py`` into a verdict.

Reads every ``*_diag.json`` under a directory and produces:

- A per-frame table (cam, ts, best, RMSE, margin)
- Vote counts per model
- Per-camera summary (median RMSE, N frames, fraction voting linear)
- Final pipeline-level verdict: which functional form to use in HAD

Usage
-----
    python scripts/aggregate_diagnostics.py --diag-dir preview/diag/

    # Stricter "linear is good enough" threshold (default 1.0 %)
    python scripts/aggregate_diagnostics.py --diag-dir preview/diag/ \\
        --equivalent-margin 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--diag-dir", required=True)
    parser.add_argument(
        "--equivalent-margin", type=float, default=1.0,
        help="treat 'best' as effectively-linear when the margin to "
        "linear is below this percentage (default 1.0%%)",
    )
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)
    paths = sorted(diag_dir.glob("*_diag.json"))
    if not paths:
        raise SystemExit(f"no *_diag.json under {diag_dir}")

    rows: list[dict] = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"  [skip]  {p.name}: {e}", file=sys.stderr)
            continue
        fits = {f["name"]: f for f in d["fits"]}
        rows.append({
            "file": p.name,
            "cam": d["cam_id"],
            "ts": d["ts_ms"],
            "n": d["n_pairs"],
            "rmse_linear": fits["linear"]["rmse_m"],
            "rmse_inverse": fits["inverse"]["rmse_m"],
            "rmse_quadratic": fits["quadratic"]["rmse_m"],
            "best": d["verdict"]["best"],
            "margin": d["verdict"]["margin_over_runner_up"],
            "confidence": d["verdict"]["confidence"],
        })

    if not rows:
        raise SystemExit("no usable diagnostic JSONs")

    # Per-frame table
    print()
    print("Per-frame results")
    fmt = "  {:<6} {:>14} {:>10} {:>9} {:>9} {:>9} {:>10} {:>9}"
    print(fmt.format(
        "cam", "ts", "n",
        "linear", "inverse", "quadratic",
        "best", "margin %",
    ))
    for r in rows:
        print(fmt.format(
            f"cam{r['cam']}", r["ts"], r["n"],
            f"{r['rmse_linear']:.2f}",
            f"{r['rmse_inverse']:.2f}",
            f"{r['rmse_quadratic']:.2f}",
            r["best"],
            f"{r['margin']*100:.2f}",
        ))

    # Vote summary
    raw_votes = Counter(r["best"] for r in rows)
    eff_votes = Counter()
    for r in rows:
        if r["best"] == "linear":
            eff_votes["linear"] += 1
        elif r["margin"] * 100.0 < args.equivalent_margin:
            eff_votes[f"~linear ({r['best']} margin <{args.equivalent_margin:.1f}%)"] += 1
        else:
            eff_votes[r["best"]] += 1

    print()
    print(f"Raw votes (best by min RMSE):")
    for k, n in raw_votes.most_common():
        print(f"  {k:<10}  {n:>3} / {len(rows)}")

    print()
    print(f"Effective votes (linear-equivalent if margin < {args.equivalent_margin:.1f}%):")
    for k, n in eff_votes.most_common():
        print(f"  {k:<50}  {n:>3} / {len(rows)}")

    # Per-camera summary
    print()
    print("Per-camera summary:")
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cam[r["cam"]].append(r)
    fmt2 = "  cam{:<3}  N={:<3}  median RMSE: lin={:>6.2f}  quad={:>6.2f}  inv={:>9.2f}"
    for cam in sorted(by_cam):
        rs = by_cam[cam]
        print(fmt2.format(
            cam, len(rs),
            median(r["rmse_linear"] for r in rs),
            median(r["rmse_quadratic"] for r in rs),
            median(r["rmse_inverse"] for r in rs),
        ))

    # Final pipeline-level verdict
    n = len(rows)
    n_linear_equiv = sum(
        1 for r in rows
        if r["best"] == "linear"
        or r["margin"] * 100.0 < args.equivalent_margin
    )
    fraction = n_linear_equiv / n
    print()
    print("=" * 64)
    if fraction >= 0.9:
        print(
            f"VERDICT: linear ({n_linear_equiv}/{n}, {fraction*100:.0f}% — "
            "use z = a · d̃ + b in HAD; no calibration head needed)."
        )
    elif fraction >= 0.7:
        print(
            f"VERDICT: linear acceptable ({n_linear_equiv}/{n}, "
            f"{fraction*100:.0f}%) but quadratic / non-linear effects exist; "
            "consider a per-camera g_φ MLP for the residual."
        )
    else:
        print(
            f"VERDICT: linear FAILS in {n - n_linear_equiv}/{n} frames "
            f"({100 - fraction*100:.0f}% non-equivalent); "
            "add g_φ calibration head before relying on HAD."
        )
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
