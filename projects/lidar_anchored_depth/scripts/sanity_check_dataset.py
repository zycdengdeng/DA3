"""Sanity-check the THICV-R2A dataset mount.

Mirrors the verification checklist from ``docs/dataset_guide.md`` §6.
Run this once on a fresh server/mount to confirm the layout matches what
the loader expects.

Usage
-----
    python scripts/sanity_check_dataset.py \
        --data-root /mnt/car_road_data_TianJin

Returns a non-zero exit code if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)


def check_root(root: Path) -> bool:
    print("[1] data root + scene count")
    if not root.is_dir():
        _fail(f"{root} does not exist or is not a directory")
        return False
    children = sorted(p.name for p in root.iterdir() if p.is_dir())
    n = len(children)
    if n != 90:
        _fail(f"expected 90 entries (89 scenes + support_info); got {n}")
        return False
    if "support_info" not in children:
        _fail("support_info/ missing under data root")
        return False
    _ok(f"found {n} entries (89 scenes + support_info)")
    return True


def check_scene_layout(root: Path, scene_glob: str = "008_*") -> bool:
    print(f"[2] scene layout ({scene_glob})")
    scenes = sorted(root.glob(scene_glob))
    if not scenes:
        _fail(f"no scene matches {scene_glob}")
        return False
    s = scenes[0]
    required = ["car", "car_labels", "road", "road_labels", "sync_info.txt"]
    missing = [r for r in required if not (s / r).exists()]
    if missing:
        _fail(f"{s.name}: missing {missing}")
        return False
    _ok(f"{s.name} has all required entries")
    return True


def check_annotations(root: Path, scene_glob: str = "008_*") -> bool:
    print(f"[3] annotation files ({scene_glob})")
    scenes = sorted(root.glob(scene_glob))
    if not scenes:
        _fail("no matching scene")
        return False
    s = scenes[0]
    label_dir = s / "road_labels" / "interpolation_labels"
    if not label_dir.is_dir():
        _fail(f"{label_dir} missing")
        return False
    n = sum(1 for _ in label_dir.glob("*.json"))
    if n == 0:
        _fail(f"{label_dir} contains 0 JSONs")
        return False
    _ok(f"{s.name}: {n} annotation JSON files")
    return True


def check_calibration(root: Path) -> bool:
    print("[4] calibration files")
    calib = root / "support_info" / "calib.json"
    if not calib.is_file():
        _fail(f"{calib} missing")
        return False
    try:
        with calib.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        _fail(f"{calib}: invalid JSON — {e}")
        return False
    needed = {"imgSize", "lidar", "camera"}
    have = set(data.keys())
    if not needed <= have:
        _fail(f"calib.json missing keys: {needed - have}")
        return False
    pinholes = [c for c, v in data["camera"].items() if not v.get("isFish", 0)]
    if sorted(pinholes) != ["0", "3", "6", "9"]:
        _fail(f"pinhole cameras should be {{0,3,6,9}}; got {sorted(pinholes)}")
        return False
    _ok("calib.json valid; pinholes = {0, 3, 6, 9}")
    return True


def check_pinhole_mapping(root: Path, scene_glob: str = "008_*") -> bool:
    print("[5] pinhole folder ↔ cam-id mapping")
    scenes = sorted(root.glob(scene_glob))
    if not scenes:
        _fail("no matching scene")
        return False
    s = scenes[0]
    expected = {"pinhole0": "cam3_", "pinhole1": "cam6_",
                "pinhole2": "cam9_", "pinhole3": "cam0_"}
    cam_root = s / "road" / "cameras"
    for folder, prefix in expected.items():
        d = cam_root / folder
        if not d.is_dir():
            _fail(f"{d} missing")
            return False
        sample = next(d.glob(f"{prefix}*.png"), None)
        if sample is None:
            files = list(d.glob("*.png"))
            _fail(
                f"{folder}/ should contain {prefix}*.png; "
                f"found e.g. {files[0].name if files else '(empty)'}"
            )
            return False
    _ok(f"{s.name}: pinhole0..3 → cam3/6/9/0 mapping verified")
    return True


def check_carid(root: Path) -> bool:
    print("[6] carid.json")
    p = root / "support_info" / "carid.json"
    if not p.is_file():
        _fail(f"{p} missing")
        return False
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        _fail("carid.json has empty results")
        return False
    sample = results[0]
    if "clip_name" not in sample or "nearest_carid" not in sample:
        _fail("carid.json entries missing clip_name / nearest_carid")
        return False
    _ok(f"carid.json has {len(results)} entries")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-root",
        default="/mnt/car_road_data_TianJin",
        help="dataset mount root (default: %(default)s)",
    )
    parser.add_argument(
        "--scene-glob",
        default="008_*",
        help="scene glob to verify (default: %(default)s)",
    )
    args = parser.parse_args()
    root = Path(args.data_root)

    checks = [
        check_root(root),
        check_scene_layout(root, args.scene_glob),
        check_annotations(root, args.scene_glob),
        check_calibration(root),
        check_pinhole_mapping(root, args.scene_glob),
        check_carid(root),
    ]
    print()
    n_ok = sum(checks)
    n = len(checks)
    if n_ok == n:
        print(f"all {n} checks passed.")
        return 0
    print(f"{n - n_ok} of {n} checks failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
