"""Tests for the upstream-artefact auto-resolution (Phase 7.1).

Each downstream stage (complete / inject / render-bev / calib /
object-accum) leaves its upstream input fields as ``None`` by
default; the stage's ``run()`` then asks the discovery helpers to
fill them in by walking the ``outputs/<scene>/<upstream>/latest/``
symlink convention. These tests cover:

* The three discovery helpers (dir, file, glob) — happy path, error
  messages, optional / required toggle.
* The 5 downstream configs surface ``None`` as a valid default.

End-to-end stage execution with real datasets isn't tested here
(GPU + V2X dataset required); the existing
``tests/integration/test_pipeline_orchestration.py`` covers
orchestration with MockStage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lidar_anchored_depth.configs import SceneConfig
from lidar_anchored_depth.engine import (
    OutputManager,
    resolve_upstream_dir,
    resolve_upstream_file,
    resolve_upstream_glob,
)


def _seed_upstream(
    root: Path, scene: str, stage: str, files: dict[str, str],
) -> Path:
    """Create an ``<root>/<scene>/<stage>/<run>/`` dir with ``latest``
    symlink and the given files."""
    om = OutputManager(root, scene=scene, stage=stage)
    for name, body in files.items():
        (om.run_dir / name).write_text(body)
    om.finalise()
    return om.run_dir


# ---- helpers: happy paths -----------------------------------------------

def test_resolve_dir_finds_latest_via_symlink(tmp_path):
    _seed_upstream(tmp_path, "008", "depth", {"008_ts1000_cam3_d.npz": ""})
    p = resolve_upstream_dir(tmp_path, "008", "depth")
    assert p.is_dir()
    assert p.parent.name == "depth"


def test_resolve_file_picks_first_match(tmp_path):
    _seed_upstream(tmp_path, "008", "calib", {
        "008_aaa_static_calib.json": "{}",
        "other_file.txt": "x",
    })
    p = resolve_upstream_file(
        tmp_path, "008", "calib", "*_static_calib.json",
    )
    assert p.name.endswith("_static_calib.json")
    assert p.is_file()


def test_resolve_glob_returns_string_under_latest(tmp_path):
    run_dir = _seed_upstream(
        tmp_path, "008", "depth", {"008_ts1000_cam3_d.npz": ""},
    )
    s = resolve_upstream_glob(tmp_path, "008", "depth", "*_d.npz")
    assert isinstance(s, str)
    # The returned glob points at the resolved 'latest' dir.
    assert s.endswith("*_d.npz")
    # And matching it from glob() should hit the seeded file.
    import glob as _glob
    matches = _glob.glob(s)
    assert matches and Path(matches[0]).parent == run_dir


# ---- helpers: error messages --------------------------------------------

def test_resolve_dir_required_missing_raises_with_hint(tmp_path):
    with pytest.raises(SystemExit) as exc:
        resolve_upstream_dir(
            tmp_path, "008", "depth", flag_hint="d-paths-glob",
        )
    msg = str(exc.value)
    assert "no upstream `depth`" in msg
    assert "lad depth --scene.scene 008" in msg
    assert "--d-paths-glob" in msg


def test_resolve_dir_optional_missing_returns_none(tmp_path):
    p = resolve_upstream_dir(tmp_path, "008", "mask", required=False)
    assert p is None


def test_resolve_file_pattern_unmatched_raises(tmp_path):
    _seed_upstream(tmp_path, "008", "calib", {"unrelated.txt": "x"})
    with pytest.raises(SystemExit, match="no files matching"):
        resolve_upstream_file(
            tmp_path, "008", "calib", "*_static_calib.json",
            flag_hint="calib-json",
        )


def test_resolve_glob_pattern_unmatched_raises(tmp_path):
    _seed_upstream(tmp_path, "008", "depth", {"unrelated.txt": "x"})
    with pytest.raises(SystemExit, match="no files matching"):
        resolve_upstream_glob(
            tmp_path, "008", "depth", "*_d.npz",
            flag_hint="d-paths-glob",
        )


# ---- config defaults: None means auto-resolve --------------------------

def test_complete_config_paths_default_to_none():
    """The 4 upstream-artefact paths default to None so the stage's
    auto-resolution kicks in; this guards against a future regression
    where someone re-introduces a hard-coded preview/ default."""
    from lidar_anchored_depth.configs.stages.complete import CompleteConfig
    cfg = CompleteConfig(scene=SceneConfig(scene="008"))
    assert cfg.d_paths_glob is None
    assert cfg.calib_json is None
    assert cfg.sam_mask_dir is None
    assert cfg.segformer_dir is None


def test_inject_config_recon_dir_defaults_to_none():
    from lidar_anchored_depth.configs.stages.inject import InjectConfig
    cfg = InjectConfig(scene=SceneConfig(scene="008"))
    assert cfg.recon_dir is None


def test_render_bev_config_recon_dir_defaults_to_none():
    from lidar_anchored_depth.configs.stages.render_bev import BevRenderConfig
    cfg = BevRenderConfig(scene=SceneConfig(scene="008"))
    assert cfg.recon_dir is None


def test_calib_config_paths_default_to_none():
    from lidar_anchored_depth.configs.stages.calib import CalibConfig
    cfg = CalibConfig(scene=SceneConfig(scene="008"))
    assert cfg.d_paths_glob is None
    assert cfg.sam_mask_dir is None


def test_object_accum_config_paths_default_to_none():
    from lidar_anchored_depth.configs.stages.object_accum import (
        ObjectAccumConfig,
    )
    cfg = ObjectAccumConfig(scene=SceneConfig(scene="008"))
    assert cfg.d_paths_glob is None
    assert cfg.sam_mask_dir is None
