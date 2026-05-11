"""Smoke tests for the Phase 1 CLI / config / engine scaffolding.

These tests don't exercise any real stage — they just prove the
plumbing (configs construct, OutputManager places runs correctly,
``lad version`` returns the package version).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lidar_anchored_depth import __version__ as PKG_VERSION
from lidar_anchored_depth.configs import (
    OutputConfig, RuntimeConfig, SceneConfig,
)
from lidar_anchored_depth.engine import OutputManager
from lidar_anchored_depth.stages import Stage, StageArtifacts


def test_scene_config_defaults():
    cfg = SceneConfig(scene="008")
    assert cfg.scene == "008"
    assert cfg.data_root == Path("/mnt/car_road_data_TianJin")
    assert cfg.cams == ("0", "3", "6", "9")


def test_output_config_run_id_override():
    cfg = OutputConfig(run_id="custom-run")
    assert cfg.run_id == "custom-run"
    assert cfg.overwrite_latest is True


def test_runtime_config_serial_default():
    cfg = RuntimeConfig()
    assert cfg.gpu_ids == ()
    assert cfg.workers == 0


def test_output_manager_creates_timestamped_dir(tmp_path):
    om = OutputManager(tmp_path, scene="008", stage="depth")
    assert om.run_dir.is_dir()
    assert om.run_dir.parent == tmp_path / "008" / "depth"
    # auto run_id is YYYY-MM-DD_HH-MM-SS, 19 chars
    assert len(om.run_id) == 19
    assert om.run_id.count("-") == 4
    assert om.run_id.count("_") == 1


def test_output_manager_dump_and_finalise(tmp_path):
    om = OutputManager(tmp_path, scene="008", stage="depth")
    cfg_path = om.dump_config(SceneConfig(scene="008"))
    assert cfg_path.is_file()
    assert '"scene": "008"' in cfg_path.read_text()
    om.finalise()
    assert om.latest_link.is_symlink()
    # relative target so the tree is portable
    assert om.latest_link.readlink().name == om.run_id


def test_resolve_latest_returns_symlink_target(tmp_path):
    om1 = OutputManager(tmp_path, scene="008", stage="depth", run_id="r1")
    om1.finalise()
    om2 = OutputManager(tmp_path, scene="008", stage="depth", run_id="r2")
    om2.finalise()
    latest = OutputManager.resolve_latest(tmp_path, "008", "depth")
    assert latest == om2.run_dir


def test_resolve_latest_returns_none_for_unknown(tmp_path):
    assert OutputManager.resolve_latest(tmp_path, "nope", "depth") is None


def test_stage_abc_cannot_instantiate(tmp_path):
    with pytest.raises(TypeError):
        Stage(cfg=None, output_dir=tmp_path)  # type: ignore[abstract]


def test_stage_artifacts_dataclass(tmp_path):
    a = StageArtifacts(output_dir=tmp_path)
    assert a.files == {}
    assert a.summary == {}


# ---- CLI subprocess smoke test --------------------------------------

def _lad(*args: str) -> subprocess.CompletedProcess:
    """Invoke the CLI via ``python -m`` so it works even without ``pip
    install -e .``. Adds the repo's src/ to PYTHONPATH for the child."""
    repo_root = Path(__file__).resolve().parent.parent
    env = {
        "PYTHONPATH": str(repo_root / "src"),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(
        [sys.executable, "-m", "lidar_anchored_depth.cli", *args],
        capture_output=True, text=True, env=env, timeout=30,
    )


def test_cli_version_subcommand():
    r = _lad("version")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == PKG_VERSION


def test_cli_info_no_scene():
    r = _lad("info")
    assert r.returncode == 0, r.stderr
    assert '"package": "lidar-anchored-depth"' in r.stdout
    assert '"scene": null' in r.stdout


def test_cli_top_help_lists_subcommands():
    r = _lad("--help")
    assert r.returncode == 0, r.stderr
    assert "info" in r.stdout
    assert "version" in r.stdout
