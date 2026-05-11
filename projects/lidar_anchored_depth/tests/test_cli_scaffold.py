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
    assert "complete" in r.stdout


# ---- Phase 2: complete stage --------------------------------------------

def test_complete_config_importable():
    from lidar_anchored_depth.configs.stages.complete import CompleteConfig

    # Required: scene block. Everything else has defaults.
    cfg = CompleteConfig(scene=SceneConfig(scene="008"))
    assert cfg.scene.scene == "008"
    assert cfg.voxel_size == 0.05
    assert cfg.robust_color == "median"
    assert cfg.lidar_skip_ground is True


def test_dense_completion_stage_has_correct_name():
    from lidar_anchored_depth.stages.dense_completion import (
        DenseCompletionStage,
    )

    assert DenseCompletionStage.name == "complete"


def test_cli_complete_help():
    r = _lad("complete", "--help")
    assert r.returncode == 0, r.stderr
    # Stage-specific flag
    assert "--residual-checkpoint" in r.stdout
    # Composed blocks
    assert "--scene.scene" in r.stdout
    assert "--runtime.gpu-ids" in r.stdout
    assert "--output.root" in r.stdout


# ---- Phase 3: inject + render-bev stages --------------------------------

def test_inject_config_importable():
    from lidar_anchored_depth.configs.stages.inject import InjectConfig

    cfg = InjectConfig(scene=SceneConfig(scene="008"))
    assert cfg.scene.scene == "008"
    assert cfg.object_fusion == "lidar-priority"
    assert cfg.strict_anchor is True
    assert cfg.mirror_classes == ("Car", "Suv", "Bus", "Truck")


def test_render_bev_config_importable():
    from lidar_anchored_depth.configs.stages.render_bev import BevRenderConfig

    cfg = BevRenderConfig(scene=SceneConfig(scene="008"))
    assert cfg.scene.scene == "008"
    assert cfg.ts_stride == 1
    assert cfg.image_size == (1024, 1024)
    assert cfg.x_range is None
    assert cfg.max_gap_ms == 500


def test_inject_stage_has_correct_name():
    from lidar_anchored_depth.stages.dynamic_inject import DynamicInjectStage

    assert DynamicInjectStage.name == "inject"


def test_render_bev_stage_has_correct_name():
    from lidar_anchored_depth.stages.bev_render import BevRenderStage

    assert BevRenderStage.name == "render-bev"


def test_discover_static_ply_errors_when_no_upstream(tmp_path):
    """Auto-discovery should surface a clear message when the user
    forgot to run `lad complete` first."""
    from lidar_anchored_depth.configs.stages.inject import InjectConfig
    from lidar_anchored_depth.stages.dynamic_inject import _discover_static_ply

    cfg = InjectConfig(scene=SceneConfig(scene="008"))
    with pytest.raises(SystemExit) as exc:
        _discover_static_ply(cfg, tmp_path, "008")
    assert "no upstream `complete` run" in str(exc.value)


def test_discover_static_ply_finds_latest(tmp_path):
    """When a previous `complete` run wrote hybrid.ply, the discovery
    helper resolves it via the ``latest`` symlink."""
    from lidar_anchored_depth.configs.stages.inject import InjectConfig
    from lidar_anchored_depth.engine import OutputManager
    from lidar_anchored_depth.stages.dynamic_inject import _discover_static_ply

    om = OutputManager(tmp_path, scene="008", stage="complete")
    hybrid = om.run_dir / "hybrid.ply"
    hybrid.write_text("ply\n")  # contents irrelevant for resolver
    om.finalise()

    cfg = InjectConfig(scene=SceneConfig(scene="008"))
    found = _discover_static_ply(cfg, tmp_path, "008")
    assert found == hybrid


def test_cli_inject_help():
    r = _lad("inject", "--help")
    assert r.returncode == 0, r.stderr
    assert "--recon-dir" in r.stdout
    assert "--anchor-ts-ms" in r.stdout
    assert "--static-ply" in r.stdout
    assert "--scene.scene" in r.stdout


def test_cli_render_bev_help():
    r = _lad("render-bev", "--help")
    assert r.returncode == 0, r.stderr
    assert "--ts-stride" in r.stdout
    assert "--max-gap-ms" in r.stdout
    assert "--image-size" in r.stdout
    assert "--scene.scene" in r.stdout


def test_cli_top_help_lists_phase3_subcommands():
    r = _lad("--help")
    assert r.returncode == 0, r.stderr
    assert "inject" in r.stdout
    assert "render-bev" in r.stdout
