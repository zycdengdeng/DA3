"""Smoke tests for the 5 upstream-stage wrappers added in Phase 7.

These don't invoke the underlying scripts (which need GPU + dataset);
they just verify:
* Each config dataclass constructs with sensible defaults.
* Each stage class has the right ``name``.
* The CLI surfaces all 5 subcommands with the expected flags.

End-to-end execution is covered by the existing
``tests/integration/test_pipeline_orchestration.py`` with MockStage.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lidar_anchored_depth.configs import SceneConfig


def _lad(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parent.parent
    return subprocess.run(
        [sys.executable, "-m", "lidar_anchored_depth.cli", *args],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(repo_root / "src"), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )


# ---- config dataclasses --------------------------------------------------

def test_depth_config_defaults():
    from lidar_anchored_depth.configs.stages.depth import DepthConfig
    cfg = DepthConfig(scene=SceneConfig(scene="008"))
    assert cfg.model.startswith("depth-anything/")
    assert cfg.process_res == 504
    assert cfg.skip_existing is True


def test_mask_config_defaults():
    from lidar_anchored_depth.configs.stages.mask import MaskConfig
    cfg = MaskConfig(scene=SceneConfig(scene="008"))
    assert "sam" in cfg.model_id.lower()
    assert cfg.bbox_pad_px == 8
    assert cfg.multimask is False


def test_seg_config_defaults():
    from lidar_anchored_depth.configs.stages.seg import SegConfig
    cfg = SegConfig(scene=SceneConfig(scene="008"))
    assert "segformer" in cfg.model.lower()
    assert cfg.save_viz is False
    assert cfg.skip_existing is True


def test_calib_config_defaults():
    from lidar_anchored_depth.configs.stages.calib import CalibConfig
    cfg = CalibConfig(scene=SceneConfig(scene="008"))
    assert cfg.voxel_size == 0.05
    assert cfg.aahad_pixel_stride == 2
    assert cfg.region_grid_rows == 0  # disabled by default


def test_object_accum_config_defaults():
    from lidar_anchored_depth.configs.stages.object_accum import (
        ObjectAccumConfig,
    )
    cfg = ObjectAccumConfig(scene=SceneConfig(scene="008"))
    assert cfg.solver_mode == "per-camera"
    assert cfg.lidar_color == "height"
    assert cfg.voxel_size == 0.05


# ---- stage classes have the right name ----------------------------------

@pytest.mark.parametrize("expected, import_path", [
    ("depth", "lidar_anchored_depth.stages.da3_depth.DA3DepthStage"),
    ("mask", "lidar_anchored_depth.stages.sam_mask.SAMMaskStage"),
    ("seg", "lidar_anchored_depth.stages.segformer_seg.SegFormerStage"),
    ("calib", "lidar_anchored_depth.stages.static_calib.StaticCalibStage"),
    ("object-accum",
     "lidar_anchored_depth.stages.object_accum.ObjectAccumStage"),
])
def test_stage_names(expected, import_path):
    import importlib
    module_path, cls_name = import_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    assert cls.name == expected


# ---- script_runner helpers ----------------------------------------------

def test_script_runner_resolves_existing_script():
    from lidar_anchored_depth.engine.script_runner import _resolve_script
    # Any real script in scripts/ should resolve.
    p = _resolve_script("run_da3_inference.py")
    assert p.is_file()


def test_script_runner_raises_on_missing_script():
    from lidar_anchored_depth.engine.script_runner import _resolve_script
    with pytest.raises(SystemExit, match="not found"):
        _resolve_script("does_not_exist.py")


# ---- CLI smoke tests ----------------------------------------------------

@pytest.mark.parametrize("subcommand,must_contain", [
    ("depth",      "--model"),
    ("mask",       "--bbox-pad-px"),
    ("seg",        "--save-viz"),
    ("calib",      "--d-paths-glob"),
    ("object-accum", "--solver-mode"),
])
def test_cli_help_for_upstream_stage(subcommand, must_contain):
    r = _lad(subcommand, "--help")
    assert r.returncode == 0, r.stderr
    assert must_contain in r.stdout
    assert "--scene.scene" in r.stdout         # shared block present
    assert "--runtime.gpu-ids" in r.stdout
    assert "--output.root" in r.stdout


def test_cli_top_help_lists_all_upstream():
    r = _lad("--help")
    assert r.returncode == 0, r.stderr
    for name in ("depth", "mask", "seg", "calib", "object-accum"):
        assert name in r.stdout


# ---- FullPipeline registry has all 8 stages -----------------------------

def test_full_pipeline_registry_has_all_eight_stages():
    from lidar_anchored_depth.pipelines.full_pipeline import FullPipeline
    expected = {
        "depth", "mask", "seg", "calib", "object-accum",
        "complete", "inject", "render-bev",
    }
    assert set(FullPipeline.REGISTRY.keys()) == expected
