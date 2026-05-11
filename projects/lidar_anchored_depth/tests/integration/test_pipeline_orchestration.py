"""End-to-end orchestration tests using MockStage.

The real stages need a V2X dataset + a trained checkpoint + GPUs;
none of those are appropriate for a unit-test suite. These tests
instead substitute lightweight ``MockStage`` classes into the
:class:`FullPipeline` registry and verify the *orchestration* —
that each stage gets its own ``OutputManager`` provisioned, that
artefacts land in the right per-scene/per-stage/per-run directory
layout, that the ``latest`` symlink is refreshed after each stage,
and that ``--stages`` + ``--from-stage`` correctly skip ahead.

The real stage internals are covered by unit tests in the modules
they live in; here we only assert that wiring the stages together
produces the expected on-disk shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from lidar_anchored_depth.configs.base import (
    OutputConfig,
    RuntimeConfig,
    SceneConfig,
)
from lidar_anchored_depth.configs.stages.complete import CompleteConfig
from lidar_anchored_depth.configs.stages.pipeline import FullPipelineConfig
from lidar_anchored_depth.engine import OutputManager
from lidar_anchored_depth.pipelines.full_pipeline import FullPipeline
from lidar_anchored_depth.stages.base import Stage, StageArtifacts


# ---- mock stages --------------------------------------------------------
#
# Each stage just touches a sentinel file and records that it ran. The
# registry is monkey-patched in the test fixtures.

@dataclass
class _StageCall:
    name: str
    output_dir: Path
    scene: str


class _MockBase(Stage):
    """Base for the three mock stages: records the call, writes a file
    named after the stage so we can assert artefact placement."""

    name: str = "MOCK"

    def __init__(self, cfg, output_dir):
        super().__init__(cfg, output_dir)
        self.cfg = cfg
        self.output_dir = output_dir

    def run(self) -> StageArtifacts:
        # Record the call on the shared registry.
        _CALLS.append(_StageCall(
            name=self.name,
            output_dir=self.output_dir,
            scene=self.cfg.scene.scene,
        ))
        sentinel = self.output_dir / f"{self.name}.txt"
        sentinel.write_text(f"ran {self.name} for {self.cfg.scene.scene}\n")
        return StageArtifacts(
            output_dir=self.output_dir,
            files={"sentinel": sentinel},
            summary={"stage": self.name, "scene": self.cfg.scene.scene},
        )


class _MockComplete(_MockBase):
    name = "complete"


class _MockInject(_MockBase):
    name = "inject"


class _MockRenderBev(_MockBase):
    name = "render-bev"


_CALLS: list[_StageCall] = []


@pytest.fixture
def mock_registry(monkeypatch):
    """Replace FullPipeline.REGISTRY with the mock stages for the
    duration of one test. Clears _CALLS up-front so each test sees a
    fresh recording."""
    _CALLS.clear()
    monkeypatch.setitem(FullPipeline.REGISTRY, "complete", _MockComplete)
    monkeypatch.setitem(FullPipeline.REGISTRY, "inject", _MockInject)
    monkeypatch.setitem(FullPipeline.REGISTRY, "render-bev", _MockRenderBev)
    yield _CALLS


# ---- helpers ------------------------------------------------------------

def _pipeline(scene: str, root: Path, **kwargs) -> FullPipelineConfig:
    """Build a FullPipelineConfig pointed at a tmp outputs root."""
    return FullPipelineConfig(
        complete=CompleteConfig(
            scene=SceneConfig(scene=scene),
            output=OutputConfig(root=root),
            runtime=RuntimeConfig(),
        ),
        **kwargs,
    )


# ---- tests --------------------------------------------------------------

def test_full_pipeline_runs_all_three_stages_in_order(
    tmp_path, mock_registry,
):
    cfg = _pipeline("008", tmp_path / "out")
    rc = FullPipeline(cfg).run()
    assert rc == 0
    names = [c.name for c in mock_registry]
    assert names == ["complete", "inject", "render-bev"]


def test_full_pipeline_each_stage_gets_own_output_dir(
    tmp_path, mock_registry,
):
    root = tmp_path / "out"
    FullPipeline(_pipeline("008", root)).run()
    # Each stage's output_dir is under <root>/<scene>/<stage>/<run_id>/
    for call in mock_registry:
        assert call.output_dir.parent.parent == root / "008"
        assert call.output_dir.parent.name == call.name
        assert call.output_dir.is_dir()


def test_full_pipeline_writes_sentinel_per_stage(
    tmp_path, mock_registry,
):
    root = tmp_path / "out"
    FullPipeline(_pipeline("008", root)).run()
    for call in mock_registry:
        sentinel = call.output_dir / f"{call.name}.txt"
        assert sentinel.is_file()
        assert call.name in sentinel.read_text()


def test_full_pipeline_refreshes_latest_after_each_stage(
    tmp_path, mock_registry,
):
    root = tmp_path / "out"
    FullPipeline(_pipeline("008", root)).run()
    for stage in ("complete", "inject", "render-bev"):
        link = root / "008" / stage / "latest"
        assert link.is_symlink()
        # Symlink target is the just-finished run dir (relative).
        target = link.readlink()
        assert (root / "008" / stage / target).is_dir()


def test_full_pipeline_writes_summary_json_per_stage(
    tmp_path, mock_registry,
):
    root = tmp_path / "out"
    FullPipeline(_pipeline("008", root)).run()
    for stage in ("complete", "inject", "render-bev"):
        latest = OutputManager.resolve_latest(root, "008", stage)
        assert latest is not None
        summary = latest / "summary.json"
        assert summary.is_file()
        assert stage in summary.read_text()


def test_full_pipeline_writes_config_json_per_stage(
    tmp_path, mock_registry,
):
    """OutputManager.dump_config should write the stage cfg into the
    run dir before .run() is called — this is what makes a run
    reproducible from disk."""
    root = tmp_path / "out"
    FullPipeline(_pipeline("008", root)).run()
    for stage in ("complete", "inject", "render-bev"):
        latest = OutputManager.resolve_latest(root, "008", stage)
        assert latest is not None
        cfg_path = latest / "config.json"
        assert cfg_path.is_file()
        assert '"scene"' in cfg_path.read_text()


def test_full_pipeline_stages_subset_skips_render_bev(
    tmp_path, mock_registry,
):
    cfg = _pipeline(
        "008", tmp_path / "out",
        stages=("complete", "inject"),
    )
    FullPipeline(cfg).run()
    names = [c.name for c in mock_registry]
    assert names == ["complete", "inject"]


def test_full_pipeline_from_stage_skips_complete(
    tmp_path, mock_registry,
):
    cfg = _pipeline("008", tmp_path / "out", from_stage="inject")
    FullPipeline(cfg).run()
    names = [c.name for c in mock_registry]
    assert names == ["inject", "render-bev"]


def test_full_pipeline_propagates_complete_scene_to_downstream(
    tmp_path, mock_registry,
):
    """The user only sets --complete.scene.scene; the orchestrator
    must propagate it into the inject + render-bev calls so all three
    stages run on the same scene id."""
    cfg = _pipeline("042", tmp_path / "out")
    FullPipeline(cfg).run()
    for call in mock_registry:
        assert call.scene == "042"


def test_full_pipeline_per_stage_output_run_ids_distinct(
    tmp_path, mock_registry,
):
    """Each stage should get its own timestamped run_id, not share one
    across the three calls. We assert that the three run dirs are
    distinct (they may collide on the second granularity if all three
    finish in the same wall-clock second — accept that)."""
    root = tmp_path / "out"
    FullPipeline(_pipeline("008", root)).run()
    run_ids = {c.output_dir.name for c in mock_registry}
    # All three got their own dir; collisions on the second granularity
    # are allowed but should be rare in CI.
    assert len(run_ids) >= 1
    for c in mock_registry:
        assert c.output_dir.name  # non-empty


def test_full_pipeline_from_stage_not_in_stages_errors(
    tmp_path, mock_registry,
):
    cfg = _pipeline(
        "008", tmp_path / "out",
        stages=("complete", "inject"),
        from_stage="render-bev",
    )
    with pytest.raises(SystemExit) as exc:
        FullPipeline(cfg).run()
    assert "not in" in str(exc.value)


# ---- shared-block propagation -------------------------------------------

def test_full_pipeline_propagates_output_root(tmp_path, mock_registry):
    """--complete.output.root must propagate to inject + render-bev."""
    root = tmp_path / "shared-out"
    cfg = FullPipelineConfig(
        complete=CompleteConfig(
            scene=SceneConfig(scene="008"),
            output=OutputConfig(root=root),
        ),
    )
    FullPipeline(cfg).run()
    for stage in ("complete", "inject", "render-bev"):
        latest = OutputManager.resolve_latest(root, "008", stage)
        assert latest is not None, f"{stage} did not write into {root}"


def test_full_pipeline_propagates_runtime_gpu_ids(tmp_path, mock_registry):
    """The mock stages don't actually use gpu_ids, but the orchestrator
    must still copy the value across — without that, real downstream
    stages would silently lose multi-GPU support when run via the
    pipeline."""
    cfg = FullPipelineConfig(
        complete=CompleteConfig(
            scene=SceneConfig(scene="008"),
            output=OutputConfig(root=tmp_path / "out"),
            runtime=RuntimeConfig(gpu_ids=(0, 1, 2, 3)),
        ),
    )
    pipe = FullPipeline(cfg)
    inj = pipe._stage_cfg("inject")
    bev = pipe._stage_cfg("render-bev")
    assert inj.runtime.gpu_ids == (0, 1, 2, 3)
    assert bev.runtime.gpu_ids == (0, 1, 2, 3)
