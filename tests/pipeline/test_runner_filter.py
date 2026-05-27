from pathlib import Path

from preprocessing.pipeline.config import InputConfig, OutputConfig, FilterConfig, LabelConfig, OperatorConfig, PipelineConfig
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.registry import OperatorRegistry
from preprocessing.pipeline.runner import PipelineRunner
from preprocessing.pipeline.types import OperatorResult


class AlwaysPass(FilterOperator):
    name = "always_pass"

    def process(self, sample, context):
        return OperatorResult.pass_({"operator": self.name})


class RejectName(FilterOperator):
    name = "reject_name"

    def __init__(self, name_contains: str = "bad") -> None:
        self.name_contains = name_contains

    def process(self, sample, context):
        if self.name_contains in sample.source_path.name:
            return OperatorResult.reject("name_matched", {"name_contains": self.name_contains})
        return OperatorResult.pass_()


class ShouldNotRun(FilterOperator):
    name = "should_not_run"
    calls = 0

    def process(self, sample, context):
        ShouldNotRun.calls += 1
        return OperatorResult.pass_()


def make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input=InputConfig(input_dir=tmp_path / "raw", recursive=True, image_exts=[".jpg"]),
        output=OutputConfig(run_root=tmp_path / "runs", pass_archive_dir=tmp_path / "clean", pass_archive_layout="run_subdir", overwrite=False),
        filter=FilterConfig(
            short_circuit=True,
            operators=[
                OperatorConfig(name="always_pass"),
                OperatorConfig(name="reject_name", params={"name_contains": "bad"}),
                OperatorConfig(name="should_not_run"),
            ],
        ),
        label=LabelConfig(enabled=False),
    )


def test_filter_stage_short_circuits_and_archives_passed_images(tmp_path, make_image):
    raw = tmp_path / "raw"
    make_image(raw / "good.jpg")
    make_image(raw / "bad.jpg")
    registry = OperatorRegistry()
    registry.register("always_pass", AlwaysPass)
    registry.register("reject_name", RejectName)
    registry.register("should_not_run", ShouldNotRun)
    ShouldNotRun.calls = 0

    runner = PipelineRunner(make_config(tmp_path), registry=registry, run_id="run1")
    summary = runner.run_filter()

    assert summary["passed"] == 1
    assert summary["rejected"] == 1
    assert (tmp_path / "clean" / "run1" / "good.jpg").exists()
    assert not (tmp_path / "clean" / "run1" / "bad.jpg").exists()
    assert ShouldNotRun.calls == 1

    manifest_lines = (tmp_path / "runs" / "run1" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    manifest_text = "\n".join(manifest_lines)
    assert '"status": "PASS"' in manifest_text
    assert '"status": "REJECT"' in manifest_text
    assert '"rejected_by": "reject_name"' in manifest_text
    assert (tmp_path / "runs" / "run1" / "summary.json").exists()
    assert (tmp_path / "runs" / "run1" / "config.snapshot.yaml").exists()
