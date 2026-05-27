from pathlib import Path

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import OperatorResult, Sample


def test_sample_defaults_metadata_and_archive_path():
    sample = Sample(
        sample_id="a.jpg",
        source_path=Path("/data/raw/a.jpg"),
        relative_path=Path("a.jpg"),
    )

    assert sample.archive_path is None
    assert sample.metadata == {}


def test_operator_result_pass_factory():
    result = OperatorResult.pass_(metrics={"ratio": 1.25})

    assert result.decision == "PASS"
    assert result.reason is None
    assert result.metrics == {"ratio": 1.25}
    assert result.labels is None


def test_operator_result_reject_factory():
    result = OperatorResult.reject(
        reason="too_wide",
        metrics={"ratio": 3.0, "threshold": 2.0},
    )

    assert result.decision == "REJECT"
    assert result.reason == "too_wide"
    assert result.metrics["ratio"] == 3.0


def test_pipeline_context_paths(tmp_path):
    context = PipelineContext(
        run_id="20260527-203012",
        run_dir=tmp_path / "runs" / "20260527-203012",
        pass_archive_dir=tmp_path / "clean",
        pass_archive_layout="run_subdir",
    )

    assert context.manifest_path == tmp_path / "runs" / "20260527-203012" / "manifest.jsonl"
    assert context.summary_path == tmp_path / "runs" / "20260527-203012" / "summary.json"
    assert context.annotation_dir == tmp_path / "runs" / "20260527-203012" / "annotations"
    assert context.labels_jsonl_path == tmp_path / "runs" / "20260527-203012" / "labels.jsonl"
    assert context.archive_path_for(Path("nested/a.jpg")) == tmp_path / "clean" / "20260527-203012" / "nested" / "a.jpg"
