import json

from preprocessing.pipeline.config import InputConfig, OutputConfig, FilterConfig, LabelConfig, OperatorConfig, PipelineConfig
from preprocessing.pipeline.operators import LabelOperator
from preprocessing.pipeline.registry import OperatorRegistry
from preprocessing.pipeline.runner import PipelineRunner
from preprocessing.pipeline.types import OperatorResult


class FakeLabeler(LabelOperator):
    name = "fake_labeler"

    def process(self, sample, context):
        return OperatorResult.label({"file_name": sample.source_path.name, "person_count": "0"})


def test_label_stage_writes_annotations_and_jsonl(tmp_path, make_image):
    clean_dir = tmp_path / "clean" / "run1"
    make_image(clean_dir / "a.jpg")
    registry = OperatorRegistry()
    registry.register("fake_labeler", FakeLabeler)
    config = PipelineConfig(
        input=InputConfig(input_dir=tmp_path / "raw"),
        output=OutputConfig(run_root=tmp_path / "runs", pass_archive_dir=tmp_path / "clean"),
        filter=FilterConfig(),
        label=LabelConfig(enabled=True, input_dir="auto", operator=OperatorConfig(name="fake_labeler")),
    )

    summary = PipelineRunner(config, registry=registry, run_id="run1").run_label()

    assert summary["labeled"] == 1
    annotation = json.loads((tmp_path / "runs" / "run1" / "annotations" / "a.json").read_text(encoding="utf-8"))
    assert annotation["annotation"]["person_count"] == "0"
    labels_lines = (tmp_path / "runs" / "run1" / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(labels_lines) == 1
