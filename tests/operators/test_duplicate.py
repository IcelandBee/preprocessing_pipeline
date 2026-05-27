from preprocessing.operators.filters.duplicate import DuplicateFilter
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import Sample


def test_duplicate_filter_rejects_smaller_duplicate(tmp_path, make_image):
    first = make_image(tmp_path / "first.jpg", size=(20, 20), color=(255, 0, 0))
    second = make_image(tmp_path / "second.jpg", size=(20, 20), color=(255, 0, 0))
    samples = [
        Sample(sample_id="first.jpg", source_path=first, relative_path=first.relative_to(tmp_path)),
        Sample(sample_id="second.jpg", source_path=second, relative_path=second.relative_to(tmp_path)),
    ]
    context = PipelineContext("run1", tmp_path / "runs" / "run1", tmp_path / "clean")
    operator = DuplicateFilter(threshold=0, hash_size=8)

    results = operator.process_batch(samples, context)

    decisions = {sample_id: result.decision for sample_id, result in results.items()}
    assert sorted(decisions.values()) == ["PASS", "REJECT"]
    rejected = [result for result in results.values() if result.decision == "REJECT"][0]
    assert rejected.reason == "duplicate_image"
    assert "keeper" in rejected.metrics
