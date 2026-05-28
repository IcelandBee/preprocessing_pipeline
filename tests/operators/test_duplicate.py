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


def test_duplicate_filter_window_size_default():
    operator = DuplicateFilter()
    assert operator.window_size == 50


def test_duplicate_filter_window_size_configurable():
    operator = DuplicateFilter(window_size=100)
    assert operator.window_size == 100


def test_duplicate_filter_presort_window_matches_full_pairwise(tmp_path, make_image):
    # 4 unique images + 4 duplicate pairs (same color as one of the unique ones)
    paths = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
    # 4 unique images
    for i, color in enumerate(colors):
        p = make_image(tmp_path / f"unique_{i}.jpg", size=(20, 20), color=color)
        paths.append(("unique", i, p, color))
    # 4 duplicates (same color as corresponding unique)
    for i, color in enumerate(colors):
        p = make_image(tmp_path / f"dup_{i}.jpg", size=(10, 10), color=color)
        paths.append(("dup", i, p, color))

    samples = [
        Sample(sample_id=info[2].name, source_path=info[2], relative_path=info[2].relative_to(tmp_path))
        for info in paths
    ]
    context = PipelineContext("run1", tmp_path / "runs" / "run1", tmp_path / "clean")

    # Full pairwise (window large enough to cover all 8 images)
    full_operator = DuplicateFilter(threshold=5, hash_size=8, window_size=1000)
    full_results = full_operator.process_batch(samples, context)

    # Window comparison (small window)
    window_operator = DuplicateFilter(threshold=5, hash_size=8, window_size=50)
    window_results = window_operator.process_batch(samples, context)

    full_decisions = {sid: r.decision for sid, r in full_results.items()}
    window_decisions = {sid: r.decision for sid, r in window_results.items()}
    assert full_decisions == window_decisions
