from preprocessing.operators.filters.aspect_ratio import AspectRatioFilter
from preprocessing.operators.filters.compression import CompressionQualityFilter
from preprocessing.operators.filters.gray_style import GrayStyleFilter
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import Sample


def make_context(tmp_path):
    return PipelineContext(
        run_id="run1",
        run_dir=tmp_path / "runs" / "run1",
        pass_archive_dir=tmp_path / "clean",
    )


def make_sample(path, root):
    return Sample(sample_id=path.name, source_path=path, relative_path=path.relative_to(root))


def test_aspect_ratio_filter_rejects_extreme_ratio(tmp_path, make_image):
    image = make_image(tmp_path / "wide.jpg", size=(300, 50))
    sample = make_sample(image, tmp_path)
    operator = AspectRatioFilter(ratio=2.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "REJECT"
    assert result.reason == "aspect_ratio_exceeds_threshold"
    assert result.metrics["ratio"] == 6.0


def test_aspect_ratio_filter_passes_normal_ratio(tmp_path, make_image):
    image = make_image(tmp_path / "normal.jpg", size=(100, 80))
    sample = make_sample(image, tmp_path)
    operator = AspectRatioFilter(ratio=2.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "PASS"
    assert result.metrics["ratio"] == 1.25


def test_gray_style_filter_rejects_grayscale(tmp_path, make_image):
    image = make_image(tmp_path / "gray.jpg", color=(120, 120, 120))
    sample = make_sample(image, tmp_path)
    operator = GrayStyleFilter(threshold=1.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "REJECT"
    assert result.reason == "grayscale_image"


def test_gray_style_filter_passes_color(tmp_path, make_image):
    image = make_image(tmp_path / "color.jpg", color=(255, 0, 0))
    sample = make_sample(image, tmp_path)
    operator = GrayStyleFilter(threshold=1.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "PASS"


def test_compression_quality_filter_reports_metric(tmp_path, make_image):
    image = make_image(tmp_path / "image.png", size=(20, 20), color=(20, 40, 60))
    sample = make_sample(image, tmp_path)
    operator = CompressionQualityFilter(min_quality_threshold=0.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "PASS"
    assert "compression_ratio" in result.metrics
