from preprocessing.operators.filters.blur import PersonBlurFilter
from preprocessing.operators.filters.framebox import FrameBoxFilter
from preprocessing.operators.filters.stitch import StitchLineFilterV2
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import Sample


def make_context(tmp_path):
    return PipelineContext("run1", tmp_path / "runs" / "run1", tmp_path / "clean")


def make_sample(path, root):
    return Sample(sample_id=path.name, source_path=path, relative_path=path.relative_to(root))


def test_framebox_filter_passes_plain_color_image(tmp_path, make_image):
    image = make_image(tmp_path / "plain.jpg", size=(80, 80), color=(120, 120, 120))
    result = FrameBoxFilter(min_std=15.0).process(make_sample(image, tmp_path), make_context(tmp_path))

    assert result.decision == "PASS"


def test_stitch_filter_passes_plain_color_image(tmp_path, make_image):
    image = make_image(tmp_path / "plain.jpg", size=(80, 80), color=(120, 120, 120))
    result = StitchLineFilterV2(prominence=6.0).process(make_sample(image, tmp_path), make_context(tmp_path))

    assert result.decision == "PASS"


def test_blur_filter_keeps_image_without_person_or_face_roi(tmp_path, make_image):
    image = make_image(tmp_path / "plain.jpg", size=(80, 80), color=(120, 120, 120))
    result = PersonBlurFilter().process(make_sample(image, tmp_path), make_context(tmp_path))

    assert result.decision == "PASS"
    assert result.reason is None
