from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from preprocessing.operators.labelers.object_holdable import (
    normalize_annotation,
    normalize_bbox,
    normalize_coarse_position,
    normalize_object_name,
    normalize_text,
    ObjectHoldableLabeler,
)
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import OperatorResult, Sample


# --- normalization function tests ---


def test_normalize_object_name_valid():
    assert normalize_object_name("Red Cup") == "red cup"


def test_normalize_object_name_null_strings():
    for v in [None, "", "null", "none", "N/A", "unknown"]:
        assert normalize_object_name(v) is None


def test_normalize_object_name_extra_whitespace():
    assert normalize_object_name("  small   teddy   bear  ") == "small teddy bear"


def test_normalize_text_valid():
    assert normalize_text("A handheld device") == "A handheld device"


def test_normalize_text_null_strings():
    for v in [None, "", "null", "none", "N/A", "unknown"]:
        assert normalize_text(v) is None


def test_normalize_bbox_valid():
    result = normalize_bbox({"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9})
    assert result["x_min"] == 0.1
    assert result["y_min"] == 0.2
    assert result["x_max"] == 0.8
    assert result["y_max"] == 0.9


def test_normalize_bbox_clamps_and_scales():
    result = normalize_bbox({"x_min": 10, "y_min": 200, "x_max": 80, "y_max": 90})
    assert result["x_min"] == 0.1
    assert result["y_min"] == 1.0  # 200 > 100, not in percentage range, clamp to 1.0
    assert result["x_max"] == 0.8
    assert result["y_max"] == 0.9


def test_normalize_bbox_swaps_min_max():
    result = normalize_bbox({"x_min": 0.8, "y_min": 0.9, "x_max": 0.1, "y_max": 0.2})
    assert result["x_min"] == 0.1
    assert result["x_max"] == 0.8


def test_normalize_bbox_non_dict():
    result = normalize_bbox("not a dict")
    assert result == {"x_min": None, "y_min": None, "x_max": None, "y_max": None}


def test_normalize_coarse_position_valid():
    result = normalize_coarse_position({"horizontal": "left", "vertical": "top"})
    assert result["horizontal"] == "left"
    assert result["vertical"] == "top"


def test_normalize_coarse_position_invalid_values():
    result = normalize_coarse_position({"horizontal": "diagonal", "vertical": "middle"})
    assert result["horizontal"] == "unknown"
    assert result["vertical"] == "unknown"


def test_normalize_coarse_position_non_dict():
    result = normalize_coarse_position(None)
    assert result == {"horizontal": None, "vertical": None}


def test_normalize_annotation_complete():
    raw = {
        "object_name": "mug",
        "object_category": "container",
        "object_description": "a white ceramic mug",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.7, "y_max": 0.8},
        "holdability": "handheld",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "high",
        "reason": "small handheld object",
    }
    ann = normalize_annotation(raw)
    assert ann["object_name"] == "mug"
    assert ann["object_category"] == "container"
    assert ann["holdability"] == "handheld"
    assert ann["suitable_for_holding"] is True
    assert ann["should_use"] is True
    assert ann["confidence"] == "high"


def test_normalize_annotation_null_object_name_forces_false():
    raw = {
        "object_name": None,
        "object_category": "furniture",
        "object_description": "a large sofa",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
        "holdability": "not_suitable",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "low",
        "reason": "too large",
    }
    ann = normalize_annotation(raw)
    assert ann["object_name"] is None
    assert ann["should_use"] is False
    assert ann["suitable_for_holding"] is False


# --- class integration tests ---


def _make_sample(tmp_path: Path, name: str = "test_obj.jpg") -> tuple[Sample, Path]:
    img_path = tmp_path / name
    Image.new("RGB", (100, 100), (0, 128, 255)).save(img_path)
    return Sample(
        sample_id=name,
        source_path=img_path,
        relative_path=Path(name),
    ), img_path


def test_process_returns_label_on_success(tmp_path):
    sample, _ = _make_sample(tmp_path)
    context = PipelineContext(
        run_id="test-run",
        run_dir=tmp_path / "run",
        pass_archive_dir=tmp_path / "archive",
        pass_archive_layout="run_subdir",
    )

    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
        api_key_env="API_KEY",
        max_retries=1,
        max_tokens=768,
    )
    labeler.client = MagicMock()

    fake_response = MagicMock()
    raw_annotation = json.dumps({
        "object_name": "cup",
        "object_category": "container",
        "object_description": "a white ceramic cup",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.7, "y_max": 0.8},
        "holdability": "handheld",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "high",
        "reason": "small handheld object",
    })
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = raw_annotation
    labeler.client.chat.completions.create.return_value = fake_response

    result = labeler.process(sample, context)
    assert result.decision == "LABEL"
    assert result.labels["object_name"] == "cup"
    assert result.labels["holdability"] == "handheld"
    assert result.labels["should_use"] is True


def test_process_returns_error_when_no_client(tmp_path):
    sample, _ = _make_sample(tmp_path)
    context = PipelineContext(
        run_id="test-run",
        run_dir=tmp_path / "run",
        pass_archive_dir=tmp_path / "archive",
        pass_archive_layout="run_subdir",
    )
    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
    )
    labeler.client = None
    result = labeler.process(sample, context)
    assert result.decision == "ERROR"


def test_setup_creates_client():
    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
        api_key_env="API_KEY",
    )
    context = PipelineContext(
        run_id="test",
        run_dir=Path("/tmp/run"),
        pass_archive_dir=Path("/tmp/archive"),
        pass_archive_layout="run_subdir",
    )
    with patch.dict("os.environ", {"API_KEY": "test-key"}):
        labeler.setup(context)
    assert labeler.client is not None


def test_infer_retry_on_failure(tmp_path):
    sample, _ = _make_sample(tmp_path)
    context = PipelineContext(
        run_id="test-run",
        run_dir=tmp_path / "run",
        pass_archive_dir=tmp_path / "archive",
        pass_archive_layout="run_subdir",
    )

    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
        max_retries=3,
    )
    labeler.client = MagicMock()

    good_raw = json.dumps({
        "object_name": "phone",
        "object_category": "electronics",
        "object_description": "a smartphone",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.3, "y_min": 0.4, "x_max": 0.7, "y_max": 0.8},
        "holdability": "handheld",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "high",
        "reason": "small handheld device",
    })
    good_response = MagicMock()
    good_response.choices = [MagicMock()]
    good_response.choices[0].message.content = good_raw

    labeler.client.chat.completions.create.side_effect = [
        RuntimeError("timeout"),
        RuntimeError("timeout"),
        good_response,
    ]

    image = Image.new("RGB", (10, 10))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    with patch("time.sleep"):
        raw = labeler._infer(data_url)
    assert raw["object_name"] == "phone"