from __future__ import annotations

from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class CompressionQualityFilter(FilterOperator):
    name = "compression_quality"

    def __init__(self, min_quality_threshold: float = 0.2) -> None:
        self.min_quality_threshold = float(min_quality_threshold)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        try:
            with Image.open(sample.source_path) as image:
                width, height = image.size
        except Exception as exc:
            return OperatorResult.error(f"image_open_failed: {exc!r}")

        theoretical_size = max(width * height * 3, 1)
        actual_size = sample.source_path.stat().st_size
        compression_ratio = actual_size / theoretical_size
        metrics = {
            "width": width,
            "height": height,
            "actual_size": actual_size,
            "theoretical_size": theoretical_size,
            "compression_ratio": compression_ratio,
            "threshold": self.min_quality_threshold,
        }
        if compression_ratio < self.min_quality_threshold:
            return OperatorResult.reject("compression_ratio_below_threshold", metrics)
        return OperatorResult.pass_(metrics)
