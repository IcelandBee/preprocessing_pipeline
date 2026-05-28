from __future__ import annotations

from PIL import Image

from preprocessing.operators.filters._helpers import get_pil_image
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class AspectRatioFilter(FilterOperator):
    name = "aspect_ratio"

    def __init__(self, ratio: float = 2.0) -> None:
        self.ratio = float(ratio)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        try:
            image = get_pil_image(sample, context)
            width, height = image.size
        except Exception as exc:
            return OperatorResult.error(f"image_open_failed: {exc!r}")

        if width <= 0 or height <= 0:
            return OperatorResult.error("invalid_image_size", {"width": width, "height": height})

        actual_ratio = max(width / height, height / width)
        metrics = {"width": width, "height": height, "ratio": round(actual_ratio, 6), "threshold": self.ratio}
        if actual_ratio > self.ratio:
            return OperatorResult.reject("aspect_ratio_exceeds_threshold", metrics)
        return OperatorResult.pass_(metrics)
