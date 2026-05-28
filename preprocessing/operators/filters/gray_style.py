from __future__ import annotations

import numpy as np
from PIL import Image

from preprocessing.operators.filters._helpers import get_pil_image
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class GrayStyleFilter(FilterOperator):
    name = "gray_style"

    def __init__(self, threshold: float = 1.0, max_side: int = 1600) -> None:
        self.threshold = float(threshold)
        self.max_side = int(max_side)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        try:
            image = get_pil_image(sample, context).convert("RGB")
        except Exception as exc:
            return OperatorResult.error(f"image_open_failed: {exc!r}")

        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BICUBIC)

        arr = np.asarray(image, dtype=np.int16)
        diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
        diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2])
        diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2])
        max_channel_diff = float(np.max([diff_rg.max(), diff_gb.max(), diff_rb.max()]))
        metrics = {"max_channel_diff": max_channel_diff, "threshold": self.threshold}

        if max_channel_diff < self.threshold:
            return OperatorResult.reject("grayscale_image", metrics)
        return OperatorResult.pass_(metrics)
