from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class StitchLineFilterV2(FilterOperator):
    name = "stitch_v2"

    def __init__(self, max_side: int = 1400, prominence: float = 6.0) -> None:
        self.max_side = int(max_side)
        self.prominence = float(prominence)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        bgr = self._load_bgr(sample)
        if bgr is None:
            return OperatorResult.error("image_open_failed")
        bad, reason, metrics = self._detect_stitch(bgr)
        if bad:
            return OperatorResult.reject(reason, metrics)
        return OperatorResult.pass_(metrics)

    def _load_bgr(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            image = Image.open(sample.source_path).convert("RGB")
        except Exception:
            return None
        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def _detect_stitch(self, bgr: np.ndarray) -> tuple[bool, str, dict[str, object]]:
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        proj_v = np.sum(grad_x, axis=0)
        proj_h = np.sum(grad_y, axis=1)

        found_v, metrics_v = self._check_projection(proj_v, width, "vertical")
        if found_v:
            return True, "stitch_line_vertical", metrics_v
        found_h, metrics_h = self._check_projection(proj_h, height, "horizontal")
        if found_h:
            return True, "stitch_line_horizontal", metrics_h
        return False, "clean", {"vertical": metrics_v, "horizontal": metrics_h}

    def _check_projection(self, projection: np.ndarray, length: int, axis: str) -> tuple[bool, dict[str, object]]:
        mean_val = float(np.mean(projection))
        if mean_val == 0:
            return False, {"axis": axis, "mean": mean_val}
        norm = projection / mean_val
        start = int(length * 0.2)
        end = int(length * 0.8)
        roi = norm[start:end]
        if len(roi) == 0:
            return False, {"axis": axis, "mean": mean_val}
        peak_idx_roi = int(np.argmax(roi))
        peak_val = float(roi[peak_idx_roi])
        peak_idx_global = peak_idx_roi + start
        metrics = {"axis": axis, "peak_position": peak_idx_global / length, "peak_prominence": peak_val, "threshold": self.prominence}
        return peak_val >= self.prominence, metrics
