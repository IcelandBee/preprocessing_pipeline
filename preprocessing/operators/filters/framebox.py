from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class FrameBoxFilter(FilterOperator):
    name = "framebox"

    def __init__(self, max_side: int = 1024, sides: int = 3, min_thick: int = 20, max_ratio: float = 0.30, min_std: float = 15.0) -> None:
        self.max_side = int(max_side)
        self.sides = int(sides)
        self.min_thick = int(min_thick)
        self.max_ratio = float(max_ratio)
        self.min_std = float(min_std)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        image = self._load_rgb_np(sample)
        if image is None:
            return OperatorResult.error("image_open_failed")
        bad, reason, metrics = self._check_image_content(image)
        if bad:
            return OperatorResult.reject(reason, metrics)
        return OperatorResult.pass_(metrics)

    def _load_rgb_np(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            image = Image.open(sample.source_path).convert("RGB")
        except Exception:
            return None
        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)

    def _border_thickness(self, gray: np.ndarray, side: str, color_tol: int = 15, noise_tol: float = 0.95) -> int:
        if side == "top":
            matrix = gray
        elif side == "bottom":
            matrix = gray[::-1, :]
        elif side == "left":
            matrix = gray.T
        elif side == "right":
            matrix = gray.T[::-1, :]
        else:
            return 0
        bg_color = np.median(matrix[0, :])
        diff = np.abs(matrix.astype(int) - bg_color)
        row_matches = np.mean(diff < color_tol, axis=1)
        border_rows = np.where(row_matches < noise_tol)[0]
        return int(border_rows[0]) if len(border_rows) else matrix.shape[0]

    def _check_image_content(self, image: np.ndarray) -> tuple[bool, str, dict[str, object]]:
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, std = cv2.meanStdDev(gray)
        std_value = float(std[0][0])
        if std_value < self.min_std:
            return False, "flat_image", {"std": std_value}

        top = self._border_thickness(gray, "top")
        bottom = self._border_thickness(gray, "bottom")
        left = self._border_thickness(gray, "left")
        right = self._border_thickness(gray, "right")
        metrics = {"top": top, "bottom": bottom, "left": left, "right": right, "std": std_value}

        if top > height * self.max_ratio or bottom > height * self.max_ratio or left > width * self.max_ratio or right > width * self.max_ratio:
            return False, "likely_solid_background_too_thick", metrics

        borders_found = sum([top > self.min_thick, bottom > self.min_thick, left > self.min_thick, right > self.min_thick])
        if borders_found >= self.sides:
            return True, "bordered_content", metrics
        return False, "clean", metrics
