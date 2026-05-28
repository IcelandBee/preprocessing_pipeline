from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from preprocessing.operators.filters._helpers import get_pil_image
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class FrameBoxFilter(FilterOperator):
    name = "framebox"

    def __init__(
        self,
        max_side: int = 1024,
        sides: int = 3,
        min_thick: int = 20,
        max_ratio: float = 0.45,
        min_std: float = 10.0,
    ) -> None:
        self.max_side = int(max_side)
        self.sides = int(sides)
        self.min_thick = int(min_thick)
        self.max_ratio = float(max_ratio)
        self.min_std = float(min_std)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        image = self._load_rgb_np(sample, context)
        if image is None:
            return OperatorResult.error("image_open_failed")
        bad, reason, metrics = self._check_image_content(image)
        if bad:
            return OperatorResult.reject(reason, metrics)
        return OperatorResult.pass_(metrics)

    def _load_rgb_np(self, sample: Sample, context: PipelineContext) -> Optional[np.ndarray]:
        try:
            image = get_pil_image(sample, context).convert("RGB")
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
            return False, "solid_color_image", {"std": std_value}

        top = self._border_thickness(gray, "top")
        bottom = self._border_thickness(gray, "bottom")
        left = self._border_thickness(gray, "left")
        right = self._border_thickness(gray, "right")
        metrics = {"top": top, "bottom": bottom, "left": left, "right": right, "std": std_value}

        if top > height * self.max_ratio or bottom > height * self.max_ratio or left > width * self.max_ratio or right > width * self.max_ratio:
            return False, "likely_solid_background_too_thick", metrics

        has_top = top > self.min_thick
        has_btm = bottom > self.min_thick
        has_lft = left > self.min_thick
        has_rgt = right > self.min_thick
        borders_found = sum([has_top, has_btm, has_lft, has_rgt])

        if borders_found >= self.sides:
            # Secondary check: crop center area and verify it has meaningful content
            cy_start = top if has_top else 0
            cy_end = height - bottom if has_btm else height
            cx_start = left if has_lft else 0
            cx_end = width - right if has_rgt else width

            if cy_end <= cy_start or cx_end <= cx_start:
                return False, "valid_content_too_small", metrics

            center_crop = gray[cy_start:cy_end, cx_start:cx_end]
            _, c_std = cv2.meanStdDev(center_crop)
            c_std_value = float(c_std[0][0])
            metrics["center_std"] = c_std_value

            if c_std_value < self.min_std:
                return False, "center_is_flat", metrics

            return True, "bordered_content", metrics

        return False, "clean", metrics