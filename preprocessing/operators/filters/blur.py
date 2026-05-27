from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class PersonBlurFilter(FilterOperator):
    name = "person_blur"

    def __init__(self, max_side: int = 1024, person_thr: float = 120.0, face_thr: float = 90.0, pad_frac: float = 0.08) -> None:
        self.max_side = int(max_side)
        self.person_thr = float(person_thr)
        self.face_thr = float(face_thr)
        self.pad_frac = float(pad_frac)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        bgr = self._load_bgr(sample)
        if bgr is None:
            return OperatorResult.error("image_open_failed")
        boxes = self._detect_faces(bgr)
        if not boxes:
            return OperatorResult.pass_({"used": "none", "no_roi_kept": True})
        score = self._blur_score(bgr, boxes)
        if score is None:
            return OperatorResult.pass_({"used": "face", "no_valid_roi_kept": True})
        metrics = {"used": "face", "score": score, "threshold": self.face_thr}
        if score < self.face_thr:
            return OperatorResult.reject("blur_score_below_threshold", metrics)
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

    def _detect_faces(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        face_xml = str(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(face_xml)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces[:10]]

    def _blur_score(self, bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scores: list[float] = []
        for x1, y1, x2, y2 in boxes:
            bw = x2 - x1
            bh = y2 - y1
            x1 = max(0, int(x1 - bw * self.pad_frac))
            y1 = max(0, int(y1 - bh * self.pad_frac))
            x2 = min(width, int(x2 + bw * self.pad_frac))
            y2 = min(height, int(y2 + bh * self.pad_frac))
            roi = gray[y1:y2, x1:x2]
            if roi.size >= 32 * 32:
                scores.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))
        return min(scores) if scores else None
