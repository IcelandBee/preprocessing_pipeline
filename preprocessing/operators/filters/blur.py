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

    def __init__(
        self,
        max_side: int = 1024,
        person_thr: float = 120.0,
        face_thr: float = 90.0,
        pad_frac: float = 0.08,
        agg: str = "min",
    ) -> None:
        self.max_side = int(max_side)
        self.person_thr = float(person_thr)
        self.face_thr = float(face_thr)
        self.pad_frac = float(pad_frac)
        self.agg = agg

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        bgr = self._load_bgr(sample)
        if bgr is None:
            return OperatorResult.error("image_open_failed")

        # Primary: HOG person detection
        people = self._detect_people_hog(bgr)
        used = "person"

        if people:
            score = self._blur_score(bgr, people)
            thr = self.person_thr
        else:
            # Fallback: Haar face detection
            faces = self._detect_faces_haar(bgr)
            if faces:
                score = self._blur_score(bgr, faces)
                used = "face"
                thr = self.face_thr
            else:
                return OperatorResult.pass_({"used": "none", "no_roi_kept": True})

        if score is None:
            return OperatorResult.pass_({"used": used, "no_valid_roi_kept": True})

        metrics = {"used": used, "score": score, "threshold": thr}
        if score < thr:
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

    def _detect_people_hog(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        rects, weights = hog.detectMultiScale(
            bgr,
            hitThreshold=0.0,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        img_area = float(w * h)
        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []
        for (x, y, bw, bh), sc in zip(rects, weights):
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            area = float((x2 - x1) * (y2 - y1))
            if area < 0.02 * img_area:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(float(sc))

        if not boxes:
            return []

        kept = self._nms(boxes, scores, iou_thr=0.4)
        return [boxes[i] for i in kept]

    def _detect_faces_haar(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        face_xml = str(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(face_xml)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        h, w = bgr.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for (x, y, bw, bh) in faces[:10]:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _blur_score(self, bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scores: list[float] = []
        for x1, y1, x2, y2 in boxes:
            bw = x2 - x1
            bh = y2 - y1
            px = int(self.pad_frac * bw)
            py = int(self.pad_frac * bh)
            x1 = max(0, x1 - px)
            y1 = max(0, y1 - py)
            x2 = min(width, x2 + px)
            y2 = min(height, y2 + py)
            roi = gray[y1:y2, x1:x2]
            if roi.size >= 32 * 32:
                scores.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))

        if not scores:
            return None
        if self.agg == "mean":
            return float(np.mean(scores))
        return float(np.min(scores))

    def _nms(
        self,
        boxes: list[tuple[int, int, int, int]],
        scores: list[float],
        iou_thr: float = 0.4,
    ) -> list[int]:
        if not boxes:
            return []
        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)

        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores_np.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            order = order[1:][iou < iou_thr]

        return keep