from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

from openai import OpenAI
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import LabelOperator
from preprocessing.pipeline.types import OperatorResult, Sample

FIELDS = [
    "person_count",
    "gender",
    "head_visible",
    "shot_type",
    "clothes_visible",
    "holding_object",
    "obvious_makeup",
    "expression",
    "hair_visible",
    "hair_color",
    "facial_features_clear",
    "face_direction",
    "hand_hold_feasible",
    "person_prominence",
    "person_size_in_frame",
]

CHOICES = {
    "person_count": {"0", "1", "N"},
    "gender": {"male", "female", "uncertain", "N/A"},
    "head_visible": {"yes", "no", "partial", "N/A"},
    "shot_type": {"close_up", "half_body", "full_body", "near_full_body", "N/A"},
    "clothes_visible": {"yes", "no", "N/A"},
    "holding_object": {"yes", "no", "N/A"},
    "obvious_makeup": {"yes", "no", "N/A"},
    "hair_visible": {"yes", "no", "partial", "N/A"},
    "facial_features_clear": {"yes", "no", "N/A"},
    "face_direction": {"frontal", "near_frontal", "side_view", "back_view", "N/A"},
    "hand_hold_feasible": {"yes", "no", "N/A"},
    "person_prominence": {"close", "medium", "far", "N/A"},
    "person_size_in_frame": {"large", "medium", "small", "N/A"},
}

SYSTEM_PROMPT = "You are a strict person-image tagging model. Return valid JSON only."
USER_TEXT = "Classify this image and return one JSON object with the required person attribute fields."


def normalize_annotation(raw: dict[str, Any], file_name: str) -> dict[str, str]:
    person_count = _choice(raw.get("person_count"), CHOICES["person_count"], "N/A")
    ann = {"file_name": file_name, "person_count": person_count}
    for field in FIELDS[1:]:
        value = raw.get(field, "N/A")
        if field in CHOICES:
            ann[field] = _choice(value, CHOICES[field], "N/A")
        elif field in {"expression", "hair_color"}:
            ann[field] = _one_word_or_na(value)
        else:
            ann[field] = str(value)

    if person_count in {"0", "N", "N/A"}:
        for field in FIELDS[1:]:
            ann[field] = "N/A"
    elif person_count == "1" and ann["head_visible"] == "no":
        for field in ["hair_color", "expression", "obvious_makeup", "facial_features_clear", "face_direction"]:
            ann[field] = "N/A"
        ann["hair_visible"] = "no"
    if ann.get("hair_visible") == "no":
        ann["hair_color"] = "N/A"
    return ann


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = str(value).strip() if value is not None else fallback
    return text if text in allowed else fallback


def _one_word_or_na(value: Any) -> str:
    text = str(value).strip() if value is not None else "N/A"
    if not text or text == "N/A":
        return "N/A"
    return text.split()[0]


class PersonAttributeLabeler(LabelOperator):
    name = "person_attribute_labeler"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key_env: str = "API_KEY",
        max_retries: int = 3,
        max_tokens: int = 224,
        temperature: float = 0.0,
        max_pixels: int = 589824,
        **unused: object,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.max_pixels = int(max_pixels)
        self.client: OpenAI | None = None

    def setup(self, context: PipelineContext) -> None:
        api_key = os.environ.get(self.api_key_env, "")
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        if self.client is None:
            return OperatorResult.error("labeler_not_initialized")
        try:
            image = load_image_rgb(sample.source_path, max_pixels=self.max_pixels)
            image_url = pil_to_data_url(image)
            raw = self._infer(image_url)
            labels = normalize_annotation(raw, file_name=sample.source_path.name)
            return OperatorResult.label(labels)
        except Exception as exc:
            return OperatorResult.error(f"label_inference_failed: {exc!r}")

    def _infer(self, image_url: str) -> dict[str, Any]:
        assert self.client is not None
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": USER_TEXT},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        },
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content or "{}"
                return safe_json_loads(text)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(float(attempt))
        raise RuntimeError(f"model inference failed after retries: {last_error!r}")


def load_image_rgb(path: str | os.PathLike[str], max_pixels: int = 0) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_pixels > 0:
        width, height = image.size
        pixels = width * height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)
    return image


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def safe_json_loads(text: str) -> dict[str, Any]:
    return json.loads(text)
