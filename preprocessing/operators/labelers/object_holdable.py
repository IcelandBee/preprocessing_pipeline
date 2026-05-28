from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Optional

from openai import OpenAI
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import LabelOperator
from preprocessing.pipeline.types import OperatorResult, Sample


OBJECT_PROMPT = r"""
You are labeling a dataset of OBJECT images for later synthetic triplet generation:

    person image + object image -> person holding the object

You will be given ONE image that should mainly contain an object (or a pet / small animal).

Task:
1) Select ONE final target object for downstream "person holding the object" generation.
2) Identify the chosen target object.
3) Describe it briefly but concretely.
4) Estimate where it is located in the image.
5) Decide whether this target is suitable for "human holding/carrying" generation.

Selection policy for multiple objects:
- The downstream use case is: person holding ONE target object.
- If multiple holdable objects are visible, choose ONE single best target object rather than naming a collection.
- Prefer an atomic, singular object that a person would naturally hold in hand or arms.
- Choose the most salient target using this priority:
  1) clearest and most visually dominant,
  2) most centered or largest visible,
  3) least occluded,
  4) easiest to describe as one physical item.

Granularity rules:
- Prefer a single item over a set/group/collection.
  Examples:
  - choose "screwdriver" rather than "screwdriver set"
  - choose "mug" rather than "cups"
  - choose "apple" rather than "fruit assortment"
- Only label a set/container/group as the object when it is clearly presented as one holdable unit,
  such as a bouquet, gift box, toolbox, packaged toy set, closed case, or bag of items.
- If the image mainly shows a packaged/contained unit and no single item can be cleanly singled out,
  label that packaged unit instead.
- object_name, object_description, bbox, and suitability must refer only to the selected final target.

Target naming and disambiguation policy:
- object_name must refer to ONE selected final target only.
- Use a short singular English noun phrase.
- If multiple objects of the same or similar category are visible, make object_name uniquely identifiable by adding a minimal modifier.
- Prefer the shortest modifier that clearly distinguishes the chosen target.

Modifier priority:
1) visible appearance attributes first, such as color, size, pattern, material, or obvious subtype,
2) relative position second, such as left / center / right / top / bottom,
3) combine appearance + position only when needed for clear disambiguation.

Examples:
- two different cats -> "orange cat", "white cat"
- two similar cats -> "left cat", "right cat"
- three similar mugs -> "left mug", "center mug", "right mug"
- two teddy bears of different size -> "small teddy bear", "large teddy bear"
- several nearly identical bottles -> choose one and name it "center bottle"

Naming rules:
- If one object is clearly dominant and unambiguous, use the simpler name without extra modifiers, e.g. "cat" or "mug".
- Keep object_name short; put fuller details in object_description.
- Do NOT guess brands, unreadable text, or uncertain fine-grained breeds/models.
- Do NOT use long sentence-like names.

Suitability guidance:
- suitable examples: phone, cup, mug, bottle, bag, backpack, book, notebook, plush toy,
  bouquet, umbrella, racket, small appliance, food item, box, small pet, puppy, kitten,
  rabbit, bird, etc.
- not suitable examples: car, sofa, bed, wardrobe, refrigerator, building, room scene,
  landscape, huge machine, large furniture, large architecture, cluttered multi-object scene
  without one clear target.
- pets / small animals can be considered suitable if they are reasonably holdable in arms.

Rules:
1) Select ONE final target object.
2) Name the chosen target with a short singular English noun phrase.
3) If multiple same/similar objects are visible, make object_name uniquely identifiable using a minimal modifier.
4) Prefer visible attributes first; use relative position when needed.
5) If there is no clear single target object, set object_name to null and should_use to false.
6) Record approximate object position with BOTH:
   - a coarse position
   - a normalized bounding box in [0,1]
7) Bounding box should be approximate but reasonable:
   x_min, y_min, x_max, y_max
8) Output ONLY valid JSON. No markdown. No extra text.

Return EXACTLY this JSON schema:
{
  "object_name": str|null,
  "object_category": str|null,
  "object_description": str|null,
  "coarse_position": {
    "horizontal": "left|center|right|full|unknown"|null,
    "vertical": "top|center|bottom|full|unknown"|null
  },
  "bbox_norm": {
    "x_min": float|null,
    "y_min": float|null,
    "x_max": float|null,
    "y_max": float|null
  },
  "holdability": "handheld|carryable|pet_holdable|not_suitable|unclear"|null,
  "suitable_for_holding": true|false,
  "should_use": true|false,
  "confidence": "high|medium|low",
  "reason": str
}
""".strip()

USER_TEXT = "Object image:"

REQUIRED_KEYS = [
    "object_name", "object_category", "object_description", "coarse_position",
    "bbox_norm", "holdability", "suitable_for_holding", "should_use", "confidence", "reason",
]

VALID_HOLDABILITY = {"handheld", "carryable", "pet_holdable", "not_suitable", "unclear"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_HORIZONTAL = {"left", "center", "right", "full", "unknown"}
VALID_VERTICAL = {"top", "center", "bottom", "full", "unknown"}


def strip_think(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    return text.strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    text = strip_think(text)
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"Cannot locate JSON object in model output:\n{s[:4000]}")
    return json.loads(s[l : r + 1])


def normalize_object_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s or s in {"null", "none", "n/a", "unknown"}:
        return None
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?\"'")
    return s or None


def normalize_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s)).strip()
    if not s or s.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    return s


def normalize_bbox(bbox: Any) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {"x_min": None, "y_min": None, "x_max": None, "y_max": None}
    if not isinstance(bbox, dict):
        return out

    def _norm(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            x = float(v)
            if x > 1.0 and x <= 100.0:
                x = x / 100.0
            x = max(0.0, min(1.0, x))
            return round(x, 4)
        except Exception:
            return None

    out["x_min"] = _norm(bbox.get("x_min"))
    out["y_min"] = _norm(bbox.get("y_min"))
    out["x_max"] = _norm(bbox.get("x_max"))
    out["y_max"] = _norm(bbox.get("y_max"))

    if out["x_min"] is not None and out["x_max"] is not None and out["x_min"] > out["x_max"]:
        out["x_min"], out["x_max"] = out["x_max"], out["x_min"]
    if out["y_min"] is not None and out["y_max"] is not None and out["y_min"] > out["y_max"]:
        out["y_min"], out["y_max"] = out["y_max"], out["y_min"]

    return out


def normalize_coarse_position(pos: Any) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {"horizontal": None, "vertical": None}
    if not isinstance(pos, dict):
        return out

    h = normalize_text(pos.get("horizontal"))
    v = normalize_text(pos.get("vertical"))

    if h is not None:
        h = h.lower()
        out["horizontal"] = h if h in VALID_HORIZONTAL else "unknown"
    if v is not None:
        v = v.lower()
        out["vertical"] = v if v in VALID_VERTICAL else "unknown"

    return out


def normalize_annotation(raw: dict[str, Any]) -> dict[str, Any]:
    for k in REQUIRED_KEYS:
        if k not in raw:
            raw[k] = None

    ann: dict[str, Any] = {
        "object_name": normalize_object_name(raw.get("object_name")),
        "object_category": normalize_text(raw.get("object_category")),
        "object_description": normalize_text(raw.get("object_description")),
        "coarse_position": normalize_coarse_position(raw.get("coarse_position")),
        "bbox_norm": normalize_bbox(raw.get("bbox_norm")),
    }

    holdability = normalize_text(raw.get("holdability"))
    ann["holdability"] = holdability.lower() if holdability is not None else None
    if ann["holdability"] is not None and ann["holdability"] not in VALID_HOLDABILITY:
        ann["holdability"] = "unclear"

    ann["suitable_for_holding"] = bool(raw.get("suitable_for_holding", False))
    ann["should_use"] = bool(raw.get("should_use", False))

    confidence = normalize_text(raw.get("confidence"))
    ann["confidence"] = confidence.lower() if confidence is not None else "low"
    if ann["confidence"] not in VALID_CONFIDENCE:
        ann["confidence"] = "low"

    ann["reason"] = normalize_text(raw.get("reason")) or ""

    if ann["object_name"] is None:
        ann["should_use"] = False
        ann["suitable_for_holding"] = False

    return ann


def load_image_rgb(path: str | os.PathLike[str], max_pixels: int = 0) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_pixels > 0:
        width, height = image.size
        pixels = width * height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.BICUBIC,
            )
    return image


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def safe_json_loads(text: str) -> dict[str, Any]:
    return extract_first_json_object(text)


class ObjectHoldableLabeler(LabelOperator):
    name = "object_holdable_labeler"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key_env: str = "API_KEY",
        max_retries: int = 3,
        max_tokens: int = 768,
        temperature: float = 0.0,
        max_pixels: int = 1048576,
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
            labels = normalize_annotation(raw)
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
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": USER_TEXT},
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": OBJECT_PROMPT},
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