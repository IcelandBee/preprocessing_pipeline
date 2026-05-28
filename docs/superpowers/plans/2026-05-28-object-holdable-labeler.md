# Object Holdable Labeler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 object_holdable.py 参考脚本转换为管线 LabelOperator 算子，统一使用 OpenAI-compatible API 调用 VLM，并创建两个管线配置文件。

**Architecture:** 创建 `ObjectHoldableLabeler(LabelOperator)` 类，完全仿照 `PersonAttributeLabeler` 的架构模式——OpenAI client 远程 API 调用、retry loop、per-sample process()、内联 normalization 函数。10个标注字段完整迁移，使用最终版 OBJECT_PROMPT。

**Tech Stack:** Python 3.10+, OpenAI Python client, PIL/Pillow, pytest

---

## File Structure

| 文件 | 变更类型 | 负责内容 |
|------|---------|---------|
| `preprocessing/operators/labelers/object_holdable.py` | 新建 | ObjectHoldableLabeler 类、OBJECT_PROMPT、所有 normalization 函数、load_image_rgb/pil_to_data_url/strip_think/extract_first_json_object/safe_json_loads |
| `preprocessing/pipeline/registry.py` | 修改 | 注册 `object_holdable_labeler` |
| `configs/object_pipeline.yaml` | 新建 | filter + label 全流程配置 |
| `configs/object_label_only.yaml` | 新建 | 只打标配置 |
| `tests/operators/test_object_holdable_labeler.py` | 新建 | normalization 函数和 normalize_annotation 的单元测试 |

---

### Task 1: normalization 函数与 OBJECT_PROMPT

**Files:**
- Create: `preprocessing/operators/labelers/object_holdable.py`
- Test: `tests/operators/test_object_holdable_labeler.py`

- [ ] **Step 1: Write failing tests for normalization functions**

```python
# tests/operators/test_object_holdable_labeler.py
from preprocessing.operators.labelers.object_holdable import (
    normalize_object_name,
    normalize_text,
    normalize_bbox,
    normalize_coarse_position,
    normalize_annotation,
)


def test_normalize_object_name_valid():
    assert normalize_object_name("Red Cup") == "red cup"


def test_normalize_object_name_null_strings():
    for v in [None, "", "null", "none", "N/A", "unknown"]:
        assert normalize_object_name(v) is None


def test_normalize_object_name_extra_whitespace():
    assert normalize_object_name("  small   teddy   bear  ") == "small teddy bear"


def test_normalize_text_valid():
    assert normalize_text("A handheld device") == "A handheld device"


def test_normalize_text_null_strings():
    for v in [None, "", "null", "none", "N/A", "unknown"]:
        assert normalize_text(v) is None


def test_normalize_bbox_valid():
    result = normalize_bbox({"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9})
    assert result["x_min"] == 0.1
    assert result["y_min"] == 0.2
    assert result["x_max"] == 0.8
    assert result["y_max"] == 0.9


def test_normalize_bbox_clamps_and_scales():
    result = normalize_bbox({"x_min": 10, "y_min": 200, "x_max": 80, "y_max": 90})
    assert result["x_min"] == 0.1
    assert result["y_min"] is None  # 200 > 100, not in [0,1] and not in 0-100%
    assert result["x_max"] == 0.8
    assert result["y_max"] == 0.9


def test_normalize_bbox_swaps_min_max():
    result = normalize_bbox({"x_min": 0.8, "y_min": 0.9, "x_max": 0.1, "y_max": 0.2})
    assert result["x_min"] == 0.1
    assert result["x_max"] == 0.8


def test_normalize_bbox_non_dict():
    result = normalize_bbox("not a dict")
    assert result == {"x_min": None, "y_min": None, "x_max": None, "y_max": None}


def test_normalize_coarse_position_valid():
    result = normalize_coarse_position({"horizontal": "left", "vertical": "top"})
    assert result["horizontal"] == "left"
    assert result["vertical"] == "top"


def test_normalize_coarse_position_invalid_values():
    result = normalize_coarse_position({"horizontal": "diagonal", "vertical": "middle"})
    assert result["horizontal"] == "unknown"
    assert result["vertical"] == "unknown"


def test_normalize_coarse_position_non_dict():
    result = normalize_coarse_position(None)
    assert result == {"horizontal": None, "vertical": None}


def test_normalize_annotation_complete():
    raw = {
        "object_name": "mug",
        "object_category": "container",
        "object_description": "a white ceramic mug",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.7, "y_max": 0.8},
        "holdability": "handheld",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "high",
        "reason": "small handheld object",
    }
    ann = normalize_annotation(raw)
    assert ann["object_name"] == "mug"
    assert ann["object_category"] == "container"
    assert ann["holdability"] == "handheld"
    assert ann["suitable_for_holding"] is True
    assert ann["should_use"] is True
    assert ann["confidence"] == "high"


def test_normalize_annotation_null_object_name_forces_false():
    raw = {
        "object_name": None,
        "object_category": "furniture",
        "object_description": "a large sofa",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
        "holdability": "not_suitable",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "low",
        "reason": "too large",
    }
    ann = normalize_annotation(raw)
    assert ann["object_name"] is None
    assert ann["should_use"] is False
    assert ann["suitable_for_holding"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/operators/test_object_holdable_labeler.py -v`
Expected: FAIL — module `preprocessing.operators.labelers.object_holdable` not found

- [ ] **Step 3: Write normalization functions and OBJECT_PROMPT**

Create `preprocessing/operators/labelers/object_holdable.py` with the following content:

```python
from __future__ import annotations

import re
from typing import Any, Optional


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


REQUIRED_KEYS = [
    "object_name", "object_category", "object_description", "coarse_position",
    "bbox_norm", "holdability", "suitable_for_holding", "should_use", "confidence", "reason",
]

VALID_HOLDABILITY = {"handheld", "carryable", "pet_holdable", "not_suitable", "unclear"}
VALID_CONFIDENCE = {"high", "medium", "low"}

VALID_HORIZONTAL = {"left", "center", "right", "full", "unknown"}
VALID_VERTICAL = {"top", "center", "bottom", "full", "unknown"}


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


def strip_think(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r" нанес.*?</think>\s*", "", text, flags=re.S)
    return text.strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    import json
    text = strip_think(text)
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"Cannot locate JSON object in model output:\n{s[:4000]}")
    return json.loads(s[l : r + 1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/operators/test_object_holdable_labeler.py -v`
Expected: PASS — all normalization tests pass

- [ ] **Step 5: Commit**

```bash
git add preprocessing/operators/labelers/object_holdable.py tests/operators/test_object_holdable_labeler.py
git commit -m "feat: add ObjectHoldable normalization functions and OBJECT_PROMPT"
```

---

### Task 2: ObjectHoldableLabeler 类（setup/process/_infer）

**Files:**
- Modify: `preprocessing/operators/labelers/object_holdable.py` — 添加类、load_image_rgb、pil_to_data_url
- Test: `tests/operators/test_object_holdable_labeler.py` — 添加类的集成测试

- [ ] **Step 1: Write failing test for ObjectHoldableLabeler.process() with mock API**

```python
# Append to tests/operators/test_object_holdable_labeler.py
import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from preprocessing.operators.labelers.object_holdable import ObjectHoldableLabeler
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import OperatorResult, Sample


def _make_sample(tmp_path: Path, name: str = "test_obj.jpg") -> tuple[Sample, Path]:
    img_path = tmp_path / name
    Image.new("RGB", (100, 100), (0, 128, 255)).save(img_path)
    return Sample(
        sample_id=name,
        source_path=img_path,
        relative_path=Path(name),
    ), img_path


def test_process_returns_label_on_success(tmp_path):
    sample, _ = _make_sample(tmp_path)
    context = PipelineContext(
        run_id="test-run",
        run_dir=tmp_path / "run",
        pass_archive_dir=tmp_path / "archive",
        pass_archive_layout="run_subdir",
    )

    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
        api_key_env="API_KEY",
        max_retries=1,
        max_tokens=768,
    )
    labeler.client = MagicMock()

    fake_response = MagicMock()
    raw_annotation = json.dumps({
        "object_name": "cup",
        "object_category": "container",
        "object_description": "a white ceramic cup",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.7, "y_max": 0.8},
        "holdability": "handheld",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "high",
        "reason": "small handheld object",
    })
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = raw_annotation
    labeler.client.chat.completions.create.return_value = fake_response

    result = labeler.process(sample, context)
    assert result.decision == "LABEL"
    assert result.labels["object_name"] == "cup"
    assert result.labels["holdability"] == "handheld"
    assert result.labels["should_use"] is True


def test_process_returns_error_when_no_client(tmp_path):
    sample, _ = _make_sample(tmp_path)
    context = PipelineContext(
        run_id="test-run",
        run_dir=tmp_path / "run",
        pass_archive_dir=tmp_path / "archive",
        pass_archive_layout="run_subdir",
    )
    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
    )
    labeler.client = None
    result = labeler.process(sample, context)
    assert result.decision == "ERROR"


def test_setup_creates_client():
    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
        api_key_env="API_KEY",
    )
    context = PipelineContext(
        run_id="test",
        run_dir=Path("/tmp/run"),
        pass_archive_dir=Path("/tmp/archive"),
        pass_archive_layout="run_subdir",
    )
    with patch.dict("os.environ", {"API_KEY": "test-key"}):
        labeler.setup(context)
    assert labeler.client is not None


def test_infer_retry_on_failure(tmp_path):
    sample, _ = _make_sample(tmp_path)
    context = PipelineContext(
        run_id="test-run",
        run_dir=tmp_path / "run",
        pass_archive_dir=tmp_path / "archive",
        pass_archive_layout="run_subdir",
    )

    labeler = ObjectHoldableLabeler(
        base_url="http://fake:8001/v1",
        model_name="gemma-4-31B-it",
        max_retries=3,
    )
    labeler.client = MagicMock()

    # First two calls fail, third succeeds
    good_raw = json.dumps({
        "object_name": "phone",
        "object_category": "electronics",
        "object_description": "a smartphone",
        "coarse_position": {"horizontal": "center", "vertical": "center"},
        "bbox_norm": {"x_min": 0.3, "y_min": 0.4, "x_max": 0.7, "y_max": 0.8},
        "holdability": "handheld",
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "high",
        "reason": "small handheld device",
    })
    good_response = MagicMock()
    good_response.choices = [MagicMock()]
    good_response.choices[0].message.content = good_raw

    labeler.client.chat.completions.create.side_effect = [
        RuntimeError("timeout"),
        RuntimeError("timeout"),
        good_response,
    ]

    image = Image.new("RGB", (10, 10))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    with patch("time.sleep"):
        raw = labeler._infer(data_url)
    assert raw["object_name"] == "phone"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/operators/test_object_holdable_labeler.py::test_process_returns_label_on_success -v`
Expected: FAIL — `ObjectHoldableLabeler` class not defined

- [ ] **Step 3: Write ObjectHoldableLabeler class, load_image_rgb, pil_to_data_url, safe_json_loads**

Append to `preprocessing/operators/labelers/object_holdable.py`:

```python
import base64
import io
import json
import os
import time

from openai import OpenAI
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import LabelOperator
from preprocessing.pipeline.types import OperatorResult, Sample


USER_TEXT = "Object image:"


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/operators/test_object_holdable_labeler.py -v`
Expected: PASS — all normalization + class tests pass

- [ ] **Step 5: Commit**

```bash
git add preprocessing/operators/labelers/object_holdable.py tests/operators/test_object_holdable_labeler.py
git commit -m "feat: add ObjectHoldableLabeler class with setup/process/_infer"
```

---

### Task 3: Registry 注册

**Files:**
- Modify: `preprocessing/pipeline/registry.py:25-44`

- [ ] **Step 1: Add import and register line**

In `preprocessing/pipeline/registry.py`, add after the `PersonAttributeLabeler` import (line 33):

```python
from preprocessing.operators.labelers.object_holdable import ObjectHoldableLabeler
```

And add after the `person_attribute_labeler` registration (line 43):

```python
registry.register("object_holdable_labeler", ObjectHoldableLabeler)
```

- [ ] **Step 2: Verify registry works**

Run: `pytest tests/pipeline/test_registry.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add preprocessing/pipeline/registry.py
git commit -m "feat: register object_holdable_labeler in default registry"
```

---

### Task 4: 管线配置文件

**Files:**
- Create: `configs/object_pipeline.yaml`
- Create: `configs/object_label_only.yaml`

- [ ] **Step 1: Create object_pipeline.yaml (filter + label)**

```yaml
input:
  input_dir: /DATA/raw/object_images
  recursive: true
  image_exts: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

output:
  run_root: outputs/runs
  pass_archive_dir: /DATA/clean/object_dataset_v1
  pass_archive_layout: run_subdir
  overwrite: false

filter:
  workers: auto
  short_circuit: true
  operators:
    - name: aspect_ratio
      enabled: true
      params:
        ratio: 2.0
    - name: gray_style
      enabled: true
      params:
        threshold: 1.0
    - name: compression_quality
      enabled: true
      params:
        min_quality: 30
    - name: duplicate
      enabled: true
      params:
        threshold: 9
        hash_size: 8
        window_size: 50

label:
  enabled: true
  input_dir: auto
  annotation_dir: auto
  jsonl_path: auto
  operator:
    name: object_holdable_labeler
    params:
      base_url: http://10.154.39.57:8001/v1
      api_key_env: API_KEY
      model_name: gemma-4-31B-it
      max_retries: 3
      max_tokens: 768
      temperature: 0.0
      max_pixels: 1048576
      skip_existing: true
```

- [ ] **Step 2: Create object_label_only.yaml (只打标)**

```yaml
input:
  input_dir: /DATA/clean/object_images
  recursive: true
  image_exts: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

output:
  run_root: outputs/runs
  pass_archive_dir: /DATA/clean/object_labeled_v1
  pass_archive_layout: run_subdir
  overwrite: false

filter:
  workers: 1
  short_circuit: true
  operators: []

label:
  enabled: true
  input_dir: /DATA/clean/object_images
  annotation_dir: auto
  jsonl_path: auto
  operator:
    name: object_holdable_labeler
    params:
      base_url: http://10.154.39.57:8001/v1
      api_key_env: API_KEY
      model_name: gemma-4-31B-it
      max_retries: 3
      max_tokens: 768
      temperature: 0.0
      max_pixels: 1048576
      skip_existing: true
```

- [ ] **Step 3: Verify YAML loads correctly**

Run: `python -c "from preprocessing.pipeline.config import load_config; c = load_config('configs/object_pipeline.yaml'); print(c.label.operator.name)"` (需要在有 Python 的环境中执行)
Expected: 输出 `object_holdable_labeler`

- [ ] **Step 4: Commit**

```bash
git add configs/object_pipeline.yaml configs/object_label_only.yaml
git commit -m "feat: add object_pipeline and object_label_only YAML configs"
```

---

## Self-Review

**1. Spec coverage:** All sections covered:
- Section 1 (类结构与构造参数) → Task 2
- Section 2 (Prompt 与消息结构) → Task 1 (OBJECT_PROMPT) + Task 2 (_infer 消息结构)
- Section 3 (Normalization 与标注输出) → Task 1 (所有 normalization 函数) + Task 2 (normalize_annotation 在 process 中调用)
- Section 4 (YAML 配置与注册) → Task 3 + Task 4

**2. Placeholder scan:** No TBD/TODO/placeholder patterns found. All code blocks are complete.

**3. Type consistency:** All function signatures consistent across tasks:
- `normalize_annotation(raw: dict[str, Any])` → same signature in Task 1 definition and Task 2 usage
- `_infer(image_url: str)` → called in `process()` with `str` return from `pil_to_data_url()`
- Registry name `"object_holdable_labeler"` matches class `name` attribute and YAML `operator.name`