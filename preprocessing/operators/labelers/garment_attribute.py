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


SYSTEM_PROMPT = "You are a strict garment-image annotation model. Return valid JSON only."

USER_PROMPT = r"""
Evaluate the given clothing/garment image across 10 dimensions. Follow these rules carefully:

**Rule 1: Gate condition**
First evaluate is_valid_garment_image. If "no", set all remaining fields to null and stop.

**Rule 2: Conditional fields**
If garment_count is NOT "single", set garment_position_type, garment_category, and pattern_type to null.

**Dimensions:**

1. is_valid_garment_image (yes|no)
   - yes: The image's main subject is clothing (may include accessories), and basic form is visible.
   - no: Main subject is not clothing, severely occluded/blurry, non-real clothing (hand-drawn, AI-generated without realism), or content unrelated to clothing.

2. garment_count (single|multiple|set|uncertain)
   - single: Only 1 garment item (exclude bags, shoes, accessories).
   - multiple: 2+ distinct garments visible (stacked, multi-display).
   - set: Upper/lower or inner/outer garments displayed as a coordinated set (suit, athletic set).
   - uncertain: Cannot determine count (severe cropping, angle issues).

3. display_mode (flat|hanging|mannequin|real_person|close_up|other)
   - flat: Clothing laid flat on a surface (floor, table, backdrop paper).
   - hanging: Hung on a hanger or rack or wall.
   - mannequin: Worn on a mannequin/dummy (no real person).
   - real_person: Worn by a real person (model or passerby).
   - close_up: Only showing a partial area (collar, cuff, fabric texture), full form not visible.
   - other: Cannot fit above categories (folded, handheld, etc.).

4. garment_position_type (top|bottom|one_piece|outerwear|accessory|other|uncertain)
   - ONLY evaluated when garment_count = "single", otherwise null.
   - top: Upper-body garments (T-shirt, shirt, sweater, vest, tank top, halter top).
   - bottom: Lower-body garments (pants, skirt, shorts, leggings).
   - one_piece: Connected upper+lower garments (dress, jumpsuit, romper).
   - outerwear: Outer layer garments (coat, jacket, blazer, cardigan, windbreaker).
   - accessory: Non-main clothing items (scarf, hat, belt, gloves). If only accessory with no main garment, classify here.
   - other: Cannot fit above (underwear, swimwear, special functional clothing).
   - uncertain: Cannot determine due to occlusion or angle.

5. garment_category (t_shirt|shirt|hoodie|pants|skirt|coat|dress|other)
   - ONLY evaluated when garment_count = "single", otherwise null.
   - t_shirt: Short/long-sleeve round/V-neck knit T-shirt.
   - shirt: Collared, buttoned shirt (long/short/no sleeve).
   - hoodie: Sport-style pullover or zip-up hoodie/sweatshirt.
   - pants: Long pants, shorts, jeans, dress pants, athletic pants.
   - skirt: Half skirt (not dress).
   - coat: Overcoat, trench coat, leather jacket, denim jacket, bomber jacket.
   - dress: One-piece dress (mini to maxi length).
   - other: Vest, jumpsuit, swimwear, underwear, accessories, etc. If ambiguous, prioritize main visual feature.

6. target_user_group (male|female|children|infant|neutral|uncertain)
   - male: Clearly designed for men (men's shirt, tie, suit pants).
   - female: Clearly designed for women (waist-emphasized, lace, skirt).
   - children: Clearly children's clothing (small size, cartoon patterns, child model).
   - infant: Designed for 0-3 years (diaper openings, onesie, bib).
   - neutral: No obvious gender tendency (basic T-shirt, hoodie, sweatpants).
   - uncertain: Cannot determine (close-up, no size reference).

7. background_type (white|light_solid|complex|transparent|uncertain)
   - white: Pure/near-white uniform background (typical e-commerce white background).
   - light_solid: Light-colored solid background other than white (beige, light gray, light pink).
   - complex: Textured, patterned, multi-object or natural scene background (street, messy interior).
   - transparent: PNG transparent background.
   - uncertain: Cannot determine background type (blurry or heavily cropped).

8. garment_completeness (complete|slightly_cut|severely_cut|partial)
   - complete: Entire garment outline visible, no cropping.
   - slightly_cut: Minor edge cropping (top of collar, bottom hem), overall form recognizable.
   - severely_cut: Over 1/3 of garment area cropped or occluded, overall structure unclear.
   - partial: Only showing local details (corresponds to close_up display mode).

9. pattern_type (solid|striped|plaid|printed|logo|denim|knitted|other)
   - ONLY evaluated when garment_count = "single", otherwise null.
   - solid: No pattern or texture, single color (exclude fabric sheen gradient).
   - striped: Parallel line patterns (any width/direction).
   - plaid: Checkered, chess-board, houndstooth cross-line patterns.
   - printed: Visible printed patterns (floral, letter, geometric, photo print).
   - logo: Brand logo or prominent text as main decoration.
   - denim: Denim-specific washing, whitening, distressed textures.
   - knitted: Knit/ribbed visible weaving texture (not print).
   - other: Tie-dye, gradient, lace, sequin, camouflage, etc. If multiple patterns, pick the most prominent.

10. image_quality (high|acceptable|low|unusable)
    - high: Clear, even lighting, accurate color, no occlusion, no noise. Suitable as reference image.
    - acceptable: Basically clear, slight blur/color shift/lighting bias/noise, still usable as reference.
    - low: Severely blurred, over/underexposed, heavy noise, color distortion, barely distinguishable.
    - unusable: Cannot see clothing details or color, unusable for any reference.

Return EXACTLY this JSON schema:
{
  "is_valid_garment_image": "yes|no",
  "garment_count": "single|multiple|set|uncertain"|null,
  "display_mode": "flat|hanging|mannequin|real_person|close_up|other"|null,
  "garment_position_type": "top|bottom|one_piece|outerwear|accessory|other|uncertain"|null,
  "garment_category": "t_shirt|shirt|hoodie|pants|skirt|coat|dress|other"|null,
  "target_user_group": "male|female|children|infant|neutral|uncertain"|null,
  "background_type": "white|light_solid|complex|transparent|uncertain"|null,
  "garment_completeness": "complete|slightly_cut|severely_cut|partial"|null,
  "pattern_type": "solid|striped|plaid|printed|logo|denim|knitted|other"|null,
  "image_quality": "high|acceptable|low|unusable"|null
}
""".strip()


FIELDS = [
    "is_valid_garment_image",
    "garment_count",
    "display_mode",
    "garment_position_type",
    "garment_category",
    "target_user_group",
    "background_type",
    "garment_completeness",
    "pattern_type",
    "image_quality",
]

CHOICES = {
    "is_valid_garment_image": {"yes", "no"},
    "garment_count": {"single", "multiple", "set", "uncertain"},
    "display_mode": {"flat", "hanging", "mannequin", "real_person", "close_up", "other"},
    "garment_position_type": {"top", "bottom", "one_piece", "outerwear", "accessory", "other", "uncertain"},
    "garment_category": {"t_shirt", "shirt", "hoodie", "pants", "skirt", "coat", "dress", "other"},
    "target_user_group": {"male", "female", "children", "infant", "neutral", "uncertain"},
    "background_type": {"white", "light_solid", "complex", "transparent", "uncertain"},
    "garment_completeness": {"complete", "slightly_cut", "severely_cut", "partial"},
    "pattern_type": {"solid", "striped", "plaid", "printed", "logo", "denim", "knitted", "other"},
    "image_quality": {"high", "acceptable", "low", "unusable"},
}

CONDITIONAL_FIELDS = {"garment_position_type", "garment_category", "pattern_type"}


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = str(value).strip().lower() if value is not None else fallback
    return text if text in allowed else fallback


def normalize_annotation(raw: dict[str, Any], file_name: str) -> dict[str, str]:
    is_valid = _choice(raw.get("is_valid_garment_image"), CHOICES["is_valid_garment_image"], "no")
    ann: dict[str, str] = {"file_name": file_name, "is_valid_garment_image": is_valid}

    if is_valid == "no":
        for field in FIELDS[1:]:
            ann[field] = "N/A"
        return ann

    garment_count = _choice(raw.get("garment_count"), CHOICES["garment_count"], "uncertain")
    ann["garment_count"] = garment_count

    unconditional_fields = ["display_mode", "target_user_group", "background_type", "garment_completeness", "image_quality"]
    for field in unconditional_fields:
        fallback = "uncertain" if "uncertain" in CHOICES[field] else "other"
        ann[field] = _choice(raw.get(field), CHOICES[field], fallback)

    if garment_count == "single":
        for field in CONDITIONAL_FIELDS:
            fallback = "uncertain" if "uncertain" in CHOICES[field] else "other"
            ann[field] = _choice(raw.get(field), CHOICES[field], fallback)
    else:
        for field in CONDITIONAL_FIELDS:
            ann[field] = "N/A"

    return ann


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


class GarmentAttributeLabeler(LabelOperator):
    name = "garment_attribute_labeler"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key_env: str = "API_KEY",
        max_retries: int = 3,
        max_tokens: int = 512,
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
                                {"type": "text", "text": USER_PROMPT},
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