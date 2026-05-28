from __future__ import annotations

from PIL import Image
from preprocessing.pipeline.types import Sample


def get_pil_image(sample: Sample, context: object) -> Image.Image:
    """从 context 获取共享的 PIL Image，如果没有则自行打开文件。"""
    pil = getattr(context, "pil_image", None)
    if pil is not None:
        return pil
    return Image.open(sample.source_path)