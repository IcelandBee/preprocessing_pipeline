from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_image():
    def _make_image(path: Path, size: tuple[int, int] = (20, 20), color: tuple[int, int, int] = (255, 0, 0)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    return _make_image
