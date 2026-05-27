import pytest

from preprocessing.pipeline.io import (
    copy_sample_to_archive,
    enumerate_images,
    read_jsonl,
    write_jsonl,
)
from preprocessing.pipeline.types import Sample


def test_enumerate_images_recursive(tmp_path, make_image):
    make_image(tmp_path / "a.jpg")
    make_image(tmp_path / "nested" / "b.png")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    samples = enumerate_images(tmp_path, recursive=True, image_exts=[".jpg", ".png"])

    assert [sample.relative_path.as_posix() for sample in samples] == ["a.jpg", "nested/b.png"]


def test_write_and_read_jsonl(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = [{"sample_id": "a.jpg"}, {"sample_id": "b.jpg"}]

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_copy_sample_to_archive_preserves_relative_path(tmp_path, make_image):
    source = make_image(tmp_path / "raw" / "nested" / "a.jpg")
    sample = Sample(sample_id="nested/a.jpg", source_path=source, relative_path=source.relative_to(tmp_path / "raw"))
    archive_path = tmp_path / "clean" / "run1" / "nested" / "a.jpg"

    copied = copy_sample_to_archive(sample, archive_path, overwrite=False)

    assert copied == archive_path
    assert copied.exists()
    assert source.exists()


def test_copy_sample_to_archive_rejects_conflict(tmp_path, make_image):
    source = make_image(tmp_path / "raw" / "a.jpg")
    sample = Sample(sample_id="a.jpg", source_path=source, relative_path=source.name)
    archive_path = make_image(tmp_path / "clean" / "a.jpg", color=(0, 255, 0))

    with pytest.raises(FileExistsError):
        copy_sample_to_archive(sample, archive_path, overwrite=False)
