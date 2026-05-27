from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from preprocessing.pipeline.types import Sample


def enumerate_images(input_dir: Path, recursive: bool, image_exts: list[str]) -> list[Sample]:
    root = Path(input_dir)
    exts = {ext.lower() for ext in image_exts}
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in exts
    )
    return [
        Sample(
            sample_id=path.relative_to(root).as_posix(),
            source_path=path,
            relative_path=path.relative_to(root),
        )
        for path in paths
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_sample_to_archive(sample: Sample, archive_path: Path, overwrite: bool) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"Archive path already exists: {archive_path}")
    shutil.copy2(sample.source_path, archive_path)
    return archive_path


def ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
