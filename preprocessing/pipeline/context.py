from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineContext:
    run_id: str
    run_dir: Path
    pass_archive_dir: Path
    pass_archive_layout: str = "run_subdir"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def config_snapshot_path(self) -> Path:
        return self.run_dir / "config.snapshot.yaml"

    @property
    def annotation_dir(self) -> Path:
        return self.run_dir / "annotations"

    @property
    def labels_jsonl_path(self) -> Path:
        return self.run_dir / "labels.jsonl"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    def archive_path_for(self, relative_path: Path) -> Path:
        if self.pass_archive_layout == "run_subdir":
            return self.pass_archive_dir / self.run_id / relative_path
        if self.pass_archive_layout == "flat":
            return self.pass_archive_dir / relative_path.name
        if self.pass_archive_layout == "preserve_relative":
            return self.pass_archive_dir / relative_path
        raise ValueError(f"Unsupported pass_archive_layout: {self.pass_archive_layout}")
