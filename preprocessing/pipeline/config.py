from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    input_dir: Path
    recursive: bool = True
    image_exts: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"])


@dataclass
class OutputConfig:
    run_root: Path
    pass_archive_dir: Path
    pass_archive_layout: str = "run_subdir"
    overwrite: bool = False
    run_id_prefix: str = "auto"


@dataclass
class OperatorConfig:
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterConfig:
    short_circuit: bool = True
    workers: str | int = "auto"
    operators: list[OperatorConfig] = field(default_factory=list)

    @property
    def enabled_operators(self) -> list[OperatorConfig]:
        return [operator for operator in self.operators if operator.enabled]


@dataclass
class LabelConfig:
    enabled: bool = True
    input_dir: str | Path = "auto"
    annotation_dir: str | Path = "auto"
    jsonl_path: str | Path = "auto"
    operator: OperatorConfig = field(default_factory=lambda: OperatorConfig(name="person_attribute_labeler"))


@dataclass
class PipelineConfig:
    input: InputConfig
    output: OutputConfig
    filter: FilterConfig
    label: LabelConfig


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return parse_pipeline_config(data)


def parse_pipeline_config(data: dict[str, Any]) -> PipelineConfig:
    input_data = data.get("input")
    if not isinstance(input_data, dict):
        raise ValueError("Missing required config field: input.input_dir")
    output_data = _require_mapping(data, "output")
    filter_data = data.get("filter") or {}
    label_data = data.get("label") or {}

    if "input_dir" not in input_data:
        raise ValueError("Missing required config field: input.input_dir")
    if "run_root" not in output_data:
        raise ValueError("Missing required config field: output.run_root")
    if "pass_archive_dir" not in output_data:
        raise ValueError("Missing required config field: output.pass_archive_dir")

    input_config = InputConfig(
        input_dir=Path(input_data["input_dir"]),
        recursive=bool(input_data.get("recursive", True)),
        image_exts=[str(ext).lower() for ext in input_data.get("image_exts", InputConfig(Path(".")).image_exts)],
    )
    output_config = OutputConfig(
        run_root=Path(output_data["run_root"]),
        pass_archive_dir=Path(output_data["pass_archive_dir"]),
        pass_archive_layout=str(output_data.get("pass_archive_layout", "run_subdir")),
        overwrite=bool(output_data.get("overwrite", False)),
        run_id_prefix=str(output_data.get("run_id_prefix", "auto")),
    )
    if output_config.pass_archive_layout not in {"run_subdir", "flat", "preserve_relative"}:
        raise ValueError("output.pass_archive_layout must be one of: run_subdir, flat, preserve_relative")

    filter_config = FilterConfig(
        short_circuit=bool(filter_data.get("short_circuit", True)),
        workers=filter_data.get("workers", "auto"),
        operators=[_parse_operator_config(item) for item in filter_data.get("operators", [])],
    )

    label_operator = _parse_operator_config(label_data.get("operator", {"name": "person_attribute_labeler"}))
    label_config = LabelConfig(
        enabled=bool(label_data.get("enabled", True)),
        input_dir=label_data.get("input_dir", "auto"),
        annotation_dir=label_data.get("annotation_dir", "auto"),
        jsonl_path=label_data.get("jsonl_path", "auto"),
        operator=label_operator,
    )

    return PipelineConfig(input=input_config, output=output_config, filter=filter_config, label=label_config)


def _parse_operator_config(data: dict[str, Any]) -> OperatorConfig:
    if "name" not in data:
        raise ValueError("Operator config is missing required field: name")
    return OperatorConfig(
        name=str(data["name"]),
        enabled=bool(data.get("enabled", True)),
        params=dict(data.get("params") or {}),
    )


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required config section: {key}")
    return value
