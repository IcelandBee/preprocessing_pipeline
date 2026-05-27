# Preprocessing Labeling Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pluggable image quality filtering and labeling pipeline that copies PASS images to a configured archive directory, records rejected samples only in manifests, and supports `filter`, `label`, and `all` stages.

**Architecture:** Implement a small Python package with typed pipeline data objects, YAML config loading, an operator registry, a linear runner, and focused filter/label operators. The runner owns orchestration, archiving, manifest/summary writing, and stage selection; operators only return decisions, metrics, and labels.

**Tech Stack:** Python 3.10+, pytest, PyYAML, Pillow, NumPy, OpenCV, imagehash, tqdm, OpenAI Python client.

---

## File Structure

Create these files:

- `pyproject.toml`: package metadata, dependencies, pytest config, console entry point.
- `README.md`: minimal usage notes and command examples.
- `configs/person_pipeline.yaml`: example YAML configuration.
- `preprocessing/__init__.py`: package marker.
- `preprocessing/pipeline/__init__.py`: pipeline package exports.
- `preprocessing/pipeline/types.py`: `Sample`, `OperatorResult`, `Decision`, trace and manifest dataclasses.
- `preprocessing/pipeline/context.py`: `PipelineContext` and run path helpers.
- `preprocessing/pipeline/operators.py`: base classes for filter, batch filter, and label operators.
- `preprocessing/pipeline/config.py`: config dataclasses and YAML loader/validator.
- `preprocessing/pipeline/io.py`: image enumeration, JSONL writing, config snapshot, copy helpers.
- `preprocessing/pipeline/registry.py`: operator registry.
- `preprocessing/pipeline/runner.py`: filter, label, and all stage orchestration.
- `preprocessing/pipeline/cli.py`: command-line interface.
- `preprocessing/operators/__init__.py`: operator package marker.
- `preprocessing/operators/filters/__init__.py`: filter exports.
- `preprocessing/operators/filters/aspect_ratio.py`: aspect ratio filter.
- `preprocessing/operators/filters/gray_style.py`: grayscale/style filter.
- `preprocessing/operators/filters/compression.py`: PNG/compression quality filter.
- `preprocessing/operators/filters/framebox.py`: border/frame filter.
- `preprocessing/operators/filters/stitch.py`: stitch-line filter.
- `preprocessing/operators/filters/blur.py`: person/face ROI blur filter.
- `preprocessing/operators/filters/duplicate.py`: pHash batch duplicate filter.
- `preprocessing/operators/labelers/__init__.py`: labeler exports.
- `preprocessing/operators/labelers/person_attribute.py`: OpenAI-compatible person attribute labeler and normalization helpers.

Create these tests:

- `tests/conftest.py`: fixture helpers for synthetic images and temp configs.
- `tests/pipeline/test_config.py`: config loading and validation.
- `tests/pipeline/test_registry.py`: registry behavior.
- `tests/pipeline/test_runner_filter.py`: short-circuit, manifest, pass archive, summaries.
- `tests/pipeline/test_runner_label.py`: label stage behavior with a fake labeler.
- `tests/operators/test_basic_filters.py`: aspect ratio and gray style smoke tests.
- `tests/operators/test_duplicate.py`: duplicate batch filter smoke test.
- `tests/operators/test_person_attribute_labeler.py`: label normalization and fake client tests.

Do not copy the old `components/` scripts into this repository. Reuse their logic by implementing focused operator classes.

---

### Task 1: Project Scaffold And Core Types

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `preprocessing/__init__.py`
- Create: `preprocessing/pipeline/__init__.py`
- Create: `preprocessing/pipeline/types.py`
- Create: `preprocessing/pipeline/context.py`
- Create: `preprocessing/pipeline/operators.py`
- Test: `tests/pipeline/test_types.py`

- [ ] **Step 1: Write the failing core type tests**

Create `tests/pipeline/test_types.py`:

```python
from pathlib import Path

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import OperatorResult, Sample


def test_sample_defaults_metadata_and_archive_path():
    sample = Sample(
        sample_id="a.jpg",
        source_path=Path("/data/raw/a.jpg"),
        relative_path=Path("a.jpg"),
    )

    assert sample.archive_path is None
    assert sample.metadata == {}


def test_operator_result_pass_factory():
    result = OperatorResult.pass_(metrics={"ratio": 1.25})

    assert result.decision == "PASS"
    assert result.reason is None
    assert result.metrics == {"ratio": 1.25}
    assert result.labels is None


def test_operator_result_reject_factory():
    result = OperatorResult.reject(
        reason="too_wide",
        metrics={"ratio": 3.0, "threshold": 2.0},
    )

    assert result.decision == "REJECT"
    assert result.reason == "too_wide"
    assert result.metrics["ratio"] == 3.0


def test_pipeline_context_paths(tmp_path):
    context = PipelineContext(
        run_id="20260527-203012",
        run_dir=tmp_path / "runs" / "20260527-203012",
        pass_archive_dir=tmp_path / "clean",
        pass_archive_layout="run_subdir",
    )

    assert context.manifest_path == tmp_path / "runs" / "20260527-203012" / "manifest.jsonl"
    assert context.summary_path == tmp_path / "runs" / "20260527-203012" / "summary.json"
    assert context.annotation_dir == tmp_path / "runs" / "20260527-203012" / "annotations"
    assert context.labels_jsonl_path == tmp_path / "runs" / "20260527-203012" / "labels.jsonl"
    assert context.archive_path_for(Path("nested/a.jpg")) == tmp_path / "clean" / "20260527-203012" / "nested" / "a.jpg"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/pipeline/test_types.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'preprocessing'`.

- [ ] **Step 3: Add package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "preprocessing-pipeline"
version = "0.1.0"
description = "Pluggable image preprocessing and labeling pipeline"
requires-python = ">=3.10"
dependencies = [
  "PyYAML>=6.0",
  "Pillow>=10.0",
  "numpy>=1.24",
  "opencv-python>=4.8",
  "ImageHash>=4.3",
  "tqdm>=4.66",
  "openai>=1.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
]

[project.scripts]
preprocessing-pipeline = "preprocessing.pipeline.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create `README.md`:

```markdown
# preprocessing-pipeline

Pluggable image preprocessing and labeling pipeline.

```bash
python -m preprocessing.pipeline.cli run --stage filter --config configs/person_pipeline.yaml
python -m preprocessing.pipeline.cli run --stage label --config configs/person_pipeline.yaml --run-id 20260527-203012
python -m preprocessing.pipeline.cli run --stage all --config configs/person_pipeline.yaml
```
```

- [ ] **Step 4: Add core type implementation**

Create `preprocessing/__init__.py`:

```python
"""Image preprocessing and labeling pipeline."""
```

Create `preprocessing/pipeline/__init__.py`:

```python
"""Pipeline core package."""
```

Create `preprocessing/pipeline/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Decision = Literal["PASS", "REJECT", "LABEL", "ERROR"]


@dataclass
class Sample:
    sample_id: str
    source_path: Path
    relative_path: Path
    archive_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorResult:
    decision: Decision
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] | None = None

    @classmethod
    def pass_(cls, metrics: dict[str, Any] | None = None) -> "OperatorResult":
        return cls(decision="PASS", metrics=metrics or {})

    @classmethod
    def reject(
        cls,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> "OperatorResult":
        return cls(decision="REJECT", reason=reason, metrics=metrics or {})

    @classmethod
    def label(cls, labels: dict[str, Any], metrics: dict[str, Any] | None = None) -> "OperatorResult":
        return cls(decision="LABEL", labels=labels, metrics=metrics or {})

    @classmethod
    def error(cls, reason: str, metrics: dict[str, Any] | None = None) -> "OperatorResult":
        return cls(decision="ERROR", reason=reason, metrics=metrics or {})
```

Create `preprocessing/pipeline/context.py`:

```python
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
```

Create `preprocessing/pipeline/operators.py`:

```python
from __future__ import annotations

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import OperatorResult, Sample


class FilterOperator:
    name: str

    def setup(self, context: PipelineContext) -> None:
        pass

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        raise NotImplementedError

    def teardown(self, context: PipelineContext) -> None:
        pass


class BatchFilterOperator(FilterOperator):
    def process_batch(
        self,
        samples: list[Sample],
        context: PipelineContext,
    ) -> dict[str, OperatorResult]:
        raise NotImplementedError


class LabelOperator:
    name: str

    def setup(self, context: PipelineContext) -> None:
        pass

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        raise NotImplementedError

    def teardown(self, context: PipelineContext) -> None:
        pass

```

- [ ] **Step 5: Run the test to verify it passes**

Run:

```bash
pytest tests/pipeline/test_types.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md preprocessing tests/pipeline/test_types.py
git commit -m "feat: add pipeline core types"
```

---

### Task 2: Config Loader And Registry

**Files:**
- Create: `preprocessing/pipeline/config.py`
- Create: `preprocessing/pipeline/registry.py`
- Create: `configs/person_pipeline.yaml`
- Test: `tests/pipeline/test_config.py`
- Test: `tests/pipeline/test_registry.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/pipeline/test_config.py`:

```python
import pytest

from preprocessing.pipeline.config import load_pipeline_config


def test_load_pipeline_config(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
input:
  input_dir: /data/raw
  recursive: true
  image_exts: [".jpg", ".png"]
output:
  run_root: outputs/runs
  pass_archive_dir: /data/clean
  pass_archive_layout: run_subdir
  overwrite: false
filter:
  short_circuit: true
  operators:
    - name: aspect_ratio
      enabled: true
      params:
        ratio: 2.0
    - name: gray_style
      enabled: false
      params:
        threshold: 1.0
label:
  enabled: true
  input_dir: auto
  annotation_dir: auto
  jsonl_path: auto
  operator:
    name: person_attribute_labeler
    params:
      model_name: test-model
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert str(config.input.input_dir) == "/data/raw"
    assert config.input.recursive is True
    assert config.input.image_exts == [".jpg", ".png"]
    assert str(config.output.pass_archive_dir) == "/data/clean"
    assert config.filter.operators[0].name == "aspect_ratio"
    assert config.filter.enabled_operators[0].name == "aspect_ratio"
    assert config.label.operator.name == "person_attribute_labeler"


def test_load_pipeline_config_rejects_missing_input(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
output:
  run_root: outputs/runs
  pass_archive_dir: /data/clean
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input.input_dir"):
        load_pipeline_config(config_path)
```

- [ ] **Step 2: Write failing registry tests**

Create `tests/pipeline/test_registry.py`:

```python
import pytest

from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.registry import OperatorRegistry


class DummyFilter(FilterOperator):
    name = "dummy"


def test_registry_creates_operator_by_name():
    registry = OperatorRegistry()
    registry.register("dummy", DummyFilter)

    operator = registry.create("dummy", {})

    assert isinstance(operator, DummyFilter)


def test_registry_rejects_unknown_operator():
    registry = OperatorRegistry()

    with pytest.raises(KeyError, match="unknown"):
        registry.create("unknown", {})
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/pipeline/test_config.py tests/pipeline/test_registry.py -v
```

Expected: FAIL because `config.py` and `registry.py` do not exist.

- [ ] **Step 4: Implement config loader**

Create `preprocessing/pipeline/config.py`:

```python
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


@dataclass
class OperatorConfig:
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterConfig:
    short_circuit: bool = True
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
    input_data = _require_mapping(data, "input")
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
    )
    if output_config.pass_archive_layout not in {"run_subdir", "flat", "preserve_relative"}:
        raise ValueError("output.pass_archive_layout must be one of: run_subdir, flat, preserve_relative")

    filter_config = FilterConfig(
        short_circuit=bool(filter_data.get("short_circuit", True)),
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
```

- [ ] **Step 5: Implement registry**

Create `preprocessing/pipeline/registry.py`:

```python
from __future__ import annotations

from preprocessing.pipeline.operators import BatchFilterOperator, FilterOperator, LabelOperator

OperatorInstance = FilterOperator | BatchFilterOperator | LabelOperator
OperatorClass = type[FilterOperator] | type[BatchFilterOperator] | type[LabelOperator]


class OperatorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, OperatorClass] = {}

    def register(self, name: str, factory: OperatorClass) -> None:
        self._factories[name] = factory

    def create(self, name: str, params: dict[str, object]) -> OperatorInstance:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown operator: {name}. Available operators: {available}") from exc
        return factory(**params)


DEFAULT_REGISTRY = OperatorRegistry()
```

- [ ] **Step 6: Add example config**

Create `configs/person_pipeline.yaml`:

```yaml
input:
  input_dir: /DATA/raw/person_images
  recursive: true
  image_exts: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

output:
  run_root: outputs/runs
  pass_archive_dir: /DATA/clean/person_dataset_v1
  pass_archive_layout: run_subdir
  overwrite: false

filter:
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
    - name: framebox
      enabled: true
      params:
        max_side: 1024
        sides: 3
        min_thick: 20
        max_ratio: 0.30
        min_std: 15.0
    - name: duplicate
      enabled: true
      params:
        threshold: 9
        hash_size: 8

label:
  enabled: true
  input_dir: auto
  annotation_dir: auto
  jsonl_path: auto
  operator:
    name: person_attribute_labeler
    params:
      base_url: http://10.154.39.71:8001/v1
      api_key_env: API_KEY
      model_name: Qwen3.5-27B
      workers: 8
      max_retries: 3
      max_tokens: 224
      temperature: 0.0
      max_pixels: 589824
      skip_existing: true
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/pipeline/test_config.py tests/pipeline/test_registry.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add preprocessing/pipeline/config.py preprocessing/pipeline/registry.py configs/person_pipeline.yaml tests/pipeline/test_config.py tests/pipeline/test_registry.py
git commit -m "feat: add config loader and operator registry"
```

---

### Task 3: Pipeline IO Helpers

**Files:**
- Create: `preprocessing/pipeline/io.py`
- Test: `tests/conftest.py`
- Test: `tests/pipeline/test_io.py`

- [ ] **Step 1: Write fixture helpers**

Create `tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Write failing IO tests**

Create `tests/pipeline/test_io.py`:

```python
import json

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
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/pipeline/test_io.py -v
```

Expected: FAIL because `preprocessing.pipeline.io` does not exist.

- [ ] **Step 4: Implement IO helpers**

Create `preprocessing/pipeline/io.py`:

```python
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
```

- [ ] **Step 5: Run IO tests**

Run:

```bash
pytest tests/pipeline/test_io.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add preprocessing/pipeline/io.py tests/conftest.py tests/pipeline/test_io.py
git commit -m "feat: add pipeline IO helpers"
```

---

### Task 4: Basic Filter Operators

**Files:**
- Create: `preprocessing/operators/__init__.py`
- Create: `preprocessing/operators/filters/__init__.py`
- Create: `preprocessing/operators/filters/aspect_ratio.py`
- Create: `preprocessing/operators/filters/gray_style.py`
- Create: `preprocessing/operators/filters/compression.py`
- Test: `tests/operators/test_basic_filters.py`

- [ ] **Step 1: Write failing filter tests**

Create `tests/operators/test_basic_filters.py`:

```python
from preprocessing.operators.filters.aspect_ratio import AspectRatioFilter
from preprocessing.operators.filters.compression import CompressionQualityFilter
from preprocessing.operators.filters.gray_style import GrayStyleFilter
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import Sample


def make_context(tmp_path):
    return PipelineContext(
        run_id="run1",
        run_dir=tmp_path / "runs" / "run1",
        pass_archive_dir=tmp_path / "clean",
    )


def make_sample(path, root):
    return Sample(sample_id=path.name, source_path=path, relative_path=path.relative_to(root))


def test_aspect_ratio_filter_rejects_extreme_ratio(tmp_path, make_image):
    image = make_image(tmp_path / "wide.jpg", size=(300, 50))
    sample = make_sample(image, tmp_path)
    operator = AspectRatioFilter(ratio=2.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "REJECT"
    assert result.reason == "aspect_ratio_exceeds_threshold"
    assert result.metrics["ratio"] == 6.0


def test_aspect_ratio_filter_passes_normal_ratio(tmp_path, make_image):
    image = make_image(tmp_path / "normal.jpg", size=(100, 80))
    sample = make_sample(image, tmp_path)
    operator = AspectRatioFilter(ratio=2.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "PASS"
    assert result.metrics["ratio"] == 1.25


def test_gray_style_filter_rejects_grayscale(tmp_path, make_image):
    image = make_image(tmp_path / "gray.jpg", color=(120, 120, 120))
    sample = make_sample(image, tmp_path)
    operator = GrayStyleFilter(threshold=1.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "REJECT"
    assert result.reason == "grayscale_image"


def test_gray_style_filter_passes_color(tmp_path, make_image):
    image = make_image(tmp_path / "color.jpg", color=(255, 0, 0))
    sample = make_sample(image, tmp_path)
    operator = GrayStyleFilter(threshold=1.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "PASS"


def test_compression_quality_filter_reports_metric(tmp_path, make_image):
    image = make_image(tmp_path / "image.png", size=(20, 20), color=(20, 40, 60))
    sample = make_sample(image, tmp_path)
    operator = CompressionQualityFilter(min_quality_threshold=0.0)

    result = operator.process(sample, make_context(tmp_path))

    assert result.decision == "PASS"
    assert "compression_ratio" in result.metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/operators/test_basic_filters.py -v
```

Expected: FAIL because filter modules do not exist.

- [ ] **Step 3: Implement aspect ratio filter**

Create `preprocessing/operators/__init__.py`:

```python
"""Pluggable preprocessing operators."""
```

Create `preprocessing/operators/filters/__init__.py`:

```python
"""Quality filter operators."""
```

Create `preprocessing/operators/filters/aspect_ratio.py`:

```python
from __future__ import annotations

from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class AspectRatioFilter(FilterOperator):
    name = "aspect_ratio"

    def __init__(self, ratio: float = 2.0) -> None:
        self.ratio = float(ratio)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        try:
            with Image.open(sample.source_path) as image:
                width, height = image.size
        except Exception as exc:
            return OperatorResult.error(f"image_open_failed: {exc!r}")

        if width <= 0 or height <= 0:
            return OperatorResult.error("invalid_image_size", {"width": width, "height": height})

        actual_ratio = max(width / height, height / width)
        metrics = {"width": width, "height": height, "ratio": round(actual_ratio, 6), "threshold": self.ratio}
        if actual_ratio > self.ratio:
            return OperatorResult.reject("aspect_ratio_exceeds_threshold", metrics)
        return OperatorResult.pass_(metrics)
```

- [ ] **Step 4: Implement gray style filter**

Create `preprocessing/operators/filters/gray_style.py`:

```python
from __future__ import annotations

import numpy as np
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class GrayStyleFilter(FilterOperator):
    name = "gray_style"

    def __init__(self, threshold: float = 1.0, max_side: int = 1600) -> None:
        self.threshold = float(threshold)
        self.max_side = int(max_side)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        try:
            image = Image.open(sample.source_path).convert("RGB")
        except Exception as exc:
            return OperatorResult.error(f"image_open_failed: {exc!r}")

        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BICUBIC)

        arr = np.asarray(image, dtype=np.int16)
        diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
        diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2])
        diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2])
        max_channel_diff = float(np.max([diff_rg.max(), diff_gb.max(), diff_rb.max()]))
        metrics = {"max_channel_diff": max_channel_diff, "threshold": self.threshold}

        if max_channel_diff < self.threshold:
            return OperatorResult.reject("grayscale_image", metrics)
        return OperatorResult.pass_(metrics)
```

- [ ] **Step 5: Implement compression quality filter**

Create `preprocessing/operators/filters/compression.py`:

```python
from __future__ import annotations

from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class CompressionQualityFilter(FilterOperator):
    name = "compression_quality"

    def __init__(self, min_quality_threshold: float = 0.2) -> None:
        self.min_quality_threshold = float(min_quality_threshold)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        try:
            with Image.open(sample.source_path) as image:
                width, height = image.size
        except Exception as exc:
            return OperatorResult.error(f"image_open_failed: {exc!r}")

        theoretical_size = max(width * height * 3, 1)
        actual_size = sample.source_path.stat().st_size
        compression_ratio = actual_size / theoretical_size
        metrics = {
            "width": width,
            "height": height,
            "actual_size": actual_size,
            "theoretical_size": theoretical_size,
            "compression_ratio": compression_ratio,
            "threshold": self.min_quality_threshold,
        }
        if compression_ratio < self.min_quality_threshold:
            return OperatorResult.reject("compression_ratio_below_threshold", metrics)
        return OperatorResult.pass_(metrics)
```

- [ ] **Step 6: Run filter tests**

Run:

```bash
pytest tests/operators/test_basic_filters.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add preprocessing/operators tests/operators/test_basic_filters.py
git commit -m "feat: add basic quality filters"
```

---

### Task 5: Runner Filter Stage

**Files:**
- Create: `preprocessing/pipeline/runner.py`
- Modify: `preprocessing/pipeline/registry.py`
- Test: `tests/pipeline/test_runner_filter.py`

- [ ] **Step 1: Write failing runner filter tests**

Create `tests/pipeline/test_runner_filter.py`:

```python
from pathlib import Path

from preprocessing.pipeline.config import InputConfig, OutputConfig, FilterConfig, LabelConfig, OperatorConfig, PipelineConfig
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.registry import OperatorRegistry
from preprocessing.pipeline.runner import PipelineRunner
from preprocessing.pipeline.types import OperatorResult


class AlwaysPass(FilterOperator):
    name = "always_pass"

    def process(self, sample, context):
        return OperatorResult.pass_({"operator": self.name})


class RejectName(FilterOperator):
    name = "reject_name"

    def __init__(self, name_contains: str = "bad") -> None:
        self.name_contains = name_contains

    def process(self, sample, context):
        if self.name_contains in sample.source_path.name:
            return OperatorResult.reject("name_matched", {"name_contains": self.name_contains})
        return OperatorResult.pass_()


class ShouldNotRun(FilterOperator):
    name = "should_not_run"
    calls = 0

    def process(self, sample, context):
        ShouldNotRun.calls += 1
        return OperatorResult.pass_()


def make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input=InputConfig(input_dir=tmp_path / "raw", recursive=True, image_exts=[".jpg"]),
        output=OutputConfig(run_root=tmp_path / "runs", pass_archive_dir=tmp_path / "clean", pass_archive_layout="run_subdir", overwrite=False),
        filter=FilterConfig(
            short_circuit=True,
            operators=[
                OperatorConfig(name="always_pass"),
                OperatorConfig(name="reject_name", params={"name_contains": "bad"}),
                OperatorConfig(name="should_not_run"),
            ],
        ),
        label=LabelConfig(enabled=False),
    )


def test_filter_stage_short_circuits_and_archives_passed_images(tmp_path, make_image):
    raw = tmp_path / "raw"
    make_image(raw / "good.jpg")
    make_image(raw / "bad.jpg")
    registry = OperatorRegistry()
    registry.register("always_pass", AlwaysPass)
    registry.register("reject_name", RejectName)
    registry.register("should_not_run", ShouldNotRun)
    ShouldNotRun.calls = 0

    runner = PipelineRunner(make_config(tmp_path), registry=registry, run_id="run1")
    summary = runner.run_filter()

    assert summary["passed"] == 1
    assert summary["rejected"] == 1
    assert (tmp_path / "clean" / "run1" / "good.jpg").exists()
    assert not (tmp_path / "clean" / "run1" / "bad.jpg").exists()
    assert ShouldNotRun.calls == 1

    manifest_lines = (tmp_path / "runs" / "run1" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    manifest_text = "\n".join(manifest_lines)
    assert '"status": "PASS"' in manifest_text
    assert '"status": "REJECT"' in manifest_text
    assert '"rejected_by": "reject_name"' in manifest_text
    assert (tmp_path / "runs" / "run1" / "summary.json").exists()
    assert (tmp_path / "runs" / "run1" / "config.snapshot.yaml").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/pipeline/test_runner_filter.py -v
```

Expected: FAIL because `PipelineRunner` does not exist.

- [ ] **Step 3: Implement filter runner**

Create `preprocessing/pipeline/runner.py`:

```python
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from preprocessing.pipeline.config import PipelineConfig
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.io import copy_sample_to_archive, ensure_run_dirs, enumerate_images, write_json, write_jsonl
from preprocessing.pipeline.operators import BatchFilterOperator
from preprocessing.pipeline.registry import DEFAULT_REGISTRY, OperatorRegistry
from preprocessing.pipeline.types import OperatorResult, Sample


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        registry: OperatorRegistry | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or DEFAULT_REGISTRY
        self.run_id = run_id or generate_run_id()
        self.context = PipelineContext(
            run_id=self.run_id,
            run_dir=self.config.output.run_root / self.run_id,
            pass_archive_dir=self.config.output.pass_archive_dir,
            pass_archive_layout=self.config.output.pass_archive_layout,
        )

    def run_filter(self) -> dict[str, Any]:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started = time.monotonic()
        ensure_run_dirs(self.context.run_dir)
        self._write_config_snapshot()

        samples = enumerate_images(
            self.config.input.input_dir,
            recursive=self.config.input.recursive,
            image_exts=self.config.input.image_exts,
        )
        states = {
            sample.sample_id: {
                "sample": sample,
                "status": "PENDING",
                "rejected_by": None,
                "reject_reason": None,
                "operator_trace": [],
            }
            for sample in samples
        }

        operators = [
            self.registry.create(operator_config.name, operator_config.params)
            for operator_config in self.config.filter.enabled_operators
        ]

        for operator in operators:
            active_samples = [
                state["sample"]
                for state in states.values()
                if state["status"] == "PENDING"
            ]
            if isinstance(operator, BatchFilterOperator):
                results = operator.process_batch(active_samples, self.context)
                for sample in active_samples:
                    result = results.get(sample.sample_id, OperatorResult.pass_())
                    self._record_filter_result(states[sample.sample_id], operator.name, result)
            else:
                operator.setup(self.context)
                try:
                    for sample in active_samples:
                        try:
                            result = operator.process(sample, self.context)
                        except Exception as exc:
                            result = OperatorResult.error(f"operator_exception: {exc!r}")
                        self._record_filter_result(states[sample.sample_id], operator.name, result)
                finally:
                    operator.teardown(self.context)

        for state in states.values():
            if state["status"] == "PENDING":
                sample = state["sample"]
                archive_path = self.context.archive_path_for(sample.relative_path)
                try:
                    sample.archive_path = copy_sample_to_archive(sample, archive_path, overwrite=self.config.output.overwrite)
                    state["status"] = "PASS"
                except Exception as exc:
                    state["status"] = "ERROR"
                    state["rejected_by"] = "archive"
                    state["reject_reason"] = f"archive_copy_failed: {exc!r}"

        manifest_rows = [self._manifest_row(state) for state in states.values()]
        write_jsonl(self.context.manifest_path, manifest_rows)
        summary = self._summary(manifest_rows, started_at, started, stage="filter")
        write_json(self.context.summary_path, summary)
        return summary

    def _record_filter_result(self, state: dict[str, Any], operator_name: str, result: OperatorResult) -> None:
        state["operator_trace"].append(
            {
                "name": operator_name,
                "decision": result.decision,
                "reason": result.reason,
                "metrics": result.metrics,
            }
        )
        if result.decision == "REJECT":
            state["status"] = "REJECT"
            state["rejected_by"] = operator_name
            state["reject_reason"] = result.reason
        elif result.decision == "ERROR":
            state["status"] = "ERROR"
            state["rejected_by"] = operator_name
            state["reject_reason"] = result.reason

    def _manifest_row(self, state: dict[str, Any]) -> dict[str, Any]:
        sample: Sample = state["sample"]
        return {
            "sample_id": sample.sample_id,
            "source_path": str(sample.source_path),
            "relative_path": sample.relative_path.as_posix(),
            "archive_path": str(sample.archive_path) if sample.archive_path else None,
            "status": state["status"],
            "rejected_by": state["rejected_by"],
            "reject_reason": state["reject_reason"],
            "operator_trace": state["operator_trace"],
        }

    def _summary(
        self,
        rows: list[dict[str, Any]],
        started_at: str,
        started_monotonic: float,
        stage: str,
    ) -> dict[str, Any]:
        reject_by_operator: dict[str, int] = {}
        for row in rows:
            if row["status"] == "REJECT" and row["rejected_by"]:
                reject_by_operator[row["rejected_by"]] = reject_by_operator.get(row["rejected_by"], 0) + 1
        return {
            "run_id": self.run_id,
            "stage": stage,
            "input_total": len(rows),
            "passed": sum(1 for row in rows if row["status"] == "PASS"),
            "rejected": sum(1 for row in rows if row["status"] == "REJECT"),
            "errors": sum(1 for row in rows if row["status"] == "ERROR"),
            "reject_by_operator": reject_by_operator,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - started_monotonic, 3),
        }

    def _write_config_snapshot(self) -> None:
        data = {
            "input": {
                "input_dir": str(self.config.input.input_dir),
                "recursive": self.config.input.recursive,
                "image_exts": self.config.input.image_exts,
            },
            "output": {
                "run_root": str(self.config.output.run_root),
                "pass_archive_dir": str(self.config.output.pass_archive_dir),
                "pass_archive_layout": self.config.output.pass_archive_layout,
                "overwrite": self.config.output.overwrite,
            },
            "filter": {
                "short_circuit": self.config.filter.short_circuit,
                "operators": [
                    {"name": op.name, "enabled": op.enabled, "params": op.params}
                    for op in self.config.filter.operators
                ],
            },
            "label": {
                "enabled": self.config.label.enabled,
                "input_dir": str(self.config.label.input_dir),
                "annotation_dir": str(self.config.label.annotation_dir),
                "jsonl_path": str(self.config.label.jsonl_path),
                "operator": {
                    "name": self.config.label.operator.name,
                    "enabled": self.config.label.operator.enabled,
                    "params": self.config.label.operator.params,
                },
            },
        }
        self.context.config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.context.config_snapshot_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
```

- [ ] **Step 4: Run runner filter tests**

Run:

```bash
pytest tests/pipeline/test_runner_filter.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all current tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add preprocessing/pipeline/runner.py tests/pipeline/test_runner_filter.py
git commit -m "feat: add filter stage runner"
```

---

### Task 6: Batch Duplicate Filter

**Files:**
- Create: `preprocessing/operators/filters/duplicate.py`
- Test: `tests/operators/test_duplicate.py`

- [ ] **Step 1: Write failing duplicate tests**

Create `tests/operators/test_duplicate.py`:

```python
from preprocessing.operators.filters.duplicate import DuplicateFilter
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import Sample


def test_duplicate_filter_rejects_smaller_duplicate(tmp_path, make_image):
    first = make_image(tmp_path / "first.jpg", size=(20, 20), color=(255, 0, 0))
    second = make_image(tmp_path / "second.jpg", size=(20, 20), color=(255, 0, 0))
    samples = [
        Sample(sample_id="first.jpg", source_path=first, relative_path=first.relative_to(tmp_path)),
        Sample(sample_id="second.jpg", source_path=second, relative_path=second.relative_to(tmp_path)),
    ]
    context = PipelineContext("run1", tmp_path / "runs" / "run1", tmp_path / "clean")
    operator = DuplicateFilter(threshold=0, hash_size=8)

    results = operator.process_batch(samples, context)

    decisions = {sample_id: result.decision for sample_id, result in results.items()}
    assert sorted(decisions.values()) == ["PASS", "REJECT"]
    rejected = [result for result in results.values() if result.decision == "REJECT"][0]
    assert rejected.reason == "duplicate_image"
    assert "keeper" in rejected.metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/operators/test_duplicate.py -v
```

Expected: FAIL because duplicate operator does not exist.

- [ ] **Step 3: Implement duplicate filter**

Create `preprocessing/operators/filters/duplicate.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import imagehash

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import BatchFilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


@dataclass
class ImageInfo:
    sample: Sample
    phash: imagehash.ImageHash
    size: int


class DuplicateFilter(BatchFilterOperator):
    name = "duplicate"

    def __init__(self, threshold: int = 9, hash_size: int = 8) -> None:
        self.threshold = int(threshold)
        self.hash_size = int(hash_size)

    def process_batch(
        self,
        samples: list[Sample],
        context: PipelineContext,
    ) -> dict[str, OperatorResult]:
        results: dict[str, OperatorResult] = {}
        infos: list[ImageInfo] = []

        for sample in samples:
            try:
                phash = self._compute_phash(sample.source_path)
                infos.append(ImageInfo(sample=sample, phash=phash, size=sample.source_path.stat().st_size))
            except Exception as exc:
                results[sample.sample_id] = OperatorResult.error(f"hash_failed: {exc!r}")

        kept: list[ImageInfo] = []
        for current in infos:
            duplicate_of: ImageInfo | None = None
            distance = 0
            for existing in kept:
                distance = current.phash - existing.phash
                if distance <= self.threshold:
                    duplicate_of = existing
                    break

            if duplicate_of is None:
                kept.append(current)
                results[current.sample.sample_id] = OperatorResult.pass_({"phash": str(current.phash)})
                continue

            keeper, duplicate = self._choose_keeper(duplicate_of, current)
            if keeper.sample.sample_id != duplicate_of.sample.sample_id:
                kept.remove(duplicate_of)
                kept.append(keeper)
                results[keeper.sample.sample_id] = OperatorResult.pass_({"phash": str(keeper.phash)})

            results[duplicate.sample.sample_id] = OperatorResult.reject(
                "duplicate_image",
                {
                    "keeper": keeper.sample.sample_id,
                    "hamming_distance": distance,
                    "threshold": self.threshold,
                },
            )

        return results

    def _compute_phash(self, path: Path) -> imagehash.ImageHash:
        with Image.open(path) as image:
            return imagehash.phash(image.convert("RGB"), hash_size=self.hash_size)

    def _choose_keeper(self, a: ImageInfo, b: ImageInfo) -> tuple[ImageInfo, ImageInfo]:
        if a.size != b.size:
            return (a, b) if a.size > b.size else (b, a)
        return (a, b) if str(a.sample.source_path) <= str(b.sample.source_path) else (b, a)
```

- [ ] **Step 4: Run duplicate tests**

Run:

```bash
pytest tests/operators/test_duplicate.py -v
```

Expected: PASS.

- [ ] **Step 5: Run runner filter tests**

Run:

```bash
pytest tests/pipeline/test_runner_filter.py tests/operators/test_duplicate.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add preprocessing/operators/filters/duplicate.py tests/operators/test_duplicate.py
git commit -m "feat: add duplicate batch filter"
```

---

### Task 7: Remaining Image Filters And Default Registry

**Files:**
- Create: `preprocessing/operators/filters/framebox.py`
- Create: `preprocessing/operators/filters/stitch.py`
- Create: `preprocessing/operators/filters/blur.py`
- Modify: `preprocessing/pipeline/registry.py`
- Test: `tests/operators/test_advanced_filters.py`

- [ ] **Step 1: Write smoke tests for advanced filters**

Create `tests/operators/test_advanced_filters.py`:

```python
from preprocessing.operators.filters.blur import PersonBlurFilter
from preprocessing.operators.filters.framebox import FrameBoxFilter
from preprocessing.operators.filters.stitch import StitchLineFilterV2
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.types import Sample


def make_context(tmp_path):
    return PipelineContext("run1", tmp_path / "runs" / "run1", tmp_path / "clean")


def make_sample(path, root):
    return Sample(sample_id=path.name, source_path=path, relative_path=path.relative_to(root))


def test_framebox_filter_passes_plain_color_image(tmp_path, make_image):
    image = make_image(tmp_path / "plain.jpg", size=(80, 80), color=(120, 120, 120))
    result = FrameBoxFilter(min_std=15.0).process(make_sample(image, tmp_path), make_context(tmp_path))

    assert result.decision == "PASS"


def test_stitch_filter_passes_plain_color_image(tmp_path, make_image):
    image = make_image(tmp_path / "plain.jpg", size=(80, 80), color=(120, 120, 120))
    result = StitchLineFilterV2(prominence=6.0).process(make_sample(image, tmp_path), make_context(tmp_path))

    assert result.decision == "PASS"


def test_blur_filter_keeps_image_without_person_or_face_roi(tmp_path, make_image):
    image = make_image(tmp_path / "plain.jpg", size=(80, 80), color=(120, 120, 120))
    result = PersonBlurFilter().process(make_sample(image, tmp_path), make_context(tmp_path))

    assert result.decision == "PASS"
    assert result.reason is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/operators/test_advanced_filters.py -v
```

Expected: FAIL because the advanced filter modules do not exist.

- [ ] **Step 3: Implement framebox filter**

Create `preprocessing/operators/filters/framebox.py` by migrating the focused logic from the old `framebox_filter.py`:

```python
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class FrameBoxFilter(FilterOperator):
    name = "framebox"

    def __init__(self, max_side: int = 1024, sides: int = 3, min_thick: int = 20, max_ratio: float = 0.30, min_std: float = 15.0) -> None:
        self.max_side = int(max_side)
        self.sides = int(sides)
        self.min_thick = int(min_thick)
        self.max_ratio = float(max_ratio)
        self.min_std = float(min_std)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        image = self._load_rgb_np(sample)
        if image is None:
            return OperatorResult.error("image_open_failed")
        bad, reason, metrics = self._check_image_content(image)
        if bad:
            return OperatorResult.reject(reason, metrics)
        return OperatorResult.pass_(metrics)

    def _load_rgb_np(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            image = Image.open(sample.source_path).convert("RGB")
        except Exception:
            return None
        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)

    def _border_thickness(self, gray: np.ndarray, side: str, color_tol: int = 15, noise_tol: float = 0.95) -> int:
        if side == "top":
            matrix = gray
        elif side == "bottom":
            matrix = gray[::-1, :]
        elif side == "left":
            matrix = gray.T
        elif side == "right":
            matrix = gray.T[::-1, :]
        else:
            return 0
        bg_color = np.median(matrix[0, :])
        diff = np.abs(matrix.astype(int) - bg_color)
        row_matches = np.mean(diff < color_tol, axis=1)
        border_rows = np.where(row_matches < noise_tol)[0]
        return int(border_rows[0]) if len(border_rows) else matrix.shape[0]

    def _check_image_content(self, image: np.ndarray) -> tuple[bool, str, dict[str, object]]:
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, std = cv2.meanStdDev(gray)
        std_value = float(std[0][0])
        if std_value < self.min_std:
            return False, "flat_image", {"std": std_value}

        top = self._border_thickness(gray, "top")
        bottom = self._border_thickness(gray, "bottom")
        left = self._border_thickness(gray, "left")
        right = self._border_thickness(gray, "right")
        metrics = {"top": top, "bottom": bottom, "left": left, "right": right, "std": std_value}

        if top > height * self.max_ratio or bottom > height * self.max_ratio or left > width * self.max_ratio or right > width * self.max_ratio:
            return False, "likely_solid_background_too_thick", metrics

        borders_found = sum([top > self.min_thick, bottom > self.min_thick, left > self.min_thick, right > self.min_thick])
        if borders_found >= self.sides:
            return True, "bordered_content", metrics
        return False, "clean", metrics
```

- [ ] **Step 4: Implement stitch filter**

Create `preprocessing/operators/filters/stitch.py` by migrating the focused v2 gradient projection logic:

```python
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class StitchLineFilterV2(FilterOperator):
    name = "stitch_v2"

    def __init__(self, max_side: int = 1400, prominence: float = 6.0) -> None:
        self.max_side = int(max_side)
        self.prominence = float(prominence)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        bgr = self._load_bgr(sample)
        if bgr is None:
            return OperatorResult.error("image_open_failed")
        bad, reason, metrics = self._detect_stitch(bgr)
        if bad:
            return OperatorResult.reject(reason, metrics)
        return OperatorResult.pass_(metrics)

    def _load_bgr(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            image = Image.open(sample.source_path).convert("RGB")
        except Exception:
            return None
        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def _detect_stitch(self, bgr: np.ndarray) -> tuple[bool, str, dict[str, object]]:
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        proj_v = np.sum(grad_x, axis=0)
        proj_h = np.sum(grad_y, axis=1)

        found_v, metrics_v = self._check_projection(proj_v, width, "vertical")
        if found_v:
            return True, "stitch_line_vertical", metrics_v
        found_h, metrics_h = self._check_projection(proj_h, height, "horizontal")
        if found_h:
            return True, "stitch_line_horizontal", metrics_h
        return False, "clean", {"vertical": metrics_v, "horizontal": metrics_h}

    def _check_projection(self, projection: np.ndarray, length: int, axis: str) -> tuple[bool, dict[str, object]]:
        mean_val = float(np.mean(projection))
        if mean_val == 0:
            return False, {"axis": axis, "mean": mean_val}
        norm = projection / mean_val
        start = int(length * 0.2)
        end = int(length * 0.8)
        roi = norm[start:end]
        if len(roi) == 0:
            return False, {"axis": axis, "mean": mean_val}
        peak_idx_roi = int(np.argmax(roi))
        peak_val = float(roi[peak_idx_roi])
        peak_idx_global = peak_idx_roi + start
        metrics = {"axis": axis, "peak_position": peak_idx_global / length, "peak_prominence": peak_val, "threshold": self.prominence}
        return peak_val >= self.prominence, metrics
```

- [ ] **Step 5: Implement blur filter**

Create `preprocessing/operators/filters/blur.py` with the migrated person/face ROI logic:

```python
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image

from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.operators import FilterOperator
from preprocessing.pipeline.types import OperatorResult, Sample


class PersonBlurFilter(FilterOperator):
    name = "person_blur"

    def __init__(self, max_side: int = 1024, person_thr: float = 120.0, face_thr: float = 90.0, pad_frac: float = 0.08) -> None:
        self.max_side = int(max_side)
        self.person_thr = float(person_thr)
        self.face_thr = float(face_thr)
        self.pad_frac = float(pad_frac)

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        bgr = self._load_bgr(sample)
        if bgr is None:
            return OperatorResult.error("image_open_failed")
        boxes = self._detect_faces(bgr)
        if not boxes:
            return OperatorResult.pass_({"used": "none", "no_roi_kept": True})
        score = self._blur_score(bgr, boxes)
        if score is None:
            return OperatorResult.pass_({"used": "face", "no_valid_roi_kept": True})
        metrics = {"used": "face", "score": score, "threshold": self.face_thr}
        if score < self.face_thr:
            return OperatorResult.reject("blur_score_below_threshold", metrics)
        return OperatorResult.pass_(metrics)

    def _load_bgr(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            image = Image.open(sample.source_path).convert("RGB")
        except Exception:
            return None
        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((int(round(width * scale)), int(round(height * scale))), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def _detect_faces(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        face_xml = str(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(face_xml)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces[:10]]

    def _blur_score(self, bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scores: list[float] = []
        for x1, y1, x2, y2 in boxes:
            bw = x2 - x1
            bh = y2 - y1
            x1 = max(0, int(x1 - bw * self.pad_frac))
            y1 = max(0, int(y1 - bh * self.pad_frac))
            x2 = min(width, int(x2 + bw * self.pad_frac))
            y2 = min(height, int(y2 + bh * self.pad_frac))
            roi = gray[y1:y2, x1:x2]
            if roi.size >= 32 * 32:
                scores.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))
        return min(scores) if scores else None
```

- [ ] **Step 6: Register default operators**

Modify `preprocessing/pipeline/registry.py`:

```python
from __future__ import annotations

from preprocessing.pipeline.operators import BatchFilterOperator, FilterOperator, LabelOperator

OperatorInstance = FilterOperator | BatchFilterOperator | LabelOperator
OperatorClass = type[FilterOperator] | type[BatchFilterOperator] | type[LabelOperator]


class OperatorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, OperatorClass] = {}

    def register(self, name: str, factory: OperatorClass) -> None:
        self._factories[name] = factory

    def create(self, name: str, params: dict[str, object]) -> OperatorInstance:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown operator: {name}. Available operators: {available}") from exc
        return factory(**params)


def build_default_registry() -> OperatorRegistry:
    from preprocessing.operators.filters.aspect_ratio import AspectRatioFilter
    from preprocessing.operators.filters.blur import PersonBlurFilter
    from preprocessing.operators.filters.compression import CompressionQualityFilter
    from preprocessing.operators.filters.duplicate import DuplicateFilter
    from preprocessing.operators.filters.framebox import FrameBoxFilter
    from preprocessing.operators.filters.gray_style import GrayStyleFilter
    from preprocessing.operators.filters.stitch import StitchLineFilterV2

    registry = OperatorRegistry()
    registry.register("aspect_ratio", AspectRatioFilter)
    registry.register("gray_style", GrayStyleFilter)
    registry.register("compression_quality", CompressionQualityFilter)
    registry.register("framebox", FrameBoxFilter)
    registry.register("stitch_v2", StitchLineFilterV2)
    registry.register("person_blur", PersonBlurFilter)
    registry.register("duplicate", DuplicateFilter)
    return registry


DEFAULT_REGISTRY = build_default_registry()
```

- [ ] **Step 7: Run advanced filter tests**

Run:

```bash
pytest tests/operators/test_advanced_filters.py tests/pipeline/test_registry.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add preprocessing/operators/filters/framebox.py preprocessing/operators/filters/stitch.py preprocessing/operators/filters/blur.py preprocessing/pipeline/registry.py tests/operators/test_advanced_filters.py
git commit -m "feat: add advanced image filters"
```

---

### Task 8: Labeler And Label Stage Runner

**Files:**
- Create: `preprocessing/operators/labelers/__init__.py`
- Create: `preprocessing/operators/labelers/person_attribute.py`
- Modify: `preprocessing/pipeline/registry.py`
- Modify: `preprocessing/pipeline/runner.py`
- Test: `tests/operators/test_person_attribute_labeler.py`
- Test: `tests/pipeline/test_runner_label.py`

- [ ] **Step 1: Write labeler normalization tests**

Create `tests/operators/test_person_attribute_labeler.py`:

```python
from preprocessing.operators.labelers.person_attribute import normalize_annotation


def test_normalize_annotation_for_multiple_people_sets_other_fields_to_na():
    ann = normalize_annotation({"person_count": "N", "gender": "male"}, file_name="a.jpg")

    assert ann["file_name"] == "a.jpg"
    assert ann["person_count"] == "N"
    assert ann["gender"] == "N/A"
    assert ann["head_visible"] == "N/A"


def test_normalize_annotation_for_single_person_keeps_allowed_values():
    ann = normalize_annotation(
        {
            "person_count": "1",
            "gender": "female",
            "head_visible": "yes",
            "shot_type": "half_body",
            "clothes_visible": "yes",
            "holding_object": "no",
            "obvious_makeup": "yes",
            "expression": "smile",
            "hair_visible": "yes",
            "hair_color": "black",
            "facial_features_clear": "yes",
            "face_direction": "frontal",
            "hand_hold_feasible": "yes",
            "person_prominence": "close",
            "person_size_in_frame": "large",
        },
        file_name="a.jpg",
    )

    assert ann["gender"] == "female"
    assert ann["expression"] == "smile"
    assert ann["hair_color"] == "black"
```

- [ ] **Step 2: Write failing runner label test**

Create `tests/pipeline/test_runner_label.py`:

```python
import json
from pathlib import Path

from preprocessing.pipeline.config import InputConfig, OutputConfig, FilterConfig, LabelConfig, OperatorConfig, PipelineConfig
from preprocessing.pipeline.operators import LabelOperator
from preprocessing.pipeline.registry import OperatorRegistry
from preprocessing.pipeline.runner import PipelineRunner
from preprocessing.pipeline.types import OperatorResult


class FakeLabeler(LabelOperator):
    name = "fake_labeler"

    def process(self, sample, context):
        return OperatorResult.label({"file_name": sample.source_path.name, "person_count": "0"})


def test_label_stage_writes_annotations_and_jsonl(tmp_path, make_image):
    clean_dir = tmp_path / "clean" / "run1"
    make_image(clean_dir / "a.jpg")
    registry = OperatorRegistry()
    registry.register("fake_labeler", FakeLabeler)
    config = PipelineConfig(
        input=InputConfig(input_dir=tmp_path / "raw"),
        output=OutputConfig(run_root=tmp_path / "runs", pass_archive_dir=tmp_path / "clean"),
        filter=FilterConfig(),
        label=LabelConfig(enabled=True, input_dir="auto", operator=OperatorConfig(name="fake_labeler")),
    )

    summary = PipelineRunner(config, registry=registry, run_id="run1").run_label()

    assert summary["labeled"] == 1
    annotation = json.loads((tmp_path / "runs" / "run1" / "annotations" / "a.json").read_text(encoding="utf-8"))
    assert annotation["annotation"]["person_count"] == "0"
    labels_lines = (tmp_path / "runs" / "run1" / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(labels_lines) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/operators/test_person_attribute_labeler.py tests/pipeline/test_runner_label.py -v
```

Expected: FAIL because labeler and `run_label` do not exist.

- [ ] **Step 4: Implement person attribute labeler normalization**

Create `preprocessing/operators/labelers/__init__.py`:

```python
"""Labeling operators."""
```

Create `preprocessing/operators/labelers/person_attribute.py`:

```python
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
```

- [ ] **Step 5: Implement label runner**

Modify `preprocessing/pipeline/runner.py` by adding these imports:

```python
from preprocessing.pipeline.io import copy_sample_to_archive, ensure_run_dirs, enumerate_images, read_jsonl, write_json, write_jsonl
from preprocessing.pipeline.operators import BatchFilterOperator, LabelOperator
```

Add this method to `PipelineRunner`:

```python
    def run_label(self) -> dict[str, Any]:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started = time.monotonic()
        ensure_run_dirs(self.context.run_dir)
        self._write_config_snapshot()

        input_dir = self._label_input_dir()
        samples = enumerate_images(
            input_dir,
            recursive=True,
            image_exts=self.config.input.image_exts,
        )
        operator = self.registry.create(self.config.label.operator.name, self.config.label.operator.params)
        if not isinstance(operator, LabelOperator):
            raise TypeError(f"Configured label operator is not a LabelOperator: {self.config.label.operator.name}")

        annotations: list[dict[str, Any]] = []
        failures = 0
        operator.setup(self.context)
        try:
            for sample in samples:
                result = operator.process(sample, self.context)
                if result.decision == "LABEL" and result.labels is not None:
                    output = {
                        "meta": {
                            "operator": operator.name,
                            "input_image": sample.source_path.name,
                        },
                        "annotation": result.labels,
                    }
                    annotation_path = self.context.annotation_dir / f"{sample.source_path.stem}.json"
                    write_json(annotation_path, output)
                    annotations.append(result.labels)
                else:
                    failures += 1
        finally:
            operator.teardown(self.context)

        write_jsonl(self.context.labels_jsonl_path, annotations)
        summary = {
            "run_id": self.run_id,
            "stage": "label",
            "input_total": len(samples),
            "labeled": len(annotations),
            "failed": failures,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        write_json(self.context.run_dir / "label_summary.json", summary)
        return summary

    def run_all(self) -> dict[str, Any]:
        filter_summary = self.run_filter()
        label_summary = self.run_label()
        return {"filter": filter_summary, "label": label_summary}

    def _label_input_dir(self) -> Path:
        if str(self.config.label.input_dir) == "auto":
            return self.config.output.pass_archive_dir / self.run_id
        return Path(self.config.label.input_dir)
```

- [ ] **Step 6: Register labeler**

Modify `build_default_registry()` in `preprocessing/pipeline/registry.py` to import and register `PersonAttributeLabeler`:

```python
    from preprocessing.operators.labelers.person_attribute import PersonAttributeLabeler
    registry.register("person_attribute_labeler", PersonAttributeLabeler)
```

- [ ] **Step 7: Run label tests**

Run:

```bash
pytest tests/operators/test_person_attribute_labeler.py tests/pipeline/test_runner_label.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add preprocessing/operators/labelers preprocessing/pipeline/registry.py preprocessing/pipeline/runner.py tests/operators/test_person_attribute_labeler.py tests/pipeline/test_runner_label.py
git commit -m "feat: add label stage and person attribute labeler"
```

---

### Task 9: CLI

**Files:**
- Create: `preprocessing/pipeline/cli.py`
- Test: `tests/pipeline/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/pipeline/test_cli.py`:

```python
from preprocessing.pipeline.cli import parse_args


def test_parse_filter_args():
    args = parse_args(["run", "--stage", "filter", "--config", "config.yaml"])

    assert args.command == "run"
    assert args.stage == "filter"
    assert args.config == "config.yaml"


def test_parse_label_args_with_run_id():
    args = parse_args(["run", "--stage", "label", "--config", "config.yaml", "--run-id", "run1"])

    assert args.stage == "label"
    assert args.run_id == "run1"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/pipeline/test_cli.py -v
```

Expected: FAIL because `cli.py` does not exist.

- [ ] **Step 3: Implement CLI**

Create `preprocessing/pipeline/cli.py`:

```python
from __future__ import annotations

import argparse
from typing import Sequence

from preprocessing.pipeline.config import load_pipeline_config
from preprocessing.pipeline.runner import PipelineRunner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="preprocessing-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--stage", choices=["filter", "label", "all"], required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--run-id", default=None)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_pipeline_config(args.config)
    runner = PipelineRunner(config, run_id=args.run_id)

    if args.stage == "filter":
        summary = runner.run_filter()
    elif args.stage == "label":
        summary = runner.run_label()
    else:
        summary = runner.run_all()

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest tests/pipeline/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all tests**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add preprocessing/pipeline/cli.py tests/pipeline/test_cli.py
git commit -m "feat: add pipeline CLI"
```

---

### Task 10: End-To-End Smoke Test And Documentation

**Files:**
- Modify: `README.md`
- Test: `tests/pipeline/test_e2e.py`

- [ ] **Step 1: Write E2E smoke test**

Create `tests/pipeline/test_e2e.py`:

```python
import json

from preprocessing.pipeline.cli import main


def test_filter_cli_end_to_end(tmp_path, make_image):
    raw = tmp_path / "raw"
    make_image(raw / "good.jpg", size=(100, 100), color=(255, 0, 0))
    make_image(raw / "bad.jpg", size=(300, 50), color=(255, 0, 0))
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
input:
  input_dir: {raw.as_posix()}
  recursive: true
  image_exts: [".jpg"]
output:
  run_root: {(tmp_path / "runs").as_posix()}
  pass_archive_dir: {(tmp_path / "clean").as_posix()}
  pass_archive_layout: run_subdir
  overwrite: false
filter:
  short_circuit: true
  operators:
    - name: aspect_ratio
      enabled: true
      params:
        ratio: 2.0
label:
  enabled: false
  input_dir: auto
  operator:
    name: person_attribute_labeler
    params:
      base_url: http://example.invalid/v1
      model_name: fake
""",
        encoding="utf-8",
    )

    exit_code = main(["run", "--stage", "filter", "--config", str(config), "--run-id", "run1"])

    assert exit_code == 0
    assert (tmp_path / "clean" / "run1" / "good.jpg").exists()
    assert not (tmp_path / "clean" / "run1" / "bad.jpg").exists()
    summary = json.loads((tmp_path / "runs" / "run1" / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] == 1
    assert summary["rejected"] == 1
```

- [ ] **Step 2: Run E2E test to verify current behavior**

Run:

```bash
pytest tests/pipeline/test_e2e.py -v
```

Expected: PASS.

- [ ] **Step 3: Update README with actual usage**

Replace `README.md` with:

```markdown
# preprocessing-pipeline

Pluggable image preprocessing and labeling pipeline.

## Workflow

1. Quality filters run in the order listed in the YAML config.
2. The first `REJECT` stops later filters for that image.
3. Rejected images are not copied or moved.
4. Passed images are copied to `output.pass_archive_dir`.
5. Labeling can run after filtering or as a separate stage.

## Commands

```bash
python -m preprocessing.pipeline.cli run --stage filter --config configs/person_pipeline.yaml
python -m preprocessing.pipeline.cli run --stage label --config configs/person_pipeline.yaml --run-id 20260527-203012
python -m preprocessing.pipeline.cli run --stage all --config configs/person_pipeline.yaml
```

## Outputs

Run artifacts are written under `output.run_root/<run_id>/`:

- `config.snapshot.yaml`
- `manifest.jsonl`
- `summary.json`
- `annotations/`
- `labels.jsonl`
- `logs/`

Passed images are copied to:

```text
output.pass_archive_dir/<run_id>/<relative_path>
```
```

- [ ] **Step 4: Run the full test suite**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/pipeline/test_e2e.py
git commit -m "test: add end-to-end filter smoke test"
```

---

## Self-Review Checklist

- Spec coverage:
  - Pluggable operator interfaces: Tasks 1, 2.
  - YAML-configured order: Tasks 2, 5.
  - Short-circuit filtering: Task 5.
  - REJECT images not copied or moved: Task 5 and Task 10.
  - PASS images copied to user archive path: Tasks 3, 5, 10.
  - Independent run directories and config snapshots: Task 5.
  - Manifest and summary artifacts: Task 5.
  - Batch duplicate operator: Task 6.
  - Label stage and JSON/JSONL outputs: Task 8.
  - CLI stages `filter`, `label`, `all`: Tasks 8, 9.
  - Tests for pipeline behavior and operator contracts: Tasks 1 through 10.

- Type consistency:
  - `Sample`, `OperatorResult`, `PipelineContext`, `OperatorConfig`, and `PipelineRunner` names are consistent across tasks.
  - Decisions are consistently `"PASS"`, `"REJECT"`, `"LABEL"`, and `"ERROR"`.
  - `pass_archive_layout` values match the design: `run_subdir`, `flat`, `preserve_relative`.

- Scope:
  - This plan does not implement a DAG engine, complex checkpointing, cross-run caching, or broad filter-stage parallelism.
  - Live model inference is deliberately isolated in `PersonAttributeLabeler`; normalization and runner behavior are testable without a live service.
