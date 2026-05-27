# Data Preprocessing and Labeling Pipeline Design

Date: 2026-05-27

## Goal

Build a pluggable image preprocessing and labeling pipeline from the existing scripts in `components/`.
The first version focuses on a linear workflow:

1. Run quality filter operators in the order defined by a YAML config.
2. Stop filtering a sample as soon as the first operator rejects it.
3. Do not move or copy rejected images.
4. Copy passed images to a user-specified archive directory.
5. Run labeling on the passed image archive.
6. Save run-level audit artifacts for traceability.

The design starts with the sample-flow model instead of a heavyweight DAG engine. This keeps the first implementation small while still making operators easy to plug in.

## Confirmed Decisions

- Use a standard operator base class with staged interfaces.
- Support `filter`, `label`, and `all` execution stages.
- The quality filter stage runs before the labeling stage.
- Filter order is completely controlled by the config file.
- Filtering uses short-circuit behavior: the first `REJECT` stops later quality operators for that sample.
- Rejected images are not copied, moved, or otherwise changed.
- Passed images are copied to a user-specified archive path.
- Each run creates an independent `run_id` and run artifact directory.
- The original input directory is treated as read-only.

## Architecture

```text
config.yaml
  |
  v
PipelineRunner(run_id)
  |
  +-- filter stage
  |     |
  |     +-- enumerate images from input_dir
  |     +-- create Sample objects
  |     +-- execute configured quality operators in order
  |     +-- short-circuit on first REJECT
  |     +-- record REJECT samples only in manifest
  |     +-- copy PASS samples to pass_archive_dir
  |
  +-- label stage
        |
        +-- enumerate images from the PASS archive or configured label input
        +-- execute label operator
        +-- write per-image annotation JSON
        +-- write labels.jsonl
```

The pipeline owns orchestration concerns:

- image enumeration
- operator loading
- short-circuit decisions
- pass-image archiving
- manifest writing
- summary writing
- run directory management
- config snapshotting
- stage selection

Operators should focus only on image quality decisions or labeling logic.

## Run Artifacts

Each run writes audit artifacts under `output.run_root/<run_id>/`.

```text
outputs/
  runs/
    20260526-203012/
      config.snapshot.yaml
      manifest.jsonl
      summary.json
      annotations/
      labels.jsonl
      logs/
        pipeline.log
```

Passed images are copied to the configured archive directory rather than being forced under the run directory.

Recommended default layout:

```text
pass_archive_dir/<run_id>/<relative_path>
```

Example:

```text
/DATA/clean/person_dataset_v1/20260526-203012/a/b/0001.jpg
```

The actual copied path is recorded in `manifest.jsonl` as `archive_path`.

## Core Data Types

```python
@dataclass
class Sample:
    sample_id: str
    source_path: Path
    relative_path: Path
    archive_path: Path | None = None
    metadata: dict = field(default_factory=dict)
```

```python
@dataclass
class OperatorResult:
    decision: Literal["PASS", "REJECT", "LABEL", "ERROR"]
    reason: str | None = None
    metrics: dict = field(default_factory=dict)
    labels: dict | None = None
```

`Sample` represents one image moving through the pipeline. `OperatorResult` is the only object an operator returns to the runner.

## Operator Interfaces

### Filter Operators

```python
class FilterOperator:
    name: str

    def setup(self, context: PipelineContext) -> None:
        pass

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        raise NotImplementedError

    def teardown(self, context: PipelineContext) -> None:
        pass
```

### Batch Filter Operators

Some operators need global context. The duplicate detector is the main example because it compares perceptual hashes across multiple images.

```python
class BatchFilterOperator(FilterOperator):
    def process_batch(
        self,
        samples: list[Sample],
        context: PipelineContext
    ) -> dict[str, OperatorResult]:
        raise NotImplementedError
```

Only samples that have not already been rejected by earlier operators should be sent to later batch operators.

### Label Operators

```python
class LabelOperator:
    name: str

    def setup(self, context: PipelineContext) -> None:
        pass

    def process(self, sample: Sample, context: PipelineContext) -> OperatorResult:
        raise NotImplementedError

    def teardown(self, context: PipelineContext) -> None:
        pass
```

The first implementation can keep labeling concurrency inside the label operator. The public interface can stay simple.

## Existing Script Migration

Current scripts map to pluggable operators as follows:

```text
aspect_ratio.py        -> AspectRatioFilter
gray_style.py          -> GrayStyleFilter
blur_filter.py         -> PersonBlurFilter
blur_filter_v2.py      -> CompressionQualityFilter
framebox_filter.py     -> FrameBoxFilter
concate_two_img.py     -> StitchLineFilterV1
concate_two_img_v2.py  -> StitchLineFilterV2
duplicate_filter.py    -> DuplicateFilter, BatchFilterOperator
gemma_labeler.py       -> PersonAttributeLabeler
```

Code retained inside operators:

- image loading and optional resizing
- algorithm-specific scoring
- decision logic
- metrics and reject reasons
- label normalization for labeling operators

Code moved out of operators into the pipeline:

- `argparse`
- recursive directory scanning
- `safe_move`
- `out_bad` handling
- sidecar movement
- pass-image copying
- manifest writing
- progress and summary aggregation

The first stage currently has no same-name sidecars. Sidecars are produced by the labeling stage, so quality filters should not move or copy sidecar files.

## Operator Registry

Operators are loaded by name through a registry.

```python
OPERATOR_REGISTRY = {
    "aspect_ratio": AspectRatioFilter,
    "gray_style": GrayStyleFilter,
    "compression_quality": CompressionQualityFilter,
    "framebox": FrameBoxFilter,
    "stitch_v2": StitchLineFilterV2,
    "person_blur": PersonBlurFilter,
    "duplicate": DuplicateFilter,
    "person_attribute_labeler": PersonAttributeLabeler,
}
```

Adding a new operator should require adding a class and registering it by name, not changing the runner.

## Configuration Format

Use YAML for the first version.

```yaml
input:
  input_dir: /DATA/raw/person_images
  recursive: true
  image_exts: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

output:
  run_root: outputs/runs
  pass_archive_dir: /DATA/clean/person_dataset_v1
  pass_archive_layout: run_subdir   # run_subdir | flat | preserve_relative
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

`label.input_dir: auto` means the label stage reads the pass archive for the selected run.

## CLI

```bash
python -m preprocessing.pipeline run --stage filter --config configs/person_pipeline.yaml
python -m preprocessing.pipeline run --stage label  --config configs/person_pipeline.yaml --run-id 20260526-203012
python -m preprocessing.pipeline run --stage all    --config configs/person_pipeline.yaml
```

Rules:

- `filter` creates a new `run_id` unless one is explicitly provided.
- `all` creates a new `run_id`, runs filter, then labels the passed archive for that run.
- `label --run-id <id>` reads that run's pass archive by default.
- `label.input_dir` may point to any existing clean image directory if the user wants to label data produced elsewhere.

## Manifest

`manifest.jsonl` stores one JSON object per original image.

PASS example:

```json
{
  "sample_id": "abc.jpg",
  "source_path": "/DATA/raw/person_images/abc.jpg",
  "relative_path": "abc.jpg",
  "archive_path": "/DATA/clean/person_dataset_v1/20260527-203012/abc.jpg",
  "status": "PASS",
  "rejected_by": null,
  "reject_reason": null,
  "operator_trace": [
    {"name": "aspect_ratio", "decision": "PASS", "metrics": {"ratio": 1.33}},
    {"name": "gray_style", "decision": "PASS", "metrics": {"max_channel_diff": 42}}
  ]
}
```

REJECT example:

```json
{
  "sample_id": "bad.jpg",
  "source_path": "/DATA/raw/person_images/bad.jpg",
  "relative_path": "bad.jpg",
  "archive_path": null,
  "status": "REJECT",
  "rejected_by": "person_blur",
  "reject_reason": "blur_score_below_threshold",
  "operator_trace": [
    {"name": "aspect_ratio", "decision": "PASS", "metrics": {"ratio": 1.2}},
    {"name": "person_blur", "decision": "REJECT", "metrics": {"score": 55.3, "threshold": 120.0}}
  ]
}
```

ERROR samples should also be recorded in the manifest with `status: "ERROR"`, `archive_path: null`, and an error reason.

## Summary

`summary.json` records run-level counts and timing.

```json
{
  "run_id": "20260526-203012",
  "stage": "filter",
  "input_total": 100000,
  "passed": 82341,
  "rejected": 17659,
  "errors": 0,
  "reject_by_operator": {
    "aspect_ratio": 1200,
    "gray_style": 3411,
    "framebox": 2088,
    "duplicate": 10960
  },
  "started_at": "2026-05-27T20:30:12+08:00",
  "finished_at": "2026-05-27T20:50:46+08:00",
  "duration_seconds": 1234.5
}
```

## Error Handling

- Image read failure: record `ERROR`, do not copy, do not label.
- Operator exception: record `ERROR` for that sample and continue with the next sample by default.
- PASS copy failure: record `ERROR`, do not send that sample to the label stage.
- Label failure: write the failure to the label summary and allow the label stage to be rerun.
- Unknown operator name: fail fast before processing starts.
- Invalid config: fail fast with a clear validation message.
- `overwrite: false`: do not silently replace existing pass archive files. The first implementation should report a path conflict clearly.

## Testing Strategy

Focus first on pipeline behavior and operator contracts.

Required tests:

1. Config loading
   - YAML can be loaded.
   - Disabled operators are skipped.
   - Operator order follows the config.
   - Missing required fields produce clear errors.

2. Filtering short-circuit
   - The first `REJECT` stops later operators for that sample.
   - REJECT images are not copied or moved.
   - PASS images are copied to `pass_archive_dir/<run_id>/`.

3. Manifest
   - Every input image gets one manifest row.
   - PASS rows include `archive_path`.
   - REJECT rows include `rejected_by` and `reject_reason`.
   - `operator_trace` preserves execution order.

4. Run isolation
   - Each run creates an independent run artifact directory.
   - `config.snapshot.yaml` is saved.
   - `summary.json` counts are correct.

5. Label stage
   - `--stage label` can read an existing run's pass archive.
   - Per-image annotation JSON is written.
   - `labels.jsonl` is written.
   - `skip_existing` works.

Operator smoke tests:

- `AspectRatioFilter`: use synthetic images with normal and extreme aspect ratios.
- `GrayStyleFilter`: use synthetic grayscale and colorful images.
- `DuplicateFilter`: use two identical images and verify one is rejected.
- `PersonAttributeLabeler`: use a mock client to test normalization and output writing without relying on a live model service.

## Implementation Plan Boundary

The first implementation should include:

1. Pipeline core
   - `Sample`
   - `OperatorResult`
   - `PipelineContext`
   - operator base classes
   - registry
   - config loader
   - run directory manager

2. Filter operators
   - aspect ratio
   - gray style
   - compression quality
   - framebox
   - stitch line v2
   - person blur
   - duplicate batch filter

3. Archive and audit
   - copy PASS images to configured archive directory
   - write `manifest.jsonl`
   - write `summary.json`
   - save `config.snapshot.yaml`

4. Label stage
   - migrate `gemma_labeler.py` into `PersonAttributeLabeler`
   - support `filter`, `label`, and `all` stages
   - write `annotations/*.json`
   - write `labels.jsonl`

The first version should not implement a DAG engine, complex checkpointing, cross-run caching, or broad filter-stage parallelism. Those can be added after the operator boundary is stable.

## Proposed Package Layout

```text
preprocessing/
  pipeline/
    cli.py
    config.py
    context.py
    runner.py
    registry.py
    types.py
  operators/
    filters/
      aspect_ratio.py
      gray_style.py
      compression.py
      framebox.py
      stitch.py
      blur.py
      duplicate.py
    labelers/
      person_attribute.py
configs/
  person_pipeline.yaml
tests/
```

Existing scripts in `components/` should be treated as migration sources until their logic is represented in operators. They do not need to be deleted in the first implementation.
