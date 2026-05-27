# preprocessing-pipeline

Pluggable image preprocessing and labeling pipeline.

## Workflow

1. Quality filters run in the order listed in the YAML config.
2. The first `REJECT` stops later filters for that image.
3. Rejected images are not copied or moved.
4. Passed images are copied to `output.pass_archive_dir`.
5. Labeling can run after filtering or as a separate stage.

## Launch Scripts

| Script | Stage | When to Use |
|---|---|---|
| `scripts/run_full.sh` | filter + label | 数据已就绪，一次性跑完全流程产出标注数据集 |
| `scripts/run_filter.sh` | filter only | 先筛选确认效果再决定是否打标；调试筛选参数；VLM 服务不可用时 |
| `scripts/run_label.sh` | label only | 筛选已完成，补跑打标；不同团队分阶段负责；重跑失败打标（配合 `skip_existing`） |
| `scripts/run_quick.sh` | filter (light) | 开发调试新算子；快速验证 pipeline 能跑通；参数调优需要秒级反馈 |

### Usage

```bash
# Full pipeline — 一次性跑完筛选+打标
./scripts/run_full.sh [CONFIG] [RUN_ID]

# Filter only — 仅筛选
./scripts/run_filter.sh [CONFIG] [RUN_ID]

# Label only — 仅打标（必须指定 RUN_ID 以定位筛选输出）
./scripts/run_label.sh <CONFIG> <RUN_ID>

# Quick test — 快速试跑轻量配置
./scripts/run_quick.sh [CONFIG] [INPUT_DIR] [RUN_ID]
```

All scripts default to `configs/all_filters_pipeline.yaml`. `run_label.sh` requires `RUN_ID` because the label stage needs to locate the filter output directory.

## Direct CLI

```bash
python -m preprocessing.pipeline.cli run --stage filter --config configs/all_filters_pipeline.yaml
python -m preprocessing.pipeline.cli run --stage label --config configs/all_filters_pipeline.yaml --run-id 20260527-203012
python -m preprocessing.pipeline.cli run --stage all --config configs/all_filters_pipeline.yaml
```

## Config Files

| File | Description |
|---|---|
| `configs/person_pipeline.yaml` | 原始配置，4 个 filter + person_attribute_labeler |
| `configs/all_filters_pipeline.yaml` | 全量配置，7 个 filter + person_attribute_labeler |

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