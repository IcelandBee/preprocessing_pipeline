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
