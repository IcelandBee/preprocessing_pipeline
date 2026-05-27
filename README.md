# preprocessing-pipeline

Pluggable image preprocessing and labeling pipeline.

```bash
python -m preprocessing.pipeline.cli run --stage filter --config configs/person_pipeline.yaml
python -m preprocessing.pipeline.cli run --stage label --config configs/person_pipeline.yaml --run-id 20260527-203012
python -m preprocessing.pipeline.cli run --stage all --config configs/person_pipeline.yaml
```
