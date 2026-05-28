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
  run_id_prefix: auto
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

    assert config.input.input_dir.as_posix() == "/data/raw"
    assert config.input.recursive is True
    assert config.input.image_exts == [".jpg", ".png"]
    assert config.output.pass_archive_dir.as_posix() == "/data/clean"
    assert config.filter.operators[0].name == "aspect_ratio"
    assert config.filter.enabled_operators[0].name == "aspect_ratio"
    assert config.label.operator.name == "person_attribute_labeler"


def test_filter_config_workers_default(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
input:
  input_dir: /data/raw
output:
  run_root: outputs/runs
  pass_archive_dir: /data/clean
filter:
  short_circuit: true
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    assert config.filter.workers == "auto"


def test_filter_config_workers_explicit(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
input:
  input_dir: /data/raw
output:
  run_root: outputs/runs
  pass_archive_dir: /data/clean
filter:
  short_circuit: true
  workers: 4
""",
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)
    assert config.filter.workers == 4


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
