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
