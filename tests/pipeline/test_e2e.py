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


def test_parallel_filter_matches_sequential(tmp_path, make_image):
    """并行模式的结果必须与串行模式一致"""
    raw = tmp_path / "raw"
    for i in range(6):
        color = (i * 40, 0, 0) if i < 3 else (0, i * 40, 0)
        make_image(raw / f"img{i}.jpg", size=(80, 80), color=color)

    config_template = f"""
input:
  input_dir: {raw.as_posix()}
  recursive: true
  image_exts: [".jpg"]
filter:
  short_circuit: true
  operators:
    - name: aspect_ratio
      params:
        ratio: 2.0
    - name: gray_style
      params:
        threshold: 1.0
label:
  enabled: false
"""

    # 串行运行 (workers=1)
    config_seq = tmp_path / "config_seq.yaml"
    config_seq.write_text(
        config_template.replace("__RUN_ROOT__", (tmp_path / "runs_seq").as_posix())
        .replace("__ARCHIVE__", (tmp_path / "clean_seq").as_posix())
        + f"""
output:
  run_root: {(tmp_path / "runs_seq").as_posix()}
  pass_archive_dir: {(tmp_path / "clean_seq").as_posix()}
  pass_archive_layout: run_subdir
  overwrite: true
filter:
  workers: 1
  short_circuit: true
  operators:
    - name: aspect_ratio
      params:
        ratio: 2.0
    - name: gray_style
      params:
        threshold: 1.0
label:
  enabled: false
""",
        encoding="utf-8",
    )

    # 并行运行 (workers=2)
    config_par = tmp_path / "config_par.yaml"
    config_par.write_text(
        f"""
input:
  input_dir: {raw.as_posix()}
  recursive: true
  image_exts: [".jpg"]
output:
  run_root: {(tmp_path / "runs_par").as_posix()}
  pass_archive_dir: {(tmp_path / "clean_par").as_posix()}
  pass_archive_layout: run_subdir
  overwrite: true
filter:
  workers: 2
  short_circuit: true
  operators:
    - name: aspect_ratio
      params:
        ratio: 2.0
    - name: gray_style
      params:
        threshold: 1.0
label:
  enabled: false
""",
        encoding="utf-8",
    )

    exit_seq = main(["run", "--stage", "filter", "--config", str(config_seq), "--run-id", "seq"])
    exit_par = main(["run", "--stage", "filter", "--config", str(config_par), "--run-id", "par"])
    assert exit_seq == 0
    assert exit_par == 0

    summary_seq = json.loads((tmp_path / "runs_seq" / "seq" / "summary.json").read_text(encoding="utf-8"))
    summary_par = json.loads((tmp_path / "runs_par" / "par" / "summary.json").read_text(encoding="utf-8"))
    assert summary_seq["passed"] == summary_par["passed"]
    assert summary_seq["rejected"] == summary_par["rejected"]
    assert summary_seq["input_total"] == summary_par["input_total"]
