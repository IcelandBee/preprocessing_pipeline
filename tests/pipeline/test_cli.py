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
