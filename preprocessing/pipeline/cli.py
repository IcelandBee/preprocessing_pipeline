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
    run_parser.add_argument("--workers", default=None, type=int, help="Number of parallel workers (overrides config)")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_pipeline_config(args.config)
    if args.workers is not None:
        config.filter.workers = args.workers
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
