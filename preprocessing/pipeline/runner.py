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
