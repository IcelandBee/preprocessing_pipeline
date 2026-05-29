from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

from preprocessing.pipeline.config import PipelineConfig
from preprocessing.pipeline.context import PipelineContext
from preprocessing.pipeline.io import copy_sample_to_archive, ensure_run_dirs, enumerate_images, write_json, write_jsonl
from preprocessing.pipeline.operators import BatchFilterOperator, FilterOperator, LabelOperator
from preprocessing.pipeline.registry import DEFAULT_REGISTRY, OperatorRegistry
from preprocessing.pipeline.types import OperatorResult, Sample


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_workers(workers: str | int) -> int:
    if workers == "auto":
        return min(os.cpu_count() or 4, 8)
    return int(workers)


def _run_sample_chain(args: tuple) -> dict[str, Any]:
    """在子进程中执行单个样本的完整 per-sample filter chain。

    Returns pickle-safe dict with:
        sample_id, trace, final_decision, rejected_by, reject_reason
    """
    sample_id, source_path_str, relative_path_str, op_configs, run_id, run_dir_str, pass_archive_dir_str, pass_archive_layout = args

    from preprocessing.pipeline.registry import build_default_registry

    registry = build_default_registry()
    source_path = Path(source_path_str)
    relative_path = Path(relative_path_str)

    # 统一加载 PIL Image 一次
    try:
        pil_image = Image.open(source_path)
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "trace": [{"name": "image_load", "decision": "ERROR", "reason": f"image_open_failed: {exc!r}", "metrics": {}}],
            "final_decision": "ERROR",
            "rejected_by": "image_load",
            "reject_reason": f"image_open_failed: {exc!r}",
        }

    context = PipelineContext(
        run_id=run_id,
        run_dir=Path(run_dir_str),
        pass_archive_dir=Path(pass_archive_dir_str),
        pass_archive_layout=pass_archive_layout,
        pil_image=pil_image,
    )
    sample = Sample(sample_id=sample_id, source_path=source_path, relative_path=relative_path)

    trace: list[dict[str, Any]] = []
    final_decision = "PASS"
    rejected_by = None
    reject_reason = None

    for op_cfg in op_configs:
        operator = registry.create(op_cfg["name"], op_cfg["params"])
        try:
            result = operator.process(sample, context)
        except Exception as exc:
            result = OperatorResult.error(f"operator_exception: {exc!r}")

        trace.append({
            "name": op_cfg["name"],
            "decision": result.decision,
            "reason": result.reason,
            "metrics": result.metrics,
        })

        if result.decision in ("REJECT", "ERROR"):
            final_decision = result.decision
            rejected_by = op_cfg["name"]
            reject_reason = result.reason
            break

    # 关闭共享的 PIL Image
    try:
        pil_image.close()
    except Exception:
        pass

    return {
        "sample_id": sample_id,
        "trace": trace,
        "final_decision": final_decision,
        "rejected_by": rejected_by,
        "reject_reason": reject_reason,
    }


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        registry: OperatorRegistry | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or DEFAULT_REGISTRY
        prefix = self._resolve_run_id_prefix()
        timestamp = generate_run_id()
        self.run_id = run_id or (f"{prefix}-{timestamp}" if prefix else timestamp)
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

        # 分离 per-sample 和 batch operators
        operators = [
            self.registry.create(operator_config.name, operator_config.params)
            for operator_config in self.config.filter.enabled_operators
        ]
        per_sample_ops = [op for op in operators if isinstance(op, FilterOperator) and not isinstance(op, BatchFilterOperator)]
        batch_ops = [op for op in operators if isinstance(op, BatchFilterOperator)]

        # Worker 数量
        workers = _resolve_workers(self.config.filter.workers)

        # 构建 per-sample operator 配置列表（用于传给子进程）
        per_sample_op_configs = []
        for op_cfg in self.config.filter.enabled_operators:
            op_instance = self.registry.create(op_cfg.name, op_cfg.params)
            if isinstance(op_instance, FilterOperator) and not isinstance(op_instance, BatchFilterOperator):
                per_sample_op_configs.append({"name": op_cfg.name, "params": op_cfg.params})

        # 当使用自定义 registry 时退化为串行，因为子进程无法访问自定义 operator
        use_parallel = workers > 1 and self.registry is DEFAULT_REGISTRY

        # 并行或串行执行 per-sample filter chain
        if per_sample_op_configs and len(samples) > 0:
            if use_parallel:
                chain_args = [
                    (
                        s.sample_id,
                        str(s.source_path),
                        s.relative_path.as_posix(),
                        per_sample_op_configs,
                        self.run_id,
                        str(self.context.run_dir),
                        str(self.context.pass_archive_dir),
                        self.context.pass_archive_layout,
                    )
                    for s in samples
                ]
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_run_sample_chain, args): args[0] for args in chain_args}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Filter", unit="img"):
                        try:
                            result = future.result()
                            state = states[result["sample_id"]]
                            state["operator_trace"] = result["trace"]
                            state["status"] = result["final_decision"] if result["final_decision"] in ("REJECT", "ERROR") else "PENDING"
                            state["rejected_by"] = result["rejected_by"]
                            state["reject_reason"] = result["reject_reason"]
                        except Exception as exc:
                            sample_id = futures[future]
                            state = states[sample_id]
                            state["operator_trace"] = [{
                                "name": "process_pool",
                                "decision": "ERROR",
                                "reason": f"worker_crashed: {exc!r}",
                                "metrics": {},
                            }]
                            state["status"] = "ERROR"
                            state["rejected_by"] = "process_pool"
                            state["reject_reason"] = f"worker_crashed: {exc!r}"
            else:
                # workers=1: 退化为串行执行（保持现有行为）
                for op in per_sample_ops:
                    active_samples = [state["sample"] for state in states.values() if state["status"] == "PENDING"]
                    if not active_samples:
                        continue
                    op.setup(self.context)
                    try:
                        for sample in tqdm(active_samples, desc=f"Filter:{op.name}", unit="img"):
                            try:
                                result = op.process(sample, self.context)
                            except Exception as exc:
                                result = OperatorResult.error(f"operator_exception: {exc!r}")
                            self._record_filter_result(states[sample.sample_id], op.name, result)
                    finally:
                        op.teardown(self.context)

        # 串行尾置执行 batch operators（DuplicateFilter 等）
        for batch_op in batch_ops:
            active_samples = [state["sample"] for state in states.values() if state["status"] == "PENDING"]
            if not active_samples:
                continue
            tqdm.write(f"Batch filter: {batch_op.name} on {len(active_samples)} images...")
            results = batch_op.process_batch(active_samples, self.context)
            for sample in active_samples:
                result = results.get(sample.sample_id, OperatorResult.pass_())
                self._record_filter_result(states[sample.sample_id], batch_op.name, result)

        for state in tqdm(states.values(), desc="Archive", unit="img"):
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

    def run_label(self) -> dict[str, Any]:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started = time.monotonic()
        ensure_run_dirs(self.context.run_dir)
        self._write_config_snapshot()

        input_dir = self._label_input_dir()
        samples = enumerate_images(
            input_dir,
            recursive=True,
            image_exts=self.config.input.image_exts,
        )
        operator = self.registry.create(self.config.label.operator.name, self.config.label.operator.params)
        if not isinstance(operator, LabelOperator):
            raise TypeError(f"Configured label operator is not a LabelOperator: {self.config.label.operator.name}")

        # skip_existing: 过滤已有 annotation 的样本
        if self.config.label.skip_existing:
            skipped = 0
            filtered_samples = []
            for sample in samples:
                annotation_path = self.context.annotation_dir / f"{sample.source_path.stem}.json"
                if annotation_path.exists():
                    skipped += 1
                else:
                    filtered_samples.append(sample)
            if skipped:
                tqdm.write(f"skip_existing: skipped {skipped} already-annotated images")
            samples = filtered_samples

        annotations: list[dict[str, Any]] = []
        failures = 0
        workers = _resolve_workers(self.config.label.workers)
        operator.setup(self.context)
        try:
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(operator.process, s, self.context): s for s in samples}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Label", unit="img"):
                        sample = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            tqdm.write(f"[ERR] {sample.source_path.name}: worker_crashed: {exc!r}")
                            failures += 1
                            continue
                        if result.decision == "LABEL" and result.labels is not None:
                            output = {
                                "meta": {
                                    "operator": operator.name,
                                    "input_image": sample.source_path.name,
                                },
                                "annotation": result.labels,
                            }
                            annotation_path = self.context.annotation_dir / f"{sample.source_path.stem}.json"
                            write_json(annotation_path, output)
                            annotations.append(result.labels)
                        else:
                            tqdm.write(f"[ERR] {sample.source_path.name}: {result.decision} — {result.reason}")
                            failures += 1
            else:
                for sample in tqdm(samples, desc="Label", unit="img"):
                    result = operator.process(sample, self.context)
                    if result.decision == "LABEL" and result.labels is not None:
                        output = {
                            "meta": {
                                "operator": operator.name,
                                "input_image": sample.source_path.name,
                            },
                            "annotation": result.labels,
                        }
                        annotation_path = self.context.annotation_dir / f"{sample.source_path.stem}.json"
                        write_json(annotation_path, output)
                        annotations.append(result.labels)
                    else:
                        tqdm.write(f"[ERR] {sample.source_path.name}: {result.decision} — {result.reason}")
                        failures += 1
        finally:
            operator.teardown(self.context)

        write_jsonl(self.context.labels_jsonl_path, annotations)
        summary = {
            "run_id": self.run_id,
            "stage": "label",
            "input_total": len(samples),
            "labeled": len(annotations),
            "failed": failures,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        write_json(self.context.run_dir / "label_summary.json", summary)
        return summary

    def run_all(self) -> dict[str, Any]:
        filter_summary = self.run_filter()
        label_summary = self.run_label()
        return {"filter": filter_summary, "label": label_summary}

    def _resolve_run_id_prefix(self) -> str:
        prefix = self.config.output.run_id_prefix
        if prefix == "auto":
            return self.config.input.input_dir.name
        return prefix

    def _label_input_dir(self) -> Path:
        if str(self.config.label.input_dir) == "auto":
            return self.config.output.pass_archive_dir / self.run_id
        return Path(self.config.label.input_dir)

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
                "run_id_prefix": self.config.output.run_id_prefix,
            },
            "filter": {
                "workers": self.config.filter.workers,
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