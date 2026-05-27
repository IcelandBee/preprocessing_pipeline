from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Decision = Literal["PASS", "REJECT", "LABEL", "ERROR"]


@dataclass
class Sample:
    sample_id: str
    source_path: Path
    relative_path: Path
    archive_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorResult:
    decision: Decision
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] | None = None

    @classmethod
    def pass_(cls, metrics: dict[str, Any] | None = None) -> "OperatorResult":
        return cls(decision="PASS", metrics=metrics or {})

    @classmethod
    def reject(
        cls,
        reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> "OperatorResult":
        return cls(decision="REJECT", reason=reason, metrics=metrics or {})

    @classmethod
    def label(cls, labels: dict[str, Any], metrics: dict[str, Any] | None = None) -> "OperatorResult":
        return cls(decision="LABEL", labels=labels, metrics=metrics or {})

    @classmethod
    def error(cls, reason: str, metrics: dict[str, Any] | None = None) -> "OperatorResult":
        return cls(decision="ERROR", reason=reason, metrics=metrics or {})
