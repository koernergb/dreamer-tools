"""Core records shared by adapters, manifests, and the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any


class Severity(IntEnum):
    """Diagnostic impact, ordered for exit status calculation."""

    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True)
class Diagnostic:
    severity: Severity
    message: str
    action: str | None = None


@dataclass(frozen=True)
class Discovery:
    root: Path
    checkpoint: Path | None = None
    config_path: Path | None = None
    metrics_path: Path | None = None
    config: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    upstream_commit: str | None = None
    model_preset: str | None = None
    python_version: str | None = None
    environment_versions: dict[str, str] = field(default_factory=dict)

    @property
    def severity(self) -> Severity:
        return max((item.severity for item in self.diagnostics), default=Severity.INFO)
