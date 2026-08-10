"""Portable Dreamer run manifest schema and serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dreamer_tools.adapters.dreamerv3 import UPSTREAM_URL
from dreamer_tools.model import Discovery

SCHEMA = "dreamer-tools/run-manifest"
SCHEMA_VERSION = 1


def build_manifest(run: Discovery) -> dict[str, Any]:
    config = run.config or {}
    metrics = run.latest_metrics or {}
    run_config = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream": {
            "repository": UPSTREAM_URL,
            "git_commit": run.upstream_commit,
            "compatibility": "experimental",
        },
        "run": {
            "task": config.get("task"),
            "seed": config.get("seed"),
            "model_preset": run.model_preset,
            "configured_steps": run_config.get("steps"),
            "recorded_steps": metrics.get("step"),
        },
        "environment": {
            "python": run.python_version,
            "packages": dict(sorted(run.environment_versions.items())),
        },
        "artifacts": {
            "checkpoint": _relative(run.checkpoint, run.root),
            "configuration": _relative(run.config_path, run.root),
            "metrics": _relative(run.metrics_path, run.root),
        },
        "configuration": run.config,
    }


def dump_manifest(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _relative(path: Path | None, root: Path) -> str | None:
    return path.relative_to(root).as_posix() if path else None
