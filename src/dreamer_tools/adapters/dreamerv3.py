"""Discovery adapter for the pinned upstream DreamerV3 layout."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from dreamer_tools.model import Diagnostic, Discovery, Severity

PINNED_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
UPSTREAM_URL = "https://github.com/danijar/dreamerv3"
_SHA = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def discover(
    path: Path,
    *,
    dreamer_commit: str | None = None,
    model_preset: str | None = None,
    python_version: str | None = None,
    environment_versions: dict[str, str] | None = None,
) -> Discovery:
    root = path.expanduser().resolve()
    diagnostics: list[Diagnostic] = []
    if not root.is_dir():
        diagnostics.append(
            Diagnostic(
                Severity.ERROR,
                f"Run directory not found: {root}",
                "Check the path and try again",
            )
        )
        return Discovery(root=root, diagnostics=tuple(diagnostics))

    checkpoint = _first_existing(root, ("ckpt", "checkpoint"))
    config_path = _first_existing(root, ("config.yaml", "config.yml"))
    metrics_path = _first_existing(root, ("metrics.jsonl",))

    _presence(
        diagnostics,
        checkpoint,
        "checkpoint",
        "Restore or copy the run checkpoint into ckpt",
    )
    _presence(
        diagnostics,
        config_path,
        "configuration",
        "Copy the resolved training config.yaml into the run",
    )
    _presence(
        diagnostics,
        metrics_path,
        "metrics",
        "Copy metrics.jsonl into the run, if available",
        required=False,
    )

    config = _read_config(config_path, diagnostics)
    latest = _read_latest_metrics(metrics_path, diagnostics)
    commit = dreamer_commit or _provenance_value(config, "dreamer_commit")
    if commit and not _SHA.fullmatch(commit):
        diagnostics.append(
            Diagnostic(
                Severity.ERROR,
                "Upstream Dreamer commit is not a full 40-character Git SHA",
                "Provide --dreamer-commit with the exact training revision",
            )
        )
    elif not commit:
        diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                "Upstream Dreamer commit not recorded",
                "Provide --dreamer-commit or add it to the manifest",
            )
        )
    elif commit != PINNED_COMMIT:
        diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                f"Upstream revision {commit[:12]} is not verified",
                f"Only {PINNED_COMMIT[:12]} is experimentally supported",
            )
        )
    else:
        diagnostics.append(
            Diagnostic(Severity.INFO, "Pinned experimental upstream revision recorded")
        )

    preset = model_preset or _provenance_value(config, "model_preset")
    if not preset:
        diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                "Model preset not recorded",
                "Provide --model-preset used during training",
            )
        )

    recorded_python = python_version or _provenance_value(config, "python_version")
    if not recorded_python:
        diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                "Training Python version not recorded",
                "Provide --python-version from the training environment",
            )
        )

    env_versions = dict(environment_versions or {})
    task = config.get("task") if config else None
    suite = str(task).split("_", 1)[0] if task else None
    if suite and suite not in env_versions:
        diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                f"{_display_suite(suite)} version missing",
                f"Provide --environment-version {suite}=VERSION",
            )
        )

    _check_steps(config, latest, diagnostics)
    return Discovery(
        root=root,
        checkpoint=checkpoint,
        config_path=config_path,
        metrics_path=metrics_path,
        config=config,
        latest_metrics=latest,
        diagnostics=tuple(diagnostics),
        upstream_commit=commit,
        model_preset=preset,
        python_version=recorded_python,
        environment_versions=env_versions,
    )


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    return next((root / name for name in names if (root / name).exists()), None)


def _presence(
    items: list[Diagnostic],
    path: Path | None,
    label: str,
    action: str,
    *,
    required: bool = True,
) -> None:
    if path:
        items.append(Diagnostic(Severity.INFO, f"Found {label}"))
    else:
        items.append(
            Diagnostic(
                Severity.ERROR if required else Severity.WARNING,
                f"{label.capitalize()} missing",
                action,
            )
        )


def _read_config(path: Path | None, items: list[Diagnostic]) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level value is not a mapping")
        return value
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        items.append(
            Diagnostic(
                Severity.ERROR,
                f"Configuration is unreadable: {exc}",
                "Replace config.yaml with the resolved training configuration",
            )
        )
        return None


def _read_latest_metrics(
    path: Path | None, items: list[Diagnostic]
) -> dict[str, Any] | None:
    if not path:
        return None
    latest: dict[str, Any] | None = None
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {number} is not a JSON object")
            latest = value
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        items.append(
            Diagnostic(
                Severity.ERROR,
                f"Metrics are unreadable: {exc}",
                "Repair or replace metrics.jsonl",
            )
        )
    return latest


def _provenance_value(config: dict[str, Any] | None, key: str) -> str | None:
    value = (
        (config or {}).get("provenance", {}).get(key)
        if isinstance((config or {}).get("provenance"), dict)
        else None
    )
    return str(value) if value is not None else None


def _check_steps(
    config: dict[str, Any] | None,
    latest: dict[str, Any] | None,
    items: list[Diagnostic],
) -> None:
    requested = (
        (config or {}).get("run", {}).get("steps")
        if isinstance((config or {}).get("run"), dict)
        else None
    )
    observed = (latest or {}).get("step")
    if (
        isinstance(requested, (int, float))
        and isinstance(observed, (int, float))
        and observed > requested
    ):
        items.append(
            Diagnostic(
                Severity.WARNING,
                "Latest metric step exceeds configured run.steps",
                "Confirm that config.yaml and metrics.jsonl came from the same run",
            )
        )


def _display_suite(suite: str) -> str:
    return {
        "crafter": "Crafter",
        "atari": "Atari",
        "dmc": "DMC",
        "procgen": "Procgen",
    }.get(suite, suite)
