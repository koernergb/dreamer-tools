"""Opt-in orchestration for a real upstream checkpoint load smoke test."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dreamer_tools.adapters.dreamerv3 import PINNED_COMMIT, discover


@dataclass(frozen=True)
class LoadCheckResult:
    """Outcome of validating and invoking the isolated upstream loader."""

    ok: bool
    message: str
    detail: str | None = None


def check_load(
    run_path: Path,
    upstream_path: Path,
    *,
    python: Path | None = None,
    timeout: int = 300,
) -> LoadCheckResult:
    """Load only the agent state using an exact, clean upstream checkout."""
    if timeout <= 0:
        return LoadCheckResult(False, "Timeout must be greater than zero")
    run = discover(run_path, dreamer_commit=PINNED_COMMIT)
    if not run.config_path or not run.checkpoint:
        return LoadCheckResult(
            False,
            "Run is missing a checkpoint or readable configuration",
            "Run `dreamer doctor PATH` and resolve its errors first.",
        )

    upstream = upstream_path.expanduser().resolve()
    validation = _validate_upstream(upstream)
    if validation:
        return validation

    executable = (python or Path(sys.executable)).expanduser().resolve()
    if not executable.is_file():
        return LoadCheckResult(
            False,
            f"Python executable not found: {executable}",
            "Pass --python from an environment containing the pinned "
            "DreamerV3 dependencies.",
        )

    worker = Path(__file__).with_name("_load_worker.py")
    command = [
        str(executable),
        str(worker),
        str(upstream),
        str(run.root),
        str(run.config_path),
        str(run.checkpoint),
    ]
    try:
        completed = _run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        return LoadCheckResult(
            False,
            f"Checkpoint load exceeded {timeout} seconds",
            "Retry with --timeout SECONDS or inspect JAX device initialization.",
        )
    except OSError as exc:
        return LoadCheckResult(False, "Could not start checkpoint loader", str(exc))

    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode:
        return LoadCheckResult(
            False,
            "Upstream rejected the checkpoint",
            output or f"Loader exited with status {completed.returncode}",
        )
    return LoadCheckResult(True, "Checkpoint agent state loaded successfully", output)


def _validate_upstream(upstream: Path) -> LoadCheckResult | None:
    if not (upstream / "dreamerv3" / "main.py").is_file():
        return LoadCheckResult(
            False,
            f"Not a DreamerV3 checkout: {upstream}",
            "Pass --upstream pointing to the repository root.",
        )
    try:
        revision = _run(["git", "-C", str(upstream), "rev-parse", "HEAD"], timeout=10)
        dirty = _run(["git", "-C", str(upstream), "status", "--porcelain"], timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LoadCheckResult(
            False, "Could not inspect upstream Git checkout", str(exc)
        )
    if revision.returncode:
        return LoadCheckResult(
            False, "Could not read upstream Git revision", revision.stderr.strip()
        )
    actual = revision.stdout.strip()
    if actual != PINNED_COMMIT:
        return LoadCheckResult(
            False,
            f"Unsupported upstream revision: {actual or 'unknown'}",
            f"Check out experimental revision {PINNED_COMMIT}.",
        )
    if dirty.returncode or dirty.stdout.strip():
        return LoadCheckResult(
            False,
            "Upstream checkout has local modifications",
            "Use a clean checkout so the verified revision describes "
            "the executed code.",
        )
    return None


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - arguments are explicit and shell is disabled.
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
