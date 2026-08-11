"""Executed by a DreamerV3 environment to load an agent checkpoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if len(arguments) != 4:
        print(
            "Internal usage: _load_worker UPSTREAM RUN CONFIG CHECKPOINT",
            file=sys.stderr,
        )
        return 2
    upstream, run_root, config_path, checkpoint = map(Path, arguments)
    sys.path.insert(0, str(upstream))

    try:
        elements = importlib.import_module("elements")
        yaml = importlib.import_module("ruamel.yaml")
        upstream_main = importlib.import_module("dreamerv3.main")

        config_data = _read_config(config_path, yaml)
        config = elements.Config(config_data).update(
            logdir=str(run_root), script="eval_only"
        )
        agent = upstream_main.make_agent(config)
        snapshot = _resolve_snapshot(checkpoint)
        cp = elements.Checkpoint()
        cp.agent = agent
        cp.load(snapshot, keys=["agent"])
    except Exception as exc:  # Upstream exceptions are the diagnostic payload.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded agent checkpoint: {snapshot}")
    return 0


def _read_config(path: Path, yaml_module: Any) -> dict[str, Any]:
    data = yaml_module.YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping")
    return data


def _resolve_snapshot(checkpoint: Path) -> Path:
    candidates = [checkpoint]
    if checkpoint.is_dir() and (checkpoint / "agent").is_dir():
        candidates.insert(0, checkpoint / "agent")
    for candidate in candidates:
        if (candidate / "done").is_file():
            return candidate
        latest = candidate / "latest"
        if latest.is_file():
            snapshot = candidate / latest.read_text(encoding="utf-8").strip()
            if (snapshot / "done").is_file():
                return snapshot
            raise ValueError(f"Checkpoint latest pointer is incomplete: {snapshot}")
    raise ValueError(
        f"No complete checkpoint snapshot found below {checkpoint}; "
        "expected latest and done files"
    )


if __name__ == "__main__":
    sys.exit(main())
