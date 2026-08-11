"""Execute upstream DreamerV3 evaluation-only without importing it in the CLI."""

from __future__ import annotations

import importlib
import sys
from functools import partial as bind
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from dreamer_tools._load_worker import _read_config, _resolve_snapshot


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if len(arguments) != 8:
        print(
            "Internal usage: _eval_worker UPSTREAM RUN CONFIG CHECKPOINT "
            "OUTPUT STEPS ENVS PLATFORM",
            file=sys.stderr,
        )
        return 2
    upstream, run_root, config_path, checkpoint, output = map(Path, arguments[:5])
    steps, envs = map(int, arguments[5:7])
    platform = arguments[7]
    sys.path.insert(0, str(upstream))
    try:
        elements = importlib.import_module("elements")
        yaml = importlib.import_module("ruamel.yaml")
        portal = importlib.import_module("portal")
        embodied = importlib.import_module("embodied")
        upstream_main = importlib.import_module("dreamerv3.main")
        snapshot = _resolve_snapshot(checkpoint)
        data = _prepare_config(
            _read_config(config_path, yaml),
            output=output,
            snapshot=snapshot,
            steps=steps,
            envs=envs,
            platform=platform or None,
        )
        config = elements.Config(data)
        portal.setup(
            errfile=config.errfile and elements.Path(output) / "error",
            clientkw={"logging_color": "cyan"},
            serverkw={"logging_color": "cyan"},
            ipv6=config.ipv6,
        )
        args = elements.Config(
            **config.run,
            replica=config.replica,
            replicas=config.replicas,
            logdir=config.logdir,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            report_length=config.report_length,
            consec_train=config.consec_train,
            consec_report=config.consec_report,
            replay_context=config.replay_context,
        )
        embodied.run.eval_only(
            bind(upstream_main.make_agent, config),
            bind(upstream_main.make_env, config),
            bind(upstream_main.make_logger, config),
            args,
        )
        config.save(elements.Path(output) / "config.yaml")
    except Exception as exc:  # Upstream exceptions are the diagnostic payload.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Evaluation completed: {output}")
    return 0


def _prepare_config(
    data: dict[str, Any],
    *,
    output: Path,
    snapshot: Path,
    steps: int,
    envs: int,
    platform: str | None,
) -> dict[str, Any]:
    run = data.get("run")
    if not isinstance(run, dict):
        raise ValueError("config.yaml is missing the run mapping")
    run.update(from_checkpoint=str(snapshot), steps=steps, envs=envs)
    if platform:
        jax = data.get("jax")
        if not isinstance(jax, dict):
            raise ValueError("config.yaml is missing the jax mapping")
        jax["platform"] = platform
    data.update(logdir=str(output), script="eval_only")
    return data


if __name__ == "__main__":
    sys.exit(main())
