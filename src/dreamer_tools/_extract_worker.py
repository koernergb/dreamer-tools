"""Collect pinned DreamerV3 latents and predictions into a portable bundle."""

from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from dreamer_tools._load_worker import _read_config, _resolve_snapshot
from dreamer_tools.adapters.instrument import instrument_policy
from dreamer_tools.bundle import write_bundle


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if len(arguments) != 7:
        print(
            "Internal usage: _extract_worker UPSTREAM RUN CONFIG CHECKPOINT "
            "OUTPUT STEPS PLATFORM",
            file=sys.stderr,
        )
        return 2
    upstream, run_root, config_path, checkpoint, output = map(Path, arguments[:5])
    steps = int(arguments[5])
    platform = arguments[6]
    sys.path.insert(0, str(upstream))
    driver = None
    try:
        np = importlib.import_module("numpy")
        elements = importlib.import_module("elements")
        yaml = importlib.import_module("ruamel.yaml")
        embodied = importlib.import_module("embodied")
        upstream_main = importlib.import_module("dreamerv3.main")
        agent_module = importlib.import_module("dreamerv3.agent")
        instrument_policy(agent_module)
        data = _read_config(config_path, yaml)
        data["logdir"] = str(run_root)
        data["script"] = "eval_only"
        jax = data.get("jax")
        if not isinstance(jax, dict):
            raise ValueError("config.yaml is missing the jax mapping")
        jax["precompile"] = False
        if platform:
            jax["platform"] = platform
        config = elements.Config(data)
        agent = upstream_main.make_agent(config)
        cp = elements.Checkpoint()
        cp.agent = agent
        cp.load(_resolve_snapshot(checkpoint), keys=["agent"])
        captured: dict[str, list[Any]] = defaultdict(list)

        def policy(carry: Any, obs: dict[str, Any]) -> tuple[Any, Any, Any]:
            carry, actions, outputs = agent.policy(carry, obs, mode="eval")
            for key in list(outputs):
                if key.startswith("extract/"):
                    captured[key.removeprefix("extract/")].append(
                        np.asarray(outputs.pop(key))
                    )
            for key, value in obs.items():
                array = np.asarray(value)
                if array.dtype == np.uint8 and array.ndim == 4:
                    captured[f"observation/{key}"].append(array)
            return carry, actions, outputs

        driver = embodied.Driver(
            [lambda: upstream_main.make_env(config, 0)], parallel=False
        )
        driver.reset(agent.init_policy)
        driver(policy, steps=steps)
        arrays = {
            key: np.concatenate(values, axis=0) for key, values in captured.items()
        }
        write_bundle(
            output,
            arrays,
            metadata={
                "upstream_commit": "e3f02248693a79dc8b0ebd62c93683888ddaccfe",
                "run_root": str(run_root),
                "task": data.get("task"),
                "seed": data.get("seed"),
                "rollout_steps": steps,
                "imagination_horizon": data.get("agent", {}).get("imag_length"),
            },
        )
    except Exception as exc:  # Upstream exceptions are the diagnostic payload.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if driver is not None:
            driver.close()
    print(f"Wrote extraction bundle: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
