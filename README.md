# dreamer-tools

Inspect a trained Dreamer world model and visualize its dreams in minutes.

`dreamer-tools` is a thin, read-only toolkit for discovering and diagnosing
DreamerV3 run directories. Version 0.1 deliberately does **not** train models,
implement Dreamer, download checkpoints, or claim that a discovered checkpoint
can be restored. The initially supported upstream revision is experimental.

## Install

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Use

```sh
dreamer doctor /path/to/run
dreamer manifest /path/to/run --output dreamer-run.yaml
dreamer manifest /path/to/run --dreamer-commit <40-hex-sha>
dreamer load-check /path/to/run --upstream /path/to/dreamerv3 --python /path/to/env/bin/python
```

`doctor` works on incomplete runs and exits with status 0 for complete
discovery, 1 for incomplete provenance or warnings, and 2 when required run
artifacts are missing or unreadable. `manifest` writes a portable, versioned
description; relative artifact paths remain relative to the run root.

## What is checked

- the run path and upstream-shaped `ckpt`, `config.yaml`, and `metrics.jsonl`
- readable YAML configuration and JSONL metrics
- task, seed, requested training steps, latest recorded step, and model preset
- upstream commit, Python/environment package versions, and artifact paths
- a few high-confidence consistency checks, such as manifest paths escaping the
  run root or a recorded step exceeding the configured run limit

This is structural inspection only. It does not import JAX or DreamerV3, inspect
checkpoint tensors, prove parameter-tree compatibility, validate environment
assets/ROMs, or restore a model. `load-check` is the explicit exception: it runs
the pinned upstream loader in a separate Python process and verifies that the
agent state can be restored, but does not execute a policy. See
[the architecture decision](docs/architecture.md).
Provenance values are never filled from the inspection machine: supply
`--python-version`, `--environment-version NAME=VERSION`, and `--model-preset`
when they are absent from `config.yaml`.

## Real checkpoint smoke test

Prepare a clean checkout and a separate environment using upstream's own
instructions. The checkout must be exactly at the experimental pinned commit;
the loader rejects modified source trees and other revisions.

```sh
git -C /path/to/dreamerv3 checkout e3f02248693a79dc8b0ebd62c93683888ddaccfe
dreamer load-check /path/to/run \
  --upstream /path/to/dreamerv3 \
  --python /path/to/dreamerv3-env/bin/python
```

The worker reconstructs the observation and action spaces through upstream's
`make_agent()`, resolves the checkpoint's `latest` snapshot, and asks
`elements.Checkpoint` to load only `agent`. Environment packages and assets are
therefore still required for the configured task.

## Develop

```sh
ruff format --check .
ruff check .
mypy src
pytest
```

## Compatibility

The experimental adapter is pinned to DreamerV3 commit
[`e3f02248693a79dc8b0ebd62c93683888ddaccfe`](https://github.com/danijar/dreamerv3/commit/e3f02248693a79dc8b0ebd62c93683888ddaccfe).
No other revision is claimed compatible. Discovery may still be useful for
other revisions, but `doctor` will label them unverified.

Primary upstream evidence:

- [`dreamerv3/main.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/main.py)
  saves `config.yaml` and configures `metrics.jsonl`/`scores.jsonl`.
- [`embodied/run/train.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/run/train.py)
  registers `step`, `agent`, and `replay` in the `ckpt` checkpoint.
- [`embodied/run/eval_only.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/run/eval_only.py)
  loads the `agent` key for evaluation-only execution.
- [`dreamerv3/configs.yaml`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml)
  defines presets and the architecture-shaping configuration.
