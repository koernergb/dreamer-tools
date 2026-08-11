# dreamer-tools

Inspect a trained Dreamer world model and visualize its dreams in minutes.

`dreamer-tools` is a thin toolkit for discovering and diagnosing DreamerV3 run
directories. It does **not** train models, implement Dreamer, or download
checkpoints. Inspection is read-only; opt-in load and evaluation commands
delegate execution to a separately installed, pinned upstream checkout. The
initially supported upstream revision is experimental.

## Install

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,media]'
```

## Use

```sh
dreamer doctor /path/to/run
dreamer manifest /path/to/run --output dreamer-run.yaml
dreamer manifest /path/to/run --dreamer-commit <40-hex-sha>
dreamer report /path/to/run --output report.html
dreamer load-check /path/to/run --upstream /path/to/dreamerv3 --python /path/to/env/bin/python
dreamer evaluate /path/to/run --upstream /path/to/dreamerv3 --python /path/to/env/bin/python --output ./eval-run --steps 10000
dreamer bundle-info extraction.npz
dreamer extract /path/to/run --upstream /path/to/dreamerv3 --python /path/to/env/bin/python --output extraction.npz --steps 100
dreamer export-videos extraction.npz --output-dir videos
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

## Experimental evaluation-only execution

After `load-check` succeeds, run a bounded evaluation into a new directory:

```sh
dreamer evaluate /path/to/run \
  --upstream /path/to/dreamerv3 \
  --python /path/to/dreamerv3-env/bin/python \
  --output /path/to/evaluations/run-001 \
  --steps 10000 --envs 1 --platform cpu
```

The output path must not exist, so prior results cannot be overwritten. This
delegates policy execution, environments, checkpoint loading, and metrics to
the pinned upstream implementation. It performs no training. CI verifies the
orchestration with synthetic doubles; real checkpoint compatibility remains
experimental until an integration fixture is tested and documented.

## Portable extraction bundles and videos

Extraction bundles use compressed NPZ arrays plus a JSON sidecar with schema
`dreamer-tools/extraction-bundle`, version 1. They never use pickle. Stable
array names cover `h`, `z`, `prior_logits`, `posterior_logits`, reward and
continuation predictions, and namespaced observation/reconstruction/imagination
frames. Extraction instruments the exact pinned upstream policy before JAX
compilation and refuses to run if that method's source hash changes. It does not
patch files in the upstream checkout. Install the optional renderer and run:

```sh
python -m pip install 'dreamer-tools[media]'
dreamer extract /path/to/run \
  --upstream /path/to/dreamerv3 \
  --python /path/to/dreamerv3-env/bin/python \
  --output extraction.npz --steps 100 --platform cpu
dreamer bundle-info extraction.npz
dreamer export-videos extraction.npz --output-dir videos --fps 15
dreamer report /path/to/run --bundle extraction.npz --video videos/imagination-image.mp4 --output report.html
```

Each extraction step records posterior `h`, `z`, posterior logits, the prior
logits computed from that step's deterministic state, reward and continuation
predictions, decoder reconstructions, and a policy-conditioned imagination of
the configured `agent.imag_length`. Arrays retain their batch/time axes and the
JSON sidecar records the task, seed, revision, rollout length, and horizon.

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
