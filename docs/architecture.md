# Architecture and compatibility decision (ADR-001)

Status: accepted for v0.1, 2026-08-10.

## Decision

Keep the core independent of DreamerV3 and put upstream naming/layout knowledge
in `adapters/dreamerv3.py`. The core owns immutable discovery records,
diagnostics, and the versioned manifest. The CLI is a presentation layer.
Loading and future extraction code must consume the discovery result through
the adapter boundary rather than importing internals from the CLI or manifest.

The first loading seam is `load-check`. Its small parent process validates a
clean pinned checkout, then launches `_load_worker.py` with an explicitly chosen
Python environment. This isolates heavyweight JAX/upstream imports from normal
inspection. The worker uses upstream `make_agent()` and `elements.Checkpoint`
rather than interpreting pickle or parameter-tree formats itself.

The first adapter is **experimental** and pinned to upstream commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. We depend on that source revision
only for the opt-in load check; we do not vendor it or reproduce its model code.
The user supplies a separate environment containing that checkout's declared
dependencies. The worker constructs the same observation/action spaces from
the saved config and calls upstream's checkpoint API.

## Evidence and consequences

At the pinned revision, upstream saves resolved configuration to `config.yaml`
and creates JSONL logs in
[`dreamerv3/main.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/main.py).
The standard training loop registers `step`, `agent`, and `replay` at `ckpt` in
[`embodied/run/train.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/run/train.py),
and evaluation loads the `agent` key in
[`embodied/run/eval_only.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/run/eval_only.py).
Parallel execution uses component paths beneath `ckpt`, so discovery accepts
both shapes without interpreting checkpoint bytes.
The upstream `elements`
[`Checkpoint`](https://github.com/danijar/elements/blob/main/elements/checkpoint.py)
stores a `latest` pointer to a complete snapshot marked by `done`; `load-check`
resolves that container before requesting the `agent` key.

Upstream warns in its
[`README`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/README.md)
that checkpoint reloads require a compatible config. Therefore file discovery
is never reported as load compatibility. Exact resolved config and provenance
are first-class manifest data, while unknown facts remain null rather than being
inferred. Environment package versions are recorded as a mapping because task
suites (Crafter, Atari, DMC, and others) have different compatibility surfaces.

## Extension seams

Later evaluation, reconstruction/imagination video, HTML reporting, and latent
extraction (`h`, `z`, logits, predictions) should be separate services over a
verified loader. They are explicitly outside v0.1.
