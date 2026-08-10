# Architecture and compatibility decision (ADR-001)

Status: accepted for v0.1, 2026-08-10.

## Decision

Keep the core independent of DreamerV3 and put upstream naming/layout knowledge
in `adapters/dreamerv3.py`. The core owns immutable discovery records,
diagnostics, and the versioned manifest. The CLI is a presentation layer.
Loading and future extraction code must consume the discovery result through
the adapter boundary rather than importing internals from the CLI or manifest.

The first adapter is **experimental** and pinned to upstream commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. We depend on that source revision
at runtime only when real loading is added; we do not vendor it or reproduce its
model code. A future loader should check out/install that exact revision in an
isolated environment, construct the same observation/action spaces from the
saved config, then call upstream's checkpoint API.

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

