"""Versioned, pickle-free extraction bundle contract."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "dreamer-tools/extraction-bundle"
BUNDLE_VERSION = 1
KNOWN_ARRAYS = frozenset(
    {
        "h",
        "z",
        "prior_logits",
        "posterior_logits",
        "reward_prediction",
        "continuation_prediction",
    }
)
FRAME_PREFIXES = ("reconstruction/", "imagination/", "observation/")


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    arrays: dict[str, tuple[tuple[int, ...], str]]


def write_bundle(
    path: Path,
    arrays: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> BundleInfo:
    np = _numpy()
    target = path.expanduser().resolve()
    if target.suffix != ".npz":
        raise ValueError("Extraction bundle path must end in .npz")
    if not arrays:
        raise ValueError("Extraction bundle must contain at least one array")
    normalized = {str(key): np.asarray(value) for key, value in arrays.items()}
    _validate_arrays(normalized)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **normalized)
    payload = {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_VERSION,
        **dict(metadata),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(normalized.items())
        },
    }
    metadata_path = target.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inspect_bundle(target)


def inspect_bundle(path: Path) -> BundleInfo:
    np = _numpy()
    target = path.expanduser().resolve()
    metadata_path = target.with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Bundle metadata is unreadable: {exc}") from exc
    if (
        metadata.get("schema") != BUNDLE_SCHEMA
        or metadata.get("schema_version") != BUNDLE_VERSION
    ):
        raise ValueError("Unsupported extraction bundle schema")
    try:
        with np.load(target, allow_pickle=False) as values:
            arrays = {
                key: (tuple(int(x) for x in values[key].shape), str(values[key].dtype))
                for key in values.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"Bundle arrays are unreadable: {exc}") from exc
    declared = metadata.get("arrays")
    actual = {
        key: {"shape": list(shape), "dtype": dtype}
        for key, (shape, dtype) in sorted(arrays.items())
    }
    if declared != actual:
        raise ValueError("Bundle metadata does not match stored arrays")
    return BundleInfo(target, metadata_path, metadata, arrays)


def load_arrays(path: Path) -> dict[str, Any]:
    np = _numpy()
    inspect_bundle(path)
    with np.load(path.expanduser().resolve(), allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _validate_arrays(arrays: Mapping[str, Any]) -> None:
    for key, value in arrays.items():
        if key not in KNOWN_ARRAYS and not key.startswith(FRAME_PREFIXES):
            raise ValueError(f"Unknown extraction array: {key}")
        if value.dtype.hasobject:
            raise ValueError(f"Object arrays are not portable: {key}")
        if value.ndim == 0:
            raise ValueError(f"Extraction arrays must have a time or batch axis: {key}")
        if key.startswith(FRAME_PREFIXES) and (
            value.ndim not in (4, 5) or value.shape[-1] not in (1, 3, 4)
        ):
            raise ValueError(f"Frame array must be [T,H,W,C] or [B,T,H,W,C]: {key}")


def _numpy() -> Any:
    try:
        numpy = importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "Extraction bundles require `pip install 'dreamer-tools[media]'`"
        ) from exc
    return numpy
