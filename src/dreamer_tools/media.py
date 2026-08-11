"""Video export from portable extraction bundles."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

from dreamer_tools.bundle import FRAME_PREFIXES, load_arrays


def export_videos(bundle: Path, output_dir: Path, *, fps: int = 15) -> list[Path]:
    if fps <= 0:
        raise ValueError("FPS must be greater than zero")
    np, imageio = _media_modules()
    arrays = load_arrays(bundle)
    frames = {
        key: value for key, value in arrays.items() if key.startswith(FRAME_PREFIXES)
    }
    if not frames:
        raise ValueError("Bundle contains no reconstruction or imagination frames")
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key, value in sorted(frames.items()):
        video = _uint8_frames(value, np)
        if video.ndim == 5:
            video = video.transpose(1, 2, 0, 3, 4)
            video = video.reshape(video.shape[0], video.shape[1], -1, video.shape[-1])
        target = output / f"{_slug(key)}.mp4"
        with imageio.get_writer(target, fps=fps, codec="libx264", quality=8) as writer:
            for frame in video:
                writer.append_data(frame)
        written.append(target)
    return written


def _uint8_frames(value: Any, np: Any) -> Any:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Frames must be uint8 or floating point, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Frames contain NaN or infinite values")
    if array.size and (array.min() < 0 or array.max() > 1):
        raise ValueError("Floating-point frames must be normalized to [0, 1]")
    return np.clip(array * 255, 0, 255).astype(np.uint8)


def _slug(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")


def _media_modules() -> tuple[Any, Any]:
    try:
        imageio = importlib.import_module("imageio.v2")
        numpy = importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            "Video export requires `pip install 'dreamer-tools[media]'`"
        ) from exc
    return numpy, imageio
