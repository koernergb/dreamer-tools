import json
from pathlib import Path

import numpy as np

from dreamer_tools.bundle import inspect_bundle, load_arrays, write_bundle
from dreamer_tools.media import _uint8_frames, export_videos


def test_bundle_round_trip_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "extract.npz"
    arrays = {
        "h": np.zeros((2, 3, 4), np.float32),
        "z": np.ones((2, 3, 2, 4), np.float32),
        "posterior_logits": np.ones((2, 3, 2, 4), np.float32),
    }
    info = write_bundle(path, arrays, metadata={"task": "dummy_disc"})
    assert info.metadata["schema_version"] == 1
    assert info.arrays["h"] == ((2, 3, 4), "float32")
    loaded = load_arrays(path)
    np.testing.assert_array_equal(loaded["z"], arrays["z"])


def test_bundle_rejects_unknown_object_and_bad_frames(tmp_path: Path) -> None:
    cases = (
        {"unknown": np.zeros((1,))},
        {"h": np.array([object()], dtype=object)},
        {"reconstruction/image": np.zeros((2, 3, 4))},
    )
    for index, arrays in enumerate(cases):
        try:
            write_bundle(tmp_path / f"bad{index}.npz", arrays, metadata={})
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_bundle_detects_metadata_tampering(tmp_path: Path) -> None:
    path = tmp_path / "extract.npz"
    write_bundle(path, {"h": np.zeros((2, 3))}, metadata={})
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["arrays"]["h"]["shape"] = [99]
    path.with_suffix(".json").write_text(json.dumps(metadata))
    try:
        inspect_bundle(path)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_video_export_writes_reconstruction_and_imagination(tmp_path: Path) -> None:
    bundle = tmp_path / "extract.npz"
    frames = np.zeros((3, 16, 16, 3), np.uint8)
    frames[:, :, :, 1] = 180
    write_bundle(
        bundle,
        {
            "reconstruction/image": frames,
            "imagination/image": frames.astype(np.float32) / 255,
        },
        metadata={},
    )
    outputs = export_videos(bundle, tmp_path / "videos", fps=5)
    assert [path.name for path in outputs] == [
        "imagination-image.mp4",
        "reconstruction-image.mp4",
    ]
    assert all(path.stat().st_size > 0 for path in outputs)


def test_frame_normalization_rejects_invalid_values() -> None:
    for value in (
        np.array([[[[2.0, 0.0, 0.0]]]]),
        np.array([[[[np.nan, 0.0, 0.0]]]]),
        np.array([[[[1, 2, 3]]]], dtype=np.int16),
    ):
        try:
            _uint8_frames(value, np)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
