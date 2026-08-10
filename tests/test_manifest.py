from pathlib import Path

import yaml

from dreamer_tools.adapters.dreamerv3 import PINNED_COMMIT, discover
from dreamer_tools.manifest import SCHEMA, build_manifest, dump_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "complete"


def test_manifest_is_portable_and_versioned(tmp_path: Path) -> None:
    run = discover(
        FIXTURE,
        dreamer_commit=PINNED_COMMIT,
        model_preset="size50m",
        python_version="3.11.9",
        environment_versions={"crafter": "1.8.3"},
    )
    data = build_manifest(run)
    assert data["schema"] == SCHEMA
    assert data["schema_version"] == 1
    assert data["artifacts"]["checkpoint"] == "ckpt"
    assert data["run"]["seed"] == 7
    assert data["environment"]["python"] == "3.11.9"
    output = tmp_path / "nested" / "run.yaml"
    dump_manifest(data, output)
    assert yaml.safe_load(output.read_text()) == data
