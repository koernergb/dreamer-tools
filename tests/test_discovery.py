from pathlib import Path

from dreamer_tools.adapters.dreamerv3 import PINNED_COMMIT, discover
from dreamer_tools.model import Severity

FIXTURES = Path(__file__).parent / "fixtures"


def test_complete_fixture_discovers_artifacts_and_fields() -> None:
    run = discover(
        FIXTURES / "complete",
        dreamer_commit=PINNED_COMMIT,
        model_preset="size50m",
        python_version="3.11.9",
        environment_versions={"crafter": "1.8.3"},
    )
    assert run.severity == Severity.INFO
    assert run.checkpoint and run.checkpoint.name == "ckpt"
    assert run.model_preset == "size50m"
    assert run.python_version == "3.11.9"
    assert run.latest_metrics == {"step": 900, "episode/score": 2.0}


def test_incomplete_fixture_produces_actionable_errors() -> None:
    run = discover(FIXTURES / "incomplete")
    assert run.severity == Severity.ERROR
    assert any(item.message == "Checkpoint missing" for item in run.diagnostics)
    assert any(item.action for item in run.diagnostics)


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    run = discover(tmp_path / "absent")
    assert run.severity == Severity.ERROR
    assert "not found" in run.diagnostics[0].message


def test_bad_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "ckpt").write_text("x")
    (tmp_path / "config.yaml").write_text("- not-a-map")
    (tmp_path / "metrics.jsonl").write_text("nope")
    run = discover(tmp_path, dreamer_commit="short")
    messages = " ".join(item.message for item in run.diagnostics)
    assert "Configuration is unreadable" in messages
    assert "Metrics are unreadable" in messages
    assert "40-character" in messages


def test_unverified_commit_and_step_mismatch_warn(tmp_path: Path) -> None:
    (tmp_path / "ckpt").write_text("x")
    (tmp_path / "config.yaml").write_text("task: dummy_disc\nrun: {steps: 1}\n")
    (tmp_path / "metrics.jsonl").write_text('{"step": 2}\n')
    run = discover(
        tmp_path,
        dreamer_commit="a" * 40,
        model_preset="custom",
        python_version="3.11",
        environment_versions={"dummy": "1"},
    )
    assert run.severity == Severity.WARNING
    assert any("exceeds" in item.message for item in run.diagnostics)
