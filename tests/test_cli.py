from pathlib import Path

import yaml

import dreamer_tools.cli as cli
from dreamer_tools.adapters.dreamerv3 import PINNED_COMMIT
from dreamer_tools.cli import main
from dreamer_tools.loader import LoadCheckResult

FIXTURES = Path(__file__).parent / "fixtures"


def test_doctor_complete(capsys: object) -> None:
    code = main(
        [
            "doctor",
            str(FIXTURES / "complete"),
            "--dreamer-commit",
            PINNED_COMMIT,
            "--model-preset",
            "size50m",
            "--python-version",
            "3.11.9",
            "--environment-version",
            "crafter=1.8.3",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "✓ Found checkpoint" in output
    assert "Status: ready" in output


def test_doctor_incomplete(capsys: object) -> None:
    code = main(["doctor", str(FIXTURES / "incomplete")])
    assert code == 2
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "✗ Checkpoint missing" in output
    assert "Next action:" in output


def test_manifest_command(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "manifest.yaml"
    code = main(
        [
            "manifest",
            str(FIXTURES / "complete"),
            "--output",
            str(output),
            "--dreamer-commit",
            PINNED_COMMIT,
            "--model-preset",
            "size50m",
            "--python-version",
            "3.11.9",
            "--environment-version",
            "crafter=1.8.3",
        ]
    )
    assert code == 0
    assert yaml.safe_load(output.read_text())["schema_version"] == 1
    assert "Wrote manifest" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_help(capsys: object) -> None:
    assert main([]) == 0
    assert "doctor" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_load_check_command(monkeypatch: object, capsys: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "check_load",
        lambda *args, **kwargs: LoadCheckResult(True, "loaded"),
    )
    assert main(["load-check", "/run", "--upstream", "/upstream"]) == 0
    assert "✓ loaded" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_evaluate_command(monkeypatch: object, capsys: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        cli,
        "evaluate",
        lambda *args, **kwargs: LoadCheckResult(True, "evaluated"),
    )
    code = main(
        [
            "evaluate",
            "/run",
            "--upstream",
            "/upstream",
            "--output",
            "/evaluation",
            "--steps",
            "100",
        ]
    )
    assert code == 0
    assert "✓ evaluated" in capsys.readouterr().out  # type: ignore[attr-defined]
