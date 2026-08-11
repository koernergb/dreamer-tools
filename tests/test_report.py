from pathlib import Path

from dreamer_tools.adapters.dreamerv3 import PINNED_COMMIT, discover
from dreamer_tools.report import _read_metrics, _sparkline, write_report

FIXTURE = Path(__file__).parent / "fixtures" / "complete"


def test_report_is_self_contained_and_escaped(tmp_path: Path) -> None:
    run = discover(
        FIXTURE,
        dreamer_commit=PINNED_COMMIT,
        model_preset="size50m<script>",
        python_version="3.11",
        environment_versions={"crafter": "1.8.3"},
    )
    output = tmp_path / "nested" / "report.html"
    write_report(run, output)
    document = output.read_text()
    assert "<!doctype html>" in document
    assert "episode/score" in document
    assert "<svg" in document
    assert "size50m&lt;script&gt;" in document
    assert "https://" not in document


def test_metrics_reader_handles_missing_and_invalid(tmp_path: Path) -> None:
    assert _read_metrics(None) == []
    path = tmp_path / "metrics.jsonl"
    path.write_text("invalid")
    assert _read_metrics(path) == []


def test_sparkline_handles_single_constant_value() -> None:
    svg = _sparkline([2.0])
    assert 'points="0.0,58.0"' in svg
