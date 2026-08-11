import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import dreamer_tools._load_worker as worker
import dreamer_tools.loader as loader
from dreamer_tools._load_worker import _resolve_snapshot
from dreamer_tools.adapters.dreamerv3 import PINNED_COMMIT

FIXTURE = Path(__file__).parent / "fixtures" / "complete"


def _upstream(tmp_path: Path) -> Path:
    upstream = tmp_path / "upstream"
    (upstream / "dreamerv3").mkdir(parents=True)
    (upstream / "dreamerv3" / "main.py").write_text("# synthetic")
    return upstream


def test_load_check_runs_isolated_worker(tmp_path: Path, monkeypatch: object) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, PINNED_COMMIT + "\n", "")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "loaded", "")

    monkeypatch.setattr(loader, "_run", fake_run)  # type: ignore[attr-defined]
    result = loader.check_load(
        FIXTURE, _upstream(tmp_path), python=Path(sys.executable)
    )
    assert result.ok
    assert "loaded" in (result.detail or "")
    assert calls[-1][0] == str(Path(sys.executable).resolve())


def test_load_check_rejects_wrong_revision(tmp_path: Path, monkeypatch: object) -> None:
    def fake_run(
        command: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "a" * 40, "")

    monkeypatch.setattr(loader, "_run", fake_run)  # type: ignore[attr-defined]
    result = loader.check_load(FIXTURE, _upstream(tmp_path))
    assert not result.ok
    assert "Unsupported upstream revision" in result.message


def test_load_check_reports_dirty_checkout(tmp_path: Path, monkeypatch: object) -> None:
    def fake_run(
        command: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        output = PINNED_COMMIT if "rev-parse" in command else " M dreamerv3/main.py"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(loader, "_run", fake_run)  # type: ignore[attr-defined]
    result = loader.check_load(FIXTURE, _upstream(tmp_path))
    assert not result.ok
    assert "local modifications" in result.message


def test_load_check_reports_worker_failure(tmp_path: Path, monkeypatch: object) -> None:
    def fake_run(
        command: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, PINNED_COMMIT, "")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "missing dependency")

    monkeypatch.setattr(loader, "_run", fake_run)  # type: ignore[attr-defined]
    result = loader.check_load(FIXTURE, _upstream(tmp_path))
    assert not result.ok
    assert result.detail == "missing dependency"


def test_load_check_validates_inputs(tmp_path: Path) -> None:
    assert not loader.check_load(FIXTURE, tmp_path, timeout=0).ok
    assert not loader.check_load(tmp_path / "missing", tmp_path).ok
    assert not loader.check_load(FIXTURE, tmp_path).ok


def test_resolve_snapshot_layouts(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "done").touch()
    assert _resolve_snapshot(direct) == direct

    snapshot = tmp_path / "container" / "20260810-0001"
    snapshot.mkdir(parents=True)
    (snapshot / "done").touch()
    (snapshot.parent / "latest").write_text(snapshot.name)
    assert _resolve_snapshot(snapshot.parent) == snapshot

    parallel = tmp_path / "parallel" / "agent" / "one"
    parallel.mkdir(parents=True)
    (parallel / "done").touch()
    (parallel.parent / "latest").write_text("one")
    assert _resolve_snapshot(tmp_path / "parallel") == parallel


def test_resolve_snapshot_rejects_incomplete(tmp_path: Path) -> None:
    (tmp_path / "latest").write_text("missing")
    try:
        _resolve_snapshot(tmp_path)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_worker_loads_agent_with_upstream_apis(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("task: dummy_disc\n")
    snapshot = tmp_path / "ckpt" / "one"
    snapshot.mkdir(parents=True)
    (snapshot / "done").touch()
    (snapshot.parent / "latest").write_text("one")
    loaded: list[tuple[Path, list[str]]] = []

    class FakeConfig(dict[str, object]):
        def update(self, **values: object) -> "FakeConfig":
            super().update(values)
            return self

    class FakeCheckpoint:
        agent: object

        def load(self, path: Path, keys: list[str]) -> None:
            loaded.append((Path(path), keys))

    class FakeYaml:
        def __init__(self, typ: str) -> None:
            assert typ == "safe"

        def load(self, text: str) -> dict[str, str]:
            assert "dummy_disc" in text
            return {"task": "dummy_disc"}

    modules = {
        "elements": SimpleNamespace(Config=FakeConfig, Checkpoint=FakeCheckpoint),
        "ruamel.yaml": SimpleNamespace(YAML=FakeYaml),
        "dreamerv3.main": SimpleNamespace(make_agent=lambda config: object()),
    }
    monkeypatch.setattr(  # type: ignore[attr-defined]
        worker.importlib, "import_module", lambda name: modules[name]
    )
    code = worker.main(
        [str(tmp_path), str(tmp_path), str(config), str(snapshot.parent)]
    )
    assert code == 0
    assert loaded == [(snapshot, ["agent"])]
    assert "Loaded agent checkpoint" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_worker_reports_bad_usage_and_upstream_error(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    assert worker.main([]) == 2
    monkeypatch.setattr(  # type: ignore[attr-defined]
        worker.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    assert worker.main([str(tmp_path)] * 4) == 1
    assert "ModuleNotFoundError" in capsys.readouterr().err  # type: ignore[attr-defined]
