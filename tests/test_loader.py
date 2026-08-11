import hashlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import dreamer_tools._eval_worker as eval_worker
import dreamer_tools._extract_worker as extract_worker
import dreamer_tools._load_worker as worker
import dreamer_tools.adapters.instrument as instrumentation
import dreamer_tools.loader as loader
from dreamer_tools._eval_worker import _prepare_config
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


def test_evaluate_runs_isolated_worker(tmp_path: Path, monkeypatch: object) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, PINNED_COMMIT, "")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "evaluated", "")

    monkeypatch.setattr(loader, "_run", fake_run)  # type: ignore[attr-defined]
    result = loader.evaluate(
        FIXTURE,
        _upstream(tmp_path),
        tmp_path / "evaluation",
        steps=100,
        envs=2,
        platform="cpu",
        python=Path(sys.executable),
    )
    assert result.ok
    assert calls[-1][-3:] == ["100", "2", "cpu"]


def test_evaluate_refuses_invalid_options(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    assert not loader.evaluate(FIXTURE, upstream, tmp_path / "out", steps=0).ok
    assert not loader.evaluate(
        FIXTURE, upstream, tmp_path / "out", steps=1, platform="metal"
    ).ok


def test_extract_runs_pinned_worker(tmp_path: Path, monkeypatch: object) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, PINNED_COMMIT, "")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "extracted", "")

    monkeypatch.setattr(loader, "_run", fake_run)  # type: ignore[attr-defined]
    result = loader.extract(
        FIXTURE,
        _upstream(tmp_path),
        tmp_path / "extract.npz",
        steps=8,
        platform="cpu",
        python=Path(sys.executable),
    )
    assert result.ok
    assert calls[-1][-2:] == ["8", "cpu"]


def test_extract_validates_options(tmp_path: Path) -> None:
    upstream = _upstream(tmp_path)
    assert not loader.extract(FIXTURE, upstream, tmp_path / "x.npz", steps=0).ok
    assert not loader.extract(FIXTURE, upstream, tmp_path / "x.bin", steps=1).ok


def test_prepare_evaluation_config(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    data = {"run": {"steps": 999}, "jax": {"platform": "cuda"}}
    result = _prepare_config(
        data,
        output=tmp_path / "output",
        snapshot=snapshot,
        steps=25,
        envs=2,
        platform="cpu",
    )
    assert result["script"] == "eval_only"
    assert result["run"] == {
        "steps": 25,
        "from_checkpoint": str(snapshot),
        "envs": 2,
    }
    assert result["jax"] == {"platform": "cpu"}


def test_prepare_evaluation_config_requires_run_mapping(tmp_path: Path) -> None:
    try:
        _prepare_config(
            {},
            output=tmp_path,
            snapshot=tmp_path,
            steps=1,
            envs=1,
            platform=None,
        )
    except ValueError as exc:
        assert "run mapping" in str(exc)
    else:
        raise AssertionError("expected ValueError")


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


def test_eval_worker_delegates_to_upstream(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic")
    snapshot = tmp_path / "ckpt" / "one"
    snapshot.mkdir(parents=True)
    (snapshot / "done").touch()
    output = tmp_path / "evaluation"
    delegated: list[object] = []

    class FakeConfig(dict[str, object]):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            for key, value in list(self.items()):
                if isinstance(value, dict):
                    self[key] = FakeConfig(value)

        def __getattr__(self, name: str) -> object:
            return self[name]

        def save(self, path: Path) -> None:
            delegated.append(Path(path))

    config_data = {
        "run": {"steps": 10, "usage": {}},
        "jax": {"platform": "cpu"},
        "logger": {"outputs": ["jsonl"]},
        "errfile": False,
        "ipv6": False,
        "replica": 0,
        "replicas": 1,
        "batch_size": 1,
        "batch_length": 1,
        "report_length": 1,
        "consec_train": 1,
        "consec_report": 1,
        "replay_context": 1,
    }

    class FakeYaml:
        def __init__(self, typ: str) -> None:
            assert typ == "safe"

        def load(self, text: str) -> dict[str, object]:
            return config_data

    def eval_only(*args: object) -> None:
        delegated.append(args)

    modules = {
        "elements": SimpleNamespace(Config=FakeConfig, Path=Path),
        "ruamel.yaml": SimpleNamespace(YAML=FakeYaml),
        "portal": SimpleNamespace(setup=lambda **kwargs: None),
        "embodied": SimpleNamespace(run=SimpleNamespace(eval_only=eval_only)),
        "dreamerv3.main": SimpleNamespace(
            make_agent=lambda config: object(),
            make_env=lambda config, index: object(),
            make_logger=lambda config: object(),
        ),
    }
    monkeypatch.setattr(  # type: ignore[attr-defined]
        eval_worker.importlib, "import_module", lambda name: modules[name]
    )
    code = eval_worker.main(
        [
            str(tmp_path),
            str(tmp_path),
            str(config_path),
            str(snapshot),
            str(output),
            "10",
            "1",
            "cpu",
        ]
    )
    assert code == 0
    assert len(delegated) == 2
    assert delegated[-1] == output / "config.yaml"
    assert "Evaluation completed" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_eval_worker_reports_usage_and_import_errors(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    assert eval_worker.main([]) == 2
    monkeypatch.setattr(  # type: ignore[attr-defined]
        eval_worker.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    assert eval_worker.main([str(tmp_path)] * 5 + ["1", "1", "cpu"]) == 1
    assert "ModuleNotFoundError" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_extract_worker_reports_usage_and_import_errors(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    assert extract_worker.main([]) == 2
    monkeypatch.setattr(  # type: ignore[attr-defined]
        extract_worker.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    assert extract_worker.main([str(tmp_path)] * 5 + ["1", "cpu"]) == 1
    assert "ModuleNotFoundError" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_policy_instrumentation_is_source_hash_guarded(monkeypatch: object) -> None:
    class FakeAgent:
        def policy(self: object) -> tuple[None, None, dict[str, object]]:
            out: dict[str, object] = {}
            carry = None
            return carry, None, out

    source = textwrap.dedent(inspect.getsource(FakeAgent.policy))
    digest = hashlib.sha256(source.encode()).hexdigest()
    monkeypatch.setattr(  # type: ignore[attr-defined]
        instrumentation, "POLICY_SOURCE_SHA256", digest
    )
    original = FakeAgent.policy
    instrumentation.instrument_policy(SimpleNamespace(Agent=FakeAgent))  # type: ignore[arg-type]
    assert FakeAgent.policy is not original


def test_extract_worker_writes_bundle(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    import numpy as np

    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic")
    snapshot = tmp_path / "ckpt" / "one"
    snapshot.mkdir(parents=True)
    (snapshot / "done").touch()
    output = tmp_path / "extract.npz"

    class FakeConfig(dict[str, object]):
        def __init__(self, values: dict[str, object]) -> None:
            super().__init__(values)
            self.jax = values["jax"]

    class FakeCheckpoint:
        agent: object

        def load(self, path: Path, keys: list[str]) -> None:
            assert path == snapshot

    class FakeAgent:
        def init_policy(self, size: int) -> None:
            return None

        def policy(
            self, carry: object, obs: dict[str, object], mode: str
        ) -> tuple[object, dict[str, np.ndarray], dict[str, np.ndarray]]:
            return (
                carry,
                {"action": np.zeros((1,), np.int32)},
                {
                    "extract/h": np.zeros((1, 4), np.float32),
                    "extract/z": np.zeros((1, 2, 3), np.float32),
                    "extract/reconstruction/image": np.zeros(
                        (1, 16, 16, 3), np.float32
                    ),
                },
            )

    class FakeDriver:
        def __init__(self, factories: object, parallel: bool) -> None:
            self.closed = False

        def reset(self, init_policy: object) -> None:
            pass

        def __call__(self, policy: object, steps: int) -> None:
            policy(  # type: ignore[operator]
                None,
                {
                    "image": np.zeros((1, 16, 16, 3), np.uint8),
                    "is_first": np.ones((1,), bool),
                },
            )

        def close(self) -> None:
            self.closed = True

    config_data: dict[str, object] = {
        "jax": {"platform": "cpu"},
        "agent": {"imag_length": 5},
        "task": "dummy_disc",
        "seed": 1,
    }

    class FakeYaml:
        def __init__(self, typ: str) -> None:
            pass

        def load(self, text: str) -> dict[str, object]:
            return config_data

    modules = {
        "numpy": np,
        "elements": SimpleNamespace(Config=FakeConfig, Checkpoint=FakeCheckpoint),
        "ruamel.yaml": SimpleNamespace(YAML=FakeYaml),
        "embodied": SimpleNamespace(Driver=FakeDriver),
        "dreamerv3.main": SimpleNamespace(
            make_agent=lambda config: FakeAgent(),
            make_env=lambda config, index: object(),
        ),
        "dreamerv3.agent": SimpleNamespace(Agent=FakeAgent),
    }
    monkeypatch.setattr(  # type: ignore[attr-defined]
        extract_worker.importlib, "import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        extract_worker, "instrument_policy", lambda module: None
    )
    code = extract_worker.main(
        [
            str(tmp_path),
            str(tmp_path),
            str(config_path),
            str(snapshot),
            str(output),
            "1",
            "cpu",
        ]
    )
    assert code == 0
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert "Wrote extraction bundle" in capsys.readouterr().out  # type: ignore[attr-defined]
