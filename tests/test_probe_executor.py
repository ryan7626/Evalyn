from pathlib import Path

import pytest

from evalyn.adapters.base import ProbeArtifact
from evalyn.adapters.pytest import PytestAdapter
from evalyn.probe_executor import (
    ContainerProbeExecutor,
    _looks_invalid,
    _stable_failure_signature,
)


def artifact(source: str = "def test_one():\n    assert True\n") -> ProbeArtifact:
    return PytestAdapter().build_artifact(source)


def test_materialize_uses_adapter_owned_contained_path(tmp_path: Path) -> None:
    item = artifact()

    target = ContainerProbeExecutor._materialize(tmp_path, item)

    assert target.is_relative_to(tmp_path)
    assert target.read_text(encoding="utf-8") == item.source


def test_materialize_rejects_conflicting_content(tmp_path: Path) -> None:
    item = artifact()
    target = tmp_path / item.relative_path
    target.parent.mkdir(parents=True)
    target.write_text("different", encoding="utf-8")

    with pytest.raises(ValueError, match="different content"):
        ContainerProbeExecutor._materialize(tmp_path, item)


def test_invalid_output_detection() -> None:
    assert _looks_invalid("pytest", "ERROR collecting test_file.py")
    assert _looks_invalid("vitest", "Failed to parse source")
    assert not _looks_invalid("pytest", "AssertionError: expected 1")


def test_default_pytest_setup_uses_workspace_venv(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='1'\n")
    executor = ContainerProbeExecutor(
        PytestAdapter(), engine="docker", run_dir=tmp_path / "artifacts"
    )

    commands = executor._default_setup(tmp_path)

    assert commands[0] == ("python", "-m", "venv", ".evalyn/venv")
    assert commands[-1][-2:] == ("-e", ".")


def test_setup_environment_requires_every_allowlisted_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NPM_TOKEN", raising=False)
    executor = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "artifacts",
        setup_env_names=("NPM_TOKEN",),
    )

    with pytest.raises(RuntimeError, match="NPM_TOKEN"):
        executor._setup_secret_environment()


def test_setup_environment_reads_only_named_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPM_TOKEN", "package-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    executor = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "artifacts",
        setup_env_names=("NPM_TOKEN",),
    )

    assert executor._setup_secret_environment() == {"NPM_TOKEN": "package-secret"}


def test_execution_budget_exhaustion_is_fail_closed(tmp_path: Path) -> None:
    executor = ContainerProbeExecutor(
        PytestAdapter(),
        engine="docker",
        run_dir=tmp_path / "artifacts",
        timeout_seconds=1,
    )
    executor._deadline = 0

    with pytest.raises(TimeoutError, match="budget"):
        executor.remaining_timeout()


def test_pytest_failure_signature_ignores_only_timing_summary() -> None:
    first = _stable_failure_signature(
        "pytest",
        "E assert 106 == 105\n===== 1 failed in 0.02s =====\n",
        "",
    )
    repeated = _stable_failure_signature(
        "pytest",
        "E assert 106 == 105\n1 failed in 0.19s\n",
        "",
    )
    different = _stable_failure_signature(
        "pytest",
        "E assert 107 == 105\n===== 1 failed in 0.02s =====\n",
        "",
    )

    assert first == repeated
    assert first != different


def test_vitest_failure_signature_ignores_timing_lines() -> None:
    first = _stable_failure_signature(
        "vitest",
        "expected 106 to be 105\n Start at 12:01:02\n Duration 80ms\n",
        "",
    )
    repeated = _stable_failure_signature(
        "vitest",
        "expected 106 to be 105\n Start at 12:02:03\n Duration 120ms\n",
        "",
    )

    assert first == repeated
