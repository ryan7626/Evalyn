"""Tests for deterministic pytest adapter detection and artifacts."""

import pytest

from evalyn.adapters.base import AdapterError
from evalyn.adapters.pytest import PytestAdapter


def test_detects_pytest_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        encoding="utf-8",
    )

    assert PytestAdapter().detect(tmp_path) is True


def test_detects_pytest_test_file(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    assert PytestAdapter().detect(tmp_path) is True


def test_empty_repo_is_not_detected(tmp_path):
    assert PytestAdapter().detect(tmp_path) is False


def test_builds_adapter_owned_path_and_argv():
    artifact = PytestAdapter(python_executable="python3").build_artifact(
        "def test_generated():\n    assert True"
    )

    assert artifact.adapter == "pytest"
    assert artifact.relative_path.as_posix() == (
        f".evalyn/probes/pytest/test_evalyn_{artifact.probe_id}.py"
    )
    assert artifact.command == (
        "python3",
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "--maxfail=1",
        "-p",
        "no:cacheprovider",
        "-p",
        "evalyn_pytest_reporter",
        artifact.relative_path.as_posix(),
    )
    assert artifact.source not in artifact.command


def test_rejects_assertion_free_or_process_probe():
    adapter = PytestAdapter()

    with pytest.raises(AdapterError, match="meaningful assertion"):
        adapter.build_artifact("def test_value():\n    value()")
    with pytest.raises(AdapterError, match="prohibited"):
        adapter.build_artifact("import subprocess\ndef test_value():\n    assert True")


def test_rejects_invalid_python_probe():
    with pytest.raises(AdapterError, match="valid Python"):
        PytestAdapter().build_artifact("def test_value(:\n    assert True")
