import sys
from dataclasses import replace
from pathlib import Path

import pytest

import bench.runner as benchmark_runner
from bench.runner import BenchmarkCase, CorpusError


def _case(tmp_path: Path, *, command: tuple[str, ...]) -> BenchmarkCase:
    fixture_root = tmp_path / "fixture"
    (fixture_root / "base").mkdir(parents=True)
    probe = fixture_root / "probe.py"
    probe.write_text("# generated probe\n", encoding="utf-8")
    return BenchmarkCase(
        case_id="bounded-case",
        framework="pytest",
        summary="bounded execution",
        fixture_root=fixture_root,
        probe_path=probe,
        probe_target=Path(".evalyn/probe.py"),
        command=command,
        availability=(sys.executable, "--version"),
        expected={"base": "pass", "buggy": "fail", "fixed": "pass"},
        repetitions=1,
    )


def test_benchmark_execution_bounds_output_and_drops_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    script = (
        "import os, sys; "
        "sys.stderr.write('BEGIN-' + ('x' * 100_000) + '-' + "
        "os.environ.get('OPENAI_API_KEY', 'unset')); "
        "raise SystemExit(7)"
    )
    case = _case(tmp_path, command=(sys.executable, "-c", script))

    outcome = benchmark_runner._execute_revision(case, "base")

    result = outcome.results[0]
    assert result.exit_code == 7
    assert len(result.stderr.encode("utf-8")) <= benchmark_runner._MAX_CAPTURE_BYTES
    assert result.stderr.startswith("BEGIN-")
    assert result.stderr.endswith("-unset")
    assert "output truncated" in result.stderr
    assert "ambient-secret" not in result.stderr


def test_benchmark_availability_check_drops_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    case = _case(tmp_path, command=(sys.executable, "-c", "pass"))
    case = replace(
        case,
        availability=(
            sys.executable,
            "-c",
            "import os, sys; raise SystemExit(bool(os.getenv('OPENAI_API_KEY')))",
        ),
    )

    assert benchmark_runner._tool_available(case)


def test_benchmark_timeout_is_an_explicit_corpus_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark_runner, "_COMMAND_TIMEOUT_SECONDS", 0.05)
    case = _case(
        tmp_path,
        command=(sys.executable, "-c", "import time; time.sleep(60)"),
    )

    with pytest.raises(CorpusError, match="timed out"):
        benchmark_runner._execute_revision(case, "base")
