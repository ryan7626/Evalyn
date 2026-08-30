"""Tests for structured, isolated command execution."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from evalyn.execution.process import CommandResult, CommandRunner


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


class TestStructuredArguments:
    @pytest.mark.parametrize("argv", ["echo unsafe", b"echo unsafe", bytearray(b"echo")])
    def test_rejects_shell_text(self, workspace: Path, argv: object) -> None:
        with (
            CommandRunner(workspace) as runner,
            pytest.raises(TypeError, match="sequence of strings"),
        ):
            runner.run(argv)  # type: ignore[arg-type]

    def test_rejects_iterators_and_non_string_elements(self, workspace: Path) -> None:
        with CommandRunner(workspace) as runner:
            with pytest.raises(TypeError, match="sequence of strings"):
                runner.run(iter([sys.executable, "--version"]))  # type: ignore[arg-type]
            with pytest.raises(TypeError, match="every argv element"):
                runner.run([sys.executable, 123])  # type: ignore[list-item]

    def test_rejects_empty_and_nul_arguments(self, workspace: Path) -> None:
        with CommandRunner(workspace) as runner:
            with pytest.raises(ValueError, match="cannot be empty"):
                runner.run([])
            with pytest.raises(ValueError, match="NUL"):
                runner.run([sys.executable, "bad\0argument"])

    def test_arguments_are_not_interpreted_by_a_shell(self, workspace: Path) -> None:
        sentinel = workspace / "should-not-exist"
        payload = f"literal; touch {sentinel}"
        with CommandRunner(workspace) as runner:
            result = runner.run([sys.executable, "-c", "import sys; print(sys.argv[1])", payload])

        assert result.succeeded
        assert payload in result.stdout
        assert not sentinel.exists()


class TestWorkspaceContainment:
    def test_runs_in_workspace_or_contained_relative_directory(self, workspace: Path) -> None:
        nested = workspace / "nested"
        nested.mkdir()
        with CommandRunner(workspace) as runner:
            root_result = runner.run([sys.executable, "-c", "import os; print(os.getcwd())"])
            nested_result = runner.run(
                [sys.executable, "-c", "import os; print(os.getcwd())"],
                cwd="nested",
            )

        assert root_result.cwd == workspace.resolve()
        assert root_result.stdout.strip() == str(workspace.resolve())
        assert nested_result.cwd == nested.resolve()
        assert nested_result.stdout.strip() == str(nested.resolve())

    def test_rejects_absolute_relative_and_missing_escape_paths(self, workspace: Path) -> None:
        outside = workspace.parent / "outside"
        outside.mkdir()
        with CommandRunner(workspace) as runner:
            with pytest.raises(ValueError, match="escapes workspace"):
                runner.run([sys.executable, "--version"], cwd=outside)
            with pytest.raises(ValueError, match="escapes workspace"):
                runner.run([sys.executable, "--version"], cwd="../outside")
            with pytest.raises(ValueError, match="does not exist"):
                runner.run([sys.executable, "--version"], cwd="missing")

    def test_rejects_symlink_escape(self, workspace: Path) -> None:
        outside = workspace.parent / "outside"
        outside.mkdir()
        escape = workspace / "escape"
        try:
            escape.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks are unavailable")

        with (
            CommandRunner(workspace) as runner,
            pytest.raises(ValueError, match="escapes workspace"),
        ):
            runner.run([sys.executable, "--version"], cwd=escape)


class TestSanitizedEnvironment:
    def test_uses_sandbox_home_and_drops_ambient_credentials(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-secret")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-aws-secret")
        monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ambient-agent.sock")
        sandbox_home = workspace.parent / "sandbox-home"
        names = [
            "HOME",
            "XDG_CONFIG_HOME",
            "ANTHROPIC_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "SSH_AUTH_SOCK",
            "EVALYN_TEST_PORT",
            "GIT_CONFIG_GLOBAL",
            "GIT_NO_REPLACE_OBJECTS",
        ]
        script = (
            "import json, os, sys; "
            "print(json.dumps({name: os.environ.get(name) for name in sys.argv[1:]}))"
        )

        with CommandRunner(workspace, sandbox_home=sandbox_home) as runner:
            result = runner.run(
                [sys.executable, "-c", script, *names],
                env={"EVALYN_TEST_PORT": "4123"},
            )

        child_env = json.loads(result.stdout)
        assert result.succeeded
        assert child_env["HOME"] == str(sandbox_home.resolve())
        assert child_env["XDG_CONFIG_HOME"] == str((sandbox_home / "config").resolve())
        assert child_env["ANTHROPIC_API_KEY"] is None
        assert child_env["AWS_SECRET_ACCESS_KEY"] is None
        assert child_env["SSH_AUTH_SOCK"] is None
        assert child_env["EVALYN_TEST_PORT"] == "4123"
        assert child_env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert child_env["GIT_NO_REPLACE_OBJECTS"] == "1"

    @pytest.mark.parametrize(
        "name",
        ["HOME", "OPENAI_API_KEY", "MY_SERVICE_TOKEN", "DATABASE_PASSWORD"],
    )
    def test_rejects_explicit_credentials_and_sandbox_overrides(
        self, workspace: Path, name: str
    ) -> None:
        with CommandRunner(workspace) as runner, pytest.raises(ValueError, match="not permitted"):
            runner.run([sys.executable, "--version"], env={name: "secret"})

    def test_owned_sandbox_home_is_cleaned_up(self, workspace: Path) -> None:
        runner = CommandRunner(workspace)
        sandbox_home = runner.sandbox_home
        assert sandbox_home.is_dir()

        runner.close()

        assert not sandbox_home.exists()
        with pytest.raises(RuntimeError, match="closed"):
            runner.run([sys.executable, "--version"])


class TestCommandResults:
    def test_success_and_failure_are_structured(self, workspace: Path) -> None:
        with CommandRunner(workspace) as runner:
            passed = runner.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr)",
                ]
            )
            failed = runner.run([sys.executable, "-c", "raise SystemExit(7)"])

        assert isinstance(passed, CommandResult)
        assert passed.argv[0] == sys.executable
        assert passed.exit_code == passed.returncode == 0
        assert passed.succeeded
        assert "out" in passed.stdout
        assert "err" in passed.stderr
        assert passed.duration_seconds > 0
        assert not passed.output_truncated
        assert failed.exit_code == 7
        assert not failed.succeeded
        assert not failed.timed_out

    def test_spawn_error_is_a_result(self, workspace: Path) -> None:
        missing = str(workspace / "definitely-not-an-executable")
        with CommandRunner(workspace) as runner:
            result = runner.run([missing])

        assert result.exit_code is None
        assert result.error
        assert result.stderr == result.error
        assert not result.succeeded
        assert not result.timed_out


class TestBoundedOutput:
    def test_capture_is_bounded_while_full_log_keeps_all_bytes(self, workspace: Path) -> None:
        stdout = "STDOUT-BEGIN-" + ("a" * 20_000) + "-STDOUT-END"
        stderr = "STDERR-BEGIN-" + ("b" * 20_000) + "-STDERR-END"
        script = (
            "import os, sys; os.write(1, sys.argv[1].encode()); os.write(2, sys.argv[2].encode())"
        )
        log_path = workspace.parent / "artifacts" / "complete.log"

        with CommandRunner(workspace, max_output_bytes=256) as runner:
            result = runner.run(
                [sys.executable, "-c", script, stdout, stderr],
                full_log_path=log_path,
            )

        assert result.succeeded
        assert result.stdout_truncated
        assert result.stderr_truncated
        assert result.output_truncated
        assert len(result.stdout.encode()) <= 256
        assert len(result.stderr.encode()) <= 256
        assert result.stdout.startswith("STDOUT-BEGIN-")
        assert result.stdout.endswith("-STDOUT-END")
        assert result.stderr.startswith("STDERR-BEGIN-")
        assert result.stderr.endswith("-STDERR-END")
        assert "output truncated" in result.stdout
        assert result.full_log_path == log_path.resolve()
        assert result.log_error is None
        full_log = log_path.read_bytes()
        assert stdout.encode() in full_log
        assert stderr.encode() in full_log

    def test_full_log_has_a_hard_disk_bound(self, workspace: Path) -> None:
        log_path = workspace.parent / "artifacts" / "bounded.log"
        with CommandRunner(
            workspace,
            max_output_bytes=64,
            max_full_log_bytes=128,
        ) as runner:
            result = runner.run(
                [sys.executable, "-c", "print('x' * 10000, end='')"],
                full_log_path=log_path,
            )

        assert result.succeeded
        assert log_path.stat().st_size == 128
        assert result.log_error == "full log truncated at 128 bytes"

    def test_full_log_refuses_a_symlink_target(self, workspace: Path) -> None:
        outside = workspace.parent / "outside.log"
        outside.write_text("keep", encoding="utf-8")
        link = workspace / "linked.log"
        link.symlink_to(outside)

        with CommandRunner(workspace) as runner, pytest.raises(ValueError, match="symlink"):
            runner.run(
                [sys.executable, "-c", "print('unsafe')"],
                full_log_path=link,
            )

        assert outside.read_text(encoding="utf-8") == "keep"

    def test_full_log_refuses_a_symlinked_parent(self, workspace: Path) -> None:
        outside = workspace.parent / "outside-logs"
        outside.mkdir()
        linked_parent = workspace.parent / "linked-logs"
        linked_parent.symlink_to(outside, target_is_directory=True)

        with CommandRunner(workspace) as runner, pytest.raises(ValueError, match="parent"):
            runner.run(
                [sys.executable, "-c", "print('unsafe')"],
                full_log_path=linked_parent / "unsafe.log",
            )

        assert list(outside.iterdir()) == []

    def test_full_log_atomically_replaces_a_hardlink_without_truncating_its_peer(
        self, workspace: Path
    ) -> None:
        victim = workspace.parent / "victim.log"
        victim.write_text("keep", encoding="utf-8")
        log_path = workspace.parent / "linked.log"
        try:
            os.link(victim, log_path)
        except OSError:
            pytest.skip("hardlinks are unavailable")

        with CommandRunner(workspace) as runner:
            result = runner.run(
                [sys.executable, "-c", "print('safe-log')"],
                full_log_path=log_path,
            )

        assert result.succeeded
        assert victim.read_text(encoding="utf-8") == "keep"
        assert log_path.read_text(encoding="utf-8").strip() == "safe-log"
        assert not os.path.samefile(victim, log_path)

    @pytest.mark.parametrize("limit", [1, 27, 28, 64])
    def test_even_tiny_capture_limits_remain_hard_bounds(self, workspace: Path, limit: int) -> None:
        with CommandRunner(workspace, max_output_bytes=limit) as runner:
            result = runner.run([sys.executable, "-c", "print('x' * 1000, end='')"])

        assert result.stdout_truncated
        assert len(result.stdout.encode()) <= limit


class TestTimeoutCleanup:
    @pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
    def test_timeout_kills_grandchild_process_group(self, workspace: Path) -> None:
        child_pid_path = workspace / "child.pid"
        child_script = (
            "import os, pathlib, signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
            "print('child-ready', flush=True); "
            "time.sleep(60)"
        )
        parent_script = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
            "print('parent-ready', flush=True); "
            "time.sleep(60)"
        )

        with CommandRunner(
            workspace,
            default_timeout_seconds=0.4,
            termination_grace_seconds=0.15,
        ) as runner:
            result = runner.run(
                [sys.executable, "-c", parent_script, child_script, str(child_pid_path)]
            )

        assert result.timed_out
        assert not result.succeeded
        assert result.error and "timed out" in result.error
        assert "parent-ready" in result.stdout
        assert "child-ready" in result.stdout
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text())

        deadline = time.monotonic() + 2
        while _pid_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _pid_is_running(child_pid), "timed-out grandchild was left alive"


def _pid_is_running(pid: int) -> bool:
    """Return false for absent processes and Linux zombies."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            return stat_path.read_text().split()[2] != "Z"
        except (OSError, IndexError):
            pass
    return True
