"""Tests for runner/test_executor.py — command execution and output truncation."""

import tempfile
from pathlib import Path

import pytest

from evalyn.runner.test_executor import run_test_command, _truncate_output


@pytest.fixture
def repo(tmp_path):
    return tmp_path


class TestRunTestCommand:
    def test_passing_command(self, repo):
        result = run_test_command(repo, "echo hello", "cat")
        assert result.status == "passed"
        assert result.exit_code == 0
        assert result.is_success is True
        assert "hello" in result.stdout

    def test_failing_command(self, repo):
        result = run_test_command(repo, "exit 1", "cat")
        assert result.status == "failed"
        assert result.exit_code == 1
        assert result.is_success is False

    def test_timeout(self, repo):
        result = run_test_command(repo, "sleep 10", "cat", timeout_seconds=1)
        assert result.status == "timeout"
        assert result.exit_code == -1
        assert "timed out" in result.stderr

    def test_category_is_recorded(self, repo):
        result = run_test_command(repo, "echo hi", "my_category")
        assert result.category == "my_category"

    def test_command_is_recorded(self, repo):
        result = run_test_command(repo, "echo hi", "cat")
        assert result.command == "echo hi"

    def test_duration_is_positive(self, repo):
        result = run_test_command(repo, "echo hi", "cat")
        assert result.duration_seconds > 0

    def test_failure_summary_empty_on_pass(self, repo):
        result = run_test_command(repo, "echo hi", "cat")
        assert result.failure_summary == ""

    def test_failure_summary_on_timeout(self, repo):
        result = run_test_command(repo, "sleep 10", "cat", timeout_seconds=1)
        assert "timed out" in result.failure_summary

    def test_failure_summary_on_failure(self, repo):
        result = run_test_command(repo, "echo 'err msg' >&2; exit 1", "cat")
        assert result.failure_summary  # not empty

    def test_max_output_lines_applied(self, repo):
        # Generate many lines and verify output is capped
        cmd = "for i in $(seq 1 100); do echo line_$i; done"
        result = run_test_command(repo, cmd, "cat", max_output_lines=20)
        lines = result.stdout.strip().split("\n")
        # With truncation plus the omitted marker, total lines should be <= 21
        # (10 head + omitted marker + 10 tail)
        assert len(lines) <= 22


class TestTruncateOutput:
    def test_short_output_unchanged(self):
        text = "line1\nline2\nline3"
        assert _truncate_output(text, 10) == text

    def test_empty_string_returns_empty(self):
        assert _truncate_output("", 10) == ""

    def test_truncates_and_shows_head_and_tail(self):
        lines = [f"line {i}" for i in range(100)]
        text = "\n".join(lines)
        result = _truncate_output(text, 20)
        assert "omitted" in result
        assert "line 0" in result    # head preserved
        assert "line 99" in result   # tail preserved
        assert "line 50" not in result  # middle dropped

    def test_exact_limit_not_truncated(self):
        text = "\n".join(f"line {i}" for i in range(10))
        assert _truncate_output(text, 10) == text

    def test_single_line_not_truncated(self):
        assert _truncate_output("only one line", 5) == "only one line"
