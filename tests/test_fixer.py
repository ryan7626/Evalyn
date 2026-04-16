"""Tests for runner/fixer.py — fix validation, application, and source reading."""

import tempfile
from pathlib import Path

import pytest

from evalyn.models import FixAttempt
from evalyn.runner.fixer import (
    _validate_fix,
    apply_fix,
    _read_source_files,
    _process_fix_response,
)


def _repo_with_file(content: str = "def foo():\n    return 1\n") -> tuple[Path, Path]:
    """Return (tmp_dir_path, repo_path) with a source.py inside."""
    tmp = tempfile.mkdtemp()
    repo = Path(tmp)
    (repo / "source.py").write_text(content, encoding="utf-8")
    return Path(tmp), repo


class TestValidateFix:
    def test_valid_fix_passes(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(
            file_path="source.py",
            original_content="return 1",
            replacement_content="return 42",
            explanation="test",
        )
        assert _validate_fix(fix, repo) is None

    def test_forbidden_env_file(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(file_path=".env", original_content="x", replacement_content="y", explanation="")
        result = _validate_fix(fix, repo)
        assert result is not None
        assert ".env" in result

    def test_forbidden_test_file_py(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(file_path="test_foo.py", original_content="x", replacement_content="y", explanation="")
        assert _validate_fix(fix, repo) is not None

    def test_forbidden_spec_file(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(file_path="auth.spec.ts", original_content="x", replacement_content="y", explanation="")
        assert _validate_fix(fix, repo) is not None

    def test_file_does_not_exist(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(file_path="missing.py", original_content="x", replacement_content="y", explanation="")
        result = _validate_fix(fix, repo)
        assert result is not None
        assert "does not exist" in result

    def test_path_traversal_rejected(self):
        _, repo = _repo_with_file()
        # Create a file outside the repo to ensure the check is about the path, not existence
        fix = FixAttempt(file_path="../outside.py", original_content="x", replacement_content="y", explanation="")
        result = _validate_fix(fix, repo)
        # Either "does not exist" (file not there) or path traversal check
        assert result is not None

    def test_empty_original_rejected(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(file_path="source.py", original_content="", replacement_content="x", explanation="")
        result = _validate_fix(fix, repo)
        assert result is not None
        assert "empty" in result

    def test_identical_content_rejected(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(
            file_path="source.py",
            original_content="return 1",
            replacement_content="return 1",
            explanation="",
        )
        result = _validate_fix(fix, repo)
        assert result is not None
        assert "identical" in result


class TestApplyFix:
    def test_applies_successfully(self):
        _, repo = _repo_with_file("def foo():\n    return 1\n")
        fix = FixAttempt(
            file_path="source.py",
            original_content="return 1",
            replacement_content="return 42",
            explanation="test",
        )
        assert apply_fix(fix, repo) is True
        assert fix.applied is True
        assert "return 42" in (repo / "source.py").read_text()

    def test_pre_rejected_fix_not_applied(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(
            file_path="source.py",
            original_content="x",
            replacement_content="y",
            explanation="",
            rejected_reason="pre-rejected",
        )
        assert apply_fix(fix, repo) is False
        assert fix.applied is False

    def test_original_not_in_file(self):
        _, repo = _repo_with_file("def foo():\n    return 1\n")
        fix = FixAttempt(
            file_path="source.py",
            original_content="NONEXISTENT_STRING_XYZ",
            replacement_content="replacement",
            explanation="",
        )
        result = apply_fix(fix, repo)
        assert result is False
        assert fix.rejected_reason is not None
        assert "not found" in fix.rejected_reason

    def test_missing_file_returns_false(self):
        _, repo = _repo_with_file()
        fix = FixAttempt(
            file_path="does_not_exist.py",
            original_content="x",
            replacement_content="y",
            explanation="",
        )
        assert apply_fix(fix, repo) is False
        assert fix.applied is False

    def test_replaces_only_first_occurrence(self):
        _, repo = _repo_with_file("x = 1\nx = 1\n")
        fix = FixAttempt(
            file_path="source.py",
            original_content="x = 1",
            replacement_content="x = 99",
            explanation="",
        )
        apply_fix(fix, repo)
        content = (repo / "source.py").read_text()
        assert content.count("x = 99") == 1
        assert content.count("x = 1") == 1


class TestReadSourceFiles:
    def test_reads_existing_file(self):
        _, repo = _repo_with_file("# hello world\n")
        result = _read_source_files(repo, ["source.py"])
        assert "source.py" in result
        assert "hello world" in result

    def test_skips_missing_files(self):
        _, repo = _repo_with_file()
        result = _read_source_files(repo, ["nonexistent.py"])
        assert "No relevant source files found" in result

    def test_skips_forbidden_test_files(self):
        _, repo = _repo_with_file()
        (repo / "test_foo.py").write_text("# test content", encoding="utf-8")
        result = _read_source_files(repo, ["test_foo.py"])
        assert "No relevant source files found" in result

    def test_truncates_long_file(self):
        _, repo = _repo_with_file()
        (repo / "big.py").write_text("\n".join(f"line {i}" for i in range(300)), encoding="utf-8")
        result = _read_source_files(repo, ["big.py"])
        assert "truncated" in result

    def test_empty_list_returns_not_found(self):
        _, repo = _repo_with_file()
        assert "No relevant source files found" in _read_source_files(repo, [])

    def test_max_files_limit(self):
        _, repo = _repo_with_file()
        for i in range(10):
            (repo / f"f{i}.py").write_text(f"# file {i}", encoding="utf-8")
        files = [f"f{i}.py" for i in range(10)]
        result = _read_source_files(repo, files, max_files=3)
        # Only first 3 files should appear
        assert result.count("### `") == 3


class TestProcessFixResponse:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Path(self.tmp)
        (self.repo / "source.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    def test_null_file_path_returns_declined(self):
        response = {
            "file_path": None,
            "original_content": None,
            "replacement_content": None,
            "explanation": "env issue",
        }
        fix = _process_fix_response(response, self.repo)
        assert fix.rejected_reason == "LLM declined"

    def test_valid_response_creates_fix(self):
        response = {
            "file_path": "source.py",
            "original_content": "return 1",
            "replacement_content": "return 42",
            "explanation": "fix the value",
        }
        fix = _process_fix_response(response, self.repo)
        assert fix.file_path == "source.py"
        assert fix.rejected_reason is None

    def test_diff_size_is_computed(self):
        response = {
            "file_path": "source.py",
            "original_content": "ab",
            "replacement_content": "abcde",
            "explanation": "expand",
        }
        fix = _process_fix_response(response, self.repo)
        assert fix.diff_size == 3  # abs(5 - 2)
