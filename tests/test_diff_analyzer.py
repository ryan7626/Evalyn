"""Tests for planner/diff_analyzer.py — ChangeSet, grouping, and analysis."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from evalyn.planner.diff_analyzer import ChangeSet, analyze_changes, _group_by_directory
from evalyn.utils.git import CommitInfo


# ── Git repo fixture ─────────────────────────────────────────────────────────


def _init_repo(path: Path) -> Path:
    """Create a minimal git repo with one commit."""
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@test.com"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test User"],
        check=True, capture_output=True,
    )
    (path / "README.md").write_text("# Hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "Initial commit"],
        check=True, capture_output=True,
    )
    return path


# ── ChangeSet.is_empty ───────────────────────────────────────────────────────


class TestChangeSetIsEmpty:
    def test_empty_with_no_commits_or_files(self):
        cs = ChangeSet(since_sha="abc", head_sha="abc")
        assert cs.is_empty is True

    def test_not_empty_with_commits(self):
        cs = ChangeSet(
            since_sha="abc",
            head_sha="def",
            commits=[CommitInfo(sha="def", message="add feature", author="a", timestamp="2026-01-01")],
        )
        assert cs.is_empty is False

    def test_not_empty_with_files(self):
        cs = ChangeSet(since_sha="abc", head_sha="def", files_changed=["src/foo.py"])
        assert cs.is_empty is False


# ── ChangeSet.summary_lines ──────────────────────────────────────────────────


class TestChangeSetSummaryLines:
    def test_includes_commit_count(self):
        cs = ChangeSet(since_sha="abcd1234", head_sha="def")
        lines = cs.summary_lines
        assert any("0 commit" in l for l in lines)

    def test_includes_file_count(self):
        cs = ChangeSet(
            since_sha="abcd1234",
            head_sha="def",
            files_changed=["a.py", "b.py", "c.py"],
        )
        lines = cs.summary_lines
        assert any("3 file" in l for l in lines)

    def test_includes_directory_breakdown(self):
        cs = ChangeSet(
            since_sha="abcd1234",
            head_sha="def",
            files_changed=["src/a.py", "src/b.py"],
            files_by_directory={"src": ["src/a.py", "src/b.py"]},
        )
        lines = cs.summary_lines
        assert any("src" in l for l in lines)


# ── _group_by_directory ──────────────────────────────────────────────────────


class TestGroupByDirectory:
    def test_groups_by_top_level_dir(self):
        files = ["src/foo.py", "src/bar.py", "tests/test_foo.py"]
        result = _group_by_directory(files)
        assert "src" in result
        assert len(result["src"]) == 2
        assert "tests" in result
        assert len(result["tests"]) == 1

    def test_root_files_grouped_under_dot(self):
        files = ["README.md", "pyproject.toml"]
        result = _group_by_directory(files)
        assert "." in result
        assert len(result["."]) == 2

    def test_mixed_root_and_subdir(self):
        files = ["README.md", "src/app.py"]
        result = _group_by_directory(files)
        assert "." in result
        assert "src" in result

    def test_empty_input(self):
        assert _group_by_directory([]) == {}

    def test_deeply_nested_uses_top_level(self):
        files = ["a/b/c/d.py"]
        result = _group_by_directory(files)
        assert "a" in result
        assert len(result["a"]) == 1


# ── analyze_changes ──────────────────────────────────────────────────────────


class TestAnalyzeChanges:
    def test_returns_changeset_first_run(self, tmp_path):
        repo = _init_repo(tmp_path)
        cs = analyze_changes(repo, since_sha=None)
        assert isinstance(cs, ChangeSet)
        # The initial commit should be present
        assert not cs.is_empty

    def test_head_sha_is_populated(self, tmp_path):
        repo = _init_repo(tmp_path)
        cs = analyze_changes(repo, since_sha=None)
        assert len(cs.head_sha) == 40  # full SHA

    def test_files_by_directory_populated(self, tmp_path):
        repo = _init_repo(tmp_path)
        cs = analyze_changes(repo, since_sha=None)
        assert isinstance(cs.files_by_directory, dict)

    def test_no_changes_since_head_is_empty(self, tmp_path):
        repo = _init_repo(tmp_path)
        # Use HEAD itself as since_sha — no new commits
        from evalyn.utils.git import get_head_sha
        head = get_head_sha(repo)
        cs = analyze_changes(repo, since_sha=head)
        # No commits since HEAD, no files changed
        assert cs.is_empty is True
