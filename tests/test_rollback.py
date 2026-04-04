"""Tests for evalyn.runner.rollback module."""

import subprocess
from pathlib import Path

import pytest

from evalyn.runner.rollback import RollbackManager


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )

    (repo / "main.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"], cwd=repo, capture_output=True,
    )

    return repo


@pytest.fixture
def run_dir(tmp_path):
    """Create a temporary run directory."""
    d = tmp_path / "run_001"
    d.mkdir()
    return d


class TestRollbackManager:
    def test_snapshot_captures_changes(self, git_repo, run_dir):
        rm = RollbackManager(run_dir, git_repo)

        # Make a change
        (git_repo / "main.py").write_text("print('modified')\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)

        patch = rm.create_snapshot("test_change")
        assert patch is not None
        assert patch.is_file()
        assert rm.patch_count == 1

    def test_snapshot_returns_none_when_clean(self, git_repo, run_dir):
        rm = RollbackManager(run_dir, git_repo)
        patch = rm.create_snapshot("no_changes")
        assert patch is None
        assert rm.patch_count == 0

    def test_rollback_last_reverts_change(self, git_repo, run_dir):
        rm = RollbackManager(run_dir, git_repo)

        # Make and snapshot a change
        (git_repo / "main.py").write_text("print('modified')\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        rm.create_snapshot("change_1")

        # Rollback
        success = rm.rollback_last()
        assert success is True
        assert rm.patch_count == 0
        assert (git_repo / "main.py").read_text() == "print('hello')\n"

    def test_rollback_all_clears_everything(self, git_repo, run_dir):
        rm = RollbackManager(run_dir, git_repo)

        # Make changes
        (git_repo / "main.py").write_text("print('v2')\n")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
        rm.create_snapshot("change_1")

        rm.rollback_all()
        assert rm.patch_count == 0
        assert (git_repo / "main.py").read_text() == "print('hello')\n"

    def test_rollback_last_returns_false_when_empty(self, git_repo, run_dir):
        rm = RollbackManager(run_dir, git_repo)
        assert rm.rollback_last() is False

    def test_patches_dir_created(self, git_repo, run_dir):
        rm = RollbackManager(run_dir, git_repo)
        assert (run_dir / "patches").is_dir()
