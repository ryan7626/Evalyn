"""Tests for v3 re-run resilience: branch helpers, state reset, stale recovery."""

import subprocess
from pathlib import Path

import pytest

from evalyn.utils.git import branch_exists, list_branches, create_branch, delete_branch
from evalyn.state import AgentState, load_state, save_state, reset_state


def _init_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo for testing."""
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    # Need at least one commit for branch operations
    (tmp_path / "README.md").write_text("init")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "init"],
        capture_output=True, check=True,
    )
    return tmp_path


class TestBranchExists:
    def test_returns_true_for_existing_branch(self, tmp_path):
        repo = _init_repo(tmp_path)
        create_branch(repo, "evalyn/test")
        assert branch_exists(repo, "evalyn/test") is True

    def test_returns_false_for_nonexistent_branch(self, tmp_path):
        repo = _init_repo(tmp_path)
        assert branch_exists(repo, "evalyn/nope") is False


class TestListBranches:
    def test_lists_matching_branches(self, tmp_path):
        repo = _init_repo(tmp_path)
        create_branch(repo, "evalyn/run1")
        subprocess.run(["git", "-C", str(repo), "checkout", "main"], capture_output=True)
        create_branch(repo, "evalyn/run2")
        subprocess.run(["git", "-C", str(repo), "checkout", "main"], capture_output=True)
        create_branch(repo, "other/branch")

        result = list_branches(repo, "evalyn/*")
        assert "evalyn/run1" in result
        assert "evalyn/run2" in result
        assert "other/branch" not in result

    def test_returns_empty_when_no_match(self, tmp_path):
        repo = _init_repo(tmp_path)
        result = list_branches(repo, "evalyn/*")
        assert result == []


class TestResetState:
    def test_resets_to_defaults(self, tmp_path):
        repo = tmp_path
        save_state(repo, AgentState(
            last_run_commit="abc123",
            last_run_status="completed",
            current_branch="evalyn/old",
            run_history=[{"timestamp": "old"}],
        ))

        reset_state(repo)

        state = load_state(repo)
        assert state.last_run_commit is None
        assert state.last_run_status is None
        assert state.current_branch is None
        assert state.run_history == []


class TestRecoverStaleState:
    """Tests for _recover_stale_state in runner.py."""

    def test_recovers_in_progress_state(self, tmp_path):
        from evalyn.runner.plan_reader import recover_stale_state as _recover_stale_state

        repo = _init_repo(tmp_path)
        stale_branch = "evalyn/stale"
        create_branch(repo, stale_branch)
        subprocess.run(["git", "-C", str(repo), "checkout", "main"], capture_output=True)

        state = AgentState(
            last_run_status="in_progress",
            current_branch=stale_branch,
        )

        _recover_stale_state(repo, state, "evalyn/new", "main")

        assert state.last_run_status == "recovered"
        assert state.current_branch is None
        assert branch_exists(repo, stale_branch) is False

    def test_removes_stale_target_branch(self, tmp_path):
        from evalyn.runner.plan_reader import recover_stale_state as _recover_stale_state

        repo = _init_repo(tmp_path)
        target = "evalyn/2026-01-01"
        create_branch(repo, target)
        subprocess.run(["git", "-C", str(repo), "checkout", "main"], capture_output=True)

        state = AgentState(last_run_status="completed")

        _recover_stale_state(repo, state, target, "main")

        assert branch_exists(repo, target) is False

    def test_no_op_when_clean(self, tmp_path):
        from evalyn.runner.plan_reader import recover_stale_state as _recover_stale_state

        repo = _init_repo(tmp_path)
        state = AgentState(last_run_status="completed")

        # Should not crash
        _recover_stale_state(repo, state, "evalyn/fresh", "main")

        assert state.last_run_status == "completed"

    def test_recovers_when_on_stale_branch(self, tmp_path):
        from evalyn.runner.plan_reader import recover_stale_state as _recover_stale_state
        from evalyn.utils.git import get_current_branch

        repo = _init_repo(tmp_path)
        target = "evalyn/current"
        create_branch(repo, target)
        # We're now ON the stale branch
        assert get_current_branch(repo) == target

        state = AgentState(last_run_status="completed")
        _recover_stale_state(repo, state, target, "main")

        assert branch_exists(repo, target) is False
        assert get_current_branch(repo) == "main"
