"""Tests for evalyn.state module."""

import json
import tempfile
from pathlib import Path

import pytest

from evalyn.state import AgentState, load_state, save_state, get_latest_run_dir, is_run_in_progress


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo directory."""
    return tmp_path


class TestLoadState:
    def test_returns_default_when_no_file(self, tmp_repo):
        state = load_state(tmp_repo)
        assert state.last_run_commit is None
        assert state.last_run_timestamp is None
        assert state.last_run_status is None
        assert state.current_branch is None
        assert state.run_history == []

    def test_loads_existing_state(self, tmp_repo):
        agent_dir = tmp_repo / ".agent"
        agent_dir.mkdir()
        state_data = {
            "last_run_commit": "abc123",
            "last_run_timestamp": "2026-04-03_220000",
            "last_run_status": "completed",
            "current_branch": "evalyn/2026-04-03_220000",
            "run_history": [{"timestamp": "2026-04-03_220000"}],
        }
        (agent_dir / "state.json").write_text(json.dumps(state_data))

        state = load_state(tmp_repo)
        assert state.last_run_commit == "abc123"
        assert state.last_run_status == "completed"
        assert len(state.run_history) == 1

    def test_handles_corrupted_json(self, tmp_repo):
        agent_dir = tmp_repo / ".agent"
        agent_dir.mkdir()
        (agent_dir / "state.json").write_text("not valid json {{{")

        state = load_state(tmp_repo)
        assert state.last_run_commit is None  # fresh state

    def test_handles_non_dict_json(self, tmp_repo):
        agent_dir = tmp_repo / ".agent"
        agent_dir.mkdir()
        (agent_dir / "state.json").write_text('"just a string"')

        state = load_state(tmp_repo)
        assert state.last_run_commit is None


class TestSaveState:
    def test_creates_agent_dir_and_file(self, tmp_repo):
        state = AgentState(
            last_run_commit="abc123",
            last_run_status="completed",
        )
        save_state(tmp_repo, state)

        assert (tmp_repo / ".agent").is_dir()
        assert (tmp_repo / ".agent" / "state.json").is_file()

    def test_saved_state_is_loadable(self, tmp_repo):
        state = AgentState(
            last_run_commit="def456",
            last_run_timestamp="2026-04-03",
            last_run_status="in_progress",
            current_branch="evalyn/test",
            run_history=[{"x": 1}],
        )
        save_state(tmp_repo, state)

        loaded = load_state(tmp_repo)
        assert loaded.last_run_commit == "def456"
        assert loaded.last_run_status == "in_progress"
        assert loaded.current_branch == "evalyn/test"
        assert loaded.run_history == [{"x": 1}]

    def test_overwrites_existing(self, tmp_repo):
        save_state(tmp_repo, AgentState(last_run_commit="first"))
        save_state(tmp_repo, AgentState(last_run_commit="second"))

        loaded = load_state(tmp_repo)
        assert loaded.last_run_commit == "second"


class TestGetLatestRunDir:
    def test_returns_none_when_no_runs(self, tmp_repo):
        assert get_latest_run_dir(tmp_repo) is None

    def test_returns_latest_dir(self, tmp_repo):
        runs_dir = tmp_repo / ".agent" / "runs"
        (runs_dir / "2026-04-01_100000").mkdir(parents=True)
        (runs_dir / "2026-04-03_220000").mkdir(parents=True)
        (runs_dir / "2026-04-02_150000").mkdir(parents=True)

        latest = get_latest_run_dir(tmp_repo)
        assert latest is not None
        assert latest.name == "2026-04-03_220000"


class TestIsRunInProgress:
    def test_false_when_no_state(self, tmp_repo):
        assert is_run_in_progress(tmp_repo) is False

    def test_true_when_in_progress(self, tmp_repo):
        save_state(tmp_repo, AgentState(last_run_status="in_progress"))
        assert is_run_in_progress(tmp_repo) is True

    def test_false_when_completed(self, tmp_repo):
        save_state(tmp_repo, AgentState(last_run_status="completed"))
        assert is_run_in_progress(tmp_repo) is False
