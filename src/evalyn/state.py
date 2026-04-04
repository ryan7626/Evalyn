"""State management for Evalyn.

Persists agent state in .agent/state.json within the target repository.
Uses atomic writes (write to temp, then rename) to prevent corruption.
"""

import json
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class AgentState:
    """Persistent state for the Evalyn agent."""

    last_run_commit: Optional[str] = None
    last_run_timestamp: Optional[str] = None
    last_run_status: Optional[str] = None  # "completed" | "failed" | "in_progress"
    current_branch: Optional[str] = None
    run_history: list[dict] = field(default_factory=list)


def _state_path(repo_path: Path) -> Path:
    """Get the path to state.json for a repo."""
    return repo_path / ".agent" / "state.json"


def load_state(repo_path: Path) -> AgentState:
    """Load agent state from .agent/state.json.

    Returns a fresh AgentState if the file doesn't exist or is corrupted.
    """
    path = _state_path(repo_path)

    if not path.is_file():
        return AgentState()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupted state file — start fresh but don't delete the old one
        return AgentState()

    if not isinstance(raw, dict):
        return AgentState()

    return AgentState(
        last_run_commit=raw.get("last_run_commit"),
        last_run_timestamp=raw.get("last_run_timestamp"),
        last_run_status=raw.get("last_run_status"),
        current_branch=raw.get("current_branch"),
        run_history=raw.get("run_history", []),
    )


def save_state(repo_path: Path, state: AgentState) -> None:
    """Save agent state to .agent/state.json atomically.

    Creates .agent/ directory if it doesn't exist.
    Writes to a temp file first, then renames to prevent corruption.
    """
    agent_dir = repo_path / ".agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    path = _state_path(repo_path)
    data = json.dumps(asdict(state), indent=2, ensure_ascii=False)

    # Atomic write: write to temp file in same directory, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=agent_dir,
        prefix=".state_",
        suffix=".tmp",
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.write("\n")
        Path(tmp_path).replace(path)
    except BaseException:
        # Clean up temp file on failure
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def get_latest_run_dir(repo_path: Path) -> Optional[Path]:
    """Find the most recent timestamped run directory.

    Returns None if no runs exist.
    """
    runs_dir = repo_path / ".agent" / "runs"
    if not runs_dir.is_dir():
        return None

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def is_run_in_progress(repo_path: Path) -> bool:
    """Check if a run was interrupted (status is 'in_progress')."""
    state = load_state(repo_path)
    return state.last_run_status == "in_progress"
