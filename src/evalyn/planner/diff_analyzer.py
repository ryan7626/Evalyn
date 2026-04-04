"""Diff analyzer for Evalyn.

Analyzes git changes since the last run to determine what has changed
and needs testing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evalyn.utils.git import (
    CommitInfo,
    get_commits_since,
    get_changed_files,
    get_diff,
    get_head_sha,
)
from evalyn.utils.log import get_logger

logger = get_logger("planner.diff_analyzer")


@dataclass
class ChangeSet:
    """Represents all changes since the last run."""

    since_sha: str
    head_sha: str
    commits: list[CommitInfo] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    diff_text: str = ""
    files_by_directory: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True if there are no changes to analyze."""
        return len(self.commits) == 0 and len(self.files_changed) == 0

    @property
    def summary_lines(self) -> list[str]:
        """Human-readable summary lines."""
        lines = []
        lines.append(
            f"{len(self.commits)} commit(s) since {self.since_sha[:8]}"
        )
        lines.append(f"{len(self.files_changed)} file(s) changed")
        if self.files_by_directory:
            lines.append("Directories affected:")
            for dir_name, files in sorted(self.files_by_directory.items()):
                lines.append(f"  {dir_name}/ ({len(files)} files)")
        return lines


def analyze_changes(
    repo_path: Path,
    since_sha: Optional[str] = None,
    *,
    max_diff_lines: int = 500,
) -> ChangeSet:
    """Analyze all changes in the repo since a given commit.

    Args:
        repo_path: Path to the target repository.
        since_sha: Commit SHA to compare against. If None, analyzes
            all commits (uses the initial empty tree SHA).
        max_diff_lines: Maximum diff lines to include (for LLM context limits).

    Returns:
        ChangeSet with all relevant change information.
    """
    head_sha = get_head_sha(repo_path)

    # If no previous run, use git's empty tree SHA to capture everything
    if since_sha is None:
        since_sha = _get_empty_tree_sha()
        logger.info("No previous run — analyzing all commits")
    else:
        logger.info(f"Analyzing changes since {since_sha[:8]}")

    # Get commits
    commits = get_commits_since(repo_path, since_sha)
    logger.info(f"Found {len(commits)} new commit(s)")

    # Get changed files
    files_changed = get_changed_files(repo_path, since_sha)
    logger.info(f"Found {len(files_changed)} changed file(s)")

    # Get diff (truncated for LLM context)
    diff_text = get_diff(repo_path, since_sha, max_lines=max_diff_lines)

    # Group files by top-level directory
    files_by_directory = _group_by_directory(files_changed)

    changeset = ChangeSet(
        since_sha=since_sha,
        head_sha=head_sha,
        commits=commits,
        files_changed=files_changed,
        diff_text=diff_text,
        files_by_directory=files_by_directory,
    )

    for line in changeset.summary_lines:
        logger.info(line)

    return changeset


def _group_by_directory(files: list[str]) -> dict[str, list[str]]:
    """Group file paths by their top-level directory.

    Files at the repo root are grouped under '.'.
    """
    groups: dict[str, list[str]] = {}
    for f in files:
        parts = f.split("/")
        dir_name = parts[0] if len(parts) > 1 else "."
        groups.setdefault(dir_name, []).append(f)
    return groups


def _get_empty_tree_sha() -> str:
    """Return git's well-known empty tree SHA.

    This is a constant in git, representing an empty tree object.
    Used as the 'since' ref when there's no previous run.
    """
    return "4b825dc642cb6eb9a060e54bf899d69f82cf7c06"
