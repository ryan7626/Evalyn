"""Git operations for Evalyn.

All operations use subprocess calls to git directly — no gitpython dependency.
Every function takes repo_path as its first argument.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class GitError(Exception):
    """Raised when a git operation fails."""
    pass


@dataclass
class CommitInfo:
    """Represents a single git commit."""
    sha: str
    message: str
    author: str
    timestamp: str


def _run_git(repo_path: Path, *args: str, check: bool = True) -> str:
    """Run a git command and return stdout.

    Args:
        repo_path: Path to the git repository.
        *args: Git subcommand and arguments.
        check: If True, raise GitError on non-zero exit code.

    Returns:
        Stripped stdout from the command.
    """
    cmd = ["git", "-C", str(repo_path)] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise GitError(f"Git command timed out: {' '.join(cmd)}")
    except FileNotFoundError:
        raise GitError("git is not installed or not in PATH")

    if check and result.returncode != 0:
        raise GitError(
            f"Git command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )

    return result.stdout.strip()


def get_current_branch(repo_path: Path) -> str:
    """Return the name of the current branch."""
    return _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")


def get_head_sha(repo_path: Path) -> str:
    """Return the full SHA of HEAD."""
    return _run_git(repo_path, "rev-parse", "HEAD")


def get_commits_since(
    repo_path: Path,
    since_sha: str,
) -> list[CommitInfo]:
    """List commits since a given SHA (exclusive).

    Returns commits in reverse chronological order (newest first).
    """
    # Format: SHA|message|author|timestamp
    sep = "|||"
    fmt = f"%H{sep}%s{sep}%an{sep}%aI"
    output = _run_git(
        repo_path,
        "log",
        f"--format={fmt}",
        f"{since_sha}..HEAD",
    )

    if not output:
        return []

    commits = []
    for line in output.split("\n"):
        parts = line.split(sep, maxsplit=3)
        if len(parts) == 4:
            commits.append(CommitInfo(
                sha=parts[0],
                message=parts[1],
                author=parts[2],
                timestamp=parts[3],
            ))

    return commits


def get_changed_files(
    repo_path: Path,
    since_sha: str,
) -> list[str]:
    """List files changed since a given SHA.

    Returns relative paths to the repo root.
    """
    output = _run_git(
        repo_path,
        "diff",
        "--name-only",
        since_sha,
        "HEAD",
    )
    if not output:
        return []
    return output.split("\n")


def get_diff(
    repo_path: Path,
    since_sha: str,
    *,
    max_lines: Optional[int] = None,
) -> str:
    """Get the full diff between a SHA and HEAD.

    Args:
        repo_path: Path to the repo.
        since_sha: Starting commit SHA.
        max_lines: If set, truncate diff to this many lines.

    Returns:
        Diff text.
    """
    output = _run_git(repo_path, "diff", since_sha, "HEAD")
    if max_lines and output:
        lines = output.split("\n")
        if len(lines) > max_lines:
            output = "\n".join(lines[:max_lines])
            output += f"\n\n... (truncated, {len(lines) - max_lines} lines omitted)"
    return output


def create_branch(repo_path: Path, name: str) -> None:
    """Create and checkout a new branch."""
    _run_git(repo_path, "checkout", "-b", name)


def checkout_branch(repo_path: Path, name: str) -> None:
    """Checkout an existing branch."""
    _run_git(repo_path, "checkout", name)


def stage_files(repo_path: Path, paths: list[str]) -> None:
    """Stage specific files (git add)."""
    if not paths:
        return
    _run_git(repo_path, "add", "--", *paths)


def create_patch(repo_path: Path, output_path: Path) -> bool:
    """Export current uncommitted changes as a .patch file.

    Returns True if there were changes to export, False if the working
    tree is clean.
    """
    # Check if there are any changes
    status = _run_git(repo_path, "status", "--porcelain")
    if not status:
        return False

    # Get diff of all changes (staged + unstaged)
    diff = _run_git(repo_path, "diff", "HEAD", check=False)
    if not diff:
        # Only untracked files — use diff with /dev/null style
        diff = _run_git(repo_path, "diff", "--cached", check=False)

    if diff:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(diff, encoding="utf-8")
        return True

    return False


def apply_patch(repo_path: Path, patch_path: Path) -> None:
    """Apply a .patch file to the working tree."""
    _run_git(repo_path, "apply", "--whitespace=nowarn", str(patch_path))


def reset_hard(repo_path: Path) -> None:
    """Reset the working tree to HEAD (git reset --hard HEAD)."""
    _run_git(repo_path, "reset", "--hard", "HEAD")
    # Also clean untracked files that the agent may have created
    _run_git(repo_path, "clean", "-fd")
