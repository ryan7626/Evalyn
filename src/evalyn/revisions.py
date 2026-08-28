"""Resolve immutable Git comparisons without changing repository state."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from evalyn.execution.bounded import run_bounded_process
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment

_GIT_STDOUT_LIMIT = 8 * 1024 * 1024
_GIT_STDERR_LIMIT = 256 * 1024


class RevisionError(ValueError):
    pass


@dataclass(frozen=True)
class RevisionRange:
    base_ref: str
    candidate_ref: str
    base_sha: str
    candidate_sha: str


@dataclass(frozen=True)
class _GitResult:
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False


def validate_git_repository(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if not candidate.is_dir():
        raise RevisionError(f"Repository directory does not exist: {candidate}")
    root = _git(candidate, "rev-parse", "--show-toplevel")
    if not root:
        raise RevisionError(f"Not a Git repository: {candidate}")
    return Path(root).resolve()


def resolve_range(
    repo_path: Path, value: str | None = None, *, allow_same: bool = False
) -> RevisionRange:
    repo = validate_git_repository(repo_path)
    if value is not None and not value.strip():
        value = None

    if value is None:
        candidate_ref = "HEAD"
        base_ref = _default_base_ref(repo)
    elif "..." in value:
        if value.count("...") != 1:
            raise RevisionError("Range must use exactly one base...candidate separator")
        base_ref, candidate_ref = value.split("...", 1)
        if not base_ref or not candidate_ref:
            raise RevisionError("Range must include both base and candidate refs")
    elif ".." in value:
        raise RevisionError("Use three-dot base...candidate range syntax")
    else:
        base_ref, candidate_ref = value, "HEAD"

    base_tip = _resolve_commit(repo, base_ref)
    candidate_sha = _resolve_commit(repo, candidate_ref)
    base_sha = _git(repo, "merge-base", base_tip, candidate_sha)
    if not base_sha:
        raise RevisionError(f"No merge base between {base_ref!r} and {candidate_ref!r}")
    if base_sha == candidate_sha and not allow_same:
        raise RevisionError("The selected range contains no committed candidate changes")
    return RevisionRange(
        base_ref=base_ref,
        candidate_ref=candidate_ref,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
    )


def get_diff(repo_path: Path, revisions: RevisionRange, *, max_chars: int = 120_000) -> str:
    char_limit = max(0, int(max_chars))
    # Four UTF-8 bytes per requested character plus one sentinel byte lets us
    # preserve the character-based API without ever buffering an arbitrary diff.
    byte_limit = (char_limit * 4) + 1
    result = _git_result(
        repo_path,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--find-renames",
        revisions.base_sha,
        revisions.candidate_sha,
        "--",
        max_stdout_bytes=byte_limit,
        allow_stdout_truncation=True,
    )
    if result.returncode != 0 and not result.stdout_truncated:
        raise RevisionError("git diff failed: " + (result.stderr.strip() or result.stdout.strip()))
    diff = result.stdout.strip()
    if not result.stdout_truncated and len(diff) <= char_limit:
        return diff
    if result.stdout_truncated:
        detail = f"diff exceeded {byte_limit} captured bytes"
    else:
        detail = f"{len(diff) - char_limit} characters"
    return f"{diff[:char_limit]}\n\n[truncated {detail}]"


def changed_files(repo_path: Path, revisions: RevisionRange) -> tuple[str, ...]:
    output = _git(
        repo_path,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        revisions.base_sha,
        revisions.candidate_sha,
        "--",
    )
    return tuple(line for line in output.splitlines() if line)


def is_dirty(repo_path: Path) -> bool:
    return bool(_git(repo_path, "status", "--porcelain=v1", "--untracked-files=normal"))


def require_dirty_candidate_is_head(repo_path: Path, revisions: RevisionRange) -> None:
    head = _resolve_commit(repo_path, "HEAD")
    if revisions.candidate_sha != head:
        raise RevisionError("--include-dirty requires the candidate revision to be HEAD")


def _default_base_ref(repo_path: Path) -> str:
    result = _git_result(
        repo_path, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    if result.returncode == 0:
        upstream = result.stdout.strip()
        if upstream:
            return upstream
    for candidate in ("origin/main", "origin/master", "main", "master"):
        if (
            _git_result(repo_path, "rev-parse", "--verify", f"{candidate}^{{commit}}").returncode
            == 0
        ):
            return candidate
    raise RevisionError("Could not infer a base ref; pass one explicitly, for example main...HEAD")


def _resolve_commit(repo_path: Path, ref: str) -> str:
    if ref.startswith("-") or any(character.isspace() for character in ref):
        raise RevisionError(f"Invalid Git ref: {ref!r}")
    result = _git_result(repo_path, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if result.returncode != 0:
        raise RevisionError(f"Could not resolve Git ref {ref!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git(repo_path: Path, *args: str) -> str:
    result = _git_result(repo_path, *args)
    if result.returncode != 0:
        raise RevisionError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _git_result(
    repo_path: Path,
    *args: str,
    max_stdout_bytes: int = _GIT_STDOUT_LIMIT,
    allow_stdout_truncation: bool = False,
) -> _GitResult:
    argv = sanitized_git_command(repo_path, *args)
    try:
        result = run_bounded_process(
            argv,
            env=sanitized_git_environment(),
            timeout=60,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=_GIT_STDERR_LIMIT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RevisionError(f"Could not run Git: {exc}") from exc
    if result.stderr_truncated:
        raise RevisionError(f"Git stderr exceeded {_GIT_STDERR_LIMIT} bytes")
    if result.stdout_truncated and not allow_stdout_truncation:
        raise RevisionError(f"Git output exceeded {max_stdout_bytes} bytes")
    return _GitResult(
        returncode=(
            0 if result.stdout_truncated and allow_stdout_truncation else result.returncode
        ),
        stdout=result.stdout.decode("utf-8", errors="replace"),
        stderr=result.stderr.decode("utf-8", errors="replace"),
        stdout_truncated=result.stdout_truncated,
    )
