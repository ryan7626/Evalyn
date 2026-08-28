"""Sanitized host-Git invocation for read-only repository inspection."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from evalyn.execution.bounded import run_bounded_process


class UnsafeGitConfiguration(ValueError):
    """Repository Git configuration cannot be made non-executable safely."""


_FILTER_KEY = re.compile(
    r"filter\.([A-Za-z0-9][A-Za-z0-9._-]{0,100})\."
    r"(?:clean|smudge|process|required)\Z",
    re.IGNORECASE,
)
_FILTER_CONFIG_LIMIT = 256 * 1024
_GIT_STDERR_LIMIT = 64 * 1024


def sanitized_git_environment(*, home: Path | None = None) -> dict[str, str]:
    """Return a minimal environment without user/system Git configuration."""
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str((home or Path(tempfile.gettempdir())).resolve()),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", os.defpath),
    }


def sanitized_git_command(repo_path: Path, *args: str) -> list[str]:
    """Build a Git argv that neutralizes host-executable repository config."""
    repo = Path(repo_path).resolve()
    base_config = [
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "submodule.recurse=false",
    ]
    names = _configured_filter_names(repo, base_config)
    filter_config: list[str] = []
    for name in names:
        filter_config.extend(
            (
                "-c",
                f"filter.{name}.clean=",
                "-c",
                f"filter.{name}.smudge=",
                "-c",
                f"filter.{name}.process=",
                "-c",
                f"filter.{name}.required=false",
            )
        )
    return ["git", *base_config, *filter_config, "-C", str(repo), *args]


def _configured_filter_names(repo: Path, base_config: list[str]) -> tuple[str, ...]:
    try:
        completed = run_bounded_process(
            [
                "git",
                *base_config,
                "-C",
                str(repo),
                "config",
                "--includes",
                "--name-only",
                "--get-regexp",
                r"^filter\..*\.(clean|smudge|process|required)$",
            ],
            timeout=10,
            env=sanitized_git_environment(),
            max_stdout_bytes=_FILTER_CONFIG_LIMIT,
            max_stderr_bytes=_GIT_STDERR_LIMIT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UnsafeGitConfiguration(f"Could not inspect repository Git filters: {exc}") from exc
    if completed.stdout_truncated or completed.stderr_truncated:
        raise UnsafeGitConfiguration("Repository Git filter configuration is too large")
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode not in {0, 1}:
        detail = stderr.strip() or "git config failed"
        raise UnsafeGitConfiguration(f"Could not inspect repository Git filters: {detail}")
    names: set[str] = set()
    for key in stdout.splitlines():
        match = _FILTER_KEY.fullmatch(key)
        if match is None:
            raise UnsafeGitConfiguration(f"Unsafe configured Git filter name: {key!r}")
        names.add(match.group(1))
    return tuple(sorted(names, key=str.casefold))
