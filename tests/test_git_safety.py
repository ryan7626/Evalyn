"""Tests for host Git configuration neutralization."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import evalyn.execution.git_safety as git_safety
from evalyn.execution.bounded import BoundedProcessResult
from evalyn.execution.git_safety import UnsafeGitConfiguration
from evalyn.revisions import get_diff, is_dirty, resolve_range


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def test_read_only_git_inspection_neutralizes_filters_and_textconv(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "evalyn@example.test")
    _git(repo, "config", "user.name", "Evalyn Test")
    (repo / ".gitattributes").write_text(
        "value.txt filter=hostile diff=hostile\n",
        encoding="utf-8",
    )
    (repo / "value.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".gitattributes", "value.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "feature")
    (repo / "value.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "commit", "-am", "candidate")

    sentinel = tmp_path / "git-driver-ran"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import pathlib, sys\n"
        f"pathlib.Path({str(sentinel)!r}).touch()\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    command = f"{sys.executable} {driver}"
    _git(repo, "config", "filter.hostile.clean", command)
    _git(repo, "config", "filter.hostile.smudge", command)
    _git(repo, "config", "filter.hostile.required", "true")
    _git(repo, "config", "diff.hostile.textconv", command)

    revisions = resolve_range(repo, "main...HEAD")
    assert "+candidate" in get_diff(repo, revisions)
    (repo / "value.txt").write_text("dirty candidate\n", encoding="utf-8")
    assert is_dirty(repo)
    assert not sentinel.exists()


def test_filter_configuration_output_is_bounded_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        git_safety,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            args=("git",),
            returncode=125,
            stdout=b"filter.hostile.clean\n",
            stderr=b"",
            stdout_truncated=True,
        ),
    )

    with pytest.raises(UnsafeGitConfiguration, match="too large"):
        git_safety.sanitized_git_command(tmp_path, "status")
