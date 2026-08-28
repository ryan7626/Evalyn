import subprocess
from pathlib import Path

import pytest

from evalyn.revisions import (
    RevisionError,
    changed_files,
    get_diff,
    resolve_range,
    validate_git_repository,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "value.txt").write_text("base\n", encoding="utf-8")
    git(tmp_path, "add", "value.txt")
    git(tmp_path, "commit", "-m", "base")
    git(tmp_path, "checkout", "-b", "feature")
    (tmp_path / "value.txt").write_text("candidate\n", encoding="utf-8")
    git(tmp_path, "commit", "-am", "candidate")
    return tmp_path


def test_resolves_single_base_to_head(repo: Path) -> None:
    revisions = resolve_range(repo, "main")

    assert revisions.base_sha == git(repo, "rev-parse", "main")
    assert revisions.candidate_sha == git(repo, "rev-parse", "HEAD")
    assert changed_files(repo, revisions) == ("value.txt",)
    assert "+candidate" in get_diff(repo, revisions)


def test_resolves_explicit_three_dot_range(repo: Path) -> None:
    revisions = resolve_range(repo, "main...feature")

    assert revisions.candidate_ref == "feature"


def test_default_falls_back_to_main(repo: Path) -> None:
    assert resolve_range(repo).base_ref == "main"


@pytest.mark.parametrize("value", ["main..feature", "main...", "...feature"])
def test_rejects_invalid_ranges(repo: Path, value: str) -> None:
    with pytest.raises(RevisionError):
        resolve_range(repo, value)


def test_rejects_no_change_range(repo: Path) -> None:
    with pytest.raises(RevisionError, match="no committed candidate changes"):
        resolve_range(repo, "HEAD...HEAD")


def test_allows_same_commit_for_an_explicit_dirty_snapshot(repo: Path) -> None:
    revisions = resolve_range(repo, "HEAD...HEAD", allow_same=True)

    assert revisions.base_sha == revisions.candidate_sha


def test_validate_returns_repository_root(repo: Path) -> None:
    child = repo / "nested"
    child.mkdir()

    assert validate_git_repository(child) == repo


def test_large_diff_is_stream_truncated_to_the_requested_character_budget(repo: Path) -> None:
    (repo / "value.txt").write_text("candidate\n" + ("x" * 1_000_000), encoding="utf-8")
    git(repo, "commit", "-am", "large candidate")
    revisions = resolve_range(repo, "main...HEAD")

    rendered = get_diff(repo, revisions, max_chars=256)

    body, marker = rendered.split("\n\n", 1)
    assert len(body) <= 256
    assert marker.startswith("[truncated")
