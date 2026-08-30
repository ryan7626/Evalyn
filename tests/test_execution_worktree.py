"""Tests for disposable baseline/candidate worktree sessions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from evalyn.execution.errors import WorktreeCleanupError, WorktreeError
from evalyn.execution.worktree import WorktreeSession, _run_bounded_process


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def revision_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "evalyn@example.test")
    _git(repo, "config", "user.name", "Evalyn Test")

    (repo / "app.py").write_text("VALUE = 'base'\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")

    (repo / "app.py").write_text("VALUE = 'candidate'\n", encoding="utf-8")
    (repo / "candidate.py").write_text("CANDIDATE = True\n", encoding="utf-8")
    _git(repo, "add", "app.py", "candidate.py")
    _git(repo, "commit", "-m", "candidate")
    candidate_sha = _git(repo, "rev-parse", "HEAD")
    return repo, base_sha, candidate_sha


def test_dual_worktrees_are_detached_and_source_checkout_is_untouched(
    revision_repo: tuple[Path, str, str],
    tmp_path: Path,
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    original_branch = _git(repo, "branch", "--show-current")
    temp_parent = tmp_path / "sessions"

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        temp_parent=temp_parent,
    ) as session:
        root = session.session_root
        baseline_path = session.baseline_path
        candidate_path = session.candidate_path
        assert session.baseline_sha == base_sha
        assert session.candidate_sha == candidate_sha
        assert (baseline_path / "app.py").read_text() == "VALUE = 'base'\n"
        assert not (baseline_path / "candidate.py").exists()
        assert (candidate_path / "app.py").read_text() == "VALUE = 'candidate'\n"
        assert _git(baseline_path, "branch", "--show-current") == ""
        assert _git(candidate_path, "branch", "--show-current") == ""

    assert root is not None and not root.exists()
    assert not baseline_path.exists()
    assert not candidate_path.exists()
    assert _git(repo, "branch", "--show-current") == original_branch
    assert _git(repo, "rev-parse", "HEAD") == candidate_sha


def test_worktree_git_environment_disables_replacement_objects(
    revision_repo: tuple[Path, str, str],
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    session = WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
    )

    assert session._git_environment()["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_dirty_overlay_is_candidate_only_and_survives_candidate_restore(
    revision_repo: tuple[Path, str, str],
    tmp_path: Path,
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    (repo / "app.py").write_text("VALUE = 'dirty user work'\n", encoding="utf-8")
    (repo / "notes.txt").write_text("untracked user work\n", encoding="utf-8")
    source_status = _git(repo, "status", "--short")

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        include_dirty_overlay=True,
        temp_parent=tmp_path / "sessions",
    ) as session:
        assert not session.overlay.is_empty
        assert session.overlay.untracked_paths == ("notes.txt",)
        assert (session.baseline_path / "app.py").read_text() == "VALUE = 'base'\n"
        assert not (session.baseline_path / "notes.txt").exists()
        assert (session.candidate_path / "app.py").read_text() == "VALUE = 'dirty user work'\n"
        assert (session.candidate_path / "notes.txt").read_text() == "untracked user work\n"

        (session.candidate_path / "app.py").write_text("VALUE = 'bad repair'\n", encoding="utf-8")
        (session.candidate_path / "repair.tmp").write_text("remove me\n", encoding="utf-8")
        session.restore_candidate()

        assert (session.candidate_path / "app.py").read_text() == "VALUE = 'dirty user work'\n"
        assert (session.candidate_path / "notes.txt").read_text() == "untracked user work\n"
        assert not (session.candidate_path / "repair.tmp").exists()

    assert (repo / "app.py").read_text() == "VALUE = 'dirty user work'\n"
    assert (repo / "notes.txt").read_text() == "untracked user work\n"
    assert _git(repo, "status", "--short") == source_status


def test_candidate_patch_can_be_limited_to_validated_fix_paths(
    revision_repo: tuple[Path, str, str],
    tmp_path: Path,
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    output = tmp_path / "artifacts" / "fix.patch"

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
    ) as session:
        (session.candidate_path / "app.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
        (session.candidate_path / "candidate.py").write_text(
            "CANDIDATE = False\n", encoding="utf-8"
        )
        written = session.write_candidate_patch(output, paths=["app.py"])

        assert written == output
        patch = output.read_text(encoding="utf-8")
        assert "VALUE = 'fixed'" in patch
        assert "candidate.py" not in patch


def test_dirty_overlay_rejects_non_head_candidate(
    revision_repo: tuple[Path, str, str],
) -> None:
    repo, base_sha, _ = revision_repo

    with pytest.raises(WorktreeError, match="only be applied"):
        WorktreeSession(
            repo,
            baseline_revision=base_sha,
            candidate_revision=base_sha,
            include_dirty_overlay=True,
        ).start()


def test_invalid_patch_path_is_rejected(
    revision_repo: tuple[Path, str, str],
) -> None:
    repo, base_sha, candidate_sha = revision_repo

    with (
        WorktreeSession(
            repo,
            baseline_revision=base_sha,
            candidate_revision=candidate_sha,
        ) as session,
        pytest.raises(WorktreeError, match="Unsafe Git-relative"),
    ):
        session.candidate_diff(["../outside.py"])


def test_session_root_cannot_be_created_inside_source_checkout(
    revision_repo: tuple[Path, str, str],
) -> None:
    repo, base_sha, candidate_sha = revision_repo

    with pytest.raises(WorktreeError, match="outside the source checkout"):
        WorktreeSession(
            repo,
            baseline_revision=base_sha,
            candidate_revision=candidate_sha,
            temp_parent=repo / ".agent" / "worktrees",
        ).start()


def test_cleanup_refuses_unknown_files_and_retains_recovery_manifest(
    revision_repo: tuple[Path, str, str],
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    session = WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
    ).start()
    assert session.session_root is not None
    assert session.manifest_path is not None
    unexpected = session.session_root / "not-owned-by-evalyn.txt"
    unexpected.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(WorktreeCleanupError, match="unowned files"):
        session.cleanup()

    assert unexpected.exists()
    assert session.manifest_path.exists()
    assert not session.baseline_path.exists()
    assert not session.candidate_path.exists()

    unexpected.unlink()
    session.cleanup()
    assert not session.session_root.exists()


def test_worktree_creation_suppresses_repository_checkout_hooks(
    revision_repo: tuple[Path, str, str],
    tmp_path: Path,
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    hooks = tmp_path / "hostile-hooks"
    hooks.mkdir()
    sentinel = tmp_path / "hook-ran"
    hook = hooks / "post-checkout"
    hook.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(sentinel)!r}).touch()\n",
        encoding="utf-8",
    )
    hook.chmod(0o700)
    _git(repo, "config", "core.hooksPath", str(hooks))

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        temp_parent=tmp_path / "sessions",
    ):
        pass

    assert not sentinel.exists()


def test_worktree_creation_disables_configured_smudge_filters(
    revision_repo: tuple[Path, str, str],
    tmp_path: Path,
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    sentinel = tmp_path / "filter-ran"
    script = tmp_path / "filter.py"
    script.write_text(
        "import pathlib, sys\n"
        f"pathlib.Path({str(sentinel)!r}).touch()\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    (repo / ".gitattributes").write_text("app.py filter=hostile\n", encoding="utf-8")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-m", "declare filter")
    filtered_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "config", "filter.hostile.smudge", f"{sys.executable} {script}")
    _git(repo, "config", "filter.hostile.required", "true")

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=filtered_sha,
        temp_parent=tmp_path / "sessions",
    ) as session:
        assert (session.candidate_path / "app.py").read_text(encoding="utf-8") == (
            "VALUE = 'candidate'\n"
        )

    assert not sentinel.exists()


def test_dirty_overlay_rejects_oversized_untracked_file_before_opening_it(
    revision_repo: tuple[Path, str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base_sha, candidate_sha = revision_repo
    oversized = repo / "oversized.bin"
    oversized.write_bytes(b"x" * 4096)
    real_open = __import__("os").open

    def guarded_open(path, flags, *args, **kwargs):
        if not isinstance(path, int) and Path(path) == oversized:
            pytest.fail("oversized untracked content was opened before its size was rejected")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("evalyn.execution.worktree.os.open", guarded_open)

    with pytest.raises(WorktreeError, match="Dirty overlay exceeds 128 bytes"):
        WorktreeSession(
            repo,
            baseline_revision=base_sha,
            candidate_revision=candidate_sha,
            include_dirty_overlay=True,
            temp_parent=tmp_path / "sessions",
            max_overlay_bytes=128,
        ).start()


def test_candidate_diff_fails_closed_at_its_streaming_byte_limit(
    revision_repo: tuple[Path, str, str],
) -> None:
    repo, base_sha, candidate_sha = revision_repo

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        max_candidate_diff_bytes=128,
    ) as session:
        (session.candidate_path / "app.py").write_text("x" * 10_000, encoding="utf-8")

        with pytest.raises(WorktreeError, match="Candidate diff exceeds 128 bytes"):
            session.candidate_diff()


def test_bounded_capture_drains_stderr_without_deadlocking() -> None:
    result = _run_bounded_process(
        [
            sys.executable,
            "-c",
            "import os; os.write(2, b'e' * 2_000_000); os.write(1, b'done')",
        ],
        env={"PATH": ""},
        timeout=5,
        max_stdout_bytes=64,
        max_stderr_bytes=64,
    )

    assert result.returncode != 0
    assert result.stderr_truncated is True
    assert result.stderr == b"e" * 64
