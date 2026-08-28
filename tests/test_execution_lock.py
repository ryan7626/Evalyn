"""Tests for repository-scoped execution locking."""

from pathlib import Path

import pytest

import evalyn.execution.lock as lock_module
from evalyn.execution.errors import RunLockError
from evalyn.execution.lock import RepoRunLock, read_lock_metadata


def test_lock_records_owner_and_releases_for_next_owner(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lock_path = tmp_path / "locks" / "evalyn.lock"

    with RepoRunLock(repo, lock_path=lock_path) as first:
        assert first.is_acquired
        metadata = read_lock_metadata(lock_path)
        assert metadata is not None
        assert metadata["repository"] == str(repo.resolve())
        assert metadata["token"] == first.token

    assert not first.is_acquired
    with RepoRunLock(repo, lock_path=lock_path) as second:
        assert second.is_acquired
        assert second.token != first.token


def test_second_lock_in_same_process_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lock_path = tmp_path / "run.lock"

    with RepoRunLock(repo, lock_path=lock_path), pytest.raises(RunLockError, match="already holds"):
        RepoRunLock(repo, lock_path=lock_path).acquire()


def test_release_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lock = RepoRunLock(repo, lock_path=tmp_path / "run.lock").acquire()

    lock.release()
    lock.release()

    assert not lock.is_acquired


def test_preplanted_lock_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    lock_path = tmp_path / "locks" / "run.lock"
    lock_path.parent.mkdir()
    lock_path.symlink_to(victim)

    with pytest.raises(RunLockError, match="Unsafe|regular file"):
        RepoRunLock(repo, lock_path=lock_path).acquire()

    assert victim.read_text(encoding="utf-8") == "keep me"
    assert read_lock_metadata(lock_path) is None


def test_lock_symlink_added_after_construction_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    lock_path = tmp_path / "locks" / "run.lock"
    lock = RepoRunLock(repo, lock_path=lock_path)
    lock_path.parent.mkdir()
    lock_path.symlink_to(victim)

    with pytest.raises(RunLockError):
        lock.acquire()

    assert victim.read_text(encoding="utf-8") == "keep me"


def test_lock_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "locks"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RunLockError):
        RepoRunLock(repo, lock_path=linked_parent / "run.lock").acquire()

    assert not (outside / "run.lock").exists()


def test_lock_rejects_hardlink_without_truncating_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    lock_path = tmp_path / "run.lock"
    lock_path.hardlink_to(victim)

    with pytest.raises(RunLockError, match="regular file"):
        RepoRunLock(repo, lock_path=lock_path).acquire()

    assert victim.read_text(encoding="utf-8") == "keep me"


def test_final_path_swap_after_open_never_truncates_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    lock_path = tmp_path / "run.lock"
    original_lock = lock_module._lock_file

    def swap_after_open(handle) -> None:
        original_lock(handle)
        lock_path.unlink()
        lock_path.symlink_to(victim)

    monkeypatch.setattr(lock_module, "_lock_file", swap_after_open)

    with pytest.raises(RunLockError, match="changed"):
        RepoRunLock(repo, lock_path=lock_path).acquire()

    assert victim.read_text(encoding="utf-8") == "keep me"


def test_oversized_lock_metadata_is_not_loaded_unbounded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    lock_path = tmp_path / "run.lock"
    lock_path.write_bytes(b"x" * (16 * 1024 + 1))

    assert read_lock_metadata(lock_path) is None
    with RepoRunLock(repo, lock_path=lock_path):
        metadata = read_lock_metadata(lock_path)
        assert metadata is not None
        assert metadata["repository"] == str(repo.resolve())
