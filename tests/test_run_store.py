import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evalyn.execution.lock import RepoRunLock
from evalyn.run_store import RunStore, repository_id


def test_run_store_lifecycle_is_external_to_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    store = RunStore(repo, state)

    run = store.create_run(base="abc", candidate="def")
    finished = store.finish_run(run, status="complete", extra={"finding_count": 1})

    assert not (repo / ".agent").exists()
    assert finished.path.is_relative_to(state)
    assert store.get(run.run_id).manifest["finding_count"] == 1


def test_find_finding_searches_recent_runs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    run = store.create_run(base="abc", candidate="def")
    store.write_json(
        run.path / "findings.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "findings": [{"id": "EVL-1234"}],
        },
    )

    found = store.find_finding("EVL-1234")

    assert found is not None
    assert found[0].run_id == run.run_id


def test_clean_removes_only_owned_expired_runs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    old = store.finish_run(store.create_run(base="a", candidate="b"), status="complete")
    manifest = dict(old.manifest)
    manifest["finished_at"] = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    store.write_json(old.path / "manifest.json", manifest)
    unrelated = store.runs_root / "unrelated"
    unrelated.mkdir()

    assert store.clean_before(30) == 1
    assert not old.path.exists()
    assert unrelated.exists()


def test_repository_id_is_stable_and_path_specific(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()

    assert repository_id(first) == repository_id(first)
    assert repository_id(first) != repository_id(second)


def test_corrupt_manifest_is_ignored(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    bad = store.runs_root / "bad"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("not json", encoding="utf-8")

    assert store.get("bad") is None
    assert json.loads(json.dumps([run.manifest for run in store.list_runs()])) == []


def test_run_with_mismatched_identity_is_ignored(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    run = store.create_run(base="abc", candidate="def")
    manifest = dict(run.manifest)
    manifest["repository"] = str(tmp_path / "other")
    store.write_json(run.path / "manifest.json", manifest)

    assert store.get(run.run_id) is None


def test_symlinked_run_directory_is_ignored(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    outside = tmp_path / "outside"
    outside.mkdir()
    store.write_json(
        outside / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "linked",
            "repository": str(repo.resolve()),
        },
    )
    store.runs_root.mkdir(parents=True)
    (store.runs_root / "linked").symlink_to(outside, target_is_directory=True)

    assert store.get("linked") is None


def test_find_finding_rejects_mismatched_file_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    run = store.create_run(base="abc", candidate="def")
    store.write_json(
        run.path / "findings.json",
        {
            "schema_version": 1,
            "run_id": "another-run",
            "findings": [{"id": "EVL-1234"}],
        },
    )

    assert store.find_finding("EVL-1234") is None


def test_state_root_cannot_live_inside_target_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    try:
        RunStore(repo, repo / ".evalyn-state")
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("internal state root was accepted")


def test_clean_skips_old_in_progress_run(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    active = store.create_run(base="a", candidate="b")
    manifest = dict(active.manifest)
    manifest["finished_at"] = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    store.write_json(active.path / "manifest.json", manifest)

    assert store.clean_before(30) == 0
    assert active.path.is_dir()


def test_clean_skips_all_runs_while_repository_lock_is_active(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    old = store.finish_run(store.create_run(base="a", candidate="b"), status="complete")
    manifest = dict(old.manifest)
    manifest["finished_at"] = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    store.write_json(old.path / "manifest.json", manifest)

    with RepoRunLock(repo, lock_path=store.repo_root / "run.lock"):
        assert store.clean_before(30) == 0
        assert old.path.is_dir()

    assert store.clean_before(30) == 1
    assert not old.path.exists()


def test_symlinked_runs_root_is_never_traversed_or_cleaned(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    outside = tmp_path / "outside"
    victim = outside / "victim"
    victim.mkdir(parents=True)
    sentinel = victim / "keep.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    (victim / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "victim",
                "repository": str(repo.resolve()),
                "status": "complete",
                "finished_at": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    store.repo_root.mkdir(parents=True)
    store.runs_root.symlink_to(outside, target_is_directory=True)

    assert store.get("victim") is None
    assert store.list_runs() == []
    assert store.clean_before(0) == 0
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    with pytest.raises(ValueError, match="unsafe"):
        store.create_run(base="a", candidate="b")


def test_symlinked_repository_state_root_is_never_traversed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    store.repo_root.parent.mkdir(parents=True)
    store.repo_root.symlink_to(outside, target_is_directory=True)

    assert store.get() is None
    assert store.list_runs() == []
    assert store.clean_before(0) == 0
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert not (outside / "run.lock").exists()
    assert not (outside / "runs").exists()
    with pytest.raises(ValueError, match="unsafe"):
        store.create_run(base="a", candidate="b")


def test_finish_run_rejects_run_directory_swapped_to_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    run = store.create_run(base="a", candidate="b")
    original = run.path.with_name(f"{run.run_id}-original")
    run.path.rename(original)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "manifest.json"
    sentinel.write_text("keep me", encoding="utf-8")
    run.path.symlink_to(outside, target_is_directory=True)

    with pytest.raises((OSError, ValueError)):
        store.finish_run(run, status="complete")

    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_atomic_json_write_rejects_final_and_parent_symlinks(tmp_path: Path) -> None:
    victim = tmp_path / "victim.json"
    victim.write_text("keep me", encoding="utf-8")
    linked_file = tmp_path / "state.json"
    linked_file.symlink_to(victim)

    with pytest.raises(ValueError, match="linked/non-file"):
        RunStore.write_json(linked_file, {"unsafe": True})
    assert victim.read_text(encoding="utf-8") == "keep me"

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        RunStore.write_json(linked_parent / "created.json", {"unsafe": True})
    assert not (outside / "created.json").exists()


def test_clean_unlinks_nested_symlink_without_deleting_its_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    old = store.finish_run(store.create_run(base="a", candidate="b"), status="complete")
    manifest = dict(old.manifest)
    manifest["finished_at"] = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    store.write_json(old.path / "manifest.json", manifest)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    (old.path / "linked-output").symlink_to(outside, target_is_directory=True)

    assert store.clean_before(30) == 1
    assert not old.path.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize(
    ("status", "finished_at"),
    [
        ("unknown", "2020-01-01T00:00:00+00:00"),
        ("complete", None),
        ("complete", "2020-01-01T00:00:00"),
        ("complete", "not-a-timestamp"),
    ],
)
def test_clean_retains_runs_without_a_terminal_aware_finish_time(
    tmp_path: Path,
    status: str,
    finished_at: str | None,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = RunStore(repo, tmp_path / "state")
    run = store.finish_run(store.create_run(base="a", candidate="b"), status="complete")
    manifest = dict(run.manifest)
    manifest["status"] = status
    manifest["finished_at"] = finished_at
    store.write_json(run.path / "manifest.json", manifest)

    assert store.clean_before(0) == 0
    assert run.path.is_dir()
