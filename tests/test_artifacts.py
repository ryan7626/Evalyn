import json
import subprocess
import zipfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

from evalyn.adapters.registry import default_registry
from evalyn.artifacts import (
    ArtifactExportError,
    build_finding_bundle,
    promote_finding_source,
)
from evalyn.run_store import RunStore


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _stored_finding(tmp_path: Path, *, triage: str = "confirmed") -> tuple[RunStore, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    store = RunStore(repo, tmp_path / "state")
    run = store.create_run(base="a" * 40, candidate="b" * 40)
    finding_id = "EVL-1234ABCD"
    source = "def test_value():\n    assert True\n"
    artifact = default_registry().get("pytest").build_artifact(source)
    finding = {
        "finding_id": finding_id,
        "run_id": run.run_id,
        "title": "value remains compatible",
        "base_sha": "a" * 40,
        "candidate_sha": "b" * 40,
        "probe_source": source,
        "oracle": "value remains compatible",
        "oracle_provenance": "existing_test",
        "oracle_admissible": True,
        "semantic_state": "likely_regression",
        "triage_status": triage,
        "artifact": {
            "probe_id": artifact.probe_id,
            "adapter": artifact.adapter,
            "relative_path": artifact.relative_path.as_posix(),
            "command": list(artifact.command),
            "source_sha256": artifact.source_sha256,
            "byte_size": len(artifact.source.encode("utf-8")),
            "media_type": artifact.media_type,
        },
    }
    RunStore.write_json(
        run.path / "findings.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "findings": [finding],
        },
    )
    evidence_root = run.path / "findings" / finding_id
    RunStore.write_text(evidence_root / "test-source.py", source)
    RunStore.write_json(
        evidence_root / "reproduce.json",
        {
            "finding_id": finding_id,
            "base_sha": "a" * 40,
            "candidate_sha": "b" * 40,
            "adapter": artifact.adapter,
            "command": list(artifact.command),
            "source_sha256": artifact.source_sha256,
        },
    )
    store.finish_run(run, status="complete")
    return store, finding_id, source


def test_bundle_is_deterministic_and_every_payload_is_manifested(tmp_path: Path) -> None:
    store, finding_id, _source = _stored_finding(tmp_path)

    first = build_finding_bundle(store, finding_id)
    second = build_finding_bundle(store, finding_id)

    assert first == second
    with zipfile.ZipFile(BytesIO(first)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("bundle-manifest.json"))
        assert "probe/test-source.py" in names
        assert "probe/test.patch" in names
        assert set(manifest["files"]) == names - {"bundle-manifest.json"}
        for name, evidence in manifest["files"].items():
            payload = archive.read(name)
            assert sha256(payload).hexdigest() == evidence["sha256"]
            assert len(payload) == evidence["bytes"]


def test_only_confirmed_finding_can_be_promoted_without_overwrite(tmp_path: Path) -> None:
    store, finding_id, source = _stored_finding(tmp_path)
    destination = store.repo_path / "tests" / "test_evalyn_regression.py"

    promoted = promote_finding_source(store, finding_id, destination)

    assert promoted == destination
    assert destination.read_text(encoding="utf-8") == source
    with pytest.raises(ArtifactExportError, match="already exists"):
        promote_finding_source(store, finding_id, destination)


def test_unreviewed_finding_cannot_be_promoted(tmp_path: Path) -> None:
    store, finding_id, _source = _stored_finding(tmp_path, triage="unreviewed")

    with pytest.raises(ArtifactExportError, match="maintainer-confirmed"):
        promote_finding_source(store, finding_id, store.repo_path / "tests" / "test_probe.py")


def test_promotion_cannot_restore_or_replace_a_tracked_test_path(tmp_path: Path) -> None:
    store, finding_id, _source = _stored_finding(tmp_path)
    destination = store.repo_path / "tests" / "test_deleted.py"
    destination.parent.mkdir()
    destination.write_text("def test_old(): pass\n", encoding="utf-8")
    _git(store.repo_path, "add", ".")
    _git(store.repo_path, "commit", "-m", "tracked test")
    destination.unlink()

    with pytest.raises(ArtifactExportError, match="tracked repository path"):
        promote_finding_source(store, finding_id, destination)


def test_bundle_rejects_tampered_probe_source(tmp_path: Path) -> None:
    store, finding_id, _source = _stored_finding(tmp_path)
    run = store.list_runs(1)[0]
    payload = store.read_run_json(run, "findings.json")
    payload["findings"][0]["probe_source"] = "def test_tampered():\n    assert False\n"
    RunStore.write_json(run.path / "findings.json", payload)

    with pytest.raises(ArtifactExportError, match="integrity"):
        build_finding_bundle(store, finding_id)


def test_promotion_rejects_source_and_digest_tampered_together(tmp_path: Path) -> None:
    store, finding_id, _source = _stored_finding(tmp_path)
    run = store.list_runs(1)[0]
    payload = store.read_run_json(run, "findings.json")
    tampered = "def test_tampered():\n    assert False\n"
    payload["findings"][0]["probe_source"] = tampered
    payload["findings"][0]["artifact"]["source_sha256"] = sha256(
        tampered.encode("utf-8")
    ).hexdigest()
    RunStore.write_json(run.path / "findings.json", payload)
    destination = store.repo_path / "tests" / "test_evalyn_regression.py"

    with pytest.raises(ArtifactExportError, match="integrity"):
        promote_finding_source(store, finding_id, destination)

    assert not destination.exists()


def test_bundle_rejects_fully_rebuilt_json_metadata_when_run_owned_copy_differs(
    tmp_path: Path,
) -> None:
    store, finding_id, _source = _stored_finding(tmp_path)
    run = store.list_runs(1)[0]
    payload = store.read_run_json(run, "findings.json")
    tampered = "def test_tampered():\n    assert False\n"
    rebuilt = default_registry().get("pytest").build_artifact(tampered)
    payload["findings"][0]["probe_source"] = tampered
    payload["findings"][0]["artifact"] = {
        "probe_id": rebuilt.probe_id,
        "adapter": rebuilt.adapter,
        "relative_path": rebuilt.relative_path.as_posix(),
        "command": list(rebuilt.command),
        "source_sha256": rebuilt.source_sha256,
        "byte_size": len(rebuilt.source.encode("utf-8")),
        "media_type": rebuilt.media_type,
    }
    RunStore.write_json(run.path / "findings.json", payload)

    with pytest.raises(ArtifactExportError, match="Run-owned probe source"):
        build_finding_bundle(store, finding_id)
