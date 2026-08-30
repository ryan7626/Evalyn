import json
import subprocess
from pathlib import Path

import pytest

import evalyn.operations as operations_module
from evalyn.engine.models import ExecutionResult, ExecutionStatus
from evalyn.execution import RepoRunLock
from evalyn.llm.base import BaseLLM
from evalyn.operations import (
    OperationError,
    reproduce_saved_finding,
    triage_saved_finding,
)
from evalyn.run_store import RunStore
from evalyn.workflow import CheckOptions, run_check


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class LLM(BaseLLM):
    def complete(self, system_prompt, user_prompt, *, max_tokens=4096):
        raise AssertionError

    def complete_json(self, system_prompt, user_prompt, *, max_tokens=4096):
        return {
            "probes": [
                {
                    "title": "value is stable",
                    "source": "from value import value\ndef test_value():\n    assert value() == 1",
                    "oracle": "the documentation guarantees value returns one",
                    "oracle_provenance": "documentation",
                    "changed_file": "value.py",
                    "changed_symbol": "value",
                }
            ]
        }


class Executor:
    def __init__(self):
        self.setup_errors = {}

    def prepare(self, revisions):
        self.base = revisions[0][1].sha

    def execute(self, revision, artifact, *, repetition):
        if revision.sha == self.base:
            return ExecutionResult.passed(metadata={"environment_integrity": "verified"})
        return ExecutionResult.failed(
            failure_signature="changed",
            metadata={"environment_integrity": "verified"},
        )


def make_run(tmp_path: Path, *, include_dirty: bool = False):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "api.md").write_text("The documentation guarantees value returns one.\n")
    (repo / "value.py").write_text("def value(): return 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    git(repo, "checkout", "-b", "feature")
    (repo / "value.py").write_text("def value(): return 2\n")
    git(repo, "commit", "-am", "change")
    if include_dirty:
        (repo / "value.py").write_text("def value(): return 3\n")
    state = tmp_path / "state"
    outcome = run_check(
        CheckOptions(
            repo,
            range_spec="main",
            state_root=state,
            include_dirty=include_dirty,
            intent=("preserve: The documentation guarantees value returns one.",),
        ),
        llm=LLM(),
        executor_factory=lambda *_args: Executor(),
    )
    return repo, RunStore(repo, state), outcome.findings[0].finding_id


def test_triage_updates_all_persisted_primary_outputs(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)

    run, finding = triage_saved_finding(
        store, finding_id, status="dismissed", reason="documented API change"
    )

    assert finding["should_gate"] is False
    assert finding["triage_history"][0]["status"] == "dismissed"
    assert "dismissed" in (run.path / "report.md").read_text()
    markdown = (run.path / "report.md").read_text()
    assert "Oracle source:" in markdown
    assert "Static target binding:" in markdown
    assert "Environment integrity:" in markdown
    assert run.manifest["gating_finding_count"] == 0
    assert run.manifest["verdict"] == "change_observed"
    stored_manifest = RunStore.read_json(run.path / "manifest.json")
    assert stored_manifest["gating_finding_count"] == 0
    assert stored_manifest["verdict"] == "change_observed"
    payload = RunStore.read_json(run.path / "findings.json")
    assert payload["verdict"] == "change_observed"
    assert payload["results"][0]["findings"][0]["triage_status"] == "dismissed"
    assert payload["results"][0]["gating_finding_count"] == 0
    assert payload["results"][0]["verdict"] == "change_observed"
    sarif = RunStore.read_json(run.path / "evalyn.sarif")
    assert sarif["runs"][0]["results"][0]["level"] == "note"
    assert sarif["runs"][0]["results"][0]["ruleId"] == "evalyn/behavior-change"
    assert 'failures="0"' in (run.path / "junit.xml").read_text(encoding="utf-8")
    assert 'skipped="1"' in (run.path / "junit.xml").read_text(encoding="utf-8")


def test_triage_rolls_back_every_output_when_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]
    names = ("manifest.json", "findings.json", "report.md", "junit.xml", "evalyn.sarif")
    before = {name: (run.path / name).read_bytes() for name in names}
    real_replace = operations_module.os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if Path(destination).name == "findings.json" and not failed:
            failed = True
            raise OSError("injected commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(operations_module.os, "replace", fail_once)

    with pytest.raises(OperationError, match="transactionally"):
        triage_saved_finding(
            store,
            finding_id,
            status="dismissed",
            reason="must not partially persist",
        )

    assert failed
    assert {name: (run.path / name).read_bytes() for name in names} == before


def test_triage_reports_repository_lock_contention_as_operation_error(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)

    with (
        RepoRunLock(store.repo_path, lock_path=store.repo_root / "run.lock"),
        pytest.raises(OperationError, match="repository operation lock"),
    ):
        triage_saved_finding(
            store,
            finding_id,
            status="dismissed",
            reason="must wait for the active operation",
        )

    located = store.find_finding(finding_id)
    assert located is not None
    assert located[1]["triage_status"] == "unreviewed"


def test_reproduce_uses_saved_probe_and_revisions(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)

    result = reproduce_saved_finding(
        store,
        finding_id,
        executor_factory=lambda *_args: Executor(),
    )

    assert [item.status.value for item in result.base] == ["passed"] * 3
    assert [item.status.value for item in result.candidate] == ["failed"] * 3
    assert result.verdict == "reproduced"

    run = store.list_runs(limit=1)[0]
    recorded = RunStore.read_json(run.path / "reproductions" / finding_id / "latest.json")
    assert recorded["verdict"] == "reproduced"
    assert recorded["expected_candidate_status"] == "failed"
    assert recorded["expected_failure_signature"] == "changed"


def test_production_replay_requires_saved_environment_identity(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)

    with pytest.raises(OperationError, match="execution-environment identity"):
        reproduce_saved_finding(store, finding_id)


def test_production_replay_passes_saved_image_and_endpoint_expectations_to_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]
    manifest = dict(run.manifest)
    manifest["environments"] = {
        "pytest": {
            "resolved_image_identity": "sha256:" + "a" * 64,
            "endpoint_scope": "local",
        }
    }
    RunStore.write_json(run.path / "manifest.json", manifest)
    observed = {}

    def capturing_factory(*_args, **kwargs):
        observed.update(kwargs)
        return Executor()

    monkeypatch.setattr(operations_module, "_reproduce_factory", capturing_factory)

    result = reproduce_saved_finding(store, finding_id)

    assert result.reproduced
    assert observed["expected_image_identity"] == "sha256:" + "a" * 64
    assert observed["expected_endpoint_scope"] == "local"


def test_reproduction_verdict_fails_closed_for_mismatches_and_infrastructure() -> None:
    mismatch = operations_module.ReproductionResult(
        "EVL-1234ABCD",
        (ExecutionResult.passed(),),
        (ExecutionResult.passed(),),
        expected_candidate_status=ExecutionStatus.FAILED,
        expected_failure_signature="saved-signature",
    )
    incomplete = operations_module.ReproductionResult(
        "EVL-1234ABCD",
        (ExecutionResult.passed(),),
        (
            ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=1,
                stderr="runtime unavailable",
            ),
        ),
    )

    assert mismatch.verdict == "not_reproduced"
    assert mismatch.reproduced is False
    assert mismatch.incomplete is False
    assert incomplete.verdict == "incomplete"
    assert incomplete.incomplete is True


def test_reproduction_requires_the_saved_failure_status_and_signature() -> None:
    different_failure = operations_module.ReproductionResult(
        "EVL-1234ABCD",
        (ExecutionResult.passed(),),
        (ExecutionResult.failed(failure_signature="different"),),
        expected_candidate_status=ExecutionStatus.FAILED,
        expected_failure_signature="saved",
    )
    timeout_instead = operations_module.ReproductionResult(
        "EVL-1234ABCD",
        (ExecutionResult.passed(),),
        (
            ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
            ),
        ),
        expected_candidate_status=ExecutionStatus.FAILED,
        expected_failure_signature="saved",
    )

    assert different_failure.verdict == "not_reproduced"
    assert timeout_instead.verdict == "not_reproduced"


def test_remote_endpoint_replay_requires_saved_explicit_operator_opt_in(
    tmp_path: Path,
) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]
    manifest = dict(run.manifest)
    configuration = dict(manifest["configuration"])
    sandbox = dict(configuration["sandbox"])
    sandbox["allow_remote_endpoint"] = True
    configuration["sandbox"] = sandbox
    manifest["configuration"] = configuration
    RunStore.write_json(run.path / "manifest.json", manifest)

    with pytest.raises(OperationError, match="explicit operator flag"):
        reproduce_saved_finding(
            store,
            finding_id,
            executor_factory=lambda *_args: Executor(),
        )

    result = reproduce_saved_finding(
        store,
        finding_id,
        allow_remote_endpoint=True,
        executor_factory=lambda *_args: Executor(),
    )

    assert result.reproduced is True


def test_reproduce_locks_before_reading_or_creating_outputs(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]

    with (
        RepoRunLock(store.repo_path, lock_path=store.repo_root / "run.lock"),
        pytest.raises(OperationError, match="repository operation lock"),
    ):
        reproduce_saved_finding(
            store,
            finding_id,
            executor_factory=lambda *_args: Executor(),
        )

    assert not (run.path / "reproductions").exists()


def test_reproduce_rejects_tampered_derived_probe_metadata(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]
    findings_path = run.path / "findings.json"
    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    payload["findings"][0]["artifact"]["relative_path"] = "tests/other.py"
    RunStore.write_json(findings_path, payload)

    try:
        reproduce_saved_finding(
            store,
            finding_id,
            executor_factory=lambda *_args: Executor(),
        )
    except RuntimeError as exc:
        assert "relative_path" in str(exc)
    else:
        raise AssertionError("tampered artifact metadata was accepted")


def test_reproduce_uses_saved_execution_configuration(tmp_path: Path) -> None:
    repo, store, finding_id = make_run(tmp_path)
    (repo / ".evalyn.yml").write_text(
        "version: 1\n"
        "sandbox:\n"
        "  runtime: podman\n"
        "adapters:\n"
        "  - name: pytest\n"
        "    image: hostile.example/changed:latest\n"
        "    setup_argv: [changed-setup]\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def capturing_factory(_adapter, settings, _path, runtime, budget):
        captured.update(settings=settings, runtime=runtime, budget=budget)
        return Executor()

    reproduce_saved_finding(store, finding_id, executor_factory=capturing_factory)

    saved = captured["settings"]
    assert saved.image == "python:3.12-slim"
    assert saved.setup_argv == ()
    assert captured["runtime"] is None
    assert captured["budget"] == 600


def test_reproduce_rejects_non_oid_revision_metadata(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]
    manifest = dict(run.manifest)
    manifest["base"] = "main"
    RunStore.write_json(run.path / "manifest.json", manifest)

    with pytest.raises(OperationError, match="full immutable Git object IDs"):
        reproduce_saved_finding(
            store,
            finding_id,
            executor_factory=lambda *_args: Executor(),
        )


def test_reproduce_verifies_saved_dirty_overlay_digest(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path, include_dirty=True)
    run = store.list_runs(limit=1)[0]
    patch_path = run.path / "dirty.patch"
    patch_path.write_bytes(patch_path.read_bytes() + b"\n# tampered\n")

    with pytest.raises(OperationError, match="failed its integrity check"):
        reproduce_saved_finding(
            store,
            finding_id,
            executor_factory=lambda *_args: Executor(),
        )


def test_reproduce_git_ignores_ambient_repository_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, store, finding_id = make_run(tmp_path, include_dirty=True)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "hostile-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "hostile-work-tree"))

    result = reproduce_saved_finding(
        store,
        finding_id,
        executor_factory=lambda *_args: Executor(),
    )

    assert [item.status.value for item in result.base] == ["passed"] * 3
    assert [item.status.value for item in result.candidate] == ["failed"] * 3


class LookupTrap:
    def find_finding(self, finding_id: str):
        raise AssertionError(f"unsafe finding ID reached lookup: {finding_id}")


@pytest.mark.parametrize(
    "finding_id",
    (
        "../EVL-1234ABCD",
        "EVL-1234",
        "EVL-1234abcd",
        "EVL-1234ABCD/../../outside",
    ),
)
def test_reproduce_rejects_invalid_finding_id_before_lookup(finding_id: str) -> None:
    with pytest.raises(OperationError, match="must match EVL-XXXXXXXX"):
        reproduce_saved_finding(LookupTrap(), finding_id)


def test_triage_rejects_invalid_finding_id_before_lookup() -> None:
    with pytest.raises(OperationError, match="must match EVL-XXXXXXXX"):
        triage_saved_finding(
            LookupTrap(),
            "../EVL-1234ABCD",
            status="confirmed",
            reason="unsafe identifier",
        )


def test_reproduce_rejects_symlinked_output_directory(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.path / "reproductions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OperationError, match="cannot be a symlink"):
        reproduce_saved_finding(
            store,
            finding_id,
            executor_factory=lambda *_args: Executor(),
        )

    assert list(outside.iterdir()) == []


def test_reproduce_rejects_symlinked_finding_output(tmp_path: Path) -> None:
    _repo, store, finding_id = make_run(tmp_path)
    run = store.list_runs(limit=1)[0]
    outside = tmp_path / "outside"
    outside.mkdir()
    reproductions = run.path / "reproductions"
    reproductions.mkdir()
    (reproductions / finding_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(OperationError, match="cannot be a symlink"):
        reproduce_saved_finding(
            store,
            finding_id,
            executor_factory=lambda *_args: Executor(),
        )

    assert list(outside.iterdir()) == []
