"""Safety and behavior tests for the experimental patch-only repair backend."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import evalyn.repair_v2 as repair_module
from evalyn.adapters.registry import default_registry
from evalyn.engine.models import ExecutionResult, Revision
from evalyn.execution import RepoRunLock, RunLockError
from evalyn.llm.base import BaseLLM
from evalyn.repair_v2 import RepairError, _is_source, repair_saved_finding
from evalyn.run_store import RunStore

FINDING_ID = "EVL-1234ABCD"
BROKEN = "def value() -> int:\n    return 2\n"
FIXED = "def value() -> int:\n    return 1\n"


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


class ReplyLLM(BaseLLM):
    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> str:
        raise AssertionError("repair must use structured completion")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> dict | list:
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_prompt)
        if not self.replies:
            raise AssertionError("unexpected repair request")
        return self.replies.pop(0)


class RecordingFactory:
    def __init__(
        self,
        *,
        suite_passes: bool = True,
        mutate_target: bool = False,
        mutate_untracked_on_call: int | None = None,
        candidate_signature: str = "value-changed",
    ) -> None:
        self.suite_passes = suite_passes
        self.mutate_target = mutate_target
        self.mutate_untracked_on_call = mutate_untracked_on_call
        self.candidate_signature = candidate_signature
        self.workspaces: list[Path] = []
        self.prepared_contents: list[str] = []
        self.probe_repetitions: list[int] = []
        self.suite_argv: list[tuple[str, ...]] = []
        self.runtimes: list[str | None] = []
        self.adapter_settings: list[Any] = []
        self.budgets: list[int] = []
        self.calls = 0

    def __call__(self, _adapter, settings, _run_path, runtime, budget):
        self.calls += 1
        self.runtimes.append(runtime)
        self.adapter_settings.append(settings)
        self.budgets.append(budget)
        return RecordingExecutor(self, self.calls)


class RecordingExecutor:
    def __init__(self, owner: RecordingFactory, factory_call: int) -> None:
        self.owner = owner
        self.factory_call = factory_call
        self.workspace: Path | None = None

    @property
    def setup_errors(self) -> dict[Path, str]:
        return {}

    def prepare(self, revision: Revision) -> None:
        self.workspace = revision.workspace
        self.owner.workspaces.append(revision.workspace)
        self.owner.prepared_contents.append(
            (revision.workspace / "src" / "value.py").read_text(encoding="utf-8")
        )

    def validate_probe(self, revision, _artifact, *, repetitions):
        self.owner.probe_repetitions.append(repetitions)
        content = (revision.workspace / "src" / "value.py").read_text(encoding="utf-8")
        result = (
            ExecutionResult.passed()
            if content == FIXED
            else ExecutionResult.failed(failure_signature=self.owner.candidate_signature)
        )
        return tuple(result for _ in range(repetitions))

    def validate_suite(self, revision, argv):
        self.owner.suite_argv.append(argv)
        if self.owner.mutate_target:
            (revision.workspace / "src" / "value.py").write_text(
                "def value() -> int:\n    return 99\n",
                encoding="utf-8",
            )
        if self.owner.mutate_untracked_on_call == self.factory_call:
            (revision.workspace / "unexpected-validation-output.txt").write_text(
                "unexpected\n",
                encoding="utf-8",
            )
        if self.owner.suite_passes:
            return ExecutionResult.passed()
        return ExecutionResult.failed(stderr="existing suite regression")


def _proposal(content: str = FIXED, *, path: str = "src/value.py") -> dict[str, str]:
    return {
        "file_path": path,
        "replacement_content": content,
        "explanation": "restore the documented result",
    }


def _saved_finding(
    tmp_path: Path,
    *,
    extra_source_change: bool = False,
    dirty_content: str | None = None,
    recorded_runtime: str = "auto",
    recorded_repeat: int = 3,
    recorded_image: str = "python:3.12-slim",
    recorded_setup: tuple[str, ...] = (),
    recorded_suite: tuple[str, ...] = (),
    recorded_budget: int = 600,
) -> tuple[Path, RunStore, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "evalyn@example.test")
    _git(repo, "config", "user.name", "Evalyn Test")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "value.py").write_text(FIXED, encoding="utf-8")
    (repo / "src" / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    (repo / "tests" / "test_value.py").write_text(
        "from src.value import value\n\ndef test_value():\n    assert value() == 1\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname='repair-fixture'\nversion='1'\n"
        "[tool.pytest.ini_options]\npythonpath=['.']\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-b", "feature")
    (repo / "src" / "value.py").write_text(BROKEN, encoding="utf-8")
    if extra_source_change:
        (repo / "src" / "other.py").write_text("OTHER = 2\n", encoding="utf-8")
    _git(repo, "commit", "-am", "candidate change")
    candidate_sha = _git(repo, "rev-parse", "HEAD")

    state = tmp_path / "state"
    store = RunStore(repo, state)
    run = store.create_run(base=base_sha, candidate=candidate_sha)
    overlay: dict[str, object] = {"present": False}
    candidate_id = candidate_sha
    if dirty_content is not None:
        (repo / "src" / "value.py").write_text(dirty_content, encoding="utf-8")
        patch = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--binary",
                "--full-index",
                "HEAD",
            ],
            capture_output=True,
            check=True,
        ).stdout
        digest = hashlib.sha256(patch).hexdigest()
        (run.path / "dirty.patch").write_bytes(patch)
        overlay = {
            "present": True,
            "content_sha256": digest,
            "patch": "dirty.patch",
            "untracked": [],
        }
        candidate_id = f"{candidate_sha}+dirty.{digest[:12]}"

    probe_source = (
        "from src.value import value\n\ndef test_saved_regression():\n    assert value() == 1\n"
    )
    artifact = default_registry().get("pytest").build_artifact(probe_source)
    finding = {
        "finding_id": FINDING_ID,
        "run_id": run.run_id,
        "title": "value stays one",
        "base_sha": base_sha,
        "candidate_sha": candidate_id,
        "artifact": {
            "probe_id": artifact.probe_id,
            "adapter": artifact.adapter,
            "relative_path": artifact.relative_path.as_posix(),
            "command": list(artifact.command),
            "source_sha256": artifact.source_sha256,
            "byte_size": len(artifact.source.encode("utf-8")),
            "media_type": artifact.media_type,
        },
        "classification": "stable_change",
        "classification_reason": "candidate consistently fails",
        "failure_signature": "value-changed",
        "probe_source": artifact.source,
        "oracle": "the public API returns one",
        "oracle_provenance": "documentation",
        "semantic_state": "likely_regression",
        "triage_status": "unreviewed",
        # Deliberately untrusted; repair must derive the target from Git.
        "probe_metadata": {"changed_file": "../../outside.py"},
        "base_runs": [
            {
                "side": "base",
                "revision_sha": base_sha,
                "repetition": index,
                "status": "passed",
                "exit_code": 0,
                "duration_seconds": 0.1,
                "failure_signature": "",
                "metadata": {},
            }
            for index in range(1, recorded_repeat + 1)
        ],
        "candidate_runs": [
            {
                "side": "candidate",
                "revision_sha": candidate_id,
                "repetition": index,
                "status": "failed",
                "exit_code": 1,
                "duration_seconds": 0.1,
                "failure_signature": "value-changed",
                "metadata": {},
            }
            for index in range(1, recorded_repeat + 1)
        ],
    }
    RunStore.write_json(
        run.path / "findings.json",
        {
            "schema_version": 1,
            "run_id": run.run_id,
            "base_sha": base_sha,
            "candidate_sha": candidate_id,
            "findings": [finding],
        },
    )
    store.finish_run(
        run,
        status="complete",
        extra={
            "candidate_id": candidate_id,
            "dirty_overlay": overlay,
            "configuration": {
                "provider": {"kind": "anthropic", "model": "fixture"},
                "sandbox": {
                    "runtime": recorded_runtime,
                    "setup_env_names": ["NPM_TOKEN"] if recorded_setup else [],
                },
                "adapters": [
                    {
                        "name": "pytest",
                        "image": recorded_image,
                        "setup_argv": list(recorded_setup),
                        "suite_argv": list(recorded_suite),
                    }
                ],
                "limits": {
                    "max_tests": 1,
                    "repeat": recorded_repeat,
                    "budget_seconds": recorded_budget,
                },
            },
        },
    )
    return repo, store, base_sha, candidate_sha


def test_verified_repair_emits_only_patch_and_preserves_checkout(tmp_path: Path) -> None:
    repo, store, _base, candidate = _saved_finding(tmp_path)
    llm = ReplyLLM(_proposal())
    factory = RecordingFactory()
    output = tmp_path / "artifacts" / "repair.patch"
    before_status = _git(repo, "status", "--porcelain=v1")
    before_worktrees = _git(repo, "worktree", "list", "--porcelain")

    result = repair_saved_finding(
        store,
        FINDING_ID,
        output_path=output,
        llm=llm,
        executor_factory=factory,
    )

    assert result == output.resolve()
    patch = output.read_text(encoding="utf-8")
    assert "src/value.py" in patch
    assert "return 2" in patch
    assert "return 1" in patch
    assert "tests/test_value.py" not in patch
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == BROKEN
    assert _git(repo, "rev-parse", "HEAD") == candidate
    assert _git(repo, "status", "--porcelain=v1") == before_status
    assert _git(repo, "worktree", "list", "--porcelain") == before_worktrees
    assert factory.probe_repetitions == [3, 3, 3]
    assert factory.suite_argv == [
        ("python", "-m", "pytest", "-q"),
        ("python", "-m", "pytest", "-q"),
    ]
    assert factory.prepared_contents == [BROKEN, FIXED, FIXED]
    assert factory.workspaces and all(not path.exists() for path in factory.workspaces)
    assert "../../outside.py" not in llm.prompts[0].split("Allowed repair paths:", 1)[1]


def test_original_failure_is_revalidated_before_any_model_request(tmp_path: Path) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)
    llm = ReplyLLM(_proposal())
    factory = RecordingFactory(candidate_signature="different-failure")
    output = tmp_path / "repair.patch"

    with pytest.raises(RepairError, match="no longer reproduces"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=output,
            llm=llm,
            executor_factory=factory,
        )

    assert factory.probe_repetitions == [3]
    assert not llm.prompts
    assert not output.exists()


@pytest.mark.parametrize(
    "tamper",
    ["mixed-status", "mixed-signature", "finding-signature", "repeat-count"],
)
def test_tampered_repeated_evidence_is_rejected_exactly(
    tmp_path: Path,
    tamper: str,
) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)
    run = store.list_runs(limit=1)[0]
    findings_path = run.path / "findings.json"
    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    finding = payload["findings"][0]
    if tamper == "mixed-status":
        finding["candidate_runs"][1].update(
            status="timeout",
            exit_code=-1,
            failure_signature="",
        )
    elif tamper == "mixed-signature":
        finding["candidate_runs"][1]["failure_signature"] = "different"
    elif tamper == "finding-signature":
        finding["failure_signature"] = "different"
    else:
        finding["candidate_runs"].pop()
        finding["base_runs"].pop()
    RunStore.write_json(findings_path, payload)
    llm = ReplyLLM(_proposal())
    factory = RecordingFactory()

    with pytest.raises(RepairError, match="Stored"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            llm=llm,
            executor_factory=factory,
        )

    assert not llm.prompts
    assert not factory.workspaces


@pytest.mark.parametrize(
    "path",
    [
        "pytest_plugins.py",
        "src/conftest.py",
        "src/setup.ts",
        "src/setupTests.ts",
        "src/vitest.setup.ts",
        "src/jest.setup.js",
        "src/vitest.workspace.ts",
        "src/webpack.config.js",
        ".github/actions/check/index.js",
        ".githooks/pre-commit.py",
        "ci/release.py",
        "hooks/verify.rs",
        "fixtures/server.py",
    ],
)
def test_test_harness_config_and_hook_targets_are_not_source(path: str) -> None:
    assert not _is_source(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/evalyn/runner.py",
        "lib/latest.py",
        "app/routes/contest.ts",
        "app/hooks/useSession.ts",
        "app/user.configurator.ts",
        "app/specification.py",
        "main.go",
    ],
)
def test_legitimate_source_targets_remain_eligible(path: str) -> None:
    assert _is_source(path)


def test_failed_hypothesis_is_retried_in_a_fresh_candidate(tmp_path: Path) -> None:
    repo, store, _base, _candidate = _saved_finding(tmp_path)
    first = "def value() -> int:\n    return 3\n"
    llm = ReplyLLM(_proposal(first), _proposal(FIXED))
    factory = RecordingFactory()

    repair_saved_finding(
        store,
        FINDING_ID,
        output_path=tmp_path / "repair.patch",
        llm=llm,
        executor_factory=factory,
    )

    assert len(factory.workspaces) == 4
    assert len(set(factory.workspaces)) == 4
    assert all(not path.exists() for path in factory.workspaces)
    assert len(llm.prompts) == 2
    assert all("return 2" in prompt for prompt in llm.prompts)
    assert "return 3" not in llm.prompts[1]
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == BROKEN


def test_model_cannot_edit_test_or_return_command_fields(tmp_path: Path) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)
    factory = RecordingFactory()
    llm = ReplyLLM(
        _proposal(path="tests/test_value.py"),
        {**_proposal(), "run_command": "pytest"},
    )

    with pytest.raises(RepairError, match="No verified patch"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            llm=llm,
            executor_factory=factory,
            max_attempts=2,
        )

    assert len(factory.workspaces) == 1
    assert not (tmp_path / "repair.patch").exists()


def test_ambiguous_changed_source_target_fails_before_llm(tmp_path: Path) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path, extra_source_change=True)
    llm = ReplyLLM(_proposal())

    with pytest.raises(RepairError, match="target is ambiguous"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            llm=llm,
            executor_factory=RecordingFactory(),
        )

    assert not llm.prompts


def test_existing_output_is_refused_before_repair_work(tmp_path: Path) -> None:
    repo, store, _base, _candidate = _saved_finding(tmp_path)
    output = tmp_path / "repair.patch"
    output.write_bytes(b"pre-existing patch\n")
    before_status = _git(repo, "status", "--porcelain=v1")
    llm = ReplyLLM(_proposal())
    factory = RecordingFactory(suite_passes=False)

    with pytest.raises(RepairError, match="refusing to overwrite"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=output,
            llm=llm,
            executor_factory=factory,
            max_attempts=1,
        )

    assert output.read_bytes() == b"pre-existing patch\n"
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == BROKEN
    assert _git(repo, "status", "--porcelain=v1") == before_status
    assert not llm.prompts
    assert not factory.workspaces


def test_suite_failure_preserves_checkout_without_emitting_patch(tmp_path: Path) -> None:
    repo, store, _base, _candidate = _saved_finding(tmp_path)
    output = tmp_path / "repair.patch"
    before_status = _git(repo, "status", "--porcelain=v1")

    with pytest.raises(RepairError, match="No verified patch"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=output,
            llm=ReplyLLM(_proposal()),
            executor_factory=RecordingFactory(suite_passes=False),
            max_attempts=1,
        )

    assert not output.exists()
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == BROKEN
    assert _git(repo, "status", "--porcelain=v1") == before_status


@pytest.mark.parametrize(
    ("semantic_state", "triage_status"),
    [("intended_change", "unreviewed"), ("likely_regression", "dismissed")],
)
def test_repair_requires_likely_or_confirmed_disposition(
    tmp_path: Path,
    semantic_state: str,
    triage_status: str,
) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)
    run = store.list_runs(limit=1)[0]
    findings_path = run.path / "findings.json"
    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    payload["findings"][0]["semantic_state"] = semantic_state
    payload["findings"][0]["triage_status"] = triage_status
    RunStore.write_json(findings_path, payload)

    with pytest.raises(RepairError, match="likely regression"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "blocked.patch",
            llm=ReplyLLM(_proposal()),
            executor_factory=RecordingFactory(),
        )


def test_dirty_overlay_is_restored_but_not_bundled_into_patch(tmp_path: Path) -> None:
    overlay_content = "def value() -> int:\n    return 7\n"
    repo, store, _base, _candidate = _saved_finding(tmp_path, dirty_content=overlay_content)
    output = tmp_path / "repair.patch"
    llm = ReplyLLM(_proposal())
    before_status = _git(repo, "status", "--porcelain=v1")

    repair_saved_finding(
        store,
        FINDING_ID,
        output_path=output,
        llm=llm,
        executor_factory=RecordingFactory(),
    )

    patch = output.read_text(encoding="utf-8")
    assert "return 7" in patch
    assert "return 1" in patch
    assert "return 2" not in patch
    assert "return 7" in llm.prompts[0]
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == overlay_content
    assert _git(repo, "status", "--porcelain=v1") == before_status
    subprocess.run(
        ["git", "-C", str(repo), "apply", "--check", str(output)],
        capture_output=True,
        check=True,
    )


def test_saved_suite_runtime_and_image_override_current_config(tmp_path: Path) -> None:
    recorded_suite = ("python", "-m", "pytest", "tests/test_value.py")
    repo, store, _base, _candidate = _saved_finding(
        tmp_path,
        recorded_runtime="podman",
        recorded_repeat=2,
        recorded_image="registry.example.test/python@sha256:" + "a" * 64,
        recorded_setup=("python", "-m", "pip", "install", "-e", "."),
        recorded_suite=recorded_suite,
    )
    config = tmp_path / "repair.yml"
    config.write_text(
        "version: 1\n"
        "sandbox:\n"
        "  runtime: docker\n"
        "adapters:\n"
        "  - name: pytest\n"
        "    image: python:latest\n"
        "    setup_argv: []\n"
        "    suite_argv: [python, -m, pytest, -q]\n",
        encoding="utf-8",
    )
    factory = RecordingFactory()

    repair_saved_finding(
        store,
        FINDING_ID,
        output_path=tmp_path / "repair.patch",
        config_path=config,
        llm=ReplyLLM(_proposal()),
        executor_factory=factory,
    )

    assert factory.suite_argv == [recorded_suite, recorded_suite]
    assert factory.runtimes == ["podman", "podman", "podman"]
    assert factory.probe_repetitions == [3, 3, 3]
    assert all(
        settings.image == "registry.example.test/python@sha256:" + "a" * 64
        and settings.setup_argv == ("python", "-m", "pip", "install", "-e", ".")
        for settings in factory.adapter_settings
    )
    assert (repo / "src" / "value.py").read_text(encoding="utf-8") == BROKEN


def test_saved_budget_overrides_current_config(tmp_path: Path) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path, recorded_budget=17)
    config = tmp_path / "repair.yml"
    config.write_text(
        "version: 1\nlimits:\n  budget_seconds: 1\n",
        encoding="utf-8",
    )
    factory = RecordingFactory()

    repair_saved_finding(
        store,
        FINDING_ID,
        output_path=tmp_path / "repair.patch",
        config_path=config,
        llm=ReplyLLM(_proposal()),
        executor_factory=factory,
    )

    assert factory.budgets
    assert factory.budgets[0] == 17
    assert all(1 <= budget <= 17 for budget in factory.budgets)


def test_builtin_repair_provider_uses_saved_remaining_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path, recorded_budget=17)
    observed: list[tuple[str, str, float]] = []

    def fake_create_provider(settings, *, timeout_seconds=120):
        observed.append((settings.kind, settings.model, timeout_seconds))
        return ReplyLLM(_proposal())

    monkeypatch.setattr(repair_module, "create_provider", fake_create_provider)

    repair_saved_finding(
        store,
        FINDING_ID,
        output_path=tmp_path / "repair.patch",
        executor_factory=RecordingFactory(),
    )

    assert len(observed) == 1
    assert observed[0][0:2] == ("anthropic", "fixture")
    assert 0 < observed[0][2] <= 17


def test_repair_rejects_repository_provider_override(tmp_path: Path) -> None:
    repo, store, _base, _candidate = _saved_finding(tmp_path)
    config = repo / ".evalyn.yml"
    config.write_text("version: 1\nprovider:\n  model: current-worktree-model\n")

    with pytest.raises(RepairError, match="outside the repository"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            config_path=config,
            llm=ReplyLLM(_proposal()),
            executor_factory=RecordingFactory(),
        )


@pytest.mark.parametrize("budget", [None, 0, -1, True, "600"])
def test_repair_rejects_missing_or_malformed_saved_budget(
    tmp_path: Path,
    budget: object,
) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)
    run = store.list_runs(limit=1)[0]
    manifest = dict(run.manifest)
    configuration = dict(manifest["configuration"])
    limits = dict(configuration["limits"])
    if budget is None:
        limits.pop("budget_seconds")
    else:
        limits["budget_seconds"] = budget
    configuration["limits"] = limits
    manifest["configuration"] = configuration
    RunStore.write_json(run.path / "manifest.json", manifest)
    llm = ReplyLLM(_proposal())

    with pytest.raises(RepairError, match="execution-budget snapshot"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            llm=llm,
            executor_factory=RecordingFactory(),
        )

    assert not llm.prompts


def test_repair_requires_a_complete_recorded_configuration(tmp_path: Path) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)
    run = store.list_runs(limit=1)[0]
    manifest = dict(run.manifest)
    manifest.pop("configuration")
    RunStore.write_json(run.path / "manifest.json", manifest)
    llm = ReplyLLM(_proposal())

    with pytest.raises(RepairError, match="configuration snapshot"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            llm=llm,
            executor_factory=RecordingFactory(),
        )

    assert not llm.prompts


def test_repair_holds_lock_from_saved_lookup_through_patch_emission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, store, _base, _candidate = _saved_finding(tmp_path)
    stages: list[str] = []
    real_find = store.find_finding
    real_write = repair_module._write_patch_atomically

    def assert_locked(stage: str) -> None:
        with pytest.raises(RunLockError, match="already holds"):
            RepoRunLock(repo, lock_path=store.repo_root / "run.lock").acquire()
        stages.append(stage)

    def locked_find(finding_id: str):
        assert_locked("lookup")
        return real_find(finding_id)

    def locked_write(output: Path, patch: bytes) -> None:
        assert_locked("emit")
        real_write(output, patch)

    monkeypatch.setattr(store, "find_finding", locked_find)
    monkeypatch.setattr(repair_module, "_write_patch_atomically", locked_write)

    repair_saved_finding(
        store,
        FINDING_ID,
        output_path=tmp_path / "repair.patch",
        llm=ReplyLLM(_proposal()),
        executor_factory=RecordingFactory(),
    )

    assert stages == ["lookup", "emit"]


def test_repair_git_ignores_hostile_environment_and_repository_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, store, _base, _candidate = _saved_finding(tmp_path)
    marker = tmp_path / "host-git-command-ran"
    hostile_command = f"/usr/bin/touch {marker}"
    _git(repo, "config", "filter.hostile.clean", hostile_command)
    _git(repo, "config", "filter.hostile.smudge", hostile_command)
    _git(repo, "config", "filter.hostile.process", hostile_command)
    _git(repo, "config", "diff.hostile.command", hostile_command)
    _git(repo, "config", "diff.hostile.textconv", hostile_command)
    (repo / ".git" / "info" / "attributes").write_text(
        "*.py filter=hostile diff=hostile\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", hostile_command)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", hostile_command)

    real_run = repair_module.run_bounded_process
    git_environments: list[dict[str, str] | None] = []

    def inspect_run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("argv")
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "git":
            git_environments.append(kwargs.get("env"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(repair_module, "run_bounded_process", inspect_run)

    repair_saved_finding(
        store,
        FINDING_ID,
        output_path=tmp_path / "repair.patch",
        llm=ReplyLLM(_proposal()),
        executor_factory=RecordingFactory(),
    )

    assert git_environments
    assert all(environment is not None for environment in git_environments)
    for environment in git_environments:
        assert environment is not None
        assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert "GIT_EXTERNAL_DIFF" not in environment
        assert "GIT_CONFIG_COUNT" not in environment
    assert not marker.exists()


def test_runtime_override_cannot_change_the_recorded_runtime(tmp_path: Path) -> None:
    _repo, store, _base, _candidate = _saved_finding(
        tmp_path,
        recorded_runtime="docker",
    )
    llm = ReplyLLM(_proposal())

    with pytest.raises(RepairError, match="cannot override"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            runtime="podman",
            llm=llm,
            executor_factory=RecordingFactory(),
        )

    assert not llm.prompts


def test_validation_cannot_mutate_tracked_source_to_manufacture_success(
    tmp_path: Path,
) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)

    with pytest.raises(RepairError, match="Validation commands modified tracked files"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            llm=ReplyLLM(_proposal()),
            executor_factory=RecordingFactory(mutate_target=True),
            max_attempts=1,
        )

    assert not (tmp_path / "repair.patch").exists()


def test_final_fresh_verification_rejects_untracked_validation_changes(
    tmp_path: Path,
) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)
    factory = RecordingFactory(mutate_untracked_on_call=3)

    with pytest.raises(RepairError, match="unexpected untracked"):
        repair_saved_finding(
            store,
            FINDING_ID,
            output_path=tmp_path / "repair.patch",
            llm=ReplyLLM(_proposal()),
            executor_factory=factory,
            max_attempts=1,
        )

    assert factory.probe_repetitions == [3, 3, 3]
    assert len(factory.suite_argv) == 2
    assert not (tmp_path / "repair.patch").exists()


def test_invalid_finding_id_is_rejected_before_store_lookup(tmp_path: Path) -> None:
    _repo, store, _base, _candidate = _saved_finding(tmp_path)

    with pytest.raises(RepairError, match="must match"):
        repair_saved_finding(
            store,
            "../EVL-1234ABCD",
            output_path=tmp_path / "repair.patch",
            llm=ReplyLLM(_proposal()),
            executor_factory=RecordingFactory(),
        )
