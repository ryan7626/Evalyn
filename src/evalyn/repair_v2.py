"""Experimental, patch-only repair of a persisted differential finding.

The repair workflow is deliberately fail-closed: every hypothesis is applied
to a freshly reconstructed candidate worktree, model output can select only a
changed tracked source file, and a patch is emitted only after three probe
passes plus the saved configured (or trusted default) full-suite pass. The source
checkout is never checked out, reset, cleaned, or modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Protocol

from evalyn.adapters.base import ProbeArtifact, TestAdapter
from evalyn.adapters.registry import default_registry
from evalyn.engine.classifier import failure_signature
from evalyn.engine.models import ExecutionResult, ExecutionStatus, Revision
from evalyn.execution import (
    ContainerRunner,
    PathPolicyError,
    RepoRunLock,
    WorkspacePathPolicy,
    WorktreeSession,
)
from evalyn.execution.bounded import BoundedProcessResult, run_bounded_process
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment
from evalyn.execution.structured_results import PYTEST_REPORTER_SOURCE
from evalyn.llm import create_provider
from evalyn.llm.base import BaseLLM
from evalyn.probe_executor import DEFAULT_IMAGES, ContainerProbeExecutor
from evalyn.run_store import RunStore, StoredRun
from evalyn.settings import (
    MAX_BUDGET_SECONDS,
    MAX_REPEAT,
    AdapterSettings,
    EvalynSettings,
    ProviderSettings,
    load_settings,
    resolve_settings_path,
)


class RepairError(RuntimeError):
    """A repair could not be produced and no patch was emitted."""


@dataclass(frozen=True)
class RepairProposal:
    file_path: str
    replacement_content: str
    explanation: str


@dataclass(frozen=True)
class _ValidatedEvidence:
    repetitions: int
    candidate_status: ExecutionStatus
    failure_signature: str


@dataclass(frozen=True)
class _RecordedRepairConfiguration:
    adapter_settings: AdapterSettings
    runtime: str
    setup_env_names: tuple[str, ...]
    repetitions: int
    budget_seconds: int
    allow_remote_endpoint: bool
    expected_image_identity: str | None
    expected_endpoint_scope: str | None


class RepairExecutor(Protocol):
    """Injectable validation boundary used by patch-only repair."""

    @property
    def setup_errors(self) -> Mapping[Path, str]: ...

    def prepare(self, revision: Revision) -> None: ...

    def validate_probe(
        self,
        revision: Revision,
        artifact: ProbeArtifact,
        *,
        repetitions: int,
    ) -> tuple[ExecutionResult, ...]: ...

    def validate_suite(
        self,
        revision: Revision,
        argv: tuple[str, ...],
    ) -> ExecutionResult: ...


RepairExecutorFactory = Callable[
    [TestAdapter, AdapterSettings, Path, str | None, int],
    RepairExecutor,
]


_SYSTEM_PROMPT = """\
You repair a single source file for one eligible differential regression finding.

Return exactly one JSON object with exactly these keys:
- "file_path": a repository-relative path from the supplied allowed list
- "replacement_content": the complete replacement contents of that file
- "explanation": a concise explanation of the source-level repair

Do not return a command, test command, patch, path outside the allowed list,
test-file edit, configuration edit, dependency change, or any additional key.
The replacement must be minimal and preserve unrelated behavior.
"""

_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".mjs",
        ".cjs",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".swift",
        ".ts",
        ".tsx",
    }
)

_DEFAULT_SUITES: dict[str, tuple[str, ...]] = {
    "pytest": ("python", "-m", "pytest", "-q"),
    "vitest": ("npx", "--no-install", "vitest", "run"),
}

_MAX_REPLACEMENT_BYTES = 512 * 1024
_MAX_CONTEXT_FILES = 8
_MAX_CONTEXT_BYTES_PER_FILE = 64 * 1024
_MAX_OVERLAY_BYTES = 50 * 1024 * 1024
_MAX_STATE_JSON_BYTES = 10 * 1024 * 1024
_MAX_GIT_OUTPUT_BYTES = 50 * 1024 * 1024
_MAX_GIT_STDERR_BYTES = 256 * 1024
_FINDING_ID = re.compile(r"^EVL-[A-F0-9]{8}$")
_GIT_OID = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_OCI_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}$")
_SETUP_ENV_EXACT = frozenset(
    {
        "BUNDLE_GEMS__CONTRIBSYS__COM",
        "CARGO_REGISTRY_TOKEN",
        "COMPOSER_AUTH",
        "NODE_AUTH_TOKEN",
        "NPM_TOKEN",
        "PIP_EXTRA_INDEX_URL",
        "PIP_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "YARN_NPM_AUTH_TOKEN",
    }
)
_SETUP_ENV_PREFIXES = ("BUNDLE_", "CARGO_REGISTRIES_", "NUGET_", "POETRY_HTTP_BASIC_")
_PROTECTED_SOURCE_NAMES = frozenset(
    {
        "conftest.py",
        "noxfile.py",
        "pytest_plugins.py",
        "setup.py",
    }
)
_PROTECTED_SOURCE_DIRECTORIES = frozenset(
    {
        "__mocks__",
        "__tests__",
        "fixture",
        "fixtures",
        "mock",
        "mocks",
        "spec",
        "specs",
        "test",
        "tests",
    }
)
_PROTECTED_TOP_LEVEL_DIRECTORIES = frozenset(
    {
        ".azure-pipelines",
        ".buildkite",
        ".circleci",
        ".github",
        ".githooks",
        ".gitlab",
        ".husky",
        "ci",
        "hook",
        "hooks",
        "jenkins",
    }
)


def repair_saved_finding(
    store: RunStore,
    finding_id: str,
    *,
    output_path: Path,
    runtime: str | None = None,
    config_path: Path | None = None,
    llm: BaseLLM | None = None,
    executor_factory: RepairExecutorFactory | None = None,
    max_attempts: int = 3,
    allow_remote_endpoint: bool = False,
) -> Path:
    """Validate model repair hypotheses and atomically emit the first safe patch.

    ``llm`` and ``executor_factory`` are injectable test/integration seams. The
    CLI intentionally relies only on the documented arguments preceding them.
    """
    if not isinstance(finding_id, str) or not _FINDING_ID.fullmatch(finding_id):
        raise RepairError("finding_id must match EVL-XXXXXXXX")
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or not 1 <= max_attempts <= MAX_REPEAT
    ):
        raise RepairError(f"max_attempts must be an integer between 1 and {MAX_REPEAT}")
    if runtime not in {None, "docker", "podman"}:
        raise RepairError("runtime must be docker, podman, or omitted")

    requested_output = Path(output_path).expanduser()
    if requested_output.is_symlink():
        raise RepairError(f"Patch output cannot be a symlink: {requested_output}")
    output = requested_output.resolve(strict=False)
    if output.exists():
        raise RepairError(f"Patch output already exists; refusing to overwrite it: {output}")

    lock = RepoRunLock(store.repo_path, lock_path=store.repo_root / "run.lock")
    try:
        with lock:
            return _repair_saved_finding_locked(
                store,
                finding_id,
                output=output,
                runtime=runtime,
                config_path=config_path,
                llm=llm,
                executor_factory=executor_factory,
                max_attempts=max_attempts,
                allow_remote_endpoint=allow_remote_endpoint,
            )
    except RepairError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise RepairError(
            f"Could not acquire or release the repository repair lock: {exc}"
        ) from exc


def _repair_saved_finding_locked(
    store: RunStore,
    finding_id: str,
    *,
    output: Path,
    runtime: str | None,
    config_path: Path | None,
    llm: BaseLLM | None,
    executor_factory: RepairExecutorFactory | None,
    max_attempts: int,
    allow_remote_endpoint: bool,
) -> Path:
    """Run all state-dependent repair work while the repository lock is held."""
    located = store.find_finding(finding_id)
    if located is None:
        raise RepairError(f"Finding not found: {finding_id}")
    run, _located_finding = located
    _validate_stored_run(store, run)
    finding = _load_owned_finding(run, finding_id)
    repo = _validate_stored_repository(store, run)
    adapter, artifact = _load_artifact(finding)
    recorded = _load_recorded_configuration(run, adapter.name)
    settings = _load_repair_settings(repo, config_path, run=run)
    if allow_remote_endpoint != recorded.allow_remote_endpoint:
        raise RepairError(
            "Remote container endpoint opt-in must match the saved check; "
            "pass the explicit operator flag again only for a remotely-created run"
        )
    adapter_settings = recorded.adapter_settings
    base_sha, candidate_sha, candidate_id = _stored_revisions(run, finding)
    evidence = _validate_finding_evidence(
        finding,
        base_sha,
        candidate_id,
        recorded_repetitions=recorded.repetitions,
    )
    suite_argv = adapter_settings.suite_argv or _default_suite(adapter.name)
    effective_runtime = _recorded_runtime(recorded.runtime, runtime)
    factory = executor_factory or partial(
        _default_executor_factory,
        setup_env_names=recorded.setup_env_names,
        allow_remote_endpoint=allow_remote_endpoint,
        expected_image_identity=recorded.expected_image_identity,
        expected_endpoint_scope=recorded.expected_endpoint_scope,
    )
    if executor_factory is None and (
        recorded.expected_image_identity is None or recorded.expected_endpoint_scope is None
    ):
        raise RepairError(
            "Stored run has no complete execution-environment identity for exact repair replay"
        )

    feedback: list[str] = []
    deadline = time.monotonic() + recorded.budget_seconds
    repair_key = hashlib.sha256(finding_id.encode("ascii")).hexdigest()[:24]
    verified_patch: bytes | None = None
    remaining_budget = math.ceil(deadline - time.monotonic())
    if remaining_budget <= 0:
        raise RepairError("Repair execution budget was exhausted before revalidation")
    _revalidate_saved_candidate(
        repo,
        run=run,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
        candidate_id=candidate_id,
        adapter=adapter,
        adapter_settings=adapter_settings,
        artifact=artifact,
        evidence=evidence,
        factory=factory,
        runtime=effective_runtime,
        budget_seconds=remaining_budget,
        run_path=(run.path / "repairs" / repair_key / f"preflight-{uuid.uuid4().hex[:12]}"),
        worktree_parent=store.repo_root / "worktrees",
    )
    for attempt in range(1, max_attempts + 1):
        remaining_budget = math.ceil(deadline - time.monotonic())
        if remaining_budget <= 0:
            feedback.append("Repair execution budget was exhausted")
            break
        attempt_dir = (
            run.path / "repairs" / repair_key / (f"attempt-{attempt}-{uuid.uuid4().hex[:12]}")
        )
        try:
            candidate_patch = _run_repair_attempt(
                repo,
                run=run,
                finding=finding,
                base_sha=base_sha,
                candidate_sha=candidate_sha,
                candidate_id=candidate_id,
                adapter=adapter,
                adapter_settings=adapter_settings,
                artifact=artifact,
                suite_argv=suite_argv,
                provider=llm,
                provider_settings=settings.provider,
                factory=factory,
                attempt_dir=attempt_dir,
                runtime=effective_runtime,
                deadline=deadline,
                validation_repetitions=max(3, evidence.repetitions),
                feedback=feedback,
                worktree_parent=store.repo_root / "worktrees",
            )
            _verify_patch_against_fresh_candidate(
                repo,
                run=run,
                base_sha=base_sha,
                candidate_sha=candidate_sha,
                candidate_id=candidate_id,
                patch=candidate_patch,
                adapter=adapter,
                adapter_settings=adapter_settings,
                artifact=artifact,
                suite_argv=suite_argv,
                factory=factory,
                runtime=effective_runtime,
                deadline=deadline,
                repetitions=max(3, evidence.repetitions),
                run_path=(
                    run.path / "repairs" / repair_key / f"final-{attempt}-{uuid.uuid4().hex[:12]}"
                ),
                worktree_parent=store.repo_root / "worktrees",
            )
            verified_patch = candidate_patch
            break
        except _RetryableRepair as exc:
            feedback.append(f"Attempt {attempt}: {exc}")
            continue
        except RepairError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise RepairError(
                f"Repair attempt {attempt} could not complete: {type(exc).__name__}: {exc}"
            ) from exc

    if verified_patch is not None:
        try:
            _write_patch_atomically(output, verified_patch)
        except Exception as exc:
            raise RepairError(f"Could not emit repair patch: {exc}") from exc
        return output

    detail = feedback[-1] if feedback else "no valid hypothesis was produced"
    raise RepairError(f"No verified patch after {max_attempts} attempt(s): {detail}")


class _RetryableRepair(RuntimeError):
    pass


def _run_repair_attempt(
    repo: Path,
    *,
    run: StoredRun,
    finding: dict,
    base_sha: str,
    candidate_sha: str,
    candidate_id: str,
    adapter: TestAdapter,
    adapter_settings: AdapterSettings,
    artifact: ProbeArtifact,
    suite_argv: tuple[str, ...],
    provider: BaseLLM | None,
    provider_settings: ProviderSettings,
    factory: RepairExecutorFactory,
    attempt_dir: Path,
    runtime: str | None,
    deadline: float,
    validation_repetitions: int,
    feedback: Sequence[str],
    worktree_parent: Path,
) -> bytes:
    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        temp_parent=worktree_parent,
    ) as worktrees:
        _restore_saved_overlay(run, worktrees)
        tracked = worktrees.tracked_files("candidate")
        allowed = _changed_source_paths(
            worktrees,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            tracked_paths=tracked,
        )
        if not allowed:
            raise RepairError(
                "Finding has no changed tracked non-test source file eligible for repair"
            )
        if len(allowed) != 1:
            raise RepairError(
                "Repair target is ambiguous: expected exactly one changed tracked "
                f"non-test source file, found {len(allowed)}"
            )
        policy = WorkspacePathPolicy(
            worktrees.candidate_path,
            tracked_paths=tracked,
            context_allowlist=allowed,
            fix_allowlist=allowed,
        )
        prompt = _repair_prompt(finding, policy, allowed, feedback)
        request_budget = deadline - time.monotonic()
        if request_budget <= 0:
            raise RepairError("Repair execution budget was exhausted before generation")
        try:
            active_provider = provider or create_provider(
                provider_settings,
                timeout_seconds=min(120, request_budget),
            )
        except Exception as exc:
            raise RepairError(f"Could not initialize repair provider: {exc}") from exc
        proposal = _request_proposal(active_provider, prompt)
        try:
            target = policy.resolve_fix_target(proposal.file_path)
        except PathPolicyError as exc:
            raise _RetryableRepair(f"Unsafe repair target: {exc}") from exc

        relative = target.relative_to(worktrees.candidate_path).as_posix()
        original = _read_bounded_utf8_source(target, relative)
        replacement = proposal.replacement_content
        if replacement == original:
            raise _RetryableRepair("Replacement content is identical to the source")
        target_mode = target.stat().st_mode & 0o777
        patch = _build_repair_patch(
            relative,
            original,
            replacement,
            mode=target_mode,
        )
        _check_patch_applies(worktrees.candidate_path, patch)
        _atomic_replace_source(target, replacement)
        expected_tracked_diff = _tracked_diff(worktrees.candidate_path)

        revision = Revision(candidate_id, worktrees.candidate_path)
        validation_budget = math.ceil(deadline - time.monotonic())
        if validation_budget <= 0:
            raise RepairError("Repair execution budget was exhausted before validation")
        executor = factory(
            adapter,
            adapter_settings,
            attempt_dir,
            runtime,
            validation_budget,
        )
        executor.prepare(revision)
        if executor.setup_errors:
            details = "; ".join(str(value) for value in executor.setup_errors.values())
            raise RepairError(f"Repair validation setup failed: {details}")

        repetitions = validation_repetitions
        probe_results = executor.validate_probe(
            revision,
            artifact,
            repetitions=repetitions,
        )
        if len(probe_results) != repetitions:
            raise RepairError(f"Repair executor did not return exactly {repetitions} probe results")
        _raise_for_infrastructure(probe_results, "probe")
        if any(result.status != ExecutionStatus.PASSED for result in probe_results):
            statuses = ", ".join(result.status.value for result in probe_results)
            raise _RetryableRepair(
                f"Repaired candidate did not pass the probe {repetitions}x ({statuses})"
            )

        suite_result = executor.validate_suite(revision, suite_argv)
        _raise_for_infrastructure((suite_result,), "full suite")
        if suite_result.status != ExecutionStatus.PASSED:
            detail = _result_fingerprint(suite_result)
            raise _RetryableRepair(f"Repaired candidate failed full suite: {detail}")

        if _tracked_diff(worktrees.candidate_path) != expected_tracked_diff:
            raise RepairError(
                "Validation commands modified tracked files; refusing to emit a patch"
            )
        try:
            final_content = _read_bounded_utf8_source(target, relative)
        except RepairError as exc:
            raise RepairError(
                "Validation commands made the proposed repair target unreadable"
            ) from exc
        if final_content != replacement:
            raise RepairError("Validation commands modified the proposed repair target")
        return patch


def _revalidate_saved_candidate(
    repo: Path,
    *,
    run: StoredRun,
    base_sha: str,
    candidate_sha: str,
    candidate_id: str,
    adapter: TestAdapter,
    adapter_settings: AdapterSettings,
    artifact: ProbeArtifact,
    evidence: _ValidatedEvidence,
    factory: RepairExecutorFactory,
    runtime: str | None,
    budget_seconds: int,
    run_path: Path,
    worktree_parent: Path,
) -> None:
    """Reproduce the persisted failure before any repair proposal is requested."""
    repetitions = max(3, evidence.repetitions)
    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        temp_parent=worktree_parent,
    ) as worktrees:
        _restore_saved_overlay(run, worktrees)
        before = _tracked_diff(worktrees.candidate_path)
        revision = Revision(candidate_id, worktrees.candidate_path)
        executor = factory(
            adapter,
            adapter_settings,
            run_path,
            runtime,
            budget_seconds,
        )
        executor.prepare(revision)
        if executor.setup_errors:
            details = "; ".join(str(value) for value in executor.setup_errors.values())
            raise RepairError(f"Saved-candidate revalidation setup failed: {details}")
        results = executor.validate_probe(
            revision,
            artifact,
            repetitions=repetitions,
        )
        if len(results) != repetitions:
            raise RepairError(
                "Repair executor did not return exactly "
                f"{repetitions} saved-candidate probe results"
            )
        if any(
            result.status in {ExecutionStatus.INVALID_PROBE, ExecutionStatus.INFRA_ERROR}
            for result in results
        ):
            bad = next(
                result
                for result in results
                if result.status in {ExecutionStatus.INVALID_PROBE, ExecutionStatus.INFRA_ERROR}
            )
            raise RepairError("Saved-candidate revalidation was incomplete: " + _result_detail(bad))
        observed_statuses = {result.status for result in results}
        observed_signatures = {failure_signature(result) for result in results}
        if observed_statuses != {evidence.candidate_status} or observed_signatures != {
            evidence.failure_signature
        }:
            statuses = ", ".join(sorted(status.value for status in observed_statuses))
            signatures = ", ".join(sorted(observed_signatures))
            raise RepairError(
                "Saved candidate no longer reproduces the original stable failure "
                f"(statuses={statuses}; signatures={signatures})"
            )
        if _tracked_diff(worktrees.candidate_path) != before:
            raise RepairError(
                "Saved-candidate revalidation modified tracked files; refusing repair"
            )


def _verify_patch_against_fresh_candidate(
    repo: Path,
    *,
    run: StoredRun,
    base_sha: str,
    candidate_sha: str,
    candidate_id: str,
    patch: bytes,
    adapter: TestAdapter,
    adapter_settings: AdapterSettings,
    artifact: ProbeArtifact,
    suite_argv: tuple[str, ...],
    factory: RepairExecutorFactory,
    runtime: str | None,
    deadline: float,
    repetitions: int,
    run_path: Path,
    worktree_parent: Path,
) -> None:
    """Apply and fully validate the patch in a newly reconstructed candidate."""
    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        temp_parent=worktree_parent,
    ) as worktrees:
        _restore_saved_overlay(run, worktrees)
        workspace = worktrees.candidate_path
        _apply_repair_patch(workspace, patch)
        expected_tracked = _tracked_diff(workspace)
        probe_path = _materialize_saved_probe(workspace, artifact)
        support_paths = _materialize_probe_support(workspace, adapter.name)
        revision = Revision(candidate_id, workspace)
        validation_budget = math.ceil(deadline - time.monotonic())
        if validation_budget <= 0:
            raise RepairError("Repair execution budget was exhausted before final verification")
        executor = factory(
            adapter,
            adapter_settings,
            run_path,
            runtime,
            validation_budget,
        )
        executor.prepare(revision)
        if executor.setup_errors:
            details = "; ".join(str(value) for value in executor.setup_errors.values())
            raise RepairError(f"Final repair verification setup failed: {details}")
        _assert_probe_unchanged(probe_path, artifact)
        _assert_support_unchanged(support_paths)
        expected_untracked = _untracked_state(workspace)
        _assert_final_workspace_state(
            workspace,
            expected_tracked=expected_tracked,
            expected_untracked=expected_untracked,
            probe_path=probe_path,
            artifact=artifact,
            run=run,
            support_paths=support_paths,
        )

        results = executor.validate_probe(
            revision,
            artifact,
            repetitions=repetitions,
        )
        if len(results) != repetitions:
            raise RepairError(
                f"Repair executor did not return exactly {repetitions} final probe results"
            )
        _raise_for_infrastructure(results, "final probe")
        if any(result.status != ExecutionStatus.PASSED for result in results):
            statuses = ", ".join(result.status.value for result in results)
            raise _RetryableRepair(
                f"Fresh repaired candidate did not pass the probe {repetitions}x ({statuses})"
            )
        _assert_final_workspace_state(
            workspace,
            expected_tracked=expected_tracked,
            expected_untracked=expected_untracked,
            probe_path=probe_path,
            artifact=artifact,
            run=run,
            support_paths=support_paths,
        )

        suite_result = executor.validate_suite(revision, suite_argv)
        _raise_for_infrastructure((suite_result,), "final full suite")
        if suite_result.status != ExecutionStatus.PASSED:
            raise _RetryableRepair(
                "Fresh repaired candidate failed full suite: " + _result_fingerprint(suite_result)
            )
        _assert_final_workspace_state(
            workspace,
            expected_tracked=expected_tracked,
            expected_untracked=expected_untracked,
            probe_path=probe_path,
            artifact=artifact,
            run=run,
            support_paths=support_paths,
        )


def _apply_repair_patch(workspace: Path, patch: bytes) -> None:
    result = _run_sanitized_git(
        workspace,
        "apply",
        "--binary",
        "--whitespace=nowarn",
        "-",
        input_bytes=patch,
        action="apply constructed repair patch",
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepairError(
            f"Constructed repair patch does not apply to the fresh saved candidate: {detail}"
        )


def _materialize_saved_probe(workspace: Path, artifact: ProbeArtifact) -> Path:
    try:
        target = WorkspacePathPolicy(workspace).resolve_write(
            artifact.relative_path.as_posix(),
            protect_sensitive=False,
        )
        content = artifact.source.encode("utf-8")
        if target.exists():
            if RunStore.read_bytes(target, max_bytes=len(content)) != content:
                raise RepairError("Fresh candidate probe path contains unexpected content")
        else:
            RunStore.write_bytes(target, content, mode=0o644)
        return target
    except RepairError:
        raise
    except (OSError, ValueError) as exc:
        raise RepairError(
            f"Could not materialize the saved probe for final verification: {exc}"
        ) from exc


def _materialize_probe_support(
    workspace: Path, adapter_name: str
) -> tuple[tuple[Path, bytes], ...]:
    if adapter_name != "pytest":
        return ()
    content = PYTEST_REPORTER_SOURCE.encode("utf-8")
    try:
        target = WorkspacePathPolicy(workspace).resolve_write(
            ".evalyn/probes/support/evalyn_pytest_reporter.py",
            protect_sensitive=False,
        )
        if target.exists():
            if RunStore.read_bytes(target, max_bytes=len(content)) != content:
                raise RepairError("Fresh candidate reporter path contains unexpected content")
        else:
            RunStore.write_bytes(target, content, mode=0o444)
        return ((target, content),)
    except RepairError:
        raise
    except (OSError, ValueError) as exc:
        raise RepairError(
            f"Could not materialize trusted probe support for final verification: {exc}"
        ) from exc


def _untracked_state(workspace: Path) -> tuple[tuple[bytes, int, int, int], ...]:
    result = _run_sanitized_git(
        workspace,
        "ls-files",
        "--others",
        "-z",
        "--",
        action="inspect candidate untracked validation state",
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepairError(f"Could not inspect candidate untracked validation state: {detail}")
    entries: list[tuple[bytes, int, int, int]] = []
    for raw_relative in result.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        safe = _safe_relative(relative)
        target = workspace.joinpath(*PurePosixPath(safe).parts)
        try:
            metadata = target.lstat()
        except OSError as exc:
            raise RepairError(f"Could not inspect untracked validation path: {relative!r}") from exc
        entries.append((raw_relative, metadata.st_mode, metadata.st_size, metadata.st_mtime_ns))
    return tuple(entries)


def _assert_final_workspace_state(
    workspace: Path,
    *,
    expected_tracked: bytes,
    expected_untracked: tuple[tuple[bytes, int, int, int], ...],
    probe_path: Path,
    artifact: ProbeArtifact,
    run: StoredRun,
    support_paths: tuple[tuple[Path, bytes], ...],
) -> None:
    if _tracked_diff(workspace) != expected_tracked:
        raise RepairError(
            "Final repair verification modified tracked files; refusing to emit a patch"
        )
    if _untracked_state(workspace) != expected_untracked:
        raise RepairError(
            "Final repair verification created or modified unexpected untracked files"
        )
    _assert_probe_unchanged(probe_path, artifact)
    _assert_support_unchanged(support_paths)
    _assert_saved_untracked_unchanged(run, workspace)


def _assert_probe_unchanged(path: Path, artifact: ProbeArtifact) -> None:
    expected = artifact.source.encode("utf-8")
    try:
        observed = RunStore.read_bytes(path, max_bytes=len(expected))
    except (OSError, ValueError) as exc:
        raise RepairError("Final repair verification made the saved probe unreadable") from exc
    if observed != expected:
        raise RepairError("Final repair verification modified the saved probe")


def _assert_support_unchanged(support_paths: tuple[tuple[Path, bytes], ...]) -> None:
    for path, expected in support_paths:
        try:
            observed = RunStore.read_bytes(path, max_bytes=len(expected))
        except (OSError, ValueError) as exc:
            raise RepairError(
                "Final repair verification made trusted probe support unreadable"
            ) from exc
        if observed != expected:
            raise RepairError("Final repair verification modified trusted probe support")


def _assert_saved_untracked_unchanged(run: StoredRun, workspace: Path) -> None:
    metadata = run.manifest.get("dirty_overlay", {})
    raw_paths = metadata.get("untracked", []) if isinstance(metadata, dict) else []
    for raw_relative in raw_paths:
        relative = _safe_relative(raw_relative)
        stored = _contained_run_file(
            run.path,
            PurePosixPath("dirty-untracked", *PurePosixPath(relative).parts).as_posix(),
        )
        target = workspace.joinpath(*PurePosixPath(relative).parts)
        try:
            expected = RunStore.read_bytes(stored, max_bytes=_MAX_OVERLAY_BYTES)
            observed = RunStore.read_bytes(target, max_bytes=len(expected))
            expected_mode = stored.stat().st_mode & 0o777
            observed_mode = target.stat().st_mode & 0o777
        except (OSError, ValueError) as exc:
            raise RepairError(
                f"Final repair verification made saved untracked file unsafe: {relative}"
            ) from exc
        if observed != expected or observed_mode != expected_mode:
            raise RepairError(
                f"Final repair verification modified saved untracked file: {relative}"
            )


class _ContainerRepairExecutor:
    def __init__(
        self,
        adapter: TestAdapter,
        settings: AdapterSettings,
        run_path: Path,
        runtime: str | None,
        timeout_seconds: int,
        setup_env_names: tuple[str, ...] = (),
        allow_remote_endpoint: bool = False,
        expected_image_identity: str | None = None,
        expected_endpoint_scope: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.settings = settings
        self.run_path = run_path
        self.runtime = runtime
        self.timeout_seconds = min(300, timeout_seconds)
        self.image = settings.image or DEFAULT_IMAGES[adapter.name]
        self._probe = ContainerProbeExecutor(
            adapter,
            engine=runtime,
            run_dir=run_path,
            image=self.image,
            setup_argv=settings.setup_argv,
            setup_env_names=setup_env_names,
            timeout_seconds=timeout_seconds,
            allow_remote_endpoint=allow_remote_endpoint,
            expected_image_identity=expected_image_identity,
            expected_endpoint_scope=expected_endpoint_scope,
        )

    @property
    def setup_errors(self) -> Mapping[Path, str]:
        return self._probe.setup_errors

    def prepare(self, revision: Revision) -> None:
        self._probe.prepare((("candidate", revision),))

    def validate_probe(
        self,
        revision: Revision,
        artifact: ProbeArtifact,
        *,
        repetitions: int,
    ) -> tuple[ExecutionResult, ...]:
        return tuple(
            self._probe.execute(revision, artifact, repetition=index)
            for index in range(1, repetitions + 1)
        )

    def validate_suite(
        self,
        revision: Revision,
        argv: tuple[str, ...],
    ) -> ExecutionResult:
        log_path = self.run_path / "logs" / "full-suite.log"
        try:
            timeout = self._probe.remaining_timeout()
            with ContainerRunner(
                revision.workspace,
                image=self._probe.execution_image,
                engine=self.runtime,
                network="none",
                pull_policy="missing",
                workspace_read_only=True,
                default_timeout_seconds=timeout,
                allow_remote_endpoint=self._probe.allow_remote_endpoint,
            ) as runner:
                result = runner.run(
                    argv,
                    env=_suite_environment(self.adapter.name),
                    timeout_seconds=self._probe.remaining_timeout(),
                    full_log_path=log_path,
                )
        except TimeoutError as exc:
            return ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=-1,
                stderr=str(exc),
                metadata={"log_path": str(log_path), "suite_argv": list(argv)},
            )
        metadata = {"log_path": str(log_path), "suite_argv": list(argv)}
        if result.timed_out:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        if result.error is not None:
            return ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=result.exit_code if result.exit_code is not None else -1,
                stdout=result.stdout,
                stderr=result.error,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        if result.exit_code == 0:
            return ExecutionResult.passed(
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        return ExecutionResult.failed(
            exit_code=result.exit_code if result.exit_code is not None else 1,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
            failure_signature=_result_output(result.stdout, result.stderr),
            metadata=metadata,
        )


def _default_executor_factory(
    adapter: TestAdapter,
    settings: AdapterSettings,
    run_path: Path,
    runtime: str | None,
    budget_seconds: int,
    *,
    setup_env_names: tuple[str, ...] = (),
    allow_remote_endpoint: bool = False,
    expected_image_identity: str | None = None,
    expected_endpoint_scope: str | None = None,
) -> RepairExecutor:
    return _ContainerRepairExecutor(
        adapter,
        settings,
        run_path,
        runtime,
        budget_seconds,
        setup_env_names,
        allow_remote_endpoint,
        expected_image_identity,
        expected_endpoint_scope,
    )


def _validate_stored_run(store: RunStore, run: StoredRun) -> None:
    """Reject state-store indirection before using a persisted run for writes."""
    try:
        runs_root = store.runs_root.resolve(strict=True)
        run_path = run.path
        if run_path.is_symlink():
            raise RepairError("Stored run directory cannot be a symlink")
        resolved = run_path.resolve(strict=True)
    except OSError as exc:
        raise RepairError(f"Stored run is unavailable: {exc}") from exc
    if resolved.parent != runs_root or resolved.name != run.run_id:
        raise RepairError("Stored run is not owned by this repository's run store")
    try:
        manifest = RunStore.read_json(resolved / "manifest.json", max_bytes=_MAX_STATE_JSON_BYTES)
        RunStore.read_json(resolved / "findings.json", max_bytes=_MAX_STATE_JSON_BYTES)
    except (OSError, ValueError) as exc:
        raise RepairError(f"Stored run manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict) or manifest != run.manifest:
        raise RepairError("Stored run manifest changed while locating the finding")
    if run.manifest.get("schema_version") != 1:
        raise RepairError("Stored run uses an unsupported schema")
    if run.manifest.get("run_id") != run.run_id:
        raise RepairError("Stored run ID does not match its directory")
    if run.manifest.get("status") != "complete":
        raise RepairError("Only completed runs are eligible for repair")


def _load_owned_finding(run: StoredRun, finding_id: str) -> dict:
    try:
        payload = RunStore.read_json(run.path / "findings.json", max_bytes=_MAX_STATE_JSON_BYTES)
    except (OSError, ValueError) as exc:
        raise RepairError(f"Stored findings are invalid: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RepairError("Stored findings use an unsupported schema")
    candidate_id = run.manifest.get("candidate_id", run.manifest.get("candidate"))
    if (
        payload.get("run_id") != run.run_id
        or payload.get("base_sha") != run.manifest.get("base")
        or payload.get("candidate_sha") != candidate_id
    ):
        raise RepairError("Stored findings do not match their run manifest")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise RepairError("Stored findings must contain a findings list")
    matches = [
        item for item in findings if isinstance(item, dict) and item.get("finding_id") == finding_id
    ]
    if len(matches) != 1:
        raise RepairError("Stored finding must exist exactly once in its run")
    return matches[0]


def _validate_stored_repository(store: RunStore, run: StoredRun) -> Path:
    raw = run.manifest.get("repository")
    if not isinstance(raw, str) or not raw:
        raise RepairError("Stored run has no repository identity")
    repo = Path(raw).resolve()
    if repo != store.repo_path:
        raise RepairError("Stored run belongs to a different repository")
    if not (repo / ".git").exists():
        raise RepairError(f"Stored repository is unavailable: {repo}")
    return repo


def _load_repair_settings(
    repo: Path,
    config_path: Path | None,
    *,
    run: StoredRun,
) -> EvalynSettings:
    if config_path is None:
        configuration = run.manifest.get("configuration")
        provider = configuration.get("provider") if isinstance(configuration, dict) else None
        kind = provider.get("kind") if isinstance(provider, dict) else None
        model = provider.get("model") if isinstance(provider, dict) else None
        if kind not in {"anthropic", "openai-compatible"}:
            raise RepairError("Stored run has an invalid provider-kind snapshot")
        if not isinstance(model, str) or not model.strip() or len(model) > 256 or "\x00" in model:
            raise RepairError("Stored run has an invalid provider-model snapshot")
        return EvalynSettings(
            repo_path=repo,
            provider=ProviderSettings(kind=kind, model=model),
        )
    try:
        requested = resolve_settings_path(repo, config_path)
    except Exception as exc:
        raise RepairError(f"Could not resolve repair settings: {exc}") from exc
    if requested.is_relative_to(repo):
        raise RepairError(
            "Repair provider overrides require an explicit configuration outside the repository"
        )
    try:
        return load_settings(repo, requested)
    except Exception as exc:
        raise RepairError(f"Could not load repair settings: {exc}") from exc


def _load_artifact(
    finding: dict,
) -> tuple[TestAdapter, ProbeArtifact]:
    try:
        expected = finding["artifact"]
        if not isinstance(expected, dict):
            raise TypeError("artifact must be an object")
        adapter_name = str(expected["adapter"])
        source = str(finding["probe_source"])
        if len(source.encode("utf-8")) > _MAX_REPLACEMENT_BYTES:
            raise ValueError("probe source is too large")
        adapter = default_registry().get(adapter_name)
        artifact = adapter.build_artifact(source)
    except Exception as exc:
        raise RepairError(f"Stored finding has invalid probe metadata: {exc}") from exc
    trusted = {
        "probe_id": artifact.probe_id,
        "adapter": artifact.adapter,
        "relative_path": artifact.relative_path.as_posix(),
        "command": list(artifact.command),
        "source_sha256": artifact.source_sha256,
        "byte_size": len(artifact.source.encode("utf-8")),
        "media_type": artifact.media_type,
    }
    for key, value in trusted.items():
        if expected.get(key) != value:
            raise RepairError(f"Stored probe {key} does not match trusted adapter output")
    return adapter, artifact


def _load_recorded_configuration(
    run: StoredRun,
    adapter_name: str,
) -> _RecordedRepairConfiguration:
    """Load the immutable execution inputs captured by the original check."""
    configuration = run.manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise RepairError("Stored run has no valid configuration snapshot")

    sandbox = configuration.get("sandbox")
    if not isinstance(sandbox, dict):
        raise RepairError("Stored run has no valid sandbox configuration snapshot")
    recorded_runtime = sandbox.get("runtime")
    if recorded_runtime not in {"auto", "docker", "podman"}:
        raise RepairError("Stored run has an invalid runtime snapshot")
    setup_env_names = _recorded_argv(
        sandbox.get("setup_env_names"),
        "sandbox setup environment",
    )
    if any(not _is_setup_environment_name(name) for name in setup_env_names):
        raise RepairError("Stored run contains a non-package-registry setup environment name")
    allow_remote_endpoint = sandbox.get("allow_remote_endpoint", False)
    if not isinstance(allow_remote_endpoint, bool):
        raise RepairError("Stored run has an invalid remote-endpoint snapshot")

    adapters = configuration.get("adapters")
    if not isinstance(adapters, list):
        raise RepairError("Stored run has no valid adapter configuration snapshot")
    matching = [
        item for item in adapters if isinstance(item, dict) and item.get("name") == adapter_name
    ]
    if len(matching) != 1:
        raise RepairError(
            "Stored run must contain exactly one configuration snapshot for the finding adapter"
        )
    adapter_snapshot = matching[0]
    image = adapter_snapshot.get("image")
    if (
        not isinstance(image, str)
        or not image
        or image != image.strip()
        or image.startswith("-")
        or not _OCI_IMAGE.fullmatch(image)
    ):
        raise RepairError("Stored run has an invalid adapter image snapshot")
    declared_image = adapter_snapshot.get("declared_image", image)
    if (
        not isinstance(declared_image, str)
        or not declared_image
        or declared_image != declared_image.strip()
        or declared_image.startswith("-")
        or not _OCI_IMAGE.fullmatch(declared_image)
    ):
        raise RepairError("Stored run has an invalid declared adapter image snapshot")
    environments = run.manifest.get("environments", {})
    environment = environments.get(adapter_name) if isinstance(environments, dict) else None
    expected_image_identity: str | None = None
    expected_endpoint_scope: str | None = None
    if isinstance(environment, dict):
        resolved = environment.get("resolved_image_identity")
        endpoint = environment.get("endpoint_scope")
        if isinstance(resolved, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", resolved):
            expected_image_identity = resolved
        if endpoint in {"local", "remote"}:
            expected_endpoint_scope = endpoint
    setup_argv = _recorded_argv(adapter_snapshot.get("setup_argv"), "adapter setup command")
    suite_argv = _recorded_argv(adapter_snapshot.get("suite_argv"), "adapter suite command")

    limits = configuration.get("limits")
    repetitions = limits.get("repeat") if isinstance(limits, dict) else None
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or not 2 <= repetitions <= MAX_REPEAT
    ):
        raise RepairError("Stored run has an invalid repeat-count snapshot")
    budget_seconds = limits.get("budget_seconds") if isinstance(limits, dict) else None
    if (
        not isinstance(budget_seconds, int)
        or isinstance(budget_seconds, bool)
        or not 1 <= budget_seconds <= MAX_BUDGET_SECONDS
    ):
        raise RepairError("Stored run has an invalid execution-budget snapshot")

    return _RecordedRepairConfiguration(
        adapter_settings=AdapterSettings(
            name=adapter_name,
            image=declared_image,
            setup_argv=setup_argv,
            suite_argv=suite_argv,
        ),
        runtime=recorded_runtime,
        setup_env_names=setup_env_names,
        repetitions=repetitions,
        budget_seconds=budget_seconds,
        allow_remote_endpoint=allow_remote_endpoint,
        expected_image_identity=expected_image_identity,
        expected_endpoint_scope=expected_endpoint_scope,
    )


def _recorded_argv(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not all(isinstance(part, str) for part in value)
        or any(not part or "\x00" in part for part in value)
    ):
        raise RepairError(f"Stored run has an invalid {label} snapshot")
    return tuple(value)


def _is_setup_environment_name(name: str) -> bool:
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        return False
    return name in _SETUP_ENV_EXACT or name.startswith(_SETUP_ENV_PREFIXES)


def _recorded_runtime(recorded: str, requested: str | None) -> str | None:
    if requested is not None and requested != recorded:
        raise RepairError("Repair runtime cannot override the runtime captured by the saved check")
    return None if recorded == "auto" else recorded


def _stored_revisions(
    run: StoredRun,
    finding: dict,
) -> tuple[str, str, str]:
    base = run.manifest.get("base")
    candidate = run.manifest.get("candidate")
    candidate_id = run.manifest.get("candidate_id", candidate)
    if not all(isinstance(value, str) and value for value in (base, candidate, candidate_id)):
        raise RepairError("Stored run is missing pinned revision metadata")
    if not _GIT_OID.fullmatch(base) or not _GIT_OID.fullmatch(candidate):
        raise RepairError("Stored run revisions must be full immutable Git object IDs")
    valid_candidate_ids = {candidate}
    overlay = run.manifest.get("dirty_overlay", {})
    if isinstance(overlay, dict) and overlay.get("present"):
        digest = overlay.get("content_sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            valid_candidate_ids.add(f"{candidate}+dirty.{digest[:12]}")
    if candidate_id not in valid_candidate_ids:
        raise RepairError("Stored candidate identity does not match its pinned revision")
    if finding.get("finding_id") is None or finding.get("run_id") != run.run_id:
        raise RepairError("Finding identity does not match its stored run")
    if finding.get("base_sha") != base:
        raise RepairError("Finding base revision does not match its run")
    if finding.get("candidate_sha") != candidate_id:
        raise RepairError("Finding candidate revision does not match its run")
    return base, candidate, candidate_id


def _validate_finding_evidence(
    finding: dict,
    base_sha: str,
    candidate_id: str,
    *,
    recorded_repetitions: int,
) -> _ValidatedEvidence:
    if finding.get("classification") != "stable_change":
        raise RepairError("Only stable differential findings are eligible for repair")
    triage_status = finding.get("triage_status", "unreviewed")
    semantic_state = finding.get("semantic_state", "uncertain_change")
    if triage_status != "confirmed" and not (
        triage_status == "unreviewed" and semantic_state == "likely_regression"
    ):
        raise RepairError("Repair requires a likely regression or a maintainer-confirmed finding")
    base_runs = finding.get("base_runs")
    candidate_runs = finding.get("candidate_runs")
    if (
        not isinstance(base_runs, list)
        or not isinstance(candidate_runs, list)
        or len(base_runs) < 2
        or len(base_runs) != len(candidate_runs)
        or len(base_runs) != recorded_repetitions
    ):
        raise RepairError("Stored finding evidence does not match the recorded repeat count")
    expected_repetitions = list(range(1, len(base_runs) + 1))
    for label, runs, revision in (
        ("base", base_runs, base_sha),
        ("candidate", candidate_runs, candidate_id),
    ):
        if not all(isinstance(item, dict) for item in runs):
            raise RepairError(f"Stored {label} evidence is malformed")
        if [item.get("repetition") for item in runs] != expected_repetitions:
            raise RepairError(f"Stored {label} repetitions are inconsistent")
        if any(item.get("side") != label or item.get("revision_sha") != revision for item in runs):
            raise RepairError(f"Stored {label} evidence does not match the finding")
        if any(not _valid_stored_result(item) for item in runs):
            raise RepairError(f"Stored {label} execution result is malformed")

    if any(
        item.get("status") != "passed"
        or item.get("exit_code") != 0
        or item.get("failure_signature") != ""
        for item in base_runs
    ):
        raise RepairError("Stored base evidence must be uniformly passing")

    candidate_statuses = {item.get("status") for item in candidate_runs}
    if len(candidate_statuses) != 1 or not candidate_statuses <= {"failed", "timeout"}:
        raise RepairError(
            "Stored candidate evidence mixes timeout and failed outcomes or is not failing"
        )
    raw_status = next(iter(candidate_statuses))
    if raw_status == "failed":
        signatures = {item.get("failure_signature") for item in candidate_runs}
        if len(signatures) != 1 or not all(
            isinstance(value, str) and value for value in signatures
        ):
            raise RepairError("Stored candidate failure signatures are not uniform")
        stored_signature = next(iter(signatures))
        if any(item.get("exit_code") == 0 for item in candidate_runs):
            raise RepairError("Stored failed candidate evidence has a zero exit code")
        candidate_status = ExecutionStatus.FAILED
    else:
        if any(
            item.get("failure_signature") != "" or item.get("exit_code") != -1
            for item in candidate_runs
        ):
            raise RepairError("Stored timeout evidence is inconsistent")
        stored_signature = "timeout"
        candidate_status = ExecutionStatus.TIMEOUT

    if finding.get("failure_signature") != stored_signature:
        raise RepairError("Stored finding failure signature does not match its repeated evidence")
    return _ValidatedEvidence(
        repetitions=recorded_repetitions,
        candidate_status=candidate_status,
        failure_signature=stored_signature,
    )


def _valid_stored_result(item: dict) -> bool:
    exit_code = item.get("exit_code")
    duration = item.get("duration_seconds")
    return (
        isinstance(item.get("status"), str)
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration >= 0
        and isinstance(item.get("failure_signature"), str)
        and isinstance(item.get("metadata"), dict)
    )


def _restore_saved_overlay(run: StoredRun, worktrees: WorktreeSession) -> None:
    metadata = run.manifest.get("dirty_overlay", {})
    if not isinstance(metadata, dict):
        raise RepairError("Saved dirty overlay metadata must be an object")
    present = metadata.get("present", False)
    if present is False:
        return
    if present is not True:
        raise RepairError("Saved dirty overlay has invalid presence metadata")
    expected_digest = metadata.get("content_sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise RepairError("Saved dirty overlay has invalid integrity metadata")

    patch = b""
    patch_name = metadata.get("patch")
    if patch_name is not None and not isinstance(patch_name, str):
        raise RepairError("Saved dirty overlay has invalid patch metadata")
    if patch_name:
        patch_path = _contained_run_file(run.path, patch_name)
        try:
            patch = RunStore.read_bytes(patch_path, max_bytes=_MAX_OVERLAY_BYTES)
        except (OSError, ValueError) as exc:
            raise RepairError(f"Saved dirty patch is unsafe: {exc}") from exc

    untracked = metadata.get("untracked", [])
    if not isinstance(untracked, list) or not all(isinstance(item, str) for item in untracked):
        raise RepairError("Saved dirty overlay has invalid untracked metadata")
    total_bytes = len(patch)
    digest_with_modes = hashlib.sha256()
    digest_with_modes.update(patch)
    copied: list[tuple[str, Path, bytes, int]] = []
    for raw_relative in untracked:
        relative = _safe_relative(raw_relative)
        stored_relative = PurePosixPath(
            "dirty-untracked", *PurePosixPath(relative).parts
        ).as_posix()
        source = _contained_run_file(run.path, stored_relative)
        if source.is_symlink() or not source.is_file():
            raise RepairError(f"Saved dirty file is not regular: {relative}")
        remaining = _MAX_OVERLAY_BYTES - total_bytes
        try:
            content = RunStore.read_bytes(source, max_bytes=remaining)
        except (OSError, ValueError) as exc:
            raise RepairError(f"Saved dirty file is unsafe: {relative}: {exc}") from exc
        total_bytes += len(content)
        if total_bytes > _MAX_OVERLAY_BYTES:
            raise RepairError("Saved dirty overlay exceeds the repair size limit")
        mode = source.stat().st_mode & 0o777
        encoded = relative.encode("utf-8", errors="surrogateescape")
        digest_with_modes.update(encoded)
        digest_with_modes.update(b"\0")
        digest_with_modes.update(mode.to_bytes(2, byteorder="big"))
        digest_with_modes.update(content)
        copied.append((relative, source, content, mode))
    if expected_digest != digest_with_modes.hexdigest():
        raise RepairError("Saved dirty overlay failed its integrity check")

    if patch:
        result = _run_sanitized_git(
            worktrees.candidate_path,
            "apply",
            "--binary",
            "--whitespace=nowarn",
            "-",
            input_bytes=patch,
            action="restore saved dirty patch",
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RepairError(f"Could not restore saved dirty patch: {detail}")
    for relative, _source, content, mode in copied:
        target = worktrees.candidate_path.joinpath(*PurePosixPath(relative).parts)
        try:
            target.resolve(strict=False).relative_to(worktrees.candidate_path)
        except ValueError as exc:
            raise RepairError(f"Saved dirty path escapes candidate: {relative}") from exc
        if target.exists() or target.is_symlink():
            raise RepairError(f"Saved untracked path collides with candidate: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(mode)


def _changed_source_paths(
    worktrees: WorktreeSession,
    *,
    base_sha: str,
    candidate_sha: str,
    tracked_paths: Sequence[str],
) -> tuple[str, ...]:
    changed = set(_git_changed_paths(worktrees.candidate_path, base_sha, candidate_sha))
    changed.update(_git_changed_paths(worktrees.candidate_path, "HEAD"))
    tracked = set(tracked_paths)
    candidates = tuple(sorted(path for path in changed if path in tracked and _is_source(path)))
    provisional = WorkspacePathPolicy(
        worktrees.candidate_path,
        tracked_paths=tracked_paths,
        context_allowlist=candidates,
        fix_allowlist=candidates,
    )
    allowed: list[str] = []
    for path in candidates:
        try:
            provisional.resolve_fix_target(path)
        except PathPolicyError:
            continue
        allowed.append(path)
    return tuple(allowed)


def _git_changed_paths(workspace: Path, *revisions: str) -> tuple[str, ...]:
    result = _run_sanitized_git(
        workspace,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-only",
        "--diff-filter=ACMRT",
        "-z",
        *revisions,
        "--",
        action="resolve changed repair paths",
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepairError(f"Could not resolve changed repair paths: {detail}")
    return tuple(
        part.decode("utf-8", errors="surrogateescape")
        for part in result.stdout.split(b"\0")
        if part
    )


def _is_source(relative: str) -> bool:
    path = PurePosixPath(relative)
    folded = path.name.casefold()
    suffix = path.suffix.casefold()
    stem = folded[: -len(suffix)] if suffix else folded
    role_tokens = set(re.split(r"[._-]+", stem))
    directory_parts = tuple(part.casefold() for part in path.parts[:-1])
    folded_parts = set(directory_parts)
    if folded_parts & _PROTECTED_SOURCE_DIRECTORIES:
        return False
    if directory_parts and (
        directory_parts[0] in _PROTECTED_TOP_LEVEL_DIRECTORIES
        or any(part.startswith(".") for part in directory_parts)
    ):
        return False
    if folded in _PROTECTED_SOURCE_NAMES:
        return False
    if (
        stem.startswith("test_")
        or stem.startswith("test-")
        or stem.endswith("_test")
        or stem.endswith("-test")
        or bool(role_tokens & {"config", "spec", "test"})
        or stem in {"setup", "setuptests", "setup_tests", "testsetup", "test_setup"}
        or stem.startswith(("jest.setup", "pytest.", "vitest.setup", "vitest.workspace"))
    ):
        return False
    return suffix in _SOURCE_SUFFIXES


def _repair_prompt(
    finding: dict,
    policy: WorkspacePathPolicy,
    allowed: tuple[str, ...],
    feedback: Sequence[str],
) -> str:
    sections: list[str] = []
    for relative in sorted(allowed)[:_MAX_CONTEXT_FILES]:
        source = policy.resolve_context(relative)
        rendered = _read_bounded_utf8_source(source, relative)
        if len(rendered) > _MAX_CONTEXT_BYTES_PER_FILE:
            rendered = rendered[:_MAX_CONTEXT_BYTES_PER_FILE]
            suffix = "\n... [context truncated by Evalyn]"
        else:
            suffix = ""
        sections.append(f"### {relative}\n```\n{rendered}{suffix}\n```")
    base_runs = _evidence_summary(finding.get("base_runs", []))
    candidate_runs = _evidence_summary(finding.get("candidate_runs", []))
    probe_source = str(finding.get("probe_source", ""))[:_MAX_CONTEXT_BYTES_PER_FILE]
    raw_signature = str(finding.get("failure_signature", ""))
    signature_digest = (
        hashlib.sha256(raw_signature.encode("utf-8", errors="replace")).hexdigest()[:16]
        if raw_signature
        else "none"
    )
    feedback_text = "\n".join(f"- {item}" for item in feedback) or "- none"
    return (
        f"Finding: {finding.get('finding_id', '')}\n"
        f"Title: {finding.get('title', '')}\n"
        f"Failure signature digest: {signature_digest}\n"
        f"Oracle: {finding.get('oracle', '')}\n"
        f"Baseline evidence: {json.dumps(base_runs, sort_keys=True)[:4000]}\n"
        f"Candidate evidence: {json.dumps(candidate_runs, sort_keys=True)[:4000]}\n\n"
        f"Generated regression probe:\n```\n{probe_source}\n```\n\n"
        "Allowed repair paths:\n"
        + "\n".join(f"- {path}" for path in allowed)
        + "\n\nPrevious rejected hypotheses:\n"
        + feedback_text
        + "\n\nChanged source context:\n"
        + "\n\n".join(sections)
    )


def _read_bounded_utf8_source(path: Path, relative: str) -> str:
    try:
        content = RunStore.read_bytes(path, max_bytes=_MAX_REPLACEMENT_BYTES)
    except (OSError, ValueError) as exc:
        raise RepairError(f"Repair target is unsafe or too large: {relative}") from exc
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"Repair target is not readable UTF-8 source: {relative}") from exc


def _evidence_summary(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    summary: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        summary.append(
            {
                "side": item.get("side"),
                "repetition": item.get("repetition"),
                "status": item.get("status"),
                "exit_code": item.get("exit_code"),
            }
        )
    return summary


def _request_proposal(llm: BaseLLM, prompt: str) -> RepairProposal:
    try:
        raw = llm.complete_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=8192,
        )
    except Exception as exc:
        raise _RetryableRepair(f"LLM request failed: {exc}") from exc
    if not isinstance(raw, dict):
        raise _RetryableRepair("LLM repair response must be a JSON object")
    expected = {"file_path", "replacement_content", "explanation"}
    if set(raw) != expected:
        extra = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        raise _RetryableRepair(f"LLM response schema mismatch (missing={missing}, extra={extra})")
    if not all(isinstance(raw[key], str) for key in expected):
        raise _RetryableRepair("LLM repair fields must all be strings")
    file_path = raw["file_path"].strip()
    replacement = raw["replacement_content"]
    explanation = raw["explanation"].strip()
    if not file_path or not replacement or not explanation:
        raise _RetryableRepair("LLM repair fields cannot be empty")
    if "\x00" in replacement:
        raise _RetryableRepair("Replacement content cannot contain NUL bytes")
    try:
        replacement_size = len(replacement.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise _RetryableRepair("Replacement content must be valid UTF-8 text") from exc
    if replacement_size > _MAX_REPLACEMENT_BYTES:
        raise _RetryableRepair(f"Replacement content exceeds {_MAX_REPLACEMENT_BYTES} bytes")
    return RepairProposal(file_path, replacement, explanation)


def _atomic_replace_source(target: Path, replacement: str) -> None:
    mode = target.stat().st_mode & 0o777
    fd, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.evalyn-repair-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).chmod(mode)
        Path(temporary).replace(target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _tracked_diff(workspace: Path) -> bytes:
    result = _run_sanitized_git(
        workspace,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "--full-index",
        "HEAD",
        "--",
        action="inspect candidate validation state",
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepairError(f"Could not inspect candidate validation state: {detail}")
    return result.stdout


def _check_patch_applies(workspace: Path, patch: bytes) -> None:
    """Forward-check a repair-only patch against the reconstructed candidate."""
    result = _run_sanitized_git(
        workspace,
        "apply",
        "--check",
        "--binary",
        "--whitespace=nowarn",
        "-",
        input_bytes=patch,
        action="check constructed repair patch",
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepairError(
            f"Constructed repair patch does not apply to the saved candidate: {detail}"
        )


def _build_repair_patch(
    relative: str,
    original: str,
    replacement: str,
    *,
    mode: int,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="evalyn-repair-patch-") as temporary:
        repo = Path(temporary)
        target = repo.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original, encoding="utf-8", newline="")
        target.chmod(mode)
        commands = (
            ("init", "-q"),
            ("add", "--", relative),
        )
        for command in commands:
            result = _run_sanitized_git(
                repo,
                *command,
                action="construct repair patch",
            )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                raise RepairError(f"Could not construct repair patch: {detail}")
        target.write_text(replacement, encoding="utf-8", newline="")
        result = _run_sanitized_git(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "--",
            relative,
            action="construct repair patch",
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RepairError(f"Could not construct repair patch: {detail}")
        if not result.stdout:
            raise _RetryableRepair("Replacement did not produce a patch")
        return result.stdout


def _run_sanitized_git(
    workspace: Path,
    *args: str,
    input_bytes: bytes | None = None,
    action: str,
) -> BoundedProcessResult:
    """Run repair Git without executable host configuration or ambient Git state."""
    try:
        result = run_bounded_process(
            sanitized_git_command(workspace, *args),
            timeout=60,
            env=sanitized_git_environment(),
            input_bytes=input_bytes,
            max_stdout_bytes=_MAX_GIT_OUTPUT_BYTES,
            max_stderr_bytes=_MAX_GIT_STDERR_BYTES,
        )
        if result.stdout_truncated or result.stderr_truncated:
            raise RepairError(f"Could not {action}: Git output exceeded its safety limit")
        return result
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise RepairError(f"Could not {action}: {exc}") from exc


def _write_patch_atomically(output: Path, patch: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.evalyn-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(patch)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        Path(temporary).unlink()
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _raise_for_infrastructure(
    results: Sequence[ExecutionResult],
    label: str,
) -> None:
    infrastructure = {
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.INVALID_PROBE,
        ExecutionStatus.INFRA_ERROR,
    }
    bad = [result for result in results if result.status in infrastructure]
    if bad:
        raise RepairError(
            f"{label.capitalize()} validation was incomplete: {_result_detail(bad[0])}"
        )


def _result_detail(result: ExecutionResult) -> str:
    return _result_output(result.stderr, result.stdout) or result.status.value


def _result_fingerprint(result: ExecutionResult) -> str:
    output = f"{result.stderr}\n{result.stdout}".encode("utf-8", errors="replace")
    digest = hashlib.sha256(output).hexdigest()[:16]
    return f"{result.status.value} (exit {result.exit_code}, output digest {digest})"


def _result_output(first: str, second: str) -> str:
    text = (first.strip() or second.strip()).replace("\x00", "")
    return text[-2000:]


def _default_suite(adapter_name: str) -> tuple[str, ...]:
    try:
        return _DEFAULT_SUITES[adapter_name]
    except KeyError as exc:
        raise RepairError(f"No trusted default full suite for adapter: {adapter_name}") from exc


def _suite_environment(adapter_name: str) -> dict[str, str]:
    if adapter_name == "pytest":
        return {
            "PATH": "/workspace/.evalyn/venv/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": "/workspace",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    return {"CI": "1"}


def _safe_relative(raw: str) -> str:
    normalized = str(raw).replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or "\x00" in normalized
        or ".." in path.parts
        or ".git" in {part.casefold() for part in path.parts}
    ):
        raise RepairError(f"Unsafe stored relative path: {raw!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise RepairError(f"Unsafe stored relative path: {raw!r}")
    return PurePosixPath(*parts).as_posix()


def _contained_run_file(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*PurePosixPath(safe).parts)
    try:
        cursor = resolved_root
        for component in PurePosixPath(safe).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise RepairError(f"Stored run file cannot use symlinks: {relative!r}")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except RepairError:
        raise
    except (OSError, ValueError) as exc:
        raise RepairError(f"Stored run file escapes its run directory: {relative!r}") from exc
    return resolved
