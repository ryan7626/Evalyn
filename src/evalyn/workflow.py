"""End-to-end v0.2 diff-regression workflow."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path, PurePosixPath
from typing import Protocol

from evalyn.adapters.base import ProbeArtifact, TestAdapter
from evalyn.adapters.registry import default_registry
from evalyn.analysis import build_change_graph
from evalyn.engine.assessment import apply_semantic_assessment
from evalyn.engine.classifier import DifferentialClassifier
from evalyn.engine.models import (
    DifferentialRunResult,
    ExecutionStatus,
    Revision,
    RunVerdict,
)
from evalyn.engine.orchestrator import DifferentialEngine, ProbeExecutor
from evalyn.engine.rules import IntentOracleAssessor
from evalyn.execution import (
    RepoRunLock,
    WorkspacePathPolicy,
    WorktreeSession,
    resolve_container_engine,
)
from evalyn.execution.bounded import BoundedProcessResult, run_bounded_process
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment
from evalyn.generation import (
    GenerationRequest,
    ProbeInfrastructureError,
    ProbeValidationError,
    RepositoryContextSource,
    generate_probe_portfolio,
)
from evalyn.llm import create_provider
from evalyn.llm.base import BaseLLM
from evalyn.probe_executor import (
    DEFAULT_IMAGES,
    ContainerProbeExecutor,
    ProbeCollectionError,
)
from evalyn.reporting_v2 import write_reports
from evalyn.revisions import (
    RevisionRange,
    get_diff,
    is_dirty,
    resolve_range,
    validate_git_repository,
)
from evalyn.run_store import RunStore, StoredRun
from evalyn.settings import (
    MAX_CONFIG_BYTES,
    AdapterSettings,
    EvalynSettings,
    ProviderSettings,
    load_settings,
    resolve_settings_path,
)

_DIFF_CHAR_LIMIT = 120_000
_DIFF_CAPTURE_LIMIT = (_DIFF_CHAR_LIMIT * 4) + 1
_CONTEXT_FILE_CHAR_LIMIT = 20_000
_CONTEXT_FILE_BYTE_LIMIT = (_CONTEXT_FILE_CHAR_LIMIT * 4) + 1
_GIT_METADATA_LIMIT = 8 * 1024 * 1024
_GIT_STDERR_LIMIT = 256 * 1024


class CheckError(RuntimeError):
    """A valid check could not be completed; callers should use exit code 3."""

    def __init__(self, message: str, *, run_path: Path | None = None) -> None:
        super().__init__(message)
        self.run_path = run_path


@dataclass(frozen=True)
class CheckOptions:
    repo_path: Path
    range_spec: str | None = None
    include_dirty: bool = False
    intent: tuple[str, ...] = ()
    adapters: tuple[str, ...] = ()
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    max_tests: int | None = None
    repeat: int | None = None
    budget_seconds: int | None = None
    runtime: str | None = None
    provider_kind: str | None = None
    model: str | None = None
    config_path: Path | None = None
    state_root: Path | None = None
    allow_remote_container: bool = False


@dataclass(frozen=True)
class CheckInspection:
    repo_path: Path
    revisions: RevisionRange
    adapters: tuple[str, ...]
    changed_files: tuple[str, ...]
    dirty: bool
    settings: EvalynSettings


@dataclass(frozen=True)
class CheckOutcome:
    run: StoredRun
    results: tuple[DifferentialRunResult, ...]
    report_paths: dict[str, Path]
    duration_seconds: float

    @property
    def findings(self):
        return tuple(finding for result in self.results for finding in result.findings)

    @property
    def gating_findings(self):
        return tuple(finding for finding in self.findings if finding.should_gate)

    @property
    def incomplete(self) -> bool:
        return self.verdict in {RunVerdict.INCOMPLETE, RunVerdict.INSUFFICIENT_EVIDENCE}

    @property
    def verdict(self) -> RunVerdict:
        verdicts = {result.verdict for result in self.results}
        if RunVerdict.INCOMPLETE in verdicts:
            return RunVerdict.INCOMPLETE
        if RunVerdict.REGRESSION_FOUND in verdicts:
            return RunVerdict.REGRESSION_FOUND
        if RunVerdict.CHANGE_OBSERVED in verdicts:
            return RunVerdict.CHANGE_OBSERVED
        if verdicts and verdicts == {RunVerdict.NO_DIFFERENTIAL_OBSERVED}:
            return RunVerdict.NO_DIFFERENTIAL_OBSERVED
        return RunVerdict.INSUFFICIENT_EVIDENCE


class PreparedProbeExecutor(ProbeExecutor, Protocol):
    def prepare(self, revisions: tuple[tuple[str, Revision], ...]) -> None: ...

    @property
    def setup_errors(self) -> dict[Path, str]: ...


ExecutorFactory = Callable[
    [TestAdapter, AdapterSettings, Path, str | None, int], PreparedProbeExecutor
]


def inspect_check(options: CheckOptions) -> CheckInspection:
    repo = validate_git_repository(options.repo_path)
    dirty = is_dirty(repo)
    if dirty and not options.include_dirty:
        raise CheckError(
            "Repository has staged, unstaged, or untracked changes; rerun with "
            "--include-dirty to snapshot them explicitly"
        )
    revisions = resolve_range(repo, options.range_spec, allow_same=options.include_dirty)
    trusted_config_path = _require_trusted_configuration(repo, revisions, options.config_path)
    settings = (
        load_settings(repo, trusted_config_path)
        if trusted_config_path is not None
        else EvalynSettings(repo_path=repo)
    )
    registry = default_registry()
    names = options.adapters or tuple(item.name for item in settings.adapters)
    if not names:
        names = tuple(adapter.name for adapter in registry.detect(repo))
    if not names:
        raise CheckError("No supported test framework detected; configure pytest or vitest")
    for name in names:
        registry.get(name)
    diff = get_diff(repo, revisions)
    changed = _changed_paths_from_diff(repo, revisions, include_dirty=options.include_dirty)
    changed = tuple(
        path
        for path in changed
        if (
            not options.include_globs
            or any(Path(path).match(glob) for glob in options.include_globs)
        )
        and not any(Path(path).match(glob) for glob in options.exclude_globs)
    )
    if not changed:
        raise CheckError("No changed files remain after include/exclude filters")
    if not diff.strip() and not (dirty and options.include_dirty):
        raise CheckError("The selected comparison contains no source changes")
    return CheckInspection(repo, revisions, names, changed, dirty, settings)


def run_check(
    options: CheckOptions,
    *,
    llm: BaseLLM | None = None,
    executor_factory: ExecutorFactory | None = None,
) -> CheckOutcome:
    started = time.monotonic()
    inspection = inspect_check(options)
    settings = _resolved_settings(inspection.settings, options)
    store = RunStore(inspection.repo_path, options.state_root)
    run = store.create_run(
        base=inspection.revisions.base_sha,
        candidate=inspection.revisions.candidate_sha,
    )
    lock = RepoRunLock(
        inspection.repo_path,
        lock_path=store.repo_root / "run.lock",
    )
    results: list[DifferentialRunResult] = []
    report_paths: dict[str, Path] = {}
    candidate_id = inspection.revisions.candidate_sha
    overlay_metadata: dict[str, object] = {"present": False}
    recorded_runtime = settings.sandbox.runtime
    environments: dict[str, dict[str, object]] = {}

    try:
        with (
            lock,
            WorktreeSession(
                inspection.repo_path,
                baseline_revision=inspection.revisions.base_sha,
                candidate_revision=inspection.revisions.candidate_sha,
                include_dirty_overlay=options.include_dirty,
                temp_parent=store.repo_root / "worktrees",
            ) as worktrees,
        ):
            if not worktrees.overlay.is_empty:
                candidate_id = (
                    f"{worktrees.candidate_sha}+dirty.{worktrees.overlay.content_sha256[:12]}"
                )
                overlay_metadata = _persist_dirty_overlay(run.path, worktrees)
            diff = _workspace_diff(worktrees, inspection.changed_files)
            context_sources = _collect_context(worktrees, inspection.changed_files)
            repository_paths = tuple(
                sorted(
                    set(worktrees.tracked_files("baseline"))
                    | set(worktrees.tracked_files("candidate"))
                )
            )
            change_graph = build_change_graph(
                worktrees.baseline_path,
                worktrees.candidate_path,
                changed_paths=inspection.changed_files,
                diff=diff,
                repository_paths=repository_paths,
            )
            registry = default_registry()
            base_revision = Revision(worktrees.baseline_sha, worktrees.baseline_path)
            candidate_revision = Revision(candidate_id, worktrees.candidate_path)

            for name in inspection.adapters:
                remaining_budget = ceil(settings.budget_seconds - (time.monotonic() - started))
                if remaining_budget <= 0:
                    raise CheckError("Check execution budget was exhausted")
                adapter = registry.get(name)
                adapter_settings = _adapter_settings(settings, name)
                runtime = None if settings.sandbox.runtime == "auto" else settings.sandbox.runtime
                if executor_factory is None:
                    # Resolve auto once and persist the concrete engine so later
                    # reproduction and repair cannot silently switch runtimes.
                    recorded_runtime = Path(resolve_container_engine(runtime)).name
                    runtime = recorded_runtime
                    executor = _container_executor_factory(
                        adapter,
                        adapter_settings,
                        run.path,
                        runtime,
                        remaining_budget,
                        setup_env_names=settings.sandbox.setup_env,
                        allow_remote_endpoint=options.allow_remote_container,
                    )
                else:
                    executor = executor_factory(
                        adapter,
                        adapter_settings,
                        run.path,
                        runtime,
                        remaining_budget,
                    )
                executor.prepare((("base", base_revision), ("candidate", candidate_revision)))
                if executor.setup_errors:
                    detail = "; ".join(
                        f"{path.name}: {message}" for path, message in executor.setup_errors.items()
                    )
                    raise CheckError(f"{name} dependency setup failed: {detail}")
                fingerprint = getattr(executor, "environment_fingerprint", None)
                if isinstance(fingerprint, dict):
                    environments[name] = fingerprint
                elif executor_factory is None:
                    raise CheckError(f"{name} execution environment identity is unavailable")
                generation_budget = settings.budget_seconds - (time.monotonic() - started)
                if generation_budget <= 0:
                    raise CheckError("Check execution budget was exhausted before generation")
                provider = llm or create_provider(
                    settings.provider,
                    timeout_seconds=min(120, generation_budget),
                )
                portfolio = generate_probe_portfolio(
                    provider,
                    GenerationRequest(
                        adapter=name,
                        diff=diff,
                        context_sources=context_sources,
                        intent=options.intent,
                        max_tests=settings.max_tests,
                        change_graph=change_graph,
                    ),
                    preflight=_collection_preflight(executor, name, base_revision),
                )
                RunStore.write_json(
                    run.path / "analysis" / f"{name}.json",
                    {
                        "schema_version": 1,
                        "adapter": name,
                        "change_graph": change_graph.to_dict(),
                        "portfolio": portfolio.to_dict(),
                    },
                )
                if portfolio.infrastructure_error_count:
                    raise CheckError(
                        f"{name} provider infrastructure failed during portfolio generation"
                    )
                if not portfolio.probes:
                    raise CheckError(f"{name} produced no valid probes after bounded correction")
                engine = DifferentialEngine(
                    adapter,
                    executor,
                    classifier=DifferentialClassifier(settings.repeat),
                    assessor=IntentOracleAssessor(options.intent),
                )
                evaluated = engine.evaluate(
                    base_revision,
                    candidate_revision,
                    portfolio.probes,
                    run_id=run.run_id,
                )
                results.append(
                    _verify_target_interventions(
                        evaluated,
                        adapter=adapter,
                        executor=executor,
                        base_revision=base_revision,
                        candidate_revision=candidate_revision,
                        assessor=IntentOracleAssessor(options.intent),
                    )
                )

            metadata = {
                "run_id": run.run_id,
                "base_sha": worktrees.baseline_sha,
                "candidate_sha": candidate_id,
                "intent": options.intent,
                "environments": environments,
            }
            report_paths = write_reports(run.path, results, metadata=metadata)

        duration = time.monotonic() - started
        verdict = _aggregate_result_verdict(results)
        incomplete = verdict in {RunVerdict.INCOMPLETE, RunVerdict.INSUFFICIENT_EVIDENCE}
        status = "incomplete" if incomplete else "complete"
        run = store.finish_run(
            run,
            status=status,
            extra={
                "candidate_id": candidate_id,
                "adapters": list(inspection.adapters),
                "finding_count": sum(len(result.findings) for result in results),
                "gating_finding_count": sum(len(result.gating_findings) for result in results),
                "duration_seconds": round(duration, 3),
                "verdict": verdict.value,
                "dirty_overlay": overlay_metadata,
                "environments": environments,
                "configuration": _settings_snapshot(
                    settings,
                    inspection.adapters,
                    runtime=recorded_runtime,
                    environments=environments,
                    allow_remote_endpoint=options.allow_remote_container,
                ),
            },
        )
        return CheckOutcome(run, tuple(results), report_paths, duration)
    except BaseException as exc:
        store.finish_run(
            run,
            status="incomplete",
            extra={"error": f"{type(exc).__name__}: {exc}", "candidate_id": candidate_id},
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, CheckError):
            raise CheckError(str(exc), run_path=run.path) from exc
        raise CheckError(f"Check could not complete: {exc}", run_path=run.path) from exc


def _resolved_settings(settings: EvalynSettings, options: CheckOptions) -> EvalynSettings:
    provider = settings.provider
    if options.provider_kind or options.model:
        kind = options.provider_kind or provider.kind
        default_model = "claude-sonnet-4-20250514" if kind == "anthropic" else "gpt-4.1-mini"
        provider = ProviderSettings(
            kind=kind,
            model=options.model or (provider.model if provider.kind == kind else default_model),
            base_url=provider.base_url,
        )
    sandbox = settings.sandbox
    if options.runtime:
        sandbox = replace(sandbox, runtime=options.runtime)
    return replace(
        settings,
        provider=provider,
        sandbox=sandbox,
        max_tests=options.max_tests or settings.max_tests,
        repeat=options.repeat or settings.repeat,
        budget_seconds=options.budget_seconds or settings.budget_seconds,
    )


def _adapter_settings(settings: EvalynSettings, name: str) -> AdapterSettings:
    return next(
        (item for item in settings.adapters if item.name == name),
        AdapterSettings(name=name),
    )


def _settings_snapshot(
    settings: EvalynSettings,
    adapter_names: tuple[str, ...],
    *,
    runtime: str | None = None,
    environments: dict[str, dict[str, object]] | None = None,
    allow_remote_endpoint: bool = False,
) -> dict[str, object]:
    environment_values = environments or {}
    return {
        "provider": {
            "kind": settings.provider.kind,
            "model": settings.provider.model,
        },
        "sandbox": {
            "runtime": runtime or settings.sandbox.runtime,
            "setup_env_names": list(settings.sandbox.setup_env),
            "allow_remote_endpoint": allow_remote_endpoint,
        },
        "adapters": [
            {
                "name": item.name,
                "setup_argv": list(item.setup_argv),
                "suite_argv": list(item.suite_argv),
                "image": (
                    environment_values.get(item.name, {}).get("resolved_image_identity")
                    or item.image
                    or DEFAULT_IMAGES[item.name]
                ),
                "declared_image": item.image or DEFAULT_IMAGES[item.name],
                "environment_fingerprint_sha256": environment_values.get(item.name, {}).get(
                    "fingerprint_sha256"
                ),
            }
            for item in (_adapter_settings(settings, name) for name in adapter_names)
        ],
        "limits": {
            "max_tests": settings.max_tests,
            "repeat": settings.repeat,
            "budget_seconds": settings.budget_seconds,
        },
    }


def _container_executor_factory(
    adapter: TestAdapter,
    settings: AdapterSettings,
    run_path: Path,
    runtime: str | None,
    budget_seconds: int,
    *,
    setup_env_names: tuple[str, ...] = (),
    allow_remote_endpoint: bool = False,
) -> ContainerProbeExecutor:
    return ContainerProbeExecutor(
        adapter,
        engine=runtime,
        run_dir=run_path,
        image=settings.image,
        setup_argv=settings.setup_argv,
        setup_env_names=setup_env_names,
        timeout_seconds=budget_seconds,
        allow_remote_endpoint=allow_remote_endpoint,
    )


def _collection_preflight(
    executor: PreparedProbeExecutor,
    adapter_name: str,
    base_revision: Revision,
) -> Callable[[str, str], None] | None:
    """Adapt an executor's collection check to portfolio validation."""
    preflight_method = getattr(executor, "preflight", None)
    if not callable(preflight_method):
        return None

    def validate(request_adapter: str, source: str) -> None:
        if request_adapter != adapter_name:
            return
        try:
            preflight_method(base_revision, source)
        except ProbeCollectionError as exc:
            if exc.code == "collection_failed":
                raise ProbeValidationError(exc.code, exc.feedback) from exc
            raise ProbeInfrastructureError(
                f"Framework collection preflight infrastructure failed ({exc.code})."
            ) from exc
        except ProbeInfrastructureError:
            raise
        except Exception as exc:
            raise ProbeInfrastructureError(
                "Framework collection preflight infrastructure failed."
            ) from exc

    return validate


_TARGET_INTERVENTION_FILE_LIMIT = 8 * 1024 * 1024


class _TargetInterventionRestoreError(RuntimeError):
    pass


def _verify_target_interventions(
    result: DifferentialRunResult,
    *,
    adapter: TestAdapter,
    executor: PreparedProbeExecutor,
    base_revision: Revision,
    candidate_revision: Revision,
    assessor: IntentOracleAssessor,
) -> DifferentialRunResult:
    """Re-run eligible probes after reverting only their claimed target file."""
    updated = []
    for finding in result.findings:
        evidence = finding.evidence
        if not (
            evidence is not None
            and evidence.environment_integrity
            and evidence.target_binding_verified
            and evidence.oracle_binding_verified
        ):
            updated.append(finding)
            continue
        artifact = adapter.build_artifact(finding.probe_source)
        if artifact.probe_id != finding.artifact.probe_id:
            updated.append(finding)
            continue
        state, repetitions = _run_target_revert_intervention(
            executor,
            artifact,
            finding.probe_metadata,
            base_revision,
            candidate_revision,
        )
        metadata = dict(finding.probe_metadata)
        metadata["target_intervention"] = state
        metadata["target_intervention_repetitions"] = str(repetitions)
        candidate = replace(
            finding,
            probe_metadata=metadata,
            evidence=replace(
                evidence,
                target_intervention_verified=state == "candidate_target_revert_passed",
                target_intervention_repetitions=repetitions,
            ),
        )
        with suppress(Exception):
            candidate = apply_semantic_assessment(candidate, assessor.assess(candidate))
        updated.append(candidate)
    return replace(result, findings=tuple(updated))


def _run_target_revert_intervention(
    executor: PreparedProbeExecutor,
    artifact: ProbeArtifact,
    metadata: object,
    base_revision: Revision,
    candidate_revision: Revision,
) -> tuple[str, int]:
    if not isinstance(metadata, dict) or metadata.get("change_kind") != "modified":
        return "candidate_target_revert_unsupported", 0
    base_relative = metadata.get("target_base_path") or metadata.get("changed_file")
    candidate_relative = metadata.get("target_candidate_path") or metadata.get("changed_file")
    if not isinstance(base_relative, str) or not isinstance(candidate_relative, str):
        return "candidate_target_revert_unsupported", 0
    try:
        base_path = _contained_intervention_path(base_revision.workspace, base_relative)
        candidate_path = _contained_intervention_path(
            candidate_revision.workspace, candidate_relative
        )
        base_content, base_mode = _regular_target_snapshot(base_path)
        candidate_content, candidate_mode = _regular_target_snapshot(candidate_path)
        _atomic_replace_target(candidate_path, base_content, base_mode)
        results = []
        try:
            results = [
                executor.execute(candidate_revision, artifact, repetition=10_000 + index)
                for index in range(1, 4)
            ]
        finally:
            try:
                _atomic_replace_target(candidate_path, candidate_content, candidate_mode)
            except Exception as exc:
                raise _TargetInterventionRestoreError(
                    "candidate target could not be restored after intervention"
                ) from exc
        restored, restored_mode = _regular_target_snapshot(candidate_path)
        if restored != candidate_content or restored_mode != candidate_mode:
            raise _TargetInterventionRestoreError(
                "candidate target bytes changed during intervention restoration"
            )
    except _TargetInterventionRestoreError as exc:
        raise CheckError(str(exc)) from exc
    except Exception:
        return "candidate_target_revert_incomplete", 0
    if any(
        item.status in {ExecutionStatus.INVALID_PROBE, ExecutionStatus.INFRA_ERROR}
        for item in results
    ):
        return "candidate_target_revert_incomplete", len(results)
    if len(results) == 3 and all(item.status == ExecutionStatus.PASSED for item in results):
        return "candidate_target_revert_passed", 3
    return "candidate_target_revert_did_not_recover", len(results)


def _contained_intervention_path(root: Path, raw_relative: str) -> Path:
    relative = PurePosixPath(raw_relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in raw_relative
    ):
        raise ValueError("unsafe target intervention path")
    workspace = root.resolve(strict=True)
    target = workspace.joinpath(*relative.parts)
    if target.resolve(strict=True) != target or not target.is_relative_to(workspace):
        raise ValueError("target intervention path crosses a symlink")
    return target


def _regular_target_snapshot(path: Path) -> tuple[bytes, int]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > _TARGET_INTERVENTION_FILE_LIMIT
    ):
        raise ValueError("target intervention requires one bounded regular file")
    return (
        RunStore.read_bytes(path, max_bytes=_TARGET_INTERVENTION_FILE_LIMIT),
        metadata.st_mode & 0o777,
    )


def _atomic_replace_target(path: Path, content: bytes, mode: int) -> None:
    temporary = path.parent / f".evalyn-intervention-{uuid.uuid4().hex}"
    try:
        RunStore.write_bytes(temporary, content, mode=mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _workspace_diff(worktrees: WorktreeSession, changed: tuple[str, ...]) -> str:
    result = _run_host_git(
        sanitized_git_command(
            worktrees.candidate_path,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            worktrees.baseline_sha,
            "--",
            *changed,
        ),
        max_stdout_bytes=_DIFF_CAPTURE_LIMIT,
        allow_stdout_truncation=True,
        description="candidate diff",
    )
    parts: list[str] = []
    length = 0
    truncated = result.stdout_truncated

    def append(piece: str) -> bool:
        nonlocal length, truncated
        remaining = _DIFF_CHAR_LIMIT - length
        if remaining <= 0:
            truncated = True
            return False
        parts.append(piece[:remaining])
        retained = min(len(piece), remaining)
        length += retained
        if len(piece) > remaining:
            truncated = True
            return False
        return True

    append(result.stdout.decode("utf-8", errors="replace"))
    for relative in worktrees.overlay.untracked_paths:
        if truncated:
            break
        if relative not in changed:
            continue
        path = worktrees.candidate_path / relative
        remaining = _DIFF_CHAR_LIMIT - length
        byte_limit = min(_DIFF_CAPTURE_LIMIT, (remaining * 4) + 1)
        try:
            data, file_truncated = _read_file_prefix(path, byte_limit)
        except OSError as exc:
            raise CheckError(f"Could not read untracked diff file {relative!r}: {exc}") from exc
        if b"\x00" in data:
            append(f"\nBinary untracked file: {relative}\n")
            truncated = truncated or file_truncated
            continue
        text = data.decode("utf-8", errors="replace")
        added = "\n".join(f"+{line}" for line in text.splitlines())
        append(
            f"\ndiff --git a/{relative} b/{relative}\n--- /dev/null\n+++ b/{relative}\n{added}\n"
        )
        truncated = truncated or file_truncated
    combined = "".join(parts)
    if truncated:
        return f"{combined}\n\n[truncated: diff exceeded safe input limit]"
    return combined


def _collect_context(
    worktrees: WorktreeSession, changed: tuple[str, ...]
) -> tuple[RepositoryContextSource, ...]:
    tracked = worktrees.tracked_files("candidate")
    preferred = list(changed)
    for name in (
        "README.md",
        "AGENTS.md",
        "CLAUDE.md",
        "pyproject.toml",
        "package.json",
        "docs/api.md",
    ):
        if name in tracked and name not in preferred:
            preferred.append(name)
    allowlist = tuple(path for path in preferred if path in tracked)
    policy = WorkspacePathPolicy(
        worktrees.candidate_path,
        tracked_paths=tracked,
        context_allowlist=allowlist,
    )
    sources: list[RepositoryContextSource] = []
    total = 0
    for relative in allowlist:
        try:
            path = policy.resolve_context(relative)
            data, _ = _read_file_prefix(path, _CONTEXT_FILE_BYTE_LIMIT)
        except (OSError, ValueError):
            continue
        if b"\x00" in data:
            continue
        text = data.decode("utf-8", errors="replace")[:_CONTEXT_FILE_CHAR_LIMIT]
        source_size = len(relative) + len(text)
        if total + source_size > 80_000:
            break
        sources.append(RepositoryContextSource(path=relative, content=text))
        total += source_size
    return tuple(sources)


def _changed_paths_from_diff(
    repo: Path, revisions: RevisionRange, *, include_dirty: bool
) -> tuple[str, ...]:
    args = sanitized_git_command(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-only",
        "--diff-filter=ACDMRT",
        revisions.base_sha,
        revisions.candidate_sha,
        "--",
    )
    result = _run_host_git(
        args,
        max_stdout_bytes=_GIT_METADATA_LIMIT,
        description="changed-file list",
    )
    paths = {
        line
        for line in result.stdout.decode("utf-8", errors="surrogateescape").splitlines()
        if line
    }
    if include_dirty:
        dirty = _run_host_git(
            sanitized_git_command(
                repo,
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ),
            max_stdout_bytes=_GIT_METADATA_LIMIT,
            description="dirty-file list",
        )
        for line in dirty.stdout.decode("utf-8", errors="surrogateescape").splitlines():
            if len(line) > 3:
                paths.add(line[3:].split(" -> ")[-1])
    return tuple(sorted(paths))


def _run_host_git(
    argv: list[str],
    *,
    max_stdout_bytes: int,
    description: str,
    allow_stdout_truncation: bool = False,
) -> BoundedProcessResult:
    try:
        result = run_bounded_process(
            argv,
            env=sanitized_git_environment(),
            timeout=60,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=_GIT_STDERR_LIMIT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckError(f"Could not render {description}: {exc}") from exc
    if result.stderr_truncated:
        raise CheckError(
            f"Could not render {description}: Git stderr exceeded {_GIT_STDERR_LIMIT} bytes"
        )
    if result.stdout_truncated and not allow_stdout_truncation:
        raise CheckError(
            f"Could not render {description}: Git output exceeded {max_stdout_bytes} bytes"
        )
    if result.returncode != 0 and not (allow_stdout_truncation and result.stdout_truncated):
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CheckError(f"Could not render {description}: {detail or 'git failed'}")
    return result


def _read_file_prefix(path: Path, max_bytes: int) -> tuple[bytes, bool]:
    """Read at most ``max_bytes`` from a file and report omitted content."""
    if max_bytes < 0:
        raise ValueError("File read limit cannot be negative")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    return data[:max_bytes], len(data) > max_bytes


def _require_trusted_configuration(
    repo: Path,
    revisions: RevisionRange,
    config_path: Path | None,
) -> Path | None:
    """Return a trusted resolved config path, rejecting byte-level drift."""
    resolved = resolve_settings_path(repo, config_path)
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError:
        # An explicit configuration outside the checkout is operator policy.
        return resolved
    if relative in {"", "."}:
        raise CheckError("Repository configuration path must name one regular file")

    base_payload = _configuration_blob_bytes(
        repo,
        revisions.base_sha,
        relative,
        label="selected base",
    )
    candidate_payload = _configuration_blob_bytes(
        repo,
        revisions.candidate_sha,
        relative,
        label="selected candidate",
    )
    working_payload = _working_configuration_bytes(resolved, relative)
    if base_payload != candidate_payload or base_payload != working_payload:
        raise CheckError(
            f"Repository configuration {relative!r} differs from the selected base; "
            "review it and pass an explicit --config outside the repository"
        )
    if working_payload is None and config_path is None:
        return None
    return resolved


def _configuration_blob_bytes(
    repo: Path,
    revision: str,
    relative: str,
    *,
    label: str,
) -> bytes | None:
    encoded_relative = os.fsencode(relative)
    pathspec = f":(top,literal){relative}"
    tree = _run_host_git(
        sanitized_git_command(
            repo,
            "ls-tree",
            "-z",
            "--full-tree",
            revision,
            "--",
            pathspec,
        ),
        max_stdout_bytes=min(_GIT_METADATA_LIMIT, len(encoded_relative) + 512),
        description=f"{label} configuration identity",
    )
    if not tree.stdout:
        return None
    if not tree.stdout.endswith(b"\0"):
        raise CheckError(f"Could not verify {label} configuration: malformed Git tree output")
    records = tree.stdout[:-1].split(b"\0")
    if len(records) != 1:
        raise CheckError(f"Could not verify {label} configuration: ambiguous Git tree output")
    metadata, separator, observed_path = records[0].partition(b"\t")
    fields = metadata.split()
    if separator != b"\t" or len(fields) != 3 or observed_path != encoded_relative:
        raise CheckError(f"Could not verify {label} configuration: malformed Git tree entry")
    mode, object_type, object_id = fields
    if object_type != b"blob" or mode not in {b"100644", b"100755"}:
        raise CheckError(f"Repository configuration {relative!r} is not a regular {label} blob")
    if len(object_id) not in {40, 64} or any(byte not in b"0123456789abcdef" for byte in object_id):
        raise CheckError(f"Could not verify {label} configuration: malformed blob identity")
    blob = _run_host_git(
        sanitized_git_command(repo, "cat-file", "blob", object_id.decode("ascii")),
        max_stdout_bytes=MAX_CONFIG_BYTES,
        description=f"{label} configuration blob",
    )
    return blob.stdout


def _working_configuration_bytes(path: Path, relative: str) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckError(f"Could not inspect repository configuration {relative!r}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CheckError(f"Repository configuration {relative!r} is not one regular file")
    try:
        return RunStore.read_bytes(path, max_bytes=MAX_CONFIG_BYTES)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise CheckError(f"Could not read repository configuration {relative!r}: {exc}") from exc


def _aggregate_result_verdict(results: list[DifferentialRunResult]) -> RunVerdict:
    verdicts = {result.verdict for result in results}
    if RunVerdict.INCOMPLETE in verdicts:
        return RunVerdict.INCOMPLETE
    if RunVerdict.REGRESSION_FOUND in verdicts:
        return RunVerdict.REGRESSION_FOUND
    if RunVerdict.CHANGE_OBSERVED in verdicts:
        return RunVerdict.CHANGE_OBSERVED
    if verdicts and verdicts == {RunVerdict.NO_DIFFERENTIAL_OBSERVED}:
        return RunVerdict.NO_DIFFERENTIAL_OBSERVED
    return RunVerdict.INSUFFICIENT_EVIDENCE


def _persist_dirty_overlay(run_path: Path, worktrees: WorktreeSession) -> dict[str, object]:
    patch = worktrees.candidate_diff()
    patch_path = run_path / "dirty.patch"
    if patch:
        RunStore.write_bytes(patch_path, patch)
    digest = hashlib.sha256()
    digest.update(patch)
    copied: list[str] = []
    root = run_path / "dirty-untracked"
    for relative in worktrees.overlay.untracked_paths:
        source = (worktrees.candidate_path / relative).resolve(strict=True)
        source.relative_to(worktrees.candidate_path)
        metadata = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise CheckError(f"Dirty overlay source is no longer regular: {relative!r}")
        try:
            content = RunStore.read_bytes(source, max_bytes=worktrees.max_overlay_bytes)
        except (OSError, ValueError) as exc:
            raise CheckError(f"Could not persist dirty overlay file {relative!r}: {exc}") from exc
        mode = stat.S_IMODE(metadata.st_mode)
        target = root.joinpath(*Path(relative).parts)
        RunStore.write_bytes(target, content, mode=mode)
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(mode.to_bytes(2, byteorder="big"))
        digest.update(content)
        copied.append(relative)
    if digest.hexdigest() != worktrees.overlay.content_sha256:
        raise CheckError("Dirty overlay changed while its artifacts were being persisted")
    return {
        "present": True,
        "content_sha256": worktrees.overlay.content_sha256,
        "patch": "dirty.patch" if patch else None,
        "untracked": copied,
    }
