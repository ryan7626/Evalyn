"""End-to-end v0.2 diff-regression workflow."""

from __future__ import annotations

import hashlib
import stat
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from typing import Protocol

from evalyn.adapters.base import TestAdapter
from evalyn.adapters.registry import default_registry
from evalyn.engine.classifier import DifferentialClassifier
from evalyn.engine.models import (
    DifferentialRunResult,
    ExecutionStatus,
    Revision,
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
from evalyn.generation import GenerationRequest, RepositoryContextSource, generate_probes
from evalyn.llm import create_provider
from evalyn.llm.base import BaseLLM
from evalyn.probe_executor import DEFAULT_IMAGES, ContainerProbeExecutor
from evalyn.reporting_v2 import write_reports
from evalyn.revisions import (
    RevisionRange,
    get_diff,
    is_dirty,
    resolve_range,
    validate_git_repository,
)
from evalyn.run_store import RunStore, StoredRun
from evalyn.settings import AdapterSettings, EvalynSettings, ProviderSettings, load_settings

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
        return any(
            record.result.status in {ExecutionStatus.INFRA_ERROR, ExecutionStatus.INVALID_PROBE}
            for result in self.results
            for assessment in result.assessments
            for record in (*assessment.base_runs, *assessment.candidate_runs)
        )


class PreparedProbeExecutor(ProbeExecutor, Protocol):
    def prepare(self, revisions: tuple[tuple[str, Revision], ...]) -> None: ...

    @property
    def setup_errors(self) -> dict[Path, str]: ...


ExecutorFactory = Callable[
    [TestAdapter, AdapterSettings, Path, str | None, int], PreparedProbeExecutor
]


def inspect_check(options: CheckOptions) -> CheckInspection:
    repo = validate_git_repository(options.repo_path)
    settings = load_settings(repo, options.config_path)
    dirty = is_dirty(repo)
    if dirty and not options.include_dirty:
        raise CheckError(
            "Repository has staged, unstaged, or untracked changes; rerun with "
            "--include-dirty to snapshot them explicitly"
        )
    revisions = resolve_range(repo, options.range_spec, allow_same=options.include_dirty)
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
                generation_budget = settings.budget_seconds - (time.monotonic() - started)
                if generation_budget <= 0:
                    raise CheckError("Check execution budget was exhausted before generation")
                provider = llm or create_provider(
                    settings.provider,
                    timeout_seconds=min(120, generation_budget),
                )
                probes = generate_probes(
                    provider,
                    GenerationRequest(
                        adapter=name,
                        diff=diff,
                        context_sources=context_sources,
                        intent=options.intent,
                        max_tests=settings.max_tests,
                    ),
                )
                engine = DifferentialEngine(
                    adapter,
                    executor,
                    classifier=DifferentialClassifier(settings.repeat),
                    assessor=IntentOracleAssessor(options.intent),
                )
                results.append(
                    engine.evaluate(
                        base_revision,
                        candidate_revision,
                        probes,
                        run_id=run.run_id,
                    )
                )

            metadata = {
                "run_id": run.run_id,
                "base_sha": worktrees.baseline_sha,
                "candidate_sha": candidate_id,
                "intent": options.intent,
            }
            report_paths = write_reports(run.path, results, metadata=metadata)

        duration = time.monotonic() - started
        incomplete = _has_incomplete_evidence(results)
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
                "dirty_overlay": overlay_metadata,
                "configuration": _settings_snapshot(
                    settings,
                    inspection.adapters,
                    runtime=recorded_runtime,
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
) -> dict[str, object]:
    return {
        "provider": {
            "kind": settings.provider.kind,
            "model": settings.provider.model,
        },
        "sandbox": {
            "runtime": runtime or settings.sandbox.runtime,
            "setup_env_names": list(settings.sandbox.setup_env),
        },
        "adapters": [
            {
                "name": item.name,
                "setup_argv": list(item.setup_argv),
                "suite_argv": list(item.suite_argv),
                "image": item.image or DEFAULT_IMAGES[item.name],
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
) -> ContainerProbeExecutor:
    return ContainerProbeExecutor(
        adapter,
        engine=runtime,
        run_dir=run_path,
        image=settings.image,
        setup_argv=settings.setup_argv,
        setup_env_names=setup_env_names,
        timeout_seconds=budget_seconds,
    )


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
        "--diff-filter=ACMRT",
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


def _has_incomplete_evidence(results: list[DifferentialRunResult]) -> bool:
    return any(
        record.result.status in {ExecutionStatus.INFRA_ERROR, ExecutionStatus.INVALID_PROBE}
        for result in results
        for assessment in result.assessments
        for record in (*assessment.base_runs, *assessment.candidate_runs)
    )


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
