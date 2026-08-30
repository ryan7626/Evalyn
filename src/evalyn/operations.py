"""Post-run finding operations: show, triage, and reproduction."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from evalyn.adapters.base import ProbeArtifact, TestAdapter
from evalyn.adapters.registry import default_registry
from evalyn.engine.classifier import failure_signature
from evalyn.engine.models import ExecutionResult, ExecutionStatus, Revision
from evalyn.execution import RepoRunLock, RunLockError, WorktreeSession
from evalyn.execution.bounded import run_bounded_process
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment
from evalyn.probe_executor import ContainerProbeExecutor
from evalyn.run_store import RunStore, StoredRun
from evalyn.settings import MAX_BUDGET_SECONDS, MAX_REPEAT, AdapterSettings


class OperationError(RuntimeError):
    pass


_FINDING_ID = re.compile(r"EVL-[A-F0-9]{8}\Z")
_GIT_OID = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z")
_OVERLAY_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_OVERLAY_BYTES = 50 * 1024 * 1024
_MAX_REPORT_BYTES = 32 * 1024 * 1024


def validate_finding_id(value: str) -> str:
    """Return a canonical finding ID or reject it before lookup/path use."""
    if not isinstance(value, str) or _FINDING_ID.fullmatch(value) is None:
        raise OperationError("finding_id must match EVL-XXXXXXXX")
    return value


@dataclass(frozen=True)
class ReproductionResult:
    finding_id: str
    base: tuple[ExecutionResult, ...]
    candidate: tuple[ExecutionResult, ...]
    expected_candidate_status: ExecutionStatus | None = None
    expected_failure_signature: str = ""

    @property
    def incomplete(self) -> bool:
        incomplete_statuses = {
            ExecutionStatus.INVALID_PROBE,
            ExecutionStatus.INFRA_ERROR,
        }
        return any(item.status in incomplete_statuses for item in (*self.base, *self.candidate))

    @property
    def reproduced(self) -> bool:
        """Whether every selected side repeated the saved differential."""
        if self.incomplete or not (self.base or self.candidate):
            return False
        base_matches = not self.base or all(
            item.status == ExecutionStatus.PASSED for item in self.base
        )
        candidate_matches = not self.candidate or (
            self.expected_candidate_status is not None
            and bool(self.expected_failure_signature)
            and all(
                item.status == self.expected_candidate_status
                and failure_signature(item) == self.expected_failure_signature
                for item in self.candidate
            )
        )
        return base_matches and candidate_matches

    @property
    def verdict(self) -> str:
        if self.incomplete:
            return "incomplete"
        if self.reproduced:
            return "reproduced"
        return "not_reproduced"


@dataclass(frozen=True)
class _RecordedExecution:
    adapter: AdapterSettings
    runtime: str | None
    setup_env_names: tuple[str, ...]
    repeat: int
    budget_seconds: int
    allow_remote_endpoint: bool
    expected_image_identity: str | None
    expected_endpoint_scope: str | None


ReproduceFactory = Callable[
    [TestAdapter, AdapterSettings, Path, str | None, int], ContainerProbeExecutor
]


def triage_saved_finding(
    store: RunStore,
    finding_id: str,
    *,
    status: str,
    reason: str,
) -> tuple[StoredRun, dict]:
    finding_id = validate_finding_id(finding_id)
    if status not in {"confirmed", "dismissed", "uncertain"}:
        raise OperationError("triage status must be confirmed, dismissed, or uncertain")
    if not reason.strip():
        raise OperationError("triage reason cannot be empty")
    lock = RepoRunLock(store.repo_path, lock_path=store.repo_root / "run.lock")
    with _acquire_operation_lock(lock):
        return _triage_saved_finding_locked(
            store,
            finding_id,
            status=status,
            reason=reason,
        )


def _triage_saved_finding_locked(
    store: RunStore,
    finding_id: str,
    *,
    status: str,
    reason: str,
) -> tuple[StoredRun, dict]:
    located = store.find_finding(finding_id)
    if located is None:
        raise OperationError(f"Finding not found: {finding_id}")
    run, _ = located
    try:
        payload = store.read_run_json(run, "findings.json")
    except (OSError, ValueError) as exc:
        raise OperationError(f"Could not read finding data: {exc}") from exc
    if not isinstance(payload, dict):
        raise OperationError("Stored finding data must be an object")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise OperationError("Stored finding data must contain a findings list")
    updated: dict | None = None
    timestamp = datetime.now(timezone.utc).isoformat()
    seen_ids: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("finding_id"), str):
            raise OperationError("Stored finding data contains a malformed finding")
        current_id = finding["finding_id"]
        if current_id in seen_ids:
            raise OperationError(f"Stored finding data repeats finding ID: {current_id}")
        seen_ids.add(current_id)
        if finding.get("finding_id") != finding_id:
            continue
        finding["triage_status"] = status
        finding["triage_reason"] = reason.strip()
        finding["should_gate"] = status == "confirmed"
        history = finding.setdefault("triage_history", [])
        if not isinstance(history, list):
            raise OperationError("Stored finding has malformed triage history")
        history.append(
            {
                "status": status,
                "reason": reason.strip(),
                "timestamp": timestamp,
            }
        )
        updated = finding
    if updated is None:
        raise OperationError(f"Finding not found: {finding_id}")
    _synchronize_nested_findings(payload, findings)
    verdict = _synchronize_saved_verdicts(payload)
    manifest = dict(run.manifest)
    manifest["finding_count"] = len(findings)
    manifest["gating_finding_count"] = sum(
        finding.get("should_gate") is True for finding in findings
    )
    manifest["verdict"] = verdict
    _persist_triage_transaction(run.path, payload, manifest)
    updated_run = StoredRun(run_id=run.run_id, path=run.path, manifest=manifest)
    return updated_run, copy.deepcopy(updated)


def reproduce_saved_finding(
    store: RunStore,
    finding_id: str,
    *,
    side: str = "both",
    repeat: int | None = None,
    runtime: str | None = None,
    executor_factory: ReproduceFactory | None = None,
    allow_remote_endpoint: bool = False,
) -> ReproductionResult:
    finding_id = validate_finding_id(finding_id)
    if side not in {"base", "candidate", "both"}:
        raise OperationError("side must be base, candidate, or both")
    if repeat is not None and (
        not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= MAX_REPEAT
    ):
        raise OperationError(f"repeat must be an integer between 1 and {MAX_REPEAT}")
    lock = RepoRunLock(store.repo_path, lock_path=store.repo_root / "run.lock")
    with _acquire_operation_lock(lock):
        return _reproduce_saved_finding_locked(
            store,
            finding_id,
            side=side,
            repeat=repeat,
            runtime=runtime,
            executor_factory=executor_factory,
            allow_remote_endpoint=allow_remote_endpoint,
        )


def _reproduce_saved_finding_locked(
    store: RunStore,
    finding_id: str,
    *,
    side: str,
    repeat: int | None,
    runtime: str | None,
    executor_factory: ReproduceFactory | None,
    allow_remote_endpoint: bool,
) -> ReproductionResult:
    located = store.find_finding(finding_id)
    if located is None:
        raise OperationError(f"Finding not found: {finding_id}")
    run, finding = located
    repo = Path(run.manifest.get("repository", "")).resolve()
    if repo != store.repo_path:
        raise OperationError("Stored run belongs to a different repository")
    base_sha, candidate_sha, candidate_id = _stored_revisions(run, finding)
    adapter_name = str(finding["artifact"]["adapter"])
    adapter = default_registry().get(adapter_name)
    artifact = adapter.build_artifact(str(finding["probe_source"]))
    _verify_artifact(artifact, finding)
    recorded = _recorded_execution(run, adapter_name, runtime_override=runtime)
    if allow_remote_endpoint != recorded.allow_remote_endpoint:
        raise OperationError(
            "Remote container endpoint opt-in must match the saved check; "
            "pass the explicit operator flag again only for a remotely-created run"
        )
    if executor_factory is None and (
        recorded.expected_image_identity is None or recorded.expected_endpoint_scope is None
    ):
        raise OperationError(
            "Stored run has no complete execution-environment identity for exact replay"
        )
    repetitions = repeat or recorded.repeat
    expected_status, expected_signature = _saved_candidate_expectation(
        finding,
        recorded_repeat=recorded.repeat,
    )
    output_root = _contained_reproduction_output(run, finding_id)

    with WorktreeSession(
        repo,
        baseline_revision=base_sha,
        candidate_revision=candidate_sha,
        temp_parent=store.repo_root / "worktrees",
    ) as worktrees:
        _restore_saved_overlay(run, worktrees)
        base_revision = Revision(base_sha, worktrees.baseline_path)
        candidate_revision = Revision(candidate_id, worktrees.candidate_path)
        if executor_factory is None:
            executor = _reproduce_factory(
                adapter,
                recorded.adapter,
                output_root,
                recorded.runtime,
                recorded.budget_seconds,
                setup_env_names=recorded.setup_env_names,
                allow_remote_endpoint=allow_remote_endpoint,
                expected_image_identity=recorded.expected_image_identity,
                expected_endpoint_scope=recorded.expected_endpoint_scope,
            )
        else:
            executor = executor_factory(
                adapter,
                recorded.adapter,
                output_root,
                recorded.runtime,
                recorded.budget_seconds,
            )
        executor.prepare((("base", base_revision), ("candidate", candidate_revision)))
        if executor.setup_errors:
            raise OperationError(
                "Dependency setup failed: " + "; ".join(executor.setup_errors.values())
            )
        base_results = (
            tuple(
                executor.execute(base_revision, artifact, repetition=index)
                for index in range(1, repetitions + 1)
            )
            if side in {"base", "both"}
            else ()
        )
        candidate_results = (
            tuple(
                executor.execute(candidate_revision, artifact, repetition=index)
                for index in range(1, repetitions + 1)
            )
            if side in {"candidate", "both"}
            else ()
        )
        result = ReproductionResult(
            finding_id,
            base_results,
            candidate_results,
            expected_candidate_status=expected_status,
            expected_failure_signature=expected_signature,
        )
        payload = {
            "finding_id": finding_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base": [result.status.value for result in base_results],
            "candidate": [result.status.value for result in candidate_results],
            "expected_candidate_status": expected_status.value,
            "expected_failure_signature": expected_signature,
            "verdict": result.verdict,
        }
        output_root = _contained_reproduction_output(run, finding_id)
        RunStore.write_json(output_root / "latest.json", payload)
        return result


def _acquire_operation_lock(lock: RepoRunLock) -> RepoRunLock:
    """Acquire a repository operation lock with a stable public error type."""
    try:
        return lock.acquire()
    except RunLockError as exc:
        raise OperationError(f"Could not acquire repository operation lock: {exc}") from exc


def _recorded_execution(
    run: StoredRun,
    adapter_name: str,
    *,
    runtime_override: str | None,
) -> _RecordedExecution:
    configuration = run.manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise OperationError("Stored run has no valid configuration snapshot")
    sandbox = configuration.get("sandbox")
    if not isinstance(sandbox, dict):
        raise OperationError("Stored run has no valid sandbox configuration snapshot")
    runtime = sandbox.get("runtime")
    if runtime not in {"auto", "docker", "podman"}:
        raise OperationError("Stored run has an invalid runtime snapshot")
    if runtime_override is not None and runtime_override != runtime:
        raise OperationError("Runtime override does not match the saved check")
    effective_runtime = None if runtime == "auto" else runtime
    setup_env = _recorded_argv(sandbox.get("setup_env_names"), "setup environment")
    if any(not _is_package_setup_environment(name) for name in setup_env):
        raise OperationError("Stored run contains an unsafe setup environment name")
    allow_remote_endpoint = sandbox.get("allow_remote_endpoint", False)
    if not isinstance(allow_remote_endpoint, bool):
        raise OperationError("Stored run has an invalid remote-endpoint snapshot")

    adapters = configuration.get("adapters")
    if not isinstance(adapters, list):
        raise OperationError("Stored run has no valid adapter configuration snapshot")
    matches = [
        item for item in adapters if isinstance(item, dict) and item.get("name") == adapter_name
    ]
    if len(matches) != 1:
        raise OperationError("Stored run must contain one matching adapter snapshot")
    snapshot = matches[0]
    image = snapshot.get("image")
    if (
        not isinstance(image, str)
        or not image
        or image.startswith("-")
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}", image) is None
    ):
        raise OperationError("Stored run has an invalid image snapshot")
    declared_image = snapshot.get("declared_image", image)
    if (
        not isinstance(declared_image, str)
        or not declared_image
        or declared_image.startswith("-")
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}", declared_image) is None
    ):
        raise OperationError("Stored run has an invalid declared-image snapshot")
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
    setup_argv = _recorded_argv(snapshot.get("setup_argv"), "setup command")
    suite_argv = _recorded_argv(snapshot.get("suite_argv"), "suite command")

    limits = configuration.get("limits")
    if not isinstance(limits, dict):
        raise OperationError("Stored run has no valid limits snapshot")
    recorded_repeat = limits.get("repeat")
    budget = limits.get("budget_seconds")
    if (
        not isinstance(recorded_repeat, int)
        or isinstance(recorded_repeat, bool)
        or not 2 <= recorded_repeat <= 10
    ):
        raise OperationError("Stored run has an invalid repeat-count snapshot")
    if (
        not isinstance(budget, int)
        or isinstance(budget, bool)
        or not 1 <= budget <= MAX_BUDGET_SECONDS
    ):
        raise OperationError("Stored run has an invalid budget snapshot")
    return _RecordedExecution(
        adapter=AdapterSettings(
            name=adapter_name,
            image=declared_image,
            setup_argv=setup_argv,
            suite_argv=suite_argv,
        ),
        runtime=effective_runtime,
        setup_env_names=setup_env,
        repeat=recorded_repeat,
        budget_seconds=budget,
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
        raise OperationError(f"Stored run has an invalid {label} snapshot")
    return tuple(value)


def _saved_candidate_expectation(
    finding: dict,
    *,
    recorded_repeat: int,
) -> tuple[ExecutionStatus, str]:
    runs = finding.get("candidate_runs")
    signature = finding.get("failure_signature")
    if (
        not isinstance(runs, list)
        or len(runs) != recorded_repeat
        or not isinstance(signature, str)
        or not signature
    ):
        raise OperationError("Stored candidate evidence is incomplete")
    statuses = {run.get("status") if isinstance(run, dict) else None for run in runs}
    if statuses == {ExecutionStatus.FAILED.value}:
        if any(run.get("failure_signature") != signature for run in runs if isinstance(run, dict)):
            raise OperationError("Stored candidate failure signatures are inconsistent")
        return ExecutionStatus.FAILED, signature
    if statuses == {ExecutionStatus.TIMEOUT.value} and signature == "timeout":
        return ExecutionStatus.TIMEOUT, signature
    raise OperationError("Stored candidate status does not match its stable finding")


def _is_package_setup_environment(name: str) -> bool:
    exact = {
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
    prefixes = ("BUNDLE_", "CARGO_REGISTRIES_", "NUGET_", "POETRY_HTTP_BASIC_")
    return name in exact or name.startswith(prefixes)


def _contained_reproduction_output(run: StoredRun, finding_id: str) -> Path:
    """Create and return an owned, nonsymlinked reproduction directory."""
    finding_id = validate_finding_id(finding_id)
    try:
        if run.path.is_symlink():
            raise OperationError("Stored run directory cannot be a symlink")
        run_root = run.path.resolve(strict=True)
        reproductions = run.path / "reproductions"
        if reproductions.is_symlink():
            raise OperationError("Reproduction output directory cannot be a symlink")
        reproductions.mkdir(parents=True, exist_ok=True)
        reproductions_root = reproductions.resolve(strict=True)
        if reproductions_root.parent != run_root:
            raise OperationError("Reproduction output must remain inside its stored run")

        output = reproductions / finding_id
        if output.is_symlink():
            raise OperationError("Finding reproduction output cannot be a symlink")
        output.mkdir(exist_ok=True)
        resolved = output.resolve(strict=True)
        if resolved.parent != reproductions_root:
            raise OperationError("Finding reproduction output escaped its stored run")
        return resolved
    except OperationError:
        raise
    except (OSError, ValueError) as exc:
        raise OperationError(f"Could not create contained reproduction output: {exc}") from exc


def _verify_artifact(artifact: ProbeArtifact, finding: dict) -> None:
    expected = finding["artifact"]
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
            raise OperationError(f"Stored probe {key} does not match the trusted adapter output")


def _stored_revisions(run: StoredRun, finding: dict) -> tuple[str, str, str]:
    base = run.manifest.get("base")
    candidate = run.manifest.get("candidate")
    candidate_id = run.manifest.get("candidate_id", candidate)
    if not all(isinstance(value, str) and value for value in (base, candidate, candidate_id)):
        raise OperationError("Stored run is missing pinned revision metadata")
    if _GIT_OID.fullmatch(base) is None or _GIT_OID.fullmatch(candidate) is None:
        raise OperationError("Stored run revisions must be full immutable Git object IDs")

    overlay = run.manifest.get("dirty_overlay", {"present": False})
    present = overlay.get("present", False) if isinstance(overlay, dict) else None
    if not isinstance(overlay, dict) or (present is not True and present is not False):
        raise OperationError("Stored run has invalid dirty-overlay metadata")
    valid_candidate_ids = {candidate}
    if overlay.get("present") is True:
        digest = overlay.get("content_sha256")
        if not isinstance(digest, str) or _OVERLAY_DIGEST.fullmatch(digest) is None:
            raise OperationError("Stored dirty overlay has invalid integrity metadata")
        valid_candidate_ids.add(f"{candidate}+dirty.{digest[:12]}")
    if candidate_id not in valid_candidate_ids:
        raise OperationError("Stored candidate identity does not match its pinned revision")
    if finding.get("run_id") != run.run_id:
        raise OperationError("Finding identity does not match its stored run")
    if finding.get("base_sha") != base or finding.get("candidate_sha") != candidate_id:
        raise OperationError("Finding revisions do not match their stored run")
    return base, candidate, candidate_id


def _restore_saved_overlay(run: StoredRun, worktrees: WorktreeSession) -> None:
    metadata = run.manifest.get("dirty_overlay", {"present": False})
    if not isinstance(metadata, dict):
        raise OperationError("Saved dirty overlay metadata must be an object")
    present = metadata.get("present", False)
    if present is False:
        return
    if present is not True:
        raise OperationError("Saved dirty overlay has invalid presence metadata")
    expected_digest = metadata.get("content_sha256")
    if not isinstance(expected_digest, str) or _OVERLAY_DIGEST.fullmatch(expected_digest) is None:
        raise OperationError("Saved dirty overlay has invalid integrity metadata")

    patch = b""
    patch_name = metadata.get("patch")
    if patch_name is not None and not isinstance(patch_name, str):
        raise OperationError("Saved dirty overlay has invalid patch metadata")
    if patch_name:
        patch = _read_stored_file(
            run.path,
            _safe_relative(patch_name),
            limit=_MAX_OVERLAY_BYTES,
        )

    untracked = metadata.get("untracked", [])
    if not isinstance(untracked, list) or not all(isinstance(item, str) for item in untracked):
        raise OperationError("Saved dirty overlay has invalid untracked metadata")
    if len(set(untracked)) != len(untracked):
        raise OperationError("Saved dirty overlay repeats an untracked path")

    total_bytes = len(patch)
    digest = hashlib.sha256()
    digest.update(patch)
    copied: list[tuple[str, bytes, int]] = []
    for raw_relative in untracked:
        relative = _safe_relative(raw_relative)
        content, mode = _read_saved_untracked(
            run.path,
            relative,
            limit=_MAX_OVERLAY_BYTES - total_bytes,
        )
        total_bytes += len(content)
        if total_bytes > _MAX_OVERLAY_BYTES:
            raise OperationError("Saved dirty overlay exceeds the reproduction size limit")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(mode.to_bytes(2, byteorder="big"))
        digest.update(content)
        copied.append((relative, content, mode))
    if digest.hexdigest() != expected_digest:
        raise OperationError("Saved dirty overlay failed its integrity check")

    if patch:
        try:
            result = run_bounded_process(
                sanitized_git_command(
                    worktrees.candidate_path,
                    "apply",
                    "--binary",
                    "--whitespace=nowarn",
                    "-",
                ),
                input_bytes=patch,
                timeout=60,
                max_stdout_bytes=1024 * 1024,
                max_stderr_bytes=1024 * 1024,
                env=sanitized_git_environment(),
            )
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise OperationError(f"Could not restore dirty patch safely: {exc}") from exc
        if result.stdout_truncated or result.stderr_truncated:
            raise OperationError("Could not restore dirty patch: Git output exceeded 1 MiB")
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise OperationError(f"Could not restore dirty patch: {detail}")
    for relative, content, mode in copied:
        _write_untracked_file(worktrees.candidate_path, relative, content, mode)


def _read_saved_untracked(run_path: Path, relative: str, *, limit: int) -> tuple[bytes, int]:
    stored_relative = PurePosixPath("dirty-untracked", *PurePosixPath(relative).parts).as_posix()
    path = _contained_stored_path(run_path, stored_relative)
    content, mode = _read_regular_file_with_mode(path, limit=limit)
    return content, mode


def _read_stored_file(run_path: Path, relative: str, *, limit: int) -> bytes:
    payload, _mode = _read_regular_file_with_mode(
        _contained_stored_path(run_path, relative), limit=limit
    )
    return payload


def _read_regular_file_with_mode(path: Path, *, limit: int) -> tuple[bytes, int]:
    if limit < 0:
        raise OperationError("Saved dirty overlay exceeds the reproduction size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OperationError(f"Could not open saved overlay file safely: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OperationError("Saved overlay entry must be one regular file")
        if metadata.st_size > limit:
            raise OperationError("Saved dirty overlay exceeds the reproduction size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(limit + 1)
        if len(payload) > limit:
            raise OperationError("Saved dirty overlay exceeds the reproduction size limit")
        return payload, metadata.st_mode & 0o777
    finally:
        os.close(descriptor)


def _safe_relative(raw: str) -> str:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or "\x00" in normalized
        or ".." in path.parts
        or ".git" in {part.casefold() for part in path.parts}
    ):
        raise OperationError(f"Unsafe stored relative path: {raw!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise OperationError(f"Unsafe stored relative path: {raw!r}")
    return PurePosixPath(*parts).as_posix()


def _contained_stored_path(run_path: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    try:
        if run_path.is_symlink():
            raise OperationError("Stored run directory cannot be a symlink")
        root = run_path.resolve(strict=True)
        cursor = root
        for component in PurePosixPath(safe).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise OperationError(f"Stored overlay path cannot use symlinks: {relative!r}")
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
        return resolved
    except OperationError:
        raise
    except (OSError, ValueError) as exc:
        raise OperationError(f"Stored overlay path escaped its run: {relative!r}") from exc


def _write_untracked_file(workspace: Path, relative: str, content: bytes, mode: int) -> None:
    safe = _safe_relative(relative)
    root = workspace.resolve(strict=True)
    parts = PurePosixPath(safe).parts
    parent = root
    try:
        for component in parts[:-1]:
            parent = parent / component
            if parent.is_symlink():
                raise OperationError(f"Saved untracked path crosses a symlink: {relative!r}")
            parent.mkdir(exist_ok=True)
            if not parent.is_dir() or not parent.resolve(strict=True).is_relative_to(root):
                raise OperationError(f"Saved untracked path escapes candidate: {relative!r}")
        target = parent / parts[-1]
        if target.exists() or target.is_symlink():
            raise OperationError(f"Saved untracked path collides with candidate: {relative!r}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.fchmod(descriptor, mode)
        finally:
            os.close(descriptor)
    except OperationError:
        raise
    except OSError as exc:
        raise OperationError(
            f"Could not restore saved untracked file: {relative!r}: {exc}"
        ) from exc


def _reproduce_factory(
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
        expected_image_identity=expected_image_identity,
        expected_endpoint_scope=expected_endpoint_scope,
    )


def _synchronize_nested_findings(payload: dict, findings: list[dict]) -> None:
    """Keep denormalized per-adapter findings consistent with the canonical list."""
    results = payload.get("results")
    if results is None:
        return
    if not isinstance(results, list):
        raise OperationError("Stored finding data contains malformed adapter results")
    canonical = {finding["finding_id"]: finding for finding in findings}
    all_nested_ids: set[str] = set()
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("findings"), list):
            raise OperationError("Stored finding data contains a malformed adapter result")
        synchronized: list[dict] = []
        seen: set[str] = set()
        for nested in result["findings"]:
            finding_id = nested.get("finding_id") if isinstance(nested, dict) else None
            if finding_id not in canonical or finding_id in seen or finding_id in all_nested_ids:
                raise OperationError("Stored adapter findings diverge from canonical findings")
            seen.add(finding_id)
            all_nested_ids.add(finding_id)
            synchronized.append(copy.deepcopy(canonical[finding_id]))
        result["findings"] = synchronized
        result["gating_finding_count"] = sum(
            finding.get("should_gate") is True for finding in synchronized
        )
    if all_nested_ids != set(canonical):
        raise OperationError("Stored adapter findings diverge from canonical findings")


def _synchronize_saved_verdicts(payload: dict) -> str:
    """Recompute adapter and run verdicts after human triage changes a gate."""
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise OperationError("Stored finding data contains no adapter results")
    verdicts: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            raise OperationError("Stored finding data contains a malformed adapter result")
        assessments = result.get("assessments")
        findings = result.get("findings")
        if not isinstance(assessments, list) or not isinstance(findings, list):
            raise OperationError("Stored adapter result is missing assessments or findings")
        kinds: set[str] = set()
        for assessment in assessments:
            classification = (
                assessment.get("classification") if isinstance(assessment, dict) else None
            )
            kind = classification.get("kind") if isinstance(classification, dict) else None
            if not isinstance(kind, str):
                raise OperationError("Stored adapter assessment has an invalid classification")
            kinds.add(kind)
        if "invalid" in kinds:
            verdict = "incomplete"
        elif any(
            isinstance(finding, dict) and finding.get("should_gate") is True for finding in findings
        ):
            verdict = "regression_found"
        elif "stable_change" in kinds:
            verdict = "change_observed"
        elif assessments and kinds == {"unchanged"}:
            verdict = "no_differential_observed"
        else:
            verdict = "insufficient_evidence"
        result["verdict"] = verdict
        verdicts.add(verdict)

    if "incomplete" in verdicts:
        aggregate = "incomplete"
    elif "regression_found" in verdicts:
        aggregate = "regression_found"
    elif "change_observed" in verdicts:
        aggregate = "change_observed"
    elif verdicts == {"no_differential_observed"}:
        aggregate = "no_differential_observed"
    else:
        aggregate = "insufficient_evidence"
    payload["verdict"] = aggregate
    return aggregate


def _persist_triage_transaction(run_path: Path, payload: dict, manifest: dict) -> None:
    """Regenerate and commit every primary output as one rollback-safe update."""
    outputs = {
        run_path / "report.md": _render_saved_markdown(payload).encode("utf-8"),
        run_path / "junit.xml": _render_saved_junit(payload),
        run_path / "evalyn.sarif": _json_bytes(_render_saved_sarif(payload)),
        run_path / "findings.json": _json_bytes(payload),
        run_path / "manifest.json": _json_bytes(manifest),
    }
    if any(len(content) > _MAX_REPORT_BYTES for content in outputs.values()):
        raise OperationError("Regenerated triage outputs exceed the persistence size limit")
    _replace_outputs_transactionally(run_path, outputs)


def _render_saved_markdown(payload: dict) -> str:
    findings = payload["findings"]
    results = payload.get("results", [])
    assessments = [
        assessment
        for result in results
        if isinstance(result, dict)
        for assessment in result.get("assessments", [])
        if isinstance(assessment, dict)
    ]
    stable = sum(
        assessment.get("classification", {}).get("kind") == "stable_change"
        for assessment in assessments
        if isinstance(assessment.get("classification"), dict)
    )
    adapters = payload.get("adapters", [])
    adapter_text = ", ".join(item for item in adapters if isinstance(item, str)) or "none"
    gating = sum(finding.get("should_gate") is True for finding in findings)
    lines = [
        f"# Evalyn diff report — {payload.get('run_id', 'unknown')}",
        "",
        f"- Base: `{payload.get('base_sha', 'unknown')}`",
        f"- Candidate: `{payload.get('candidate_sha', 'unknown')}`",
        f"- Adapters: {adapter_text}",
        f"- Probes: {len(assessments)}",
        f"- Verdict: `{payload.get('verdict', 'insufficient_evidence')}`",
        f"- Stable changes: {stable}",
        f"- Findings: {len(findings)}",
        f"- Gating findings: {gating}",
        f"- No differential observed: {_classification_count(assessments, 'unchanged')}",
        f"- Baseline failures: {_classification_count(assessments, 'baseline_failure')}",
        f"- Unstable: {_classification_count(assessments, 'unstable')}",
        f"- Invalid/infrastructure: {_classification_count(assessments, 'invalid')}",
        "",
    ]
    environments = payload.get("environments", {})
    if isinstance(environments, dict):
        environment_lines: list[str] = []
        for adapter, fingerprint in sorted(environments.items()):
            if not isinstance(adapter, str) or not isinstance(fingerprint, dict):
                continue
            digest = fingerprint.get("fingerprint_sha256") or "incomplete"
            environment_lines.append(f"- {adapter} environment: `{digest}`")
        lines[-1:-1] = environment_lines
    if not findings:
        if payload.get("verdict") == "no_differential_observed":
            lines.extend(
                (
                    "No differential was observed by the generated probes. This is not "
                    "proof that the change is regression-free.",
                    "",
                )
            )
        else:
            lines.extend(("No stable behavior changes were found.", ""))
    for finding in findings:
        triage = finding.get("triage_status", "unreviewed")
        if triage == "confirmed":
            label = "CONFIRMED REGRESSION"
        elif finding.get("should_gate") is True:
            label = "LIKELY REGRESSION"
        else:
            label = str(finding.get("semantic_state", "uncertain_change")).upper()
        base_runs = finding.get("base_runs", [])
        candidate_runs = finding.get("candidate_runs", [])
        base_passes = sum(item.get("status") == "passed" for item in base_runs)
        candidate_failures = sum(
            item.get("status") in {"failed", "timeout"} for item in candidate_runs
        )
        metadata = finding.get("probe_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        evidence = finding.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        environment_integrity = (
            "verified" if evidence.get("environment_integrity") is True else "incomplete"
        )
        target_binding = (
            "verified" if evidence.get("target_binding_verified") is True else "unverified"
        )
        oracle_binding = (
            "verified" if evidence.get("oracle_binding_verified") is True else "unverified"
        )
        target_intervention = (
            "verified" if evidence.get("target_intervention_verified") is True else "unverified"
        )
        lines.extend(
            (
                f"## {label} — {finding['finding_id']}",
                "",
                str(finding.get("title", "")),
                "",
                f"- Oracle: {finding.get('oracle') or 'Not grounded'}",
                f"- Provenance: `{finding.get('oracle_provenance') or 'none'}`",
                f"- Oracle source: "
                f"`{metadata.get('oracle_grounding_source', 'unverified')}` "
                f"({metadata.get('oracle_source_side', 'unknown')})",
                f"- Assessment: `{finding.get('semantic_state', 'uncertain_change')}` — "
                f"{finding.get('semantic_reason', '')}",
                f"- Base: PASS {base_passes}/{len(base_runs)}",
                f"- Candidate: FAIL {candidate_failures}/{len(candidate_runs)}",
                f"- Failure signature: `{finding.get('failure_signature', '')}`",
                f"- Change target: `{metadata.get('change_target_id', 'unverified')}`",
                f"- Changed symbol: `{metadata.get('changed_symbol', 'unknown')}`",
                f"- Static target binding: {target_binding}",
                f"- Static oracle binding: {oracle_binding}",
                f"- Target-revert intervention: {target_intervention}",
                f"- Confirmed executions: base {len(base_runs)}, candidate {len(candidate_runs)}",
                f"- Corroborating probes: {evidence.get('corroboration_count', 0)}",
                f"- Environment integrity: {environment_integrity}",
                f"- Triage: `{triage}`",
                f"- Reason: {finding.get('triage_reason') or finding.get('semantic_reason', '')}",
                f"- Reproduce: `evalyn reproduce {finding['finding_id']}`",
                "",
            )
        )
    lines.extend(
        (
            "## Interpretation",
            "",
            "Findings are repeatable behavior changes. Automatic gates require trusted, "
            "machine-verified evidence and still require maintainer review.",
            "",
        )
    )
    return "\n".join(lines)


def _render_saved_junit(payload: dict) -> bytes:
    findings = payload["findings"]
    by_probe = {
        finding.get("artifact", {}).get("probe_id"): finding
        for finding in findings
        if isinstance(finding.get("artifact"), dict)
    }
    assessments = [
        assessment
        for result in payload.get("results", [])
        if isinstance(result, dict)
        for assessment in result.get("assessments", [])
        if isinstance(assessment, dict)
    ]
    inconclusive = {"baseline_failure", "unstable", "improvement"}
    skipped_count = sum(
        assessment.get("classification", {}).get("kind") == "stable_change"
        and not (
            (finding := by_probe.get(assessment.get("artifact", {}).get("probe_id")))
            and finding.get("should_gate") is True
        )
        for assessment in assessments
        if isinstance(assessment.get("classification"), dict)
        and isinstance(assessment.get("artifact"), dict)
    )
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": "evalyn",
            "tests": str(len(assessments)),
            "failures": str(sum(finding.get("should_gate") is True for finding in findings)),
            "errors": str(
                sum(
                    assessment.get("classification", {}).get("kind") in ({"invalid"} | inconclusive)
                    for assessment in assessments
                    if isinstance(assessment.get("classification"), dict)
                )
            ),
            "skipped": str(skipped_count),
        },
    )
    properties = ElementTree.SubElement(suite, "properties")
    ElementTree.SubElement(
        properties,
        "property",
        {"name": "evalyn.verdict", "value": str(payload.get("verdict", "unknown"))},
    )
    environments = payload.get("environments", {})
    if isinstance(environments, dict):
        for adapter, fingerprint in sorted(environments.items()):
            if not isinstance(adapter, str) or not isinstance(fingerprint, dict):
                continue
            digest = fingerprint.get("fingerprint_sha256")
            if isinstance(digest, str) and digest:
                ElementTree.SubElement(
                    properties,
                    "property",
                    {"name": f"evalyn.environment.{adapter}", "value": digest},
                )
    for assessment in assessments:
        artifact = assessment.get("artifact", {})
        classification = assessment.get("classification", {})
        if not isinstance(artifact, dict) or not isinstance(classification, dict):
            raise OperationError("Stored assessment is malformed")
        probe_id = str(artifact.get("probe_id", ""))
        case = ElementTree.SubElement(
            suite,
            "testcase",
            {"classname": str(artifact.get("adapter", "unknown")), "name": probe_id},
        )
        finding = by_probe.get(probe_id)
        if finding and finding.get("should_gate") is True:
            failure_type = (
                "confirmed_regression"
                if finding.get("triage_status") == "confirmed"
                else "likely_regression"
            )
            failure = ElementTree.SubElement(case, "failure", {"type": failure_type})
            failure.text = str(finding.get("title", ""))
        elif classification.get("kind") == "invalid":
            error = ElementTree.SubElement(case, "error", {"type": "invalid_probe"})
            error.text = str(classification.get("reason", ""))
        elif classification.get("kind") in inconclusive:
            error = ElementTree.SubElement(case, "error", {"type": "insufficient_evidence"})
            error.text = str(classification.get("reason", ""))
        elif classification.get("kind") == "stable_change":
            skipped = ElementTree.SubElement(
                case,
                "skipped",
                {"type": str(classification.get("kind", "inconclusive"))},
            )
            skipped.text = str(classification.get("reason", ""))
    return ElementTree.tostring(suite, encoding="utf-8", xml_declaration=True)


def _render_saved_sarif(payload: dict) -> dict:
    results: list[dict] = []
    for finding in payload["findings"]:
        triage = finding.get("triage_status", "unreviewed")
        if finding.get("should_gate") is True:
            level = "error"
        elif triage == "dismissed" or finding.get("semantic_state") == "intended_change":
            level = "note"
        else:
            level = "warning"
        item: dict = {
            "ruleId": (
                "evalyn/likely-regression"
                if finding.get("should_gate") is True
                else "evalyn/behavior-change"
            ),
            "level": level,
            "message": {"text": f"{finding.get('title', '')} ({finding['finding_id']})"},
            "properties": {
                "findingId": finding["finding_id"],
                "semanticState": finding.get("semantic_state", "uncertain_change"),
                "oracleProvenance": finding.get("oracle_provenance", ""),
                "triageStatus": triage,
                "triageReason": finding.get("triage_reason", ""),
                "evidence": finding.get("evidence"),
            },
        }
        metadata = finding.get("probe_metadata", {})
        changed_file = metadata.get("changed_file", "") if isinstance(metadata, dict) else ""
        if changed_file:
            item["locations"] = [{"physicalLocation": {"artifactLocation": {"uri": changed_file}}}]
        results.append(item)
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Evalyn",
                        "informationUri": "https://github.com/ryan7626/Evalyn",
                        "rules": [
                            {
                                "id": "evalyn/likely-regression",
                                "shortDescription": {
                                    "text": "Automatic or maintainer-confirmed regression gate"
                                },
                            },
                            {
                                "id": "evalyn/behavior-change",
                                "shortDescription": {
                                    "text": "Repeatable behavior change requiring review"
                                },
                            },
                        ],
                    }
                },
                "invocations": [
                    {
                        "executionSuccessful": payload.get("verdict")
                        not in {"incomplete", "insufficient_evidence"},
                        "properties": {"verdict": payload.get("verdict", "unknown")},
                    }
                ],
                "properties": {
                    "verdict": payload.get("verdict", "unknown"),
                    "environments": payload.get("environments", {}),
                },
                "results": results,
            }
        ],
    }


def _classification_count(assessments: list[dict], kind: str) -> int:
    return sum(
        assessment.get("classification", {}).get("kind") == kind
        for assessment in assessments
        if isinstance(assessment.get("classification"), dict)
    )


def _json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise OperationError(f"Could not serialize triage outputs: {exc}") from exc


def _replace_outputs_transactionally(run_path: Path, outputs: dict[Path, bytes]) -> None:
    """Stage all bytes first and roll every replacement back on commit failure."""
    originals: dict[Path, bytes | None] = {}
    modes: dict[Path, int] = {}
    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        root = run_path.resolve(strict=True)
        if run_path.is_symlink():
            raise OperationError("Stored run directory cannot be a symlink")
        for target, content in outputs.items():
            if target.parent.resolve(strict=True) != root or target.is_symlink():
                raise OperationError("Triage output escaped its stored run")
            if target.exists():
                originals[target] = RunStore.read_bytes(target, max_bytes=_MAX_REPORT_BYTES)
                modes[target] = target.stat().st_mode & 0o777
            else:
                originals[target] = None
                modes[target] = 0o600
            staged[target] = _stage_output(target, content, modes[target])

        for target in outputs:
            os.replace(staged[target], target)
            committed.append(target)
        _fsync_directory(root)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for target in reversed(committed):
            original = originals[target]
            try:
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    rollback = _stage_output(target, original, modes[target])
                    try:
                        os.replace(rollback, target)
                    finally:
                        rollback.unlink(missing_ok=True)
            except BaseException as rollback_exc:  # pragma: no cover - catastrophic filesystem
                rollback_errors.append(f"{target.name}: {rollback_exc}")
        detail = "; rollback failed: " + "; ".join(rollback_errors) if rollback_errors else ""
        if isinstance(exc, OperationError):
            raise OperationError(f"{exc}{detail}") from exc
        raise OperationError(
            f"Could not commit triage outputs transactionally: {exc}{detail}"
        ) from exc
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _stage_output(target: Path, content: bytes, mode: int) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.triage-",
        suffix=".tmp",
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
