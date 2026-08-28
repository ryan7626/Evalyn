"""Post-run finding operations: show, triage, and reproduction."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from evalyn.adapters.base import ProbeArtifact, TestAdapter
from evalyn.adapters.registry import default_registry
from evalyn.engine.models import ExecutionResult, Revision
from evalyn.execution import RepoRunLock, RunLockError, WorktreeSession
from evalyn.execution.bounded import run_bounded_process
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment
from evalyn.probe_executor import ContainerProbeExecutor
from evalyn.run_store import RunStore, StoredRun
from evalyn.settings import AdapterSettings


class OperationError(RuntimeError):
    pass


_FINDING_ID = re.compile(r"EVL-[A-F0-9]{8}\Z")
_GIT_OID = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z")
_OVERLAY_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_OVERLAY_BYTES = 50 * 1024 * 1024


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


@dataclass(frozen=True)
class _RecordedExecution:
    adapter: AdapterSettings
    runtime: str | None
    setup_env_names: tuple[str, ...]
    repeat: int
    budget_seconds: int


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
    updated: dict | None = None
    for finding in payload.get("findings", []):
        if finding.get("finding_id") != finding_id:
            continue
        finding["triage_status"] = status
        finding["triage_reason"] = reason.strip()
        finding["should_gate"] = status == "confirmed"
        history = finding.setdefault("triage_history", [])
        history.append(
            {
                "status": status,
                "reason": reason.strip(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        updated = finding
        break
    if updated is None:
        raise OperationError(f"Finding not found: {finding_id}")
    RunStore.write_json(run.path / "findings.json", payload)
    _refresh_saved_outputs(run.path, payload)
    return run, updated


def reproduce_saved_finding(
    store: RunStore,
    finding_id: str,
    *,
    side: str = "both",
    repeat: int | None = None,
    runtime: str | None = None,
    executor_factory: ReproduceFactory | None = None,
) -> ReproductionResult:
    finding_id = validate_finding_id(finding_id)
    if side not in {"base", "candidate", "both"}:
        raise OperationError("side must be base, candidate, or both")
    if repeat is not None and repeat < 1:
        raise OperationError("repeat must be positive")
    lock = RepoRunLock(store.repo_path, lock_path=store.repo_root / "run.lock")
    with _acquire_operation_lock(lock):
        return _reproduce_saved_finding_locked(
            store,
            finding_id,
            side=side,
            repeat=repeat,
            runtime=runtime,
            executor_factory=executor_factory,
        )


def _reproduce_saved_finding_locked(
    store: RunStore,
    finding_id: str,
    *,
    side: str,
    repeat: int | None,
    runtime: str | None,
    executor_factory: ReproduceFactory | None,
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
    repetitions = repeat or recorded.repeat
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
        payload = {
            "finding_id": finding_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base": [result.status.value for result in base_results],
            "candidate": [result.status.value for result in candidate_results],
        }
        output_root = _contained_reproduction_output(run, finding_id)
        RunStore.write_json(output_root / "latest.json", payload)
        return ReproductionResult(finding_id, base_results, candidate_results)


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
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise OperationError("Stored run has an invalid budget snapshot")
    return _RecordedExecution(
        adapter=AdapterSettings(
            name=adapter_name,
            image=image,
            setup_argv=setup_argv,
            suite_argv=suite_argv,
        ),
        runtime=effective_runtime,
        setup_env_names=setup_env,
        repeat=recorded_repeat,
        budget_seconds=budget,
    )


def _recorded_argv(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not all(isinstance(part, str) for part in value)
        or any(not part or "\x00" in part for part in value)
    ):
        raise OperationError(f"Stored run has an invalid {label} snapshot")
    return tuple(value)


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


def _refresh_saved_outputs(run_path: Path, payload: dict) -> None:
    lines = [f"# Evalyn diff report — {payload.get('run_id', 'unknown')}", ""]
    findings = payload.get("findings", [])
    gating = sum(bool(item.get("should_gate")) for item in findings)
    lines.extend((f"- Findings: {len(findings)}", f"- Gating findings: {gating}", ""))
    for finding in findings:
        status = (
            "CONFIRMED REGRESSION"
            if finding.get("triage_status") == "confirmed"
            else str(finding.get("semantic_state", "uncertain_change")).upper()
        )
        lines.extend(
            (
                f"## {status} — {finding['finding_id']}",
                "",
                str(finding.get("title", "")),
                "",
                f"- Triage: `{finding.get('triage_status', 'unreviewed')}`",
                f"- Reason: {finding.get('triage_reason') or finding.get('semantic_reason', '')}",
                f"- Reproduce: `evalyn reproduce {finding['finding_id']}`",
                "",
            )
        )
    RunStore.write_text(run_path / "report.md", "\n".join(lines))

    sarif_path = run_path / "evalyn.sarif"
    try:
        sarif = RunStore.read_json(sarif_path)
        for item in sarif["runs"][0]["results"]:
            finding = next(
                (
                    candidate
                    for candidate in findings
                    if candidate["finding_id"] == item.get("properties", {}).get("findingId")
                ),
                None,
            )
            if finding is not None:
                item["level"] = "error" if finding.get("should_gate") else "note"
        RunStore.write_json(sarif_path, sarif)
    except (OSError, ValueError, KeyError, TypeError):
        pass

    junit_path = run_path / "junit.xml"
    try:
        root = ElementTree.fromstring(RunStore.read_bytes(junit_path, max_bytes=32 * 1024 * 1024))
        by_probe = {item["artifact"]["probe_id"]: item for item in findings}
        failures = 0
        for case in root.findall("testcase"):
            finding = by_probe.get(case.attrib.get("name", ""))
            for child in list(case.findall("failure")):
                case.remove(child)
            if finding and finding.get("should_gate"):
                ElementTree.SubElement(
                    case, "failure", {"type": "confirmed_regression"}
                ).text = str(finding.get("title", ""))
                failures += 1
        root.set("failures", str(failures))
        RunStore.write_bytes(
            junit_path,
            ElementTree.tostring(root, encoding="utf-8", xml_declaration=True),
        )
    except (OSError, ElementTree.ParseError):
        pass
