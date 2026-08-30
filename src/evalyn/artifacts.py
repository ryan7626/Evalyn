"""Portable evidence bundles and explicit test promotion."""

from __future__ import annotations

import difflib
import io
import json
import os
import stat
import subprocess
import tempfile
import zipfile
from contextlib import suppress
from hashlib import sha256
from pathlib import Path

from evalyn.adapters.base import AdapterError
from evalyn.adapters.registry import default_registry
from evalyn.execution.git_safety import sanitized_git_command, sanitized_git_environment
from evalyn.operations import validate_finding_id
from evalyn.run_store import RunStore

_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class ArtifactExportError(RuntimeError):
    """Stored evidence could not be safely exported."""


def build_finding_bundle(store: RunStore, finding_id: str) -> bytes:
    """Return a deterministic ZIP containing evidence and reproduction inputs."""

    finding_id = validate_finding_id(finding_id)
    located = store.find_finding(finding_id)
    if located is None:
        raise ArtifactExportError(f"Finding not found: {finding_id}")
    run, finding = located
    source, adapter, relative_path = _validated_source(finding, run_path=run.path)
    suffix = ".py" if adapter == "pytest" else ".test.ts"
    patch = "".join(
        difflib.unified_diff(
            [],
            source.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile=f"b/{relative_path}",
        )
    )
    entries: dict[str, bytes] = {
        "README.md": _bundle_readme(finding_id, finding).encode("utf-8"),
        "finding.json": _json_bytes(finding),
        "run-manifest.json": _json_bytes(run.manifest),
        f"probe/test-source{suffix}": source.encode("utf-8"),
        "probe/test.patch": patch.encode("utf-8"),
        "probe/oracle.json": _json_bytes(
            {
                "oracle": finding.get("oracle", ""),
                "provenance": finding.get("oracle_provenance", ""),
                "admissible": finding.get("oracle_admissible", False),
                "semantic_state": finding.get("semantic_state", "uncertain_change"),
                "triage_status": finding.get("triage_status", "unreviewed"),
            }
        ),
        "probe/reproduce.json": _json_bytes(
            {
                "finding_id": finding_id,
                "base_sha": finding.get("base_sha"),
                "candidate_sha": finding.get("candidate_sha"),
                "adapter": adapter,
                "command": finding["artifact"].get("command"),
                "source_sha256": finding["artifact"].get("source_sha256"),
            }
        ),
    }
    bundle_manifest = {
        "schema_version": 1,
        "kind": "evalyn.finding-bundle",
        "finding_id": finding_id,
        "run_id": run.run_id,
        "files": {
            name: {"sha256": sha256(content).hexdigest(), "bytes": len(content)}
            for name, content in sorted(entries.items())
        },
    }
    entries["bundle-manifest.json"] = _json_bytes(bundle_manifest)

    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return destination.getvalue()


def promote_finding_source(
    store: RunStore,
    finding_id: str,
    destination: Path,
    *,
    require_confirmed: bool = True,
) -> Path:
    """Publish one reviewed probe as a normal repository test without overwriting."""

    finding_id = validate_finding_id(finding_id)
    located = store.find_finding(finding_id)
    if located is None:
        raise ArtifactExportError(f"Finding not found: {finding_id}")
    _run, finding = located
    if require_confirmed and finding.get("triage_status") != "confirmed":
        raise ArtifactExportError(
            "Only maintainer-confirmed findings can be promoted to permanent tests"
        )
    source, adapter, _relative_path = _validated_source(finding, run_path=_run.path)
    target = _validated_promotion_path(store.repo_path, destination, adapter)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _validated_promotion_path(store.repo_path, target, adapter)
    _write_exclusive(target, source.encode("utf-8"), mode=0o644)
    return target


def write_bundle_exclusive(destination: Path, payload: bytes) -> Path:
    if destination.suffix.casefold() != ".zip":
        raise ArtifactExportError("Evidence bundle output must use a .zip extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive(destination, payload, mode=0o600)
    return destination


def _validated_source(finding: dict, *, run_path: Path) -> tuple[str, str, str]:
    artifact = finding.get("artifact")
    source = finding.get("probe_source")
    if not isinstance(artifact, dict) or not isinstance(source, str) or not source.strip():
        raise ArtifactExportError("Stored finding has no valid probe artifact")
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ArtifactExportError("Stored probe source is not valid UTF-8 text") from exc
    if len(encoded) > _MAX_SOURCE_BYTES:
        raise ArtifactExportError("Stored probe source is too large to export")
    adapter_name = artifact.get("adapter")
    if not isinstance(adapter_name, str):
        raise ArtifactExportError("Stored finding uses an unsupported adapter")
    try:
        rebuilt = default_registry().get(adapter_name).build_artifact(source)
    except (AdapterError, UnicodeEncodeError) as exc:
        raise ArtifactExportError(f"Stored probe failed trusted adapter validation: {exc}") from exc

    # A source digest is not an integrity anchor when the stored source and digest
    # can be edited together. Re-derive every adapter-owned field and require the
    # remaining immutable metadata to agree before exporting executable source.
    expected = {
        "probe_id": rebuilt.probe_id,
        "adapter": rebuilt.adapter,
        "relative_path": rebuilt.relative_path.as_posix(),
        "command": list(rebuilt.command),
        "source_sha256": rebuilt.source_sha256,
        "byte_size": len(rebuilt.source.encode("utf-8")),
        "media_type": rebuilt.media_type,
    }
    metadata_changed = any(artifact.get(key) != value for key, value in expected.items())
    if source != rebuilt.source or metadata_changed:
        raise ArtifactExportError("Stored probe source or metadata failed its integrity check")
    _verify_run_owned_probe_copy(run_path, finding, rebuilt.source, expected)
    return rebuilt.source, rebuilt.adapter, rebuilt.relative_path.as_posix()


def _verify_run_owned_probe_copy(
    run_path: Path,
    finding: dict,
    source: str,
    expected_artifact: dict[str, object],
) -> None:
    finding_id = finding.get("finding_id")
    if not isinstance(finding_id, str):
        raise ArtifactExportError("Stored finding has no valid identity")
    suffix = ".py" if expected_artifact["adapter"] == "pytest" else ".test.ts"
    root = run_path.resolve(strict=True)
    source_path = run_path / "findings" / finding_id / f"test-source{suffix}"
    reproduce_path = run_path / "findings" / finding_id / "reproduce.json"
    for path in (source_path, reproduce_path):
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ArtifactExportError("Run-owned probe evidence is unavailable or unsafe") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or path.is_symlink():
            raise ArtifactExportError("Run-owned probe evidence must be one regular file")
    try:
        stored_source = RunStore.read_bytes(source_path, max_bytes=_MAX_SOURCE_BYTES).decode(
            "utf-8"
        )
        reproduction = RunStore.read_json(reproduce_path, max_bytes=256 * 1024)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ArtifactExportError("Run-owned probe evidence failed validation") from exc
    if stored_source != source or not isinstance(reproduction, dict):
        raise ArtifactExportError("Run-owned probe source failed its integrity check")
    trusted_reproduction = {
        "finding_id": finding_id,
        "base_sha": finding.get("base_sha"),
        "candidate_sha": finding.get("candidate_sha"),
        "adapter": expected_artifact["adapter"],
        "command": expected_artifact["command"],
        "source_sha256": expected_artifact["source_sha256"],
    }
    if any(reproduction.get(key) != value for key, value in trusted_reproduction.items()):
        raise ArtifactExportError("Run-owned reproduction metadata failed its integrity check")


def _validated_promotion_path(repo: Path, destination: Path, adapter: str) -> Path:
    root = repo.resolve(strict=True)
    candidate = Path(os.path.abspath(destination.expanduser()))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactExportError("Promoted test must stay inside the target repository") from exc
    if not relative.parts or relative.parts[0].casefold() in {".git", ".evalyn"}:
        raise ArtifactExportError("Promoted test cannot target metadata or the repository root")
    expected = ".py" if adapter == "pytest" else ".ts"
    if candidate.suffix.casefold() != expected:
        raise ArtifactExportError(f"{adapter} promoted tests must use a {expected} extension")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ArtifactExportError(f"Promoted test path cannot contain symlinks: {current}")
    if candidate.exists():
        raise ArtifactExportError("Promoted test destination already exists")
    if candidate.parent.exists() and not candidate.parent.is_dir():
        raise ArtifactExportError("Promoted test parent must be a directory")
    if _is_tracked_path(root, relative.as_posix()):
        raise ArtifactExportError("Promoted test cannot replace a tracked repository path")
    return candidate


def _is_tracked_path(repo: Path, relative: str) -> bool:
    try:
        completed = subprocess.run(
            sanitized_git_command(repo, "ls-files", "--", relative),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=sanitized_git_environment(),
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArtifactExportError(
            f"Could not verify promotion destination against Git: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ArtifactExportError(
            "Could not verify promotion destination against Git" + (f": {detail}" if detail else "")
        )
    return bool(completed.stdout.strip())


def _write_exclusive(destination: Path, payload: bytes, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as writer:
            writer.write(payload)
            os.fchmod(writer.fileno(), mode)
            writer.flush()
            os.fsync(writer.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _bundle_readme(finding_id: str, finding: dict) -> str:
    return "\n".join(
        (
            f"# Evalyn evidence bundle — {finding_id}",
            "",
            str(finding.get("title", "")),
            "",
            "This archive contains a content-verified generated probe, its oracle,",
            "the saved run manifest, and exact reproduction metadata. Verify every",
            "file against `bundle-manifest.json` before moving it between systems.",
            "The digest manifest detects corruption; it is not a signer-authenticity proof.",
            "",
            f"Triage status: `{finding.get('triage_status', 'unreviewed')}`",
            f"Semantic state: `{finding.get('semantic_state', 'uncertain_change')}`",
            "",
        )
    )


def _json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ArtifactExportError(f"Stored evidence is not serializable: {exc}") from exc
