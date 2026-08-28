"""External, versioned run storage.

Evalyn deliberately keeps runtime state out of the repository being checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from evalyn.execution.errors import RunLockError
from evalyn.execution.lock import RepoRunLock

SCHEMA_VERSION = 1
_MAX_STORED_JSON_BYTES = 32 * 1024 * 1024


class _UnsafeStatePath(ValueError):
    """A managed state path could not be accessed without following links."""


def default_state_root() -> Path:
    override = os.getenv("EVALYN_STATE_HOME")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.getenv("XDG_STATE_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "evalyn"
    return Path.home() / ".local" / "state" / "evalyn"


def repository_id(repo_path: Path) -> str:
    normalized = os.path.normcase(str(repo_path.resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    path: Path
    manifest: dict[str, Any]


class RunStore:
    def __init__(self, repo_path: Path, root: Path | None = None) -> None:
        self.repo_path = repo_path.resolve()
        self.root = (root or default_state_root()).resolve()
        try:
            self.root.relative_to(self.repo_path)
        except ValueError:
            pass
        else:
            raise ValueError("Evalyn state root must be outside the target repository")
        self.repo_root = self.root / "repositories" / repository_id(self.repo_path)
        self.runs_root = self.repo_root / "runs"

    def create_run(self, *, base: str, candidate: str) -> StoredRun:
        now = datetime.now(timezone.utc)
        prefix = now.strftime("%Y%m%dT%H%M%S.%fZ")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "repository": str(self.repo_path),
            "base": base,
            "candidate": candidate,
            "status": "in_progress",
            "started_at": now.isoformat(),
            "finished_at": None,
        }
        runs_fd = _open_directory_tree(self.runs_root, create=True)
        if runs_fd is None:  # pragma: no cover - Windows fallback
            path = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=self.runs_root))
            run_id = path.name
            manifest["run_id"] = run_id
            self.write_json(path / "manifest.json", manifest)
            return StoredRun(run_id=run_id, path=path, manifest=manifest)

        try:
            for _ in range(100):
                run_id = f"{prefix}-{uuid.uuid4().hex[:12]}"
                try:
                    os.mkdir(run_id, 0o700, dir_fd=runs_fd)
                except FileExistsError:
                    continue
                break
            else:  # pragma: no cover - UUID collision defense
                raise RuntimeError("Could not allocate a unique Evalyn run directory")
            manifest["run_id"] = run_id
            run_fd = _open_child_directory(runs_fd, run_id)
            try:
                _write_json_at(run_fd, "manifest.json", manifest)
            finally:
                os.close(run_fd)
        finally:
            os.close(runs_fd)
        path = self.runs_root / run_id
        return StoredRun(run_id=run_id, path=path, manifest=manifest)

    def finish_run(
        self,
        run: StoredRun,
        *,
        status: str,
        extra: dict[str, Any] | None = None,
    ) -> StoredRun:
        if run.path != self.runs_root / run.run_id:
            raise ValueError("Stored run path is not owned by this run store")
        manifest = dict(run.manifest)
        manifest.update(extra or {})
        manifest["status"] = status
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        runs_fd = _open_directory_tree(self.runs_root, create=False)
        if runs_fd is None:  # pragma: no cover - Windows fallback
            if not _is_owned_run(run.path, self.runs_root):
                raise ValueError("Stored run is no longer owned by this run store")
            self.write_json(run.path / "manifest.json", manifest)
        else:
            try:
                run_fd = _open_child_directory(runs_fd, run.run_id)
                try:
                    existing = _read_json_at(run_fd, "manifest.json")
                    if existing != run.manifest:
                        raise ValueError("Stored run manifest changed before it was finished")
                    _write_json_at(run_fd, "manifest.json", manifest)
                finally:
                    os.close(run_fd)
            finally:
                os.close(runs_fd)
        return StoredRun(run_id=run.run_id, path=run.path, manifest=manifest)

    def get(self, run_id: str | None = None) -> StoredRun | None:
        try:
            runs_fd = _open_directory_tree(self.runs_root, create=False)
        except (OSError, _UnsafeStatePath):
            return None
        if runs_fd is None:  # pragma: no cover - Windows fallback
            return self._get_fallback(run_id)
        try:
            candidates = sorted(os.listdir(runs_fd), reverse=True)
            if run_id:
                candidates = [name for name in candidates if name == run_id]
            for name in candidates:
                stored = self._load_run_at(runs_fd, name)
                if stored is not None:
                    return stored
            return None
        finally:
            os.close(runs_fd)

    def list_runs(self, limit: int = 20) -> list[StoredRun]:
        try:
            runs_fd = _open_directory_tree(self.runs_root, create=False)
        except (OSError, _UnsafeStatePath):
            return []
        if runs_fd is None:  # pragma: no cover - Windows fallback
            return self._list_runs_fallback(limit)
        try:
            return self._list_runs_at(runs_fd, limit)
        finally:
            os.close(runs_fd)

    def find_finding(self, finding_id: str) -> tuple[StoredRun, dict[str, Any]] | None:
        for run in self.list_runs(limit=100_000):
            try:
                data = self._read_run_json(run, "findings.json")
            except (OSError, ValueError):
                continue
            if (
                not isinstance(data, dict)
                or data.get("schema_version") != SCHEMA_VERSION
                or data.get("run_id") != run.run_id
            ):
                continue
            findings = data.get("findings", [])
            if not isinstance(findings, list):
                continue
            for finding in findings:
                if (
                    isinstance(finding, dict)
                    and finding.get("finding_id", finding.get("id")) == finding_id
                ):
                    return run, finding
        return None

    def read_run_json(self, run: StoredRun, name: str) -> Any:
        """Read a run-owned JSON artifact while rechecking manifest identity."""
        return self._read_run_json(run, name)

    def clean_before(self, days: int) -> int:
        if days < 0:
            raise ValueError("days must be non-negative")
        lock = RepoRunLock(self.repo_path, lock_path=self.repo_root / "run.lock")
        try:
            lock.acquire()
        except RunLockError:
            # Cleanup is opportunistic. An active check/reproduction/repair owns
            # the same lock and must never lose its artifact directory.
            return 0
        try:
            return self._clean_before_locked(days)
        finally:
            lock.release()

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        encoded = payload.encode("utf-8")
        if len(encoded) > _MAX_STORED_JSON_BYTES:
            raise ValueError(f"Refusing to persist JSON larger than {_MAX_STORED_JSON_BYTES} bytes")
        RunStore.write_bytes(path, encoded)

    @staticmethod
    def read_json(path: Path, *, max_bytes: int = _MAX_STORED_JSON_BYTES) -> Any:
        """Read bounded JSON without following a linked path component."""
        try:
            return json.loads(RunStore.read_bytes(path, max_bytes=max_bytes).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Managed state file is invalid JSON: {path}") from exc

    @staticmethod
    def read_bytes(path: Path, *, max_bytes: int) -> bytes:
        """Read one bounded regular artifact without following path links."""
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        target = Path(os.path.abspath(os.fspath(path.expanduser())))
        parent_fd = _open_directory_tree(target.parent, create=False)
        if parent_fd is not None:
            try:
                return _read_bytes_at(parent_fd, target.name, max_bytes=max_bytes)
            finally:
                os.close(parent_fd)

        if target.is_symlink():  # pragma: no cover - Windows fallback
            raise ValueError(f"Refusing to read linked state file: {target}")
        metadata = target.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Managed state entry is not one regular file: {target}")
        if metadata.st_size > max_bytes:
            raise ValueError(f"Managed state file is too large: {target}")
        with target.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError(f"Managed state file is too large: {target}")
        return payload

    @staticmethod
    def write_text(
        path: Path,
        value: str,
        *,
        encoding: str = "utf-8",
        mode: int = 0o600,
    ) -> None:
        """Atomically replace a managed text artifact without following links."""
        if not isinstance(value, str):
            raise TypeError("value must be text")
        RunStore.write_bytes(path, value.encode(encoding), mode=mode)

    @staticmethod
    def write_bytes(path: Path, value: bytes, *, mode: int = 0o600) -> None:
        """Atomically replace a managed artifact without following path links."""
        if not isinstance(value, bytes):
            raise TypeError("value must be bytes")
        if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
            raise ValueError("mode must contain only ordinary permission bits")
        target = Path(os.path.abspath(os.fspath(path.expanduser())))
        parent_fd = _open_directory_tree(target.parent, create=True)
        if parent_fd is not None:
            try:
                _atomic_write_at(parent_fd, target.name, value, mode=mode)
            finally:
                os.close(parent_fd)
            return

        if target.is_symlink():  # pragma: no cover - Windows fallback
            raise ValueError(f"Refusing to replace linked state file: {target}")
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(value)
                os.fchmod(handle.fileno(), mode)
            Path(temporary).replace(target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _load_run_at(self, runs_fd: int, name: str) -> StoredRun | None:
        if not _safe_child_name(name):
            return None
        try:
            run_fd = _open_child_directory(runs_fd, name)
        except (OSError, _UnsafeStatePath):
            return None
        try:
            try:
                manifest = _read_json_at(run_fd, "manifest.json")
            except (OSError, _UnsafeStatePath):
                return None
        finally:
            os.close(run_fd)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("run_id") != name
            or manifest.get("repository") != str(self.repo_path)
        ):
            return None
        return StoredRun(name, self.runs_root / name, manifest)

    def _list_runs_at(self, runs_fd: int, limit: int) -> list[StoredRun]:
        result: list[StoredRun] = []
        for name in sorted(os.listdir(runs_fd), reverse=True):
            if len(result) >= limit:
                break
            stored = self._load_run_at(runs_fd, name)
            if stored is not None:
                result.append(stored)
        return result

    def _read_run_json(self, run: StoredRun, name: str) -> Any:
        if run.path != self.runs_root / run.run_id or not _safe_child_name(name):
            raise ValueError("Stored run path is not owned by this run store")
        runs_fd = _open_directory_tree(self.runs_root, create=False)
        if runs_fd is None:  # pragma: no cover - Windows fallback
            path = run.path / name
            if not _is_owned_run(run.path, self.runs_root) or path.is_symlink():
                raise ValueError("Stored run file is unsafe")
            return self.read_json(path)
        try:
            run_fd = _open_child_directory(runs_fd, run.run_id)
            try:
                current = _read_json_at(run_fd, "manifest.json")
                if current != run.manifest:
                    raise ValueError("Stored run manifest changed while reading artifacts")
                return _read_json_at(run_fd, name)
            finally:
                os.close(run_fd)
        finally:
            os.close(runs_fd)

    def _clean_before_locked(self, days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            runs_fd = _open_directory_tree(self.runs_root, create=False)
        except (OSError, _UnsafeStatePath):
            return 0
        if runs_fd is None:  # pragma: no cover - Windows fallback
            removed = 0
            for run in self._list_runs_fallback(100_000):
                if not _eligible_for_cleanup(run.manifest, cutoff):
                    continue
                if _is_owned_run(run.path, self.runs_root):
                    shutil.rmtree(run.path)
                    removed += 1
            return removed

        try:
            removed = 0
            for run in self._list_runs_at(runs_fd, 100_000):
                if not _eligible_for_cleanup(run.manifest, cutoff):
                    continue
                current = self._load_run_at(runs_fd, run.run_id)
                if current is None or current.manifest != run.manifest:
                    continue
                if _remove_owned_run_at(runs_fd, run.run_id, run.manifest):
                    removed += 1
            return removed
        finally:
            os.close(runs_fd)

    def _get_fallback(self, run_id: str | None) -> StoredRun | None:
        runs = self._list_runs_fallback(100_000)
        if run_id is None:
            return runs[0] if runs else None
        return next((run for run in runs if run.run_id == run_id), None)

    def _list_runs_fallback(self, limit: int) -> list[StoredRun]:
        if not _is_plain_directory(self.runs_root):
            return []
        result: list[StoredRun] = []
        for path in sorted(self.runs_root.iterdir(), reverse=True):
            if len(result) >= limit or not _is_owned_run(path, self.runs_root):
                continue
            try:
                manifest_path = path / "manifest.json"
                manifest = self.read_json(manifest_path)
            except (OSError, ValueError):
                continue
            if (
                isinstance(manifest, dict)
                and manifest.get("schema_version") == SCHEMA_VERSION
                and manifest.get("run_id") == path.name
                and manifest.get("repository") == str(self.repo_path)
            ):
                result.append(StoredRun(path.name, path, manifest))
        return result


def _is_owned_run(path: Path, runs_root: Path) -> bool:
    path = Path(os.path.abspath(os.fspath(path)))
    runs_root = Path(os.path.abspath(os.fspath(runs_root)))
    if path.parent != runs_root or not _is_plain_directory(runs_root):
        return False
    try:
        if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
            return False
    except OSError:
        return False
    manifest = path / "manifest.json"
    try:
        metadata = manifest.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _eligible_for_cleanup(manifest: dict[str, Any], cutoff: datetime) -> bool:
    if manifest.get("status") not in {"complete", "incomplete"}:
        return False
    try:
        finished = datetime.fromisoformat(str(manifest["finished_at"]))
    except (KeyError, TypeError, ValueError):
        return False
    if finished.tzinfo is None or finished.utcoffset() is None:
        return False
    return finished < cutoff


def _open_directory_tree(path: Path, *, create: bool) -> int | None:
    """Open a canonical managed directory without following any component link."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    supports_dir_fd = os.open in os.supports_dir_fd and os.mkdir in os.supports_dir_fd
    if not supports_dir_fd:  # pragma: no cover - Windows fallback
        if create:
            _reject_existing_link_components(absolute)
            absolute.mkdir(parents=True, exist_ok=True)
        _reject_existing_link_components(absolute)
        if not absolute.is_dir():
            raise _UnsafeStatePath(f"Managed state path is not a directory: {absolute}")
        return None

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    anchor = absolute.anchor or os.sep
    current = os.open(anchor, flags)
    try:
        anchor_parts = Path(anchor).parts
        for component in absolute.parts[len(anchor_parts) :]:
            try:
                child = os.open(component, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, 0o700, dir_fd=current)
                child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except OSError as exc:
        os.close(current)
        raise _UnsafeStatePath(f"Managed state path is unsafe: {absolute}: {exc}") from exc
    except BaseException:
        os.close(current)
        raise


def _open_child_directory(parent_fd: int, name: str) -> int:
    if not _safe_child_name(name):
        raise _UnsafeStatePath(f"Unsafe managed state directory name: {name!r}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise _UnsafeStatePath(f"Managed state child is not a directory: {name}")
    return descriptor


def _read_json_at(directory_fd: int, name: str) -> Any:
    payload = _read_bytes_at(directory_fd, name, max_bytes=_MAX_STORED_JSON_BYTES)
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _UnsafeStatePath(f"Managed state file is invalid JSON: {name}") from exc


def _read_bytes_at(directory_fd: int, name: str, *, max_bytes: int) -> bytes:
    if not _safe_child_name(name):
        raise _UnsafeStatePath(f"Unsafe managed state filename: {name!r}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise _UnsafeStatePath(f"Managed state file is not one regular file: {name}")
        if metadata.st_size > max_bytes:
            raise _UnsafeStatePath(f"Managed state file is too large: {name}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise _UnsafeStatePath(f"Managed state file is too large: {name}")
        return payload
    finally:
        os.close(descriptor)


def _write_json_at(directory_fd: int, name: str, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    encoded = payload.encode("utf-8")
    if len(encoded) > _MAX_STORED_JSON_BYTES:
        raise ValueError(f"Refusing to persist JSON larger than {_MAX_STORED_JSON_BYTES} bytes")
    _atomic_write_at(directory_fd, name, encoded)


def _atomic_write_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> None:
    if not _safe_child_name(name):
        raise _UnsafeStatePath(f"Unsafe managed state filename: {name!r}")
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, mode, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        try:
            existing = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise _UnsafeStatePath(f"Refusing to replace linked/non-file state entry: {name}")
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        with suppress(OSError):
            os.fsync(directory_fd)
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory_fd)


def _remove_owned_run_at(runs_fd: int, name: str, expected: dict[str, Any]) -> bool:
    try:
        run_fd = _open_child_directory(runs_fd, name)
    except (OSError, _UnsafeStatePath):
        return False
    opened = os.fstat(run_fd)
    try:
        if _read_json_at(run_fd, "manifest.json") != expected:
            return False
        if not _remove_directory_contents(run_fd):
            return False
    except (OSError, _UnsafeStatePath):
        return False
    finally:
        os.close(run_fd)

    try:
        current = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(opened, current):
            return False
        os.rmdir(name, dir_fd=runs_fd)
        return True
    except OSError:
        return False


def _remove_directory_contents(directory_fd: int) -> bool:
    try:
        names = os.listdir(directory_fd)
    except OSError:
        return False
    for name in names:
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = _open_child_directory(directory_fd, name)
                opened = os.fstat(child_fd)
                try:
                    if not _remove_directory_contents(child_fd):
                        return False
                finally:
                    os.close(child_fd)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(opened, current):
                    return False
                os.rmdir(name, dir_fd=directory_fd)
            else:
                os.unlink(name, dir_fd=directory_fd)
        except OSError:
            return False
    return True


def _safe_child_name(name: str) -> bool:
    return bool(name) and name not in {".", ".."} and Path(name).name == name


def _is_plain_directory(path: Path) -> bool:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        if _has_link_component(absolute):
            return False
        return absolute.is_dir() and absolute.resolve(strict=True) == absolute
    except OSError:
        return False


def _has_link_component(path: Path) -> bool:
    for component in reversed([path, *path.parents]):
        if component.is_symlink():
            return True
        is_junction = getattr(component, "is_junction", lambda: False)
        if is_junction():
            return True
    return False


def _reject_existing_link_components(path: Path) -> None:
    if _has_link_component(path):
        raise _UnsafeStatePath(f"Managed state path contains a link: {path}")
