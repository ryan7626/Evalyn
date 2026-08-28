"""Cross-process repository lock for isolated Evalyn runs."""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from evalyn.execution.errors import RunLockError

_HELD_LOCKS: set[Path] = set()
_REGISTRY_LOCK = threading.Lock()
_LOCK_METADATA_LIMIT = 16 * 1024


class RepoRunLock:
    """Hold an exclusive advisory lock for one target repository.

    The lock file is intentionally retained after release. Removing a lock file
    while contenders may have it open creates an inode race where two processes
    can both believe they own the lock.
    """

    def __init__(
        self,
        repo_path: Path,
        *,
        lock_path: Path | None = None,
        timeout_seconds: float = 0.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        requested_lock = (
            Path(lock_path).expanduser()
            if lock_path is not None
            else self.repo_path / ".agent" / "run.lock"
        )
        # Keep the final path lexical. Resolving it here would turn a preplanted
        # symlink into its target before the safety checks in ``acquire`` run.
        self.lock_path = Path(os.path.abspath(os.fspath(requested_lock)))
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self.token = uuid.uuid4().hex
        self._handle: IO[str] | None = None
        self._acquired = False
        self._registry_key: Path | None = None

    @property
    def is_acquired(self) -> bool:
        return self._acquired

    def acquire(self) -> RepoRunLock:
        """Acquire the lock or raise :class:`RunLockError`."""
        if self._acquired:
            return self

        registry_key = self.lock_path

        with _REGISTRY_LOCK:
            if registry_key in _HELD_LOCKS:
                raise RunLockError(
                    f"An Evalyn run already holds the repository lock: {self.lock_path}"
                )
            _HELD_LOCKS.add(registry_key)

        handle: IO[str] | None = None
        try:
            handle = _open_lock_handle(self.lock_path, create=True)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    _lock_file(handle)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        owner = _read_owner(handle)
                        detail = f" Owner: {owner}" if owner else ""
                        raise RunLockError(
                            f"Another Evalyn run owns {self.lock_path}.{detail}"
                        ) from exc
                    time.sleep(self.poll_interval_seconds)

            if not _lock_path_matches_handle(self.lock_path, handle):
                raise RunLockError(
                    f"Repository lock path changed while it was being acquired: {self.lock_path}"
                )

            metadata = {
                "token": self.token,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "repository": str(self.repo_path),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.seek(0)
            handle.truncate()
            json.dump(metadata, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            self._handle = handle
            self._acquired = True
            self._registry_key = registry_key
            return self
        except BaseException:
            if handle is not None:
                with suppress(OSError):
                    _unlock_file(handle)
                handle.close()
            with _REGISTRY_LOCK:
                _HELD_LOCKS.discard(registry_key)
            raise

    def release(self) -> None:
        """Release the lock. Calling this more than once is safe."""
        if not self._acquired or self._handle is None:
            return

        handle = self._handle
        self._handle = None
        self._acquired = False
        try:
            _unlock_file(handle)
        finally:
            handle.close()
            with _REGISTRY_LOCK:
                if self._registry_key is not None:
                    _HELD_LOCKS.discard(self._registry_key)
            self._registry_key = None

    def __enter__(self) -> RepoRunLock:
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.release()
        return False


def read_lock_metadata(lock_path: Path) -> dict[str, Any] | None:
    """Read diagnostic lock metadata without making ownership assumptions."""
    try:
        path = Path(os.path.abspath(os.fspath(Path(lock_path).expanduser())))
        with _open_lock_handle(path, create=False, writable=False) as handle:
            raw = _read_bounded_metadata(handle)
    except (OSError, RunLockError):
        return None
    return raw if isinstance(raw, dict) else None


def _open_lock_handle(
    path: Path,
    *,
    create: bool,
    writable: bool = True,
) -> IO[str]:
    """Open one regular lock file without following any path symlinks.

    On platforms with ``openat`` support, every parent component is traversed
    through directory descriptors with ``O_NOFOLLOW``. The final file is opened
    without truncation and verified against its directory entry before callers
    are allowed to write metadata through the descriptor.
    """
    if not path.name or path.name in {".", ".."}:
        raise RunLockError(f"Invalid repository lock path: {path}")

    parent_fd = _open_directory_tree(path.parent, create=create)
    flags = os.O_RDWR if writable else os.O_RDONLY
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)

    descriptor: int | None = None
    try:
        if parent_fd is None:  # Windows fallback without dir_fd support.
            descriptor = os.open(path, flags, 0o600)
            entry = os.lstat(path)
        else:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
            entry = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(entry.st_mode)
            or opened.st_dev != entry.st_dev
            or opened.st_ino != entry.st_ino
            or opened.st_nlink != 1
        ):
            raise RunLockError(f"Repository lock must be one unlinked regular file: {path}")
        if writable:
            fchmod = getattr(os, "fchmod", None)
            if fchmod is not None:
                with suppress(OSError):
                    fchmod(descriptor, 0o600)
        mode = "r+" if writable else "r"
        handle = os.fdopen(descriptor, mode, encoding="utf-8")
        descriptor = None
        return handle
    except FileNotFoundError:
        raise RunLockError(f"Repository lock does not exist: {path}") from None
    except RunLockError:
        raise
    except OSError as exc:
        raise RunLockError(f"Unsafe or unavailable repository lock path {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def _lock_path_matches_handle(path: Path, handle: IO[str]) -> bool:
    """Revalidate the directory entry after acquiring the OS-level lock."""
    try:
        with _open_lock_handle(path, create=False, writable=False) as current:
            return os.path.samestat(os.fstat(handle.fileno()), os.fstat(current.fileno()))
    except (OSError, RunLockError):
        return False


def _open_directory_tree(path: Path, *, create: bool) -> int | None:
    """Open a directory path component-by-component without following links."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    supports_dir_fd = os.open in os.supports_dir_fd and os.mkdir in os.supports_dir_fd
    if not supports_dir_fd:  # pragma: no cover - exercised on Windows CI
        _prepare_directory_fallback(absolute, create=create)
        return None

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    anchor = absolute.anchor or os.sep
    try:
        current = os.open(anchor, flags)
    except OSError as exc:
        raise RunLockError(f"Cannot open repository lock anchor {anchor}: {exc}") from exc

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
        raise RunLockError(f"Repository lock path is unsafe or unavailable: {path}: {exc}") from exc
    except BaseException:
        os.close(current)
        raise


def _prepare_directory_fallback(path: Path, *, create: bool) -> None:
    """Best-effort no-link directory validation for platforms without openat."""
    if create:
        _reject_existing_link_components(path)
        path.mkdir(parents=True, exist_ok=True)
    _reject_existing_link_components(path)
    if not path.is_dir():
        raise RunLockError(f"Repository lock parent is not a directory: {path}")


def _reject_existing_link_components(path: Path) -> None:
    components = [path, *path.parents]
    for component in reversed(components):
        if not component.exists() and not component.is_symlink():
            continue
        is_junction = getattr(component, "is_junction", lambda: False)
        if component.is_symlink() or is_junction():
            raise RunLockError(f"Repository lock path contains a link: {component}")


def _read_owner(handle: IO[str]) -> str:
    raw = _read_bounded_metadata(handle)
    if not isinstance(raw, dict):
        return ""
    pid = raw.get("pid", "unknown")
    host = raw.get("hostname", "unknown")
    return f"pid={pid} host={host}"


def _read_bounded_metadata(handle: IO[str]) -> Any | None:
    try:
        os.lseek(handle.fileno(), 0, os.SEEK_SET)
        payload = os.read(handle.fileno(), _LOCK_METADATA_LIMIT + 1)
        if len(payload) > _LOCK_METADATA_LIMIT:
            return None
        return json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


if os.name == "nt":  # pragma: no cover - exercised on Windows CI
    import msvcrt

    def _lock_file(handle: IO[str]) -> None:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError from exc

    def _unlock_file(handle: IO[str]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_file(handle: IO[str]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(handle: IO[str]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
