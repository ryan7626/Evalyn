"""Safe, bounded execution of structured subprocess commands.

This module is intentionally lower-level than Evalyn's test runner.  It does
not parse shell text: callers must provide an argument vector and decide which
programs and arguments their policy permits.
"""

from __future__ import annotations

import errno
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_SAFE_AMBIENT_ENV = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TERM",
    "TZ",
    "WINDIR",
)

_PROTECTED_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_CLIENT_SECRET",
        "COMPOSER_AUTH",
        "DOCKER_AUTH_CONFIG",
        "DOCKER_CONFIG",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GIT_ASKPASS",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_NO_REPLACE_OBJECTS",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HOME",
        "KUBECONFIG",
        "LOGNAME",
        "NETRC",
        "NPM_TOKEN",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "PIP_CONFIG_FILE",
        "PWD",
        "SSH_ASKPASS",
        "SSH_AUTH_SOCK",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)

_SECRET_ENV_SUFFIXES = (
    "_API_KEY",
    "_ACCESS_TOKEN",
    "_AUTH_TOKEN",
    "_CLIENT_SECRET",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The complete, bounded result of one command invocation."""

    argv: tuple[str, ...]
    cwd: Path
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    full_log_path: Path | None = None
    error: str | None = None
    log_error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the command started, finished in time, and exited zero."""
        return not self.timed_out and self.error is None and self.exit_code == 0

    @property
    def returncode(self) -> int | None:
        """Alias matching :mod:`subprocess` terminology."""
        return self.exit_code

    @property
    def output_truncated(self) -> bool:
        """Whether either captured stream omitted output."""
        return self.stdout_truncated or self.stderr_truncated


class CommandRunner:
    """Run argument vectors inside one filesystem workspace.

    Each child starts in a new process session.  On timeout or cancellation,
    the session receives ``SIGTERM`` and then ``SIGKILL`` after a short grace
    period, preventing grandchildren from being left behind.

    The child receives a deliberately small environment.  Ambient credentials,
    user configuration, and authentication agents are not inherited.  ``HOME``
    and the XDG directories always point at a dedicated sandbox directory.
    Callers may add non-secret variables explicitly with ``run(..., env=...)``.

    Captured stdout and stderr are each bounded by ``max_output_bytes`` while
    they are being read. Passing ``full_log_path`` additionally streams both
    outputs to a combined log bounded by ``max_full_log_bytes``.
    """

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        sandbox_home: Path | str | None = None,
        max_output_bytes: int = 64 * 1024,
        max_full_log_bytes: int = 4 * 1024 * 1024,
        default_timeout_seconds: float | None = 300,
        termination_grace_seconds: float = 1.0,
        encoding: str = "utf-8",
    ) -> None:
        try:
            workspace = Path(workspace_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Workspace does not exist: {workspace_root}") from exc
        if not workspace.is_dir():
            raise ValueError(f"Workspace is not a directory: {workspace}")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int):
            raise TypeError("max_output_bytes must be an integer")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")
        if isinstance(max_full_log_bytes, bool) or not isinstance(max_full_log_bytes, int):
            raise TypeError("max_full_log_bytes must be an integer")
        if max_full_log_bytes <= 0:
            raise ValueError("max_full_log_bytes must be a positive integer")
        if default_timeout_seconds is not None and (
            isinstance(default_timeout_seconds, bool)
            or not math.isfinite(default_timeout_seconds)
            or default_timeout_seconds <= 0
        ):
            raise ValueError("default_timeout_seconds must be finite, positive, or None")
        if (
            isinstance(termination_grace_seconds, bool)
            or not math.isfinite(termination_grace_seconds)
            or termination_grace_seconds < 0
        ):
            raise ValueError("termination_grace_seconds cannot be negative")

        self.workspace_root = workspace
        self.max_output_bytes = int(max_output_bytes)
        self.max_full_log_bytes = int(max_full_log_bytes)
        self.default_timeout_seconds = default_timeout_seconds
        self.termination_grace_seconds = float(termination_grace_seconds)
        self.encoding = encoding
        self._closed = False
        self._temporary_home: tempfile.TemporaryDirectory | None = None

        if sandbox_home is None:
            self._temporary_home = tempfile.TemporaryDirectory(prefix="evalyn-home-")
            home = Path(self._temporary_home.name).resolve()
        else:
            home = Path(sandbox_home).expanduser()
            if not home.is_absolute():
                home = workspace / home
            home = home.resolve(strict=False)
            home.mkdir(parents=True, exist_ok=True)
            if not home.is_dir():
                raise ValueError(f"Sandbox home is not a directory: {home}")

        self.sandbox_home = home
        for child in ("cache", "config", "data", "tmp"):
            (home / child).mkdir(exist_ok=True)

    def __enter__(self) -> CommandRunner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Remove an automatically-created sandbox home."""
        if self._closed:
            return
        self._closed = True
        if self._temporary_home is not None:
            self._temporary_home.cleanup()

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str | None = None,
        timeout_seconds: float | None = None,
        env: Mapping[str, str] | None = None,
        full_log_path: Path | str | None = None,
    ) -> CommandResult:
        """Execute ``argv`` and return a bounded :class:`CommandResult`.

        ``cwd`` may be absolute or relative to ``workspace_root``, but its
        canonical path must remain inside that workspace.  Strings and bytes
        are rejected for ``argv`` so shell text cannot accidentally cross this
        boundary.

        ``timeout_seconds=None`` uses the runner's default.  Configure
        ``default_timeout_seconds=None`` to permit commands without a timeout.
        """
        if self._closed:
            raise RuntimeError("CommandRunner is closed")

        normalized_argv = _normalize_argv(argv)
        working_directory = self._resolve_cwd(cwd)
        timeout = self.default_timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be finite, positive, or None")
        child_env = self._build_environment(env)
        child_env["PWD"] = str(working_directory)
        log_path = self._resolve_log_path(full_log_path)

        stdout_capture = _BoundedCapture(self.max_output_bytes)
        stderr_capture = _BoundedCapture(self.max_output_bytes)
        log_writer = (
            _FullLogWriter(log_path, self.max_full_log_bytes) if log_path is not None else None
        )
        started_at = time.monotonic()

        try:
            process = subprocess.Popen(
                normalized_argv,
                cwd=str(working_directory),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **_process_group_options(),
            )
        except OSError as exc:
            if log_writer is not None:
                log_writer.write(str(exc).encode(self.encoding, errors="replace"))
                log_writer.close()
            duration = time.monotonic() - started_at
            message = str(exc)
            return CommandResult(
                argv=normalized_argv,
                cwd=working_directory,
                exit_code=None,
                stdout="",
                stderr=message,
                duration_seconds=duration,
                timed_out=False,
                full_log_path=log_path,
                error=message,
                log_error=log_writer.error if log_writer is not None else None,
            )

        assert process.stdout is not None
        assert process.stderr is not None
        readers = (
            threading.Thread(
                target=_drain_stream,
                args=(process.stdout, stdout_capture, log_writer),
                name=f"evalyn-stdout-{process.pid}",
                daemon=True,
            ),
            threading.Thread(
                target=_drain_stream,
                args=(process.stderr, stderr_capture, log_writer),
                name=f"evalyn-stderr-{process.pid}",
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()

        timed_out = False
        try:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_session(process, self.termination_grace_seconds)

            # A parent can exit while a background grandchild still owns its
            # output pipe. Give normal cleanup a moment, then reap that session.
            _join_threads(readers, self.termination_grace_seconds)
            if any(reader.is_alive() for reader in readers):
                _terminate_process_session(process, self.termination_grace_seconds)
                _close_pipe(process.stdout)
                _close_pipe(process.stderr)
                _join_threads(readers, max(0.1, self.termination_grace_seconds))
        except BaseException:
            _terminate_process_session(process, self.termination_grace_seconds)
            _close_pipe(process.stdout)
            _close_pipe(process.stderr)
            _join_threads(readers, max(0.1, self.termination_grace_seconds))
            raise
        finally:
            if log_writer is not None:
                log_writer.close()

        duration = time.monotonic() - started_at
        stdout_bytes, stdout_truncated = stdout_capture.render()
        stderr_bytes, stderr_truncated = stderr_capture.render()
        error = None
        if timed_out:
            error = f"Command timed out after {timeout:g} seconds"

        return CommandResult(
            argv=normalized_argv,
            cwd=working_directory,
            exit_code=process.returncode,
            stdout=stdout_bytes.decode(self.encoding, errors="replace"),
            stderr=stderr_bytes.decode(self.encoding, errors="replace"),
            duration_seconds=duration,
            timed_out=timed_out,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            full_log_path=log_path,
            error=error,
            log_error=log_writer.error if log_writer is not None else None,
        )

    def _resolve_cwd(self, cwd: Path | str | None) -> Path:
        candidate = self.workspace_root if cwd is None else Path(cwd).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Command working directory does not exist: {candidate}") from exc
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"Command working directory escapes workspace: {resolved}") from exc
        if not resolved.is_dir():
            raise ValueError(f"Command working directory is not a directory: {resolved}")
        return resolved

    def _resolve_log_path(self, path: Path | str | None) -> Path | None:
        if path is None:
            return None
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        if _has_link_component(candidate.parent):
            raise ValueError(f"Full log parent cannot contain a link: {candidate.parent}")
        if candidate.is_symlink():
            raise ValueError(f"Full log path cannot be a symlink: {candidate}")
        parent = candidate.parent.resolve(strict=False)
        parent.mkdir(parents=True, exist_ok=True)
        resolved = parent / candidate.name
        if resolved.is_symlink():
            raise ValueError(f"Full log path cannot be a symlink: {resolved}")
        if resolved.exists() and resolved.is_dir():
            raise ValueError(f"Full log path is a directory: {resolved}")
        return resolved

    def _build_environment(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        environment: dict[str, str] = {}
        for name in _SAFE_AMBIENT_ENV:
            value = os.environ.get(name)
            if value is not None and "\0" not in value:
                environment[name] = value
        environment.setdefault("PATH", os.defpath)

        environment.update(validate_environment_overrides(extra))

        home = str(self.sandbox_home)
        environment.update(
            {
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "HOME": home,
                "LOGNAME": "evalyn",
                "NPM_CONFIG_USERCONFIG": os.devnull,
                "PIP_CONFIG_FILE": os.devnull,
                "PYTHONNOUSERSITE": "1",
                "TMPDIR": str(self.sandbox_home / "tmp"),
                "USER": "evalyn",
                "XDG_CACHE_HOME": str(self.sandbox_home / "cache"),
                "XDG_CONFIG_HOME": str(self.sandbox_home / "config"),
                "XDG_DATA_HOME": str(self.sandbox_home / "data"),
            }
        )
        return environment


def validate_environment_overrides(
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate explicitly supplied child variables and reject credentials.

    Container runtimes use the same boundary before serializing variables into
    OCI arguments, preventing that wrapper from bypassing ``CommandRunner``'s
    environment policy.
    """
    validated: dict[str, str] = {}
    for name, value in (extra or {}).items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("Environment names and values must be strings")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or "\0" in value:
            raise ValueError(f"Invalid environment variable name or value: {name!r}")
        if _is_protected_environment_name(name):
            raise ValueError(f"Credential or sandbox environment variable is not permitted: {name}")
        validated[name] = value
    return validated


class _BoundedCapture:
    """Retain the beginning and end of a byte stream with fixed memory use."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.head_limit = (limit + 1) // 2
        self.tail_limit = limit // 2
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def add(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total += len(chunk)
        remaining = chunk
        if len(self.head) < self.head_limit:
            take = min(self.head_limit - len(self.head), len(remaining))
            self.head.extend(remaining[:take])
            remaining = remaining[take:]
        if remaining and self.tail_limit:
            self.tail.extend(remaining)
            overflow = len(self.tail) - self.tail_limit
            if overflow > 0:
                del self.tail[:overflow]

    def render(self) -> tuple[bytes, bool]:
        if self.total <= self.limit:
            return bytes(self.head + self.tail), False

        marker = b"\n... output truncated ...\n"
        if len(marker) >= self.limit:
            return bytes(self.head[: self.limit]), True
        payload_budget = self.limit - len(marker)
        head_size = (payload_budget + 1) // 2
        tail_size = payload_budget // 2
        tail = bytes(self.tail[-tail_size:]) if tail_size else b""
        return (
            bytes(self.head[:head_size]) + marker + tail,
            True,
        )


class _FullLogWriter:
    """Serialize raw stdout/stderr chunks into one size-bounded on-disk log."""

    def __init__(self, path: Path, limit: int) -> None:
        self._directory_fd: int | None = None
        self._temporary_path: Path | None = None
        self._temporary_name = f".{path.name}.evalyn-{uuid.uuid4().hex}.tmp"
        supports_dir_fd = os.open in os.supports_dir_fd and os.replace in os.supports_dir_fd
        if supports_dir_fd:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            self._directory_fd = os.open(path.parent, directory_flags)
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            file_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(
                    self._temporary_name,
                    file_flags,
                    0o600,
                    dir_fd=self._directory_fd,
                )
            except BaseException:
                os.close(self._directory_fd)
                self._directory_fd = None
                raise
        else:  # pragma: no cover - Windows fallback
            descriptor, temporary = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.evalyn-",
                suffix=".tmp",
            )
            self._temporary_path = Path(temporary)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            self._discard_temporary()
            if self._directory_fd is not None:
                os.close(self._directory_fd)
                self._directory_fd = None
            raise ValueError(f"Full log temporary is not one regular file: {path}")
        self._handle: BinaryIO = os.fdopen(descriptor, "wb", buffering=0)
        self._path = path
        self._limit = limit
        self._written = 0
        self._lock = threading.Lock()
        self._io_failed = False
        self._closed = False
        self.error: str | None = None

    def write(self, chunk: bytes) -> None:
        if not chunk or self.error is not None:
            return
        try:
            with self._lock:
                remaining = self._limit - self._written
                if remaining <= 0:
                    self.error = f"full log truncated at {self._limit} bytes"
                    return
                payload = chunk[:remaining]
                self._handle.write(payload)
                self._written += len(payload)
                if len(payload) < len(chunk):
                    self.error = f"full log truncated at {self._limit} bytes"
        except (OSError, ValueError) as exc:
            self._io_failed = True
            self.error = str(exc)

    def close(self) -> None:
        try:
            with self._lock:
                if self._closed:
                    return
                if not self._handle.closed:
                    self._handle.close()
                if not self._io_failed:
                    if self._directory_fd is not None:
                        os.replace(
                            self._temporary_name,
                            self._path.name,
                            src_dir_fd=self._directory_fd,
                            dst_dir_fd=self._directory_fd,
                        )
                        with suppress(OSError):
                            os.fsync(self._directory_fd)
                    else:  # pragma: no cover - Windows fallback
                        assert self._temporary_path is not None
                        self._temporary_path.replace(self._path)
        except OSError as exc:
            self._io_failed = True
            self.error = self.error or str(exc)
        finally:
            self._closed = True
            if self._io_failed:
                self._discard_temporary()
            if self._directory_fd is not None:
                os.close(self._directory_fd)
                self._directory_fd = None

    def _discard_temporary(self) -> None:
        if self._directory_fd is not None:
            with suppress(OSError):
                os.unlink(self._temporary_name, dir_fd=self._directory_fd)
        elif self._temporary_path is not None:  # pragma: no cover - Windows fallback
            with suppress(OSError):
                self._temporary_path.unlink()


def _normalize_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes, bytearray)) or not isinstance(argv, Sequence):
        raise TypeError("argv must be a sequence of strings, not shell text")
    normalized = tuple(argv)
    if not normalized:
        raise ValueError("argv cannot be empty")
    for argument in normalized:
        if not isinstance(argument, str):
            raise TypeError("every argv element must be a string")
        if "\0" in argument:
            raise ValueError("argv elements cannot contain NUL bytes")
    if not normalized[0]:
        raise ValueError("argv[0] cannot be empty")
    return normalized


def _has_link_component(path: Path) -> bool:
    for component in (path, *path.parents):
        if component.is_symlink():
            return True
        is_junction = getattr(component, "is_junction", lambda: False)
        if is_junction():
            return True
    return False


def _is_protected_environment_name(name: str) -> bool:
    upper_name = name.upper()
    return upper_name in _PROTECTED_ENV_NAMES or upper_name.endswith(_SECRET_ENV_SUFFIXES)


def _drain_stream(
    stream: BinaryIO,
    capture: _BoundedCapture,
    log_writer: _FullLogWriter | None,
) -> None:
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            capture.add(chunk)
            if log_writer is not None:
                log_writer.write(chunk)
    except (OSError, ValueError):
        # The main thread closes pipes only after terminating a stuck session.
        return
    finally:
        _close_pipe(stream)


def _process_group_options() -> dict[str, object]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def _terminate_process_session(process: subprocess.Popen, grace_seconds: float) -> None:
    if os.name == "posix":
        process.poll()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()
            if not _posix_group_exists(process.pid):
                break
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

        if _posix_group_exists(process.pid):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is final
            process.kill()
            process.wait()
        return

    # A Windows process group does not offer POSIX signal semantics. This is a
    # best-effort fallback; production Windows isolation should use a Job Object.
    if process.poll() is None:  # pragma: no cover - exercised on Windows CI
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _posix_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _join_threads(threads: Sequence[threading.Thread], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


def _close_pipe(pipe: BinaryIO | None) -> None:
    if pipe is None:
        return
    with suppress(OSError):
        pipe.close()
