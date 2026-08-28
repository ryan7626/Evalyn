"""Generic subprocess capture with hard in-memory output ceilings."""

from __future__ import annotations

import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from typing import BinaryIO

_PROCESS_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_STDERR_BYTES = 256 * 1024


@dataclass(frozen=True)
class BoundedProcessResult:
    """A subprocess result whose captured streams never exceed their limits."""

    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass
class _CaptureState:
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0
    truncated: bool = False
    error: BaseException | None = None


def run_bounded_process(
    argv: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
    input_bytes: bytes | None = None,
) -> BoundedProcessResult:
    """Run a command while concurrently draining both pipes into hard bounds."""
    if max_stdout_bytes < 0 or max_stderr_bytes < 0:
        raise ValueError("Subprocess output limits cannot be negative")

    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    termination_lock = threading.Lock()

    def terminate() -> None:
        with termination_lock:
            if process.poll() is None:
                with suppress(OSError):
                    process.terminate()

    def capture(stream: BinaryIO, state: _CaptureState, limit: int) -> None:
        try:
            while True:
                chunk = stream.read(_PROCESS_CHUNK_SIZE)
                if not chunk:
                    break
                remaining = limit - state.size
                if remaining > 0:
                    retained = chunk[:remaining]
                    state.chunks.append(retained)
                    state.size += len(retained)
                if len(chunk) > max(remaining, 0):
                    state.truncated = True
                    terminate()
        except BaseException as exc:  # pragma: no cover - defensive pipe failure
            state.error = exc
            terminate()
        finally:
            stream.close()

    stdout_state = _CaptureState()
    stderr_state = _CaptureState()
    readers = [
        threading.Thread(
            target=capture,
            args=(process.stdout, stdout_state, max_stdout_bytes),
            daemon=True,
        ),
        threading.Thread(
            target=capture,
            args=(process.stderr, stderr_state, max_stderr_bytes),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    writer_error: list[BaseException] = []

    def write_stdin() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(input_bytes or b"")
            process.stdin.flush()
        except BrokenPipeError:
            pass
        except BaseException as exc:  # pragma: no cover - defensive pipe failure
            writer_error.append(exc)
            terminate()
        finally:
            process.stdin.close()

    writer: threading.Thread | None = None
    if input_bytes is not None:
        writer = threading.Thread(target=write_stdin, daemon=True)
        writer.start()

    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=2)
        if writer is not None:
            writer.join(timeout=2)

    capture_error = stdout_state.error or stderr_state.error
    if capture_error is not None:
        raise OSError(f"Could not capture subprocess output: {capture_error}")
    if writer_error:
        raise OSError(f"Could not provide subprocess input: {writer_error[0]}")
    if any(reader.is_alive() for reader in readers) or (writer is not None and writer.is_alive()):
        raise OSError("Subprocess pipes did not close after process exit")

    output_truncated = stdout_state.truncated or stderr_state.truncated
    return BoundedProcessResult(
        args=tuple(argv),
        # A child can exit between writing the overflow byte and termination.
        # Never let that race turn a truncated command into a success.
        returncode=125 if output_truncated and returncode == 0 else returncode,
        stdout=b"".join(stdout_state.chunks),
        stderr=b"".join(stderr_state.chunks),
        stdout_truncated=stdout_state.truncated,
        stderr_truncated=stderr_state.truncated,
    )
