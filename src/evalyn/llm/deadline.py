"""Hard caller-side deadlines for synchronous provider requests."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import suppress
from queue import Queue
from threading import Thread
from typing import TypeVar, cast

T = TypeVar("T")


def run_with_hard_deadline(
    call: Callable[[], T],
    *,
    timeout_seconds: float,
    timeout_error: Callable[[], BaseException],
    cancel: Callable[[], object] | None = None,
) -> T:
    """Stop waiting at a wall-clock deadline even if a client slow-drips.

    Provider clients still receive their native socket timeout. The daemon
    worker prevents a blocking synchronous client from extending Evalyn's run
    budget; a best-effort close is launched separately on timeout.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    outcome: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put((True, call()))
        except BaseException as exc:
            outcome.put((False, exc))

    worker = Thread(target=invoke, name="evalyn-provider-request", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        if cancel is not None:
            Thread(target=_best_effort_cancel, args=(cancel,), daemon=True).start()
        raise timeout_error()

    succeeded, value = outcome.get_nowait()
    if succeeded:
        return cast(T, value)
    if not isinstance(value, BaseException):  # pragma: no cover - queue invariant
        raise RuntimeError("Provider worker returned an invalid error")
    raise value


def _best_effort_cancel(cancel: Callable[[], object]) -> None:
    with suppress(BaseException):
        cancel()
