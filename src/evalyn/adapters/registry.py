"""Registry and deterministic detection for supported test adapters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from evalyn.adapters.base import AdapterError, TestAdapter
from evalyn.adapters.pytest import PytestAdapter
from evalyn.adapters.vitest import VitestAdapter


class AdapterRegistry:
    """Ordered registry of explicitly trusted framework adapters."""

    def __init__(self, adapters: Iterable[TestAdapter] = ()) -> None:
        self._adapters: dict[str, TestAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: TestAdapter) -> None:
        if not adapter.name or adapter.name in self._adapters:
            raise AdapterError(f"adapter already registered or unnamed: {adapter.name!r}")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> TestAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            supported = ", ".join(self.names) or "none"
            raise AdapterError(f"unknown adapter {name!r}; supported: {supported}") from exc

    def detect(self, repo_path: Path) -> tuple[TestAdapter, ...]:
        return tuple(adapter for adapter in self._adapters.values() if adapter.detect(repo_path))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._adapters)


def default_registry() -> AdapterRegistry:
    """Return a fresh registry containing the v0.2 supported adapters."""
    return AdapterRegistry((PytestAdapter(), VitestAdapter()))
