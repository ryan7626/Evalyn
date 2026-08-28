"""Tests for supported-adapter registry behavior."""

import pytest

from evalyn.adapters.base import AdapterError
from evalyn.adapters.pytest import PytestAdapter
from evalyn.adapters.registry import AdapterRegistry, default_registry


def test_default_registry_contains_only_v02_adapters():
    assert default_registry().names == ("pytest", "vitest")


def test_detect_can_return_multiple_frameworks(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "vitest.config.ts").write_text("export default {}\n", encoding="utf-8")

    assert tuple(adapter.name for adapter in default_registry().detect(tmp_path)) == (
        "pytest",
        "vitest",
    )


def test_get_unknown_adapter_lists_supported_names():
    with pytest.raises(AdapterError, match="pytest, vitest"):
        default_registry().get("jest")


def test_duplicate_adapter_registration_is_rejected():
    registry = AdapterRegistry((PytestAdapter(),))

    with pytest.raises(AdapterError, match="already registered"):
        registry.register(PytestAdapter())
