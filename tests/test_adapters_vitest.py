"""Tests for deterministic Vitest adapter detection and artifacts."""

import json

import pytest

from evalyn.adapters.base import AdapterError
from evalyn.adapters.vitest import VitestAdapter


def test_detects_vitest_config(tmp_path):
    (tmp_path / "vitest.config.ts").write_text("export default {}\n", encoding="utf-8")

    assert VitestAdapter().detect(tmp_path) is True


def test_detects_vitest_dependency(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"vitest": "^2.0.0"}}),
        encoding="utf-8",
    )

    assert VitestAdapter().detect(tmp_path) is True


def test_detects_vitest_script(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )

    assert VitestAdapter().detect(tmp_path) is True


def test_invalid_package_json_is_not_detected(tmp_path):
    (tmp_path / "package.json").write_text("not json", encoding="utf-8")

    assert VitestAdapter().detect(tmp_path) is False


def test_builds_adapter_owned_path_and_argv():
    artifact = VitestAdapter().build_artifact(
        "import { expect, test } from 'vitest';\ntest('ok', () => expect(1).toBe(1));"
    )

    assert artifact.adapter == "vitest"
    assert artifact.relative_path.as_posix() == (
        f".evalyn/probes/vitest/evalyn.{artifact.probe_id}.test.ts"
    )
    assert artifact.command == (
        "npx",
        "--no-install",
        "vitest",
        "run",
        artifact.relative_path.as_posix(),
        "--reporter=json",
        "--no-cache",
    )
    assert artifact.source not in artifact.command


def test_read_only_probe_command_disables_vitest_results_cache():
    artifact = VitestAdapter().build_artifact(
        "import { expect, test } from 'vitest';\ntest('ok', () => expect(1).toBe(1));"
    )

    assert artifact.command[-1] == "--no-cache"


def test_rejects_assertion_free_or_network_probe():
    adapter = VitestAdapter()

    with pytest.raises(AdapterError, match="expect assertion"):
        adapter.build_artifact("it('runs', () => value())")
    with pytest.raises(AdapterError, match="prohibited"):
        adapter.build_artifact("import http from 'node:http'; it('x', () => expect(1).toBe(1))")
