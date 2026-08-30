"""Tests for bounded OCI image identity and environment fingerprints."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from evalyn.execution import (
    ContainerRunner,
    ImageIdentityError,
    ImageIdentityStatus,
    OciContainerRuntime,
    parse_image_identity_inspection,
)
from evalyn.execution.process import CommandResult


@pytest.fixture
def identity_engine(tmp_path: Path) -> tuple[Path, Path]:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    calls = engine_dir / "calls.jsonl"
    engine = engine_dir / "docker"
    engine.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

root = pathlib.Path(__file__).parent
with (root / "calls.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "argv": sys.argv[1:],
        "docker_host": os.environ.get("DOCKER_HOST"),
        "secret_present": "OPENAI_API_KEY" in os.environ,
    }) + "\\n")

args = sys.argv[1:]
if args and args[0] == "run":
    print("workload complete")
    raise SystemExit(0)
if args and args[0] == "rm":
    raise SystemExit(0)
if args[:2] == ["image", "inspect"]:
    mode_path = root / "inspect-mode"
    mode = mode_path.read_text(encoding="utf-8").strip() if mode_path.exists() else "valid"
    if mode == "malformed":
        print("sha256:not-a-digest")
        raise SystemExit(0)
    if mode == "truncated":
        print("x" * 4096)
        raise SystemExit(0)
    if mode == "failed":
        print("daemon detail must not be serialized", file=sys.stderr)
        raise SystemExit(2)
    print("sha256:" + "a" * 64)
    raise SystemExit(0)
raise SystemExit(3)
""",
        encoding="utf-8",
    )
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    return engine, calls


def test_explicit_resolution_uses_same_endpoint_and_is_cached(
    tmp_path: Path,
    identity_engine: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, calls_path = identity_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/trusted-docker.sock")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    with ContainerRunner(
        workspace,
        image="example/evalyn:mutable",
        engine=str(engine),
    ) as runner:
        assert not runner.environment_fingerprint.complete
        assert (
            runner.environment_fingerprint.image_resolution.error
            == ImageIdentityError.NOT_INSPECTED
        )
        assert runner.run(["python", "--version"]).succeeded
        fingerprint = runner.resolve_environment_fingerprint()
        assert runner.resolve_environment_fingerprint() is fingerprint

    assert fingerprint.complete
    assert fingerprint.resolved_image_identity == "sha256:" + "a" * 64
    assert fingerprint.fingerprint_sha256 is not None
    payload = fingerprint.to_dict()
    assert payload["engine"] == "docker"
    assert payload["declared_image"] == "example/evalyn:mutable"
    assert payload["network_policy"] == "none"
    assert payload["read_only_policy"] == {
        "root_filesystem": True,
        "workspace": False,
    }
    assert payload["pull_policy"] == "never"
    assert payload["security_policy"]["capabilities_dropped"] == ["ALL"]

    serialized = json.dumps(payload, sort_keys=True)
    assert str(engine) not in serialized
    assert "trusted-docker.sock" not in serialized
    assert "must-not-leak" not in serialized
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    assert [call["argv"][0] for call in calls] == ["run", "rm", "image"]
    assert all(call["docker_host"] == "unix:///tmp/trusted-docker.sock" for call in calls)
    assert all(not call["secret_present"] for call in calls)


def test_resolution_before_successful_launch_is_incomplete_without_inspection(
    tmp_path: Path,
    identity_engine: tuple[Path, Path],
) -> None:
    engine, calls_path = identity_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with ContainerRunner(workspace, image="example:latest", engine=str(engine)) as runner:
        fingerprint = runner.resolve_environment_fingerprint()

    assert not fingerprint.complete
    assert fingerprint.fingerprint_sha256 is None
    assert fingerprint.image_resolution.status == ImageIdentityStatus.INCOMPLETE
    assert fingerprint.image_resolution.error == ImageIdentityError.NO_SUCCESSFUL_LAUNCH
    assert not calls_path.exists()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("malformed", ImageIdentityError.MALFORMED_INSPECTION),
        ("failed", ImageIdentityError.INSPECTION_FAILED),
        ("truncated", ImageIdentityError.INSPECTION_TRUNCATED),
    ],
)
def test_bad_inspection_is_incomplete_and_has_no_fingerprint(
    tmp_path: Path,
    identity_engine: tuple[Path, Path],
    mode: str,
    expected: ImageIdentityError,
) -> None:
    engine, _calls_path = identity_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (engine.parent / "inspect-mode").write_text(mode, encoding="utf-8")

    with ContainerRunner(
        workspace,
        image="example:latest",
        engine=str(engine),
        max_output_bytes=128,
    ) as runner:
        assert runner.run(["true"]).succeeded
        fingerprint = runner.resolve_environment_fingerprint()

    assert not fingerprint.complete
    assert fingerprint.resolved_image_identity is None
    assert fingerprint.fingerprint_sha256 is None
    assert fingerprint.image_resolution.error == expected
    assert "daemon detail" not in json.dumps(fingerprint.to_dict())


def test_parser_rejects_truncation_even_with_valid_prefix(tmp_path: Path) -> None:
    result = _command_result(
        tmp_path,
        stdout="sha256:" + "b" * 64,
        stderr_truncated=True,
    )

    resolution = parse_image_identity_inspection(result)

    assert not resolution.complete
    assert resolution.error == ImageIdentityError.INSPECTION_TRUNCATED


def test_parser_canonicalizes_bare_podman_image_id(tmp_path: Path) -> None:
    resolution = parse_image_identity_inspection(_command_result(tmp_path, stdout="c" * 64 + "\n"))

    assert resolution.complete
    assert resolution.identity is not None
    assert resolution.identity.value == "sha256:" + "c" * 64


def test_runtime_fingerprint_normalizes_custom_engine_path(tmp_path: Path) -> None:
    runtime = OciContainerRuntime(
        "python:3.12-slim",
        engine=str(tmp_path / "private" / "wrapper"),
        network="bridge",
        pull_policy="missing",
        workspace_read_only=True,
    )

    payload = runtime.environment_fingerprint().to_dict()

    assert payload["engine"] == "custom-oci"
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["complete"] is False
    assert payload["fingerprint_sha256"] is None
    assert payload["endpoint_scope"] == "local"


def _command_result(
    tmp_path: Path,
    *,
    stdout: str,
    exit_code: int = 0,
    timed_out: bool = False,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> CommandResult:
    return CommandResult(
        argv=("docker", "image", "inspect"),
        cwd=tmp_path,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        duration_seconds=0.01,
        timed_out=timed_out,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        error="timed out" if timed_out else None,
    )
