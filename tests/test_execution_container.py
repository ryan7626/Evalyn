"""Tests for fail-closed Docker/Podman workload execution."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from evalyn.execution.bounded import BoundedProcessResult
from evalyn.execution.container import (
    ContainerRunner,
    _resolve_container_client_environment,
    resolve_container_engine,
)
from evalyn.execution.errors import RuntimeUnavailableError


@pytest.fixture
def fake_engine(tmp_path: Path) -> tuple[Path, Path]:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    calls = engine_dir / "calls.jsonl"
    engine = engine_dir / "docker"
    engine.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

root = pathlib.Path(__file__).parent
with (root / "calls.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")

if len(sys.argv) > 1 and sys.argv[1] == "run":
    print("container workload output")
    raise SystemExit(0)
if len(sys.argv) > 1 and sys.argv[1] == "rm":
    if (root / "fail-cleanup").exists():
        print("daemon unavailable", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0)
raise SystemExit(3)
""",
        encoding="utf-8",
    )
    engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
    return engine, calls


def test_container_runner_wraps_workload_and_force_removes_named_container(
    tmp_path: Path,
    fake_engine: tuple[Path, Path],
) -> None:
    engine, calls_path = fake_engine
    workspace = tmp_path / "workspace"
    nested = workspace / "src"
    nested.mkdir(parents=True)

    with ContainerRunner(
        workspace,
        image="evalyn-tests:local",
        engine=str(engine),
    ) as runner:
        result = runner.run(
            ["python", "-m", "pytest", ".evalyn-generated/probe.py"],
            cwd=nested,
            env={"EVALYN_REPETITION": "1"},
        )
        invocation = runner.last_invocation

    assert result.succeeded
    assert "container workload output" in result.stdout
    assert invocation is not None
    assert invocation.container_name is not None
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 2
    run_call, cleanup_call = calls
    assert run_call[0] == "run"
    assert "--pull" in run_call
    assert run_call[run_call.index("--pull") + 1] == "never"
    assert "--network" in run_call
    assert run_call[run_call.index("--network") + 1] == "none"
    assert "EVALYN_REPETITION=1" in run_call
    assert run_call[-4:] == [
        "python",
        "-m",
        "pytest",
        ".evalyn-generated/probe.py",
    ]
    assert cleanup_call == ["rm", "--force", invocation.container_name]


def test_container_runner_never_falls_back_to_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr("evalyn.execution.container.shutil.which", lambda _: None)

    with pytest.raises(RuntimeUnavailableError, match="will not execute.*host"):
        ContainerRunner(workspace, image="evalyn-tests:local")


def test_container_runner_forwards_read_only_workspace_policy(
    tmp_path: Path,
    fake_engine: tuple[Path, Path],
) -> None:
    engine, _calls = fake_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with ContainerRunner(
        workspace,
        image="evalyn-tests:local",
        engine=str(engine),
        workspace_read_only=True,
    ) as runner:
        result = runner.run(["python", "--version"])
        invocation = runner.last_invocation

    assert result.succeeded
    assert invocation is not None
    mount = invocation.argv[invocation.argv.index("--mount") + 1]
    assert mount.endswith(",target=/workspace,readonly")


def test_container_environment_rejects_credentials_before_launch(
    tmp_path: Path,
    fake_engine: tuple[Path, Path],
) -> None:
    engine, calls_path = fake_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with (
        ContainerRunner(
            workspace,
            image="evalyn-tests:local",
            engine=str(engine),
        ) as runner,
        pytest.raises(ValueError, match="not permitted"),
    ):
        runner.run(["pytest"], env={"OPENAI_API_KEY": "secret"})

    assert not calls_path.exists()


def test_cleanup_failure_turns_success_into_infrastructure_error(
    tmp_path: Path,
    fake_engine: tuple[Path, Path],
) -> None:
    engine, _ = fake_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (engine.parent / "fail-cleanup").touch()

    with ContainerRunner(
        workspace,
        image="evalyn-tests:local",
        engine=str(engine),
    ) as runner:
        result = runner.run(["pytest"])

    assert result.exit_code == 0
    assert not result.succeeded
    assert result.error is not None
    assert "cleanup failed" in result.error.casefold()


def test_explicit_engine_resolution_requires_executable(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "not-installed"

    with pytest.raises(RuntimeUnavailableError):
        resolve_container_engine(str(missing))


def test_docker_context_resolution_forwards_only_the_daemon_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return BoundedProcessResult(
            tuple(argv),
            0,
            b"unix:///tmp/docker.sock\n",
            b"",
        )

    monkeypatch.setenv("HOME", "/tmp/trusted-home")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr("evalyn.execution.container.run_bounded_process", fake_run)

    result = _resolve_container_client_environment("docker", "/usr/bin/docker")

    assert result == {"DOCKER_HOST": "unix:///tmp/docker.sock"}
    assert captured["argv"][:3] == ["/usr/bin/docker", "context", "inspect"]
    assert captured["env"] == {"HOME": "/tmp/trusted-home", "PATH": os.environ["PATH"]}


def test_absolute_custom_engine_skips_context_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(
        "evalyn.execution.container.run_bounded_process",
        lambda *args, **kwargs: pytest.fail("context discovery should not run"),
    )

    assert _resolve_container_client_environment("/tmp/docker", "/tmp/docker") == {}


@pytest.mark.parametrize(
    ("engine", "variable", "endpoint"),
    [
        ("docker", "DOCKER_HOST", "tcp://builder.example.test:2376"),
        ("docker", "DOCKER_HOST", "ssh://builder.example.test"),
        ("podman", "CONTAINER_HOST", "tcp://builder.example.test:2376"),
        ("podman", "CONTAINER_HOST", "ssh://builder.example.test/run/podman.sock"),
    ],
)
def test_remote_container_endpoints_require_explicit_operator_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    variable: str,
    endpoint: str,
) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("CONTAINER_HOST", raising=False)
    monkeypatch.setenv(variable, endpoint)

    with pytest.raises(ValueError, match="explicit operator opt-in"):
        _resolve_container_client_environment(engine, f"/usr/bin/{engine}")

    assert (
        _resolve_container_client_environment(
            engine,
            f"/usr/bin/{engine}",
            allow_remote_endpoint=True,
        )[variable]
        == endpoint
    )


def test_remote_endpoint_scope_is_recorded_without_persisting_the_endpoint(
    tmp_path: Path,
    fake_engine: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _calls = fake_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    endpoint = "tcp://builder.example.test:2376"
    monkeypatch.setenv("DOCKER_HOST", endpoint)

    with ContainerRunner(
        workspace,
        image="evalyn-tests:local",
        engine=str(engine),
        allow_remote_endpoint=True,
    ) as runner:
        payload = runner.environment_fingerprint.to_dict()

    assert payload["endpoint_scope"] == "remote"
    assert endpoint not in json.dumps(payload)


def test_allowlisted_setup_secret_uses_ephemeral_env_file_not_argv(
    tmp_path: Path,
    fake_engine: tuple[Path, Path],
) -> None:
    engine, calls_path = fake_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with ContainerRunner(
        workspace,
        image="evalyn-tests:local",
        engine=str(engine),
        allowed_secret_env_names=("NPM_TOKEN",),
    ) as runner:
        result = runner.run(["npm", "ci"], secret_env={"NPM_TOKEN": "top-secret"})
        invocation = runner.last_invocation

    assert result.succeeded
    assert invocation is not None
    serialized = calls_path.read_text(encoding="utf-8")
    assert "top-secret" not in serialized
    env_file = Path(invocation.argv[invocation.argv.index("--env-file") + 1])
    assert not env_file.exists()


def test_setup_secret_requires_explicit_name_allowlist(
    tmp_path: Path,
    fake_engine: tuple[Path, Path],
) -> None:
    engine, calls_path = fake_engine
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with (
        ContainerRunner(
            workspace,
            image="evalyn-tests:local",
            engine=str(engine),
        ) as runner,
        pytest.raises(ValueError, match="not allowlisted"),
    ):
        runner.run(["npm", "ci"], secret_env={"NPM_TOKEN": "secret"})

    assert not calls_path.exists()
