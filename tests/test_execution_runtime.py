"""Tests for host/container execution-runtime preparation."""

from pathlib import Path

import pytest

from evalyn.execution.errors import RuntimeUnavailableError
from evalyn.execution.runtime import HostRuntime, OciContainerRuntime, _portable_bind_source


def test_host_runtime_preserves_structured_command_and_contained_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    cwd = workspace / "src"
    cwd.mkdir(parents=True)

    invocation = HostRuntime().prepare(
        ["python", "-m", "pytest", "tests/unit test.py"],
        workspace_root=workspace,
        cwd=cwd,
        env={"MODE": "test"},
    )

    assert invocation.argv == ("python", "-m", "pytest", "tests/unit test.py")
    assert invocation.cwd == cwd.resolve()
    assert invocation.env == {"MODE": "test"}
    assert not invocation.containerized


def test_host_runtime_rejects_shell_string_and_outside_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(TypeError, match="shell string"):
        HostRuntime().prepare("pytest; touch sentinel", workspace_root=workspace)
    with pytest.raises(ValueError, match="escapes workspace"):
        HostRuntime().prepare(
            ["pytest"],
            workspace_root=workspace,
            cwd=tmp_path,
        )


def test_oci_runtime_builds_restrictive_invocation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    cwd = workspace / "frontend"
    cwd.mkdir(parents=True)
    runtime = OciContainerRuntime("python:3.12-slim", engine="podman")

    invocation = runtime.prepare(
        ["python", "-m", "pytest"],
        workspace_root=workspace,
        cwd=cwd,
        env={"MODE": "test"},
    )

    argv = invocation.argv
    assert argv[:2] == ("podman", "run")
    assert "--rm" in argv
    assert "--init" in argv
    assert "--read-only" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--workdir") + 1] == "/workspace/frontend"
    assert f"type=bind,source={_portable_bind_source(workspace)},target=/workspace" in argv
    assert "MODE=test" in argv
    assert "HOME=/tmp/evalyn-home" in argv
    assert argv[-4:] == ("python:3.12-slim", "python", "-m", "pytest")
    assert invocation.cwd == workspace.resolve()
    assert invocation.containerized


def test_container_availability_has_explicit_failure(monkeypatch) -> None:
    monkeypatch.setattr("evalyn.execution.runtime.shutil.which", lambda _: None)
    runtime = OciContainerRuntime("python:3.12", engine="missing-engine")

    assert not runtime.is_available
    with pytest.raises(RuntimeUnavailableError, match="missing-engine"):
        runtime.require_available()


def test_oci_runtime_accepts_registry_and_digest_image(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = "ghcr.io/example/evalyn@sha256:" + "a" * 64

    invocation = OciContainerRuntime(image).prepare(
        ["python", "--version"], workspace_root=workspace
    )

    assert image in invocation.argv


def test_oci_runtime_can_mount_workspace_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    invocation = OciContainerRuntime("python:3.12-slim", workspace_read_only=True).prepare(
        ["python", "--version"], workspace_root=workspace
    )

    mount = invocation.argv[invocation.argv.index("--mount") + 1]
    assert mount.endswith(",target=/workspace,readonly")


def test_runtime_rejects_invalid_environment_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="environment variable"):
        HostRuntime().prepare(
            ["pytest"],
            workspace_root=workspace,
            env={"BAD-NAME": "value"},
        )


def test_oci_runtime_accepts_owner_only_environment_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env_file = tmp_path / "setup.env"
    env_file.write_text("NPM_TOKEN=secret\n", encoding="utf-8")
    env_file.chmod(0o600)

    invocation = OciContainerRuntime("node:22", engine="docker").prepare(
        ["npm", "ci"],
        workspace_root=workspace,
        env_file=env_file.resolve(),
    )

    assert invocation.argv[invocation.argv.index("--env-file") + 1] == str(env_file.resolve())
    assert "secret" not in " ".join(invocation.argv)


def test_oci_runtime_rejects_world_readable_environment_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env_file = tmp_path / "setup.env"
    env_file.write_text("NPM_TOKEN=secret\n", encoding="utf-8")
    env_file.chmod(0o644)

    with pytest.raises(ValueError, match="owner-only"):
        OciContainerRuntime("node:22", engine="docker").prepare(
            ["npm", "ci"],
            workspace_root=workspace,
            env_file=env_file.resolve(),
        )


def test_portable_bind_source_preserves_canonical_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    mount_source = _portable_bind_source(workspace)

    assert mount_source.exists()
    assert mount_source.resolve() == workspace.resolve()
