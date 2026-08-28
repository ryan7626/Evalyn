"""Execution-runtime abstraction for host and OCI-container commands."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from evalyn.execution.errors import RuntimeUnavailableError
from evalyn.execution.process import validate_environment_overrides


@dataclass(frozen=True)
class RuntimeInvocation:
    """A command transformed for a concrete runtime."""

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict)
    containerized: bool = False
    container_name: str | None = None


class ExecutionRuntime(Protocol):
    """Transform a workspace command into a host-launchable invocation."""

    def prepare(
        self,
        argv: Sequence[str],
        *,
        workspace_root: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> RuntimeInvocation: ...


class HostRuntime:
    """Run directly on the host while preserving workspace containment."""

    def prepare(
        self,
        argv: Sequence[str],
        *,
        workspace_root: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> RuntimeInvocation:
        if env_file is not None:
            raise ValueError("HostRuntime does not accept container environment files")
        command = _validate_argv(argv)
        root, working_dir = _validate_workspace_cwd(workspace_root, cwd)
        return RuntimeInvocation(
            argv=command,
            cwd=working_dir,
            env=_validate_env(env),
            containerized=False,
        )


class OciContainerRuntime:
    """Build a restrictive ``docker run``/``podman run`` invocation.

    Networking is disabled by default, Linux capabilities are dropped,
    privilege escalation is disabled, and the container receives only the
    explicitly supplied environment. Callers can make the workspace bind mount
    read-only after a separate trusted setup phase.
    """

    def __init__(
        self,
        image: str,
        *,
        engine: str = "docker",
        network: str = "none",
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 256,
        read_only_root: bool = True,
        tmpfs_size: str = "512m",
        pull_policy: str = "never",
        run_as_current_user: bool = True,
        workspace_read_only: bool = False,
        container_workspace: str = "/workspace",
    ) -> None:
        if (
            not image
            or image.startswith("-")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,511}", image)
        ):
            raise ValueError(f"Invalid container image: {image!r}")
        if not engine or engine.startswith("-") or "\x00" in engine:
            raise ValueError(f"Invalid container engine: {engine!r}")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", network):
            raise ValueError(f"Invalid container network: {network!r}")
        if not container_workspace.startswith("/"):
            raise ValueError("container_workspace must be absolute")
        if pids_limit <= 0:
            raise ValueError("pids_limit must be positive")
        if pull_policy not in {"always", "missing", "never"}:
            raise ValueError("pull_policy must be always, missing, or never")

        self.image = image
        self.engine = engine
        self.network = network
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.read_only_root = read_only_root
        self.tmpfs_size = tmpfs_size
        self.pull_policy = pull_policy
        self.run_as_current_user = run_as_current_user
        self.workspace_read_only = workspace_read_only
        self.container_workspace = container_workspace.rstrip("/") or "/"

    @property
    def is_available(self) -> bool:
        return shutil.which(self.engine) is not None

    def require_available(self) -> None:
        if not self.is_available:
            raise RuntimeUnavailableError(
                f"Container runtime executable is unavailable: {self.engine}"
            )

    def prepare(
        self,
        argv: Sequence[str],
        *,
        workspace_root: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> RuntimeInvocation:
        command = _validate_argv(argv)
        root, working_dir = _validate_workspace_cwd(workspace_root, cwd)
        container_env = _validate_env(env)
        container_env.setdefault("HOME", "/tmp/evalyn-home")
        relative_cwd = working_dir.relative_to(root)
        container_cwd = self.container_workspace
        if relative_cwd.parts:
            container_cwd += "/" + relative_cwd.as_posix()

        container_name = f"evalyn-{uuid.uuid4().hex[:12]}"
        invocation: list[str] = [
            self.engine,
            "run",
            "--rm",
            "--init",
            "--name",
            container_name,
            "--pull",
            self.pull_policy,
            "--network",
            self.network,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
        ]
        if self.read_only_root:
            invocation.append("--read-only")
        workspace_mount = (
            f"type=bind,source={_portable_bind_source(root)},target={self.container_workspace}"
        )
        if self.workspace_read_only:
            workspace_mount += ",readonly"
        invocation.extend(
            [
                "--tmpfs",
                f"/tmp:rw,nosuid,nodev,size={self.tmpfs_size}",
                "--mount",
                workspace_mount,
                "--workdir",
                container_cwd,
            ]
        )

        if self.run_as_current_user and hasattr(os, "getuid") and hasattr(os, "getgid"):
            invocation.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        for key in sorted(container_env):
            invocation.extend(["--env", f"{key}={container_env[key]}"])
        if env_file is not None:
            invocation.extend(["--env-file", str(_validate_env_file(env_file))])

        invocation.append(self.image)
        invocation.extend(command)
        return RuntimeInvocation(
            argv=tuple(invocation),
            cwd=root,
            # This environment is for the OCI client, not the container. The
            # CommandRunner will add its minimal PATH and sandbox HOME.
            env={},
            containerized=True,
            container_name=container_name,
        )


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments, not a shell string")
    command = tuple(str(arg) for arg in argv)
    if not command or not command[0]:
        raise ValueError("argv must contain a non-empty executable")
    if any("\x00" in arg for arg in command):
        raise ValueError("argv cannot contain NUL bytes")
    return command


def _validate_workspace_cwd(
    workspace_root: Path,
    cwd: Path | None,
) -> tuple[Path, Path]:
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Workspace root is not a directory: {root}")
    working_dir = Path(cwd).resolve() if cwd is not None else root
    try:
        working_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Working directory escapes workspace: {working_dir}") from exc
    if not working_dir.is_dir():
        raise ValueError(f"Working directory is not a directory: {working_dir}")
    return root, working_dir


def _validate_env(env: Mapping[str, str] | None) -> dict[str, str]:
    return validate_environment_overrides(env)


def _validate_env_file(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise ValueError("Container environment file must be an absolute regular file")
    resolved = candidate.resolve(strict=True)
    if resolved.stat().st_mode & 0o077:
        raise ValueError("Container environment file permissions must be owner-only")
    return resolved


def _portable_bind_source(root: Path) -> Path:
    """Retain a daemon-visible spelling for macOS's private path aliases."""
    resolved = root.resolve(strict=True)
    for canonical_prefix, visible_prefix in (
        (Path("/private/var"), Path("/var")),
        (Path("/private/tmp"), Path("/tmp")),
    ):
        try:
            relative = resolved.relative_to(canonical_prefix)
        except ValueError:
            continue
        visible = visible_prefix / relative
        try:
            if visible.exists() and os.path.samefile(visible, resolved):
                if "," in str(visible):
                    raise ValueError("Workspace path cannot contain a comma")
                return visible
        except OSError:
            continue
    if "," in str(resolved):
        raise ValueError("Workspace path cannot contain a comma")
    return resolved
