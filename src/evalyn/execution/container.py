"""Docker/Podman workload runner with no implicit host fallback."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from evalyn.execution.bounded import run_bounded_process
from evalyn.execution.errors import RuntimeUnavailableError
from evalyn.execution.image_identity import (
    EnvironmentFingerprint,
    ImageIdentityError,
    ImageIdentityResolution,
    parse_image_identity_inspection,
)
from evalyn.execution.process import CommandResult, CommandRunner
from evalyn.execution.runtime import OciContainerRuntime, RuntimeInvocation

_NOT_FOUND_MARKERS = (
    "no such container",
    "no container with name or id",
    "container does not exist",
)


class ContainerRunner:
    """Execute structured workload argv exclusively inside Docker or Podman.

    Engine discovery is fail-closed: if neither approved OCI client is present,
    construction raises :class:`RuntimeUnavailableError`. Workloads are never
    retried directly on the host. The OCI client itself is launched through
    :class:`CommandRunner`, so timeout cancellation also terminates its process
    tree. A named-container cleanup is attempted in ``finally`` to cover daemon
    workloads left behind when the attached client is killed.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        image: str,
        engine: str | None = None,
        network: str = "none",
        memory: str = "2g",
        cpus: str = "2",
        pids_limit: int = 256,
        pull_policy: str = "never",
        sandbox_home: Path | None = None,
        max_output_bytes: int = 64 * 1024,
        default_timeout_seconds: float | None = 300,
        termination_grace_seconds: float = 1.0,
        cleanup_timeout_seconds: float = 15.0,
        allowed_secret_env_names: Sequence[str] = (),
        workspace_read_only: bool = False,
        allow_remote_endpoint: bool = False,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.engine = resolve_container_engine(engine)
        self._client_env = _resolve_container_client_environment(
            engine,
            self.engine,
            allow_remote_endpoint=allow_remote_endpoint,
        )
        self._endpoint_scope = (
            "remote"
            if any(value.startswith(("tcp://", "ssh://")) for value in self._client_env.values())
            else "local"
        )
        self.runtime = OciContainerRuntime(
            image,
            engine=self.engine,
            network=network,
            memory=memory,
            cpus=cpus,
            pids_limit=pids_limit,
            pull_policy=pull_policy,
            workspace_read_only=workspace_read_only,
        )
        self.image = image
        self.cleanup_timeout_seconds = max(1.0, float(cleanup_timeout_seconds))
        self.allowed_secret_env_names = frozenset(
            _validate_environment_name(name) for name in allowed_secret_env_names
        )
        self._client = CommandRunner(
            self.workspace_root,
            sandbox_home=sandbox_home,
            max_output_bytes=max_output_bytes,
            default_timeout_seconds=default_timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
        )
        self.last_invocation: RuntimeInvocation | None = None
        self.environment_fingerprint = self._environment_fingerprint()
        self._successful_run_generation = 0
        self._fingerprint_generation = -1

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str | None = None,
        timeout_seconds: float | None = None,
        env: Mapping[str, str] | None = None,
        secret_env: Mapping[str, str] | None = None,
        full_log_path: Path | str | None = None,
    ) -> CommandResult:
        """Run one command in the configured image and force cleanup afterward."""
        working_dir: Path | None
        if cwd is None:
            working_dir = None
        else:
            working_dir = Path(cwd)
            if not working_dir.is_absolute():
                working_dir = self.workspace_root / working_dir

        secret_file = self._write_secret_environment(secret_env)
        try:
            invocation = self.runtime.prepare(
                argv,
                workspace_root=self.workspace_root,
                cwd=working_dir,
                env=env,
                env_file=secret_file,
            )
            self.last_invocation = invocation
            result: CommandResult | None = None
            cleanup_error: str | None = None
            try:
                result = self._client.run(
                    invocation.argv,
                    cwd=invocation.cwd,
                    timeout_seconds=timeout_seconds,
                    env={**invocation.env, **self._client_env},
                    full_log_path=full_log_path,
                )
            finally:
                if invocation.container_name:
                    cleanup_error = self._force_remove(invocation.container_name)

            assert result is not None
            if cleanup_error is None:
                final_result = result
            else:
                error = (
                    cleanup_error if result.error is None else f"{result.error}; {cleanup_error}"
                )
                final_result = replace(result, error=error)
            if final_result.succeeded:
                self._successful_run_generation += 1
                self._fingerprint_generation = -1
                self.environment_fingerprint = self._environment_fingerprint()
            return final_result
        finally:
            if secret_file is not None:
                secret_file.unlink(missing_ok=True)

    def close(self) -> None:
        self._client.close()

    def resolve_environment_fingerprint(self) -> EnvironmentFingerprint:
        """Resolve and cache the image ID after a successful OCI workload.

        Inspection uses the same sanitized client and trusted local endpoint as
        the workload. A failed, timed-out, truncated, or malformed inspection
        produces an incomplete fingerprint with a stable error code.
        """
        if self._successful_run_generation == 0:
            resolution = ImageIdentityResolution.incomplete(ImageIdentityError.NO_SUCCESSFUL_LAUNCH)
            self.environment_fingerprint = self._environment_fingerprint(resolution)
            return self.environment_fingerprint
        if (
            self._fingerprint_generation == self._successful_run_generation
            and self.environment_fingerprint.complete
        ):
            return self.environment_fingerprint

        inspection = self._client.run(
            [
                self.engine,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                self.image,
            ],
            cwd=self.workspace_root,
            timeout_seconds=self.cleanup_timeout_seconds,
            env=self._client_env,
        )
        resolution = parse_image_identity_inspection(inspection)
        self.environment_fingerprint = self._environment_fingerprint(resolution)
        if self.environment_fingerprint.complete:
            self._fingerprint_generation = self._successful_run_generation
        return self.environment_fingerprint

    def _environment_fingerprint(
        self,
        resolution: ImageIdentityResolution | None = None,
    ) -> EnvironmentFingerprint:
        return replace(
            self.runtime.environment_fingerprint(resolution),
            endpoint_scope=self._endpoint_scope,
        )

    def __enter__(self) -> ContainerRunner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _force_remove(self, container_name: str) -> str | None:
        cleanup = self._client.run(
            [self.engine, "rm", "--force", container_name],
            cwd=self.workspace_root,
            timeout_seconds=self.cleanup_timeout_seconds,
            env=self._client_env,
        )
        if cleanup.succeeded:
            return None
        combined = f"{cleanup.stdout}\n{cleanup.stderr}".casefold()
        if any(marker in combined for marker in _NOT_FOUND_MARKERS):
            return None
        detail = cleanup.error or cleanup.stderr.strip() or cleanup.stdout.strip()
        return f"Container cleanup failed for {container_name}: {detail or 'unknown error'}"

    def _write_secret_environment(self, secret_env: Mapping[str, str] | None) -> Path | None:
        if not secret_env:
            return None
        normalized: dict[str, str] = {}
        for raw_name, value in secret_env.items():
            name = _validate_environment_name(raw_name)
            if name not in self.allowed_secret_env_names:
                raise ValueError(f"Secret environment variable is not allowlisted: {name}")
            if not isinstance(value, str):
                raise TypeError(f"Secret environment value for {name} must be a string")
            if "\x00" in value or "\n" in value or "\r" in value:
                raise ValueError(f"Secret environment value for {name} is not env-file safe")
            if len(value.encode("utf-8")) > 64 * 1024:
                raise ValueError(f"Secret environment value for {name} is too large")
            normalized[name] = value
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="setup-env-",
                dir=self._client.sandbox_home,
                delete=False,
            ) as handle:
                path = Path(handle.name).resolve()
                os.chmod(path, 0o600)
                for name in sorted(normalized):
                    handle.write(f"{name}={normalized[name]}\n")
            return path
        except BaseException:
            if path is not None:
                path.unlink(missing_ok=True)
            raise


def resolve_container_engine(preferred: str | None = None) -> str:
    """Return an approved OCI client path or fail without a host fallback."""
    candidates = (preferred,) if preferred else ("docker", "podman")
    for candidate in candidates:
        if candidate is None or not candidate or "\x00" in candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved is not None and os.access(resolved, os.X_OK):
            return str(Path(resolved).resolve())
    requested = preferred or "docker or podman"
    raise RuntimeUnavailableError(
        f"Required container runtime is unavailable: {requested}; "
        "Evalyn will not execute the workload on the host"
    )


def _resolve_container_client_environment(
    requested_engine: str | None,
    resolved_engine: str,
    *,
    allow_remote_endpoint: bool = False,
) -> dict[str, str]:
    """Preserve only the local daemon endpoint, never the user's registry auth.

    ``CommandRunner`` intentionally replaces HOME and XDG configuration. Docker
    Desktop and Colima select their socket through that configuration, so the
    trusted client endpoint is resolved once before entering the sanitized
    process boundary. Explicit absolute engine paths (used by tests and custom
    wrappers) are left untouched.
    """
    name = Path(resolved_engine).name
    if name == "docker":
        explicit = os.environ.get("DOCKER_HOST")
        if explicit:
            return {
                "DOCKER_HOST": _validate_container_endpoint(
                    explicit,
                    allow_remote=allow_remote_endpoint,
                )
            }
        if requested_engine not in {None, "docker"}:
            return {}
        discovery_env = {
            key: value
            for key in ("PATH", "HOME", "DOCKER_CONFIG", "DOCKER_CONTEXT")
            if (value := os.environ.get(key)) and "\x00" not in value
        }
        try:
            completed = run_bounded_process(
                [
                    resolved_engine,
                    "context",
                    "inspect",
                    "--format",
                    "{{.Endpoints.docker.Host}}",
                ],
                timeout=10,
                env=discovery_env,
                max_stdout_bytes=64 * 1024,
                max_stderr_bytes=64 * 1024,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if completed.returncode != 0 or completed.stdout_truncated or completed.stderr_truncated:
            return {}
        try:
            raw_endpoint = completed.stdout.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            return {}
        endpoint = _validate_container_endpoint(
            raw_endpoint,
            allow_remote=allow_remote_endpoint,
        )
        return {"DOCKER_HOST": endpoint}

    if name == "podman":
        result: dict[str, str] = {}
        explicit = os.environ.get("CONTAINER_HOST")
        if explicit:
            result["CONTAINER_HOST"] = _validate_container_endpoint(
                explicit,
                allow_remote=allow_remote_endpoint,
            )
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if runtime_dir and "\x00" not in runtime_dir and "\n" not in runtime_dir:
            result["XDG_RUNTIME_DIR"] = runtime_dir
        return result
    return {}


def _validate_container_endpoint(value: str, *, allow_remote: bool = False) -> str:
    endpoint = value.strip()
    if (
        not endpoint
        or endpoint != value
        or "\x00" in endpoint
        or "\n" in endpoint
        or not endpoint.startswith(("unix://", "npipe://", "tcp://", "ssh://"))
    ):
        raise ValueError("Container endpoint must be a supported structured URI")
    if endpoint.startswith(("tcp://", "ssh://")) and not allow_remote:
        raise ValueError(
            "Remote container endpoints are disabled; explicit operator opt-in is required"
        )
    return endpoint


def _validate_environment_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid environment variable name: {value!r}")
    return value
