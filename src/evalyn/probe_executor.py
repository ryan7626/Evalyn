"""Container-only materialization and execution of generated probes."""

from __future__ import annotations

import os
import re
import secrets
import time
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from evalyn.adapters.base import ProbeArtifact, TestAdapter
from evalyn.engine.models import ExecutionResult, ExecutionStatus, Revision
from evalyn.execution import (
    ContainerRunner,
    EnvironmentFingerprint,
    WorkspacePathPolicy,
)
from evalyn.execution.structured_results import (
    PYTEST_REPORTER_SOURCE,
    parse_structured_result,
)
from evalyn.run_store import RunStore

DEFAULT_IMAGES = {
    "pytest": "python:3.12-slim",
    "vitest": "node:22-bookworm-slim",
}


class ProbeCollectionError(RuntimeError):
    """A generated probe failed a bounded framework collection preflight."""

    def __init__(self, code: str, feedback: str) -> None:
        self.code = code
        self.feedback = feedback[:2_000]
        super().__init__(self.feedback)


class ContainerProbeExecutor:
    """Prepare revision dependencies and execute probes without a host fallback."""

    def __init__(
        self,
        adapter: TestAdapter,
        *,
        engine: str | None,
        run_dir: Path,
        image: str | None = None,
        setup_argv: tuple[str, ...] = (),
        setup_env_names: tuple[str, ...] = (),
        timeout_seconds: int = 300,
        allow_remote_endpoint: bool = False,
        expected_image_identity: str | None = None,
        expected_endpoint_scope: str | None = None,
    ) -> None:
        if adapter.name not in DEFAULT_IMAGES:
            raise ValueError(f"No container image for adapter {adapter.name!r}")
        self.adapter = adapter
        self.engine = engine
        self.run_dir = run_dir.resolve()
        self.image = image or DEFAULT_IMAGES[adapter.name]
        self.execution_image = self.image
        self.setup_argv = setup_argv
        self.setup_env_names = setup_env_names
        self.allow_remote_endpoint = allow_remote_endpoint
        if (
            expected_image_identity is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_identity) is None
        ):
            raise ValueError("expected_image_identity must be a full sha256 OCI image ID")
        if expected_endpoint_scope not in {None, "local", "remote"}:
            raise ValueError("expected_endpoint_scope must be local or remote")
        self.expected_image_identity = expected_image_identity
        self.expected_endpoint_scope = expected_endpoint_scope
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.budget_seconds = timeout_seconds
        self.command_timeout_seconds = min(300, timeout_seconds)
        self._deadline = time.monotonic() + timeout_seconds
        self._prepared: set[Path] = set()
        self._setup_errors: dict[Path, str] = {}
        self._labels: dict[Path, str] = {}
        self._environment_fingerprint: EnvironmentFingerprint | None = None
        self._collection_validated: set[str] = set()
        self._attested_workspaces: set[Path] = set()

    @property
    def setup_errors(self) -> dict[Path, str]:
        return dict(self._setup_errors)

    @property
    def environment_fingerprint(self) -> dict[str, object] | None:
        fingerprint = self._environment_fingerprint
        return fingerprint.to_dict() if fingerprint is not None else None

    def prepare(self, revisions: Iterable[tuple[str, Revision]]) -> None:
        for label, revision in revisions:
            workspace = revision.workspace.resolve()
            self._labels[workspace] = label
            if workspace in self._prepared or workspace in self._setup_errors:
                continue
            try:
                self._prepare_workspace(workspace)
                self._prepared.add(workspace)
            except Exception as exc:
                self._setup_errors[workspace] = f"{type(exc).__name__}: {exc}"

    def execute(
        self,
        revision: Revision,
        artifact: ProbeArtifact,
        *,
        repetition: int,
    ) -> ExecutionResult:
        workspace = revision.workspace.resolve()
        if workspace not in self._prepared and workspace not in self._setup_errors:
            self.prepare((("revision", revision),))
        if workspace in self._setup_errors:
            return ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=-1,
                stderr=f"Dependency setup failed: {self._setup_errors[workspace]}",
            )

        try:
            target = self._materialize(workspace, artifact)
            if self.adapter.name == "pytest":
                self._materialize_pytest_reporter(workspace)
        except Exception as exc:
            return ExecutionResult(
                status=ExecutionStatus.INVALID_PROBE,
                exit_code=-1,
                stderr=f"Probe materialization failed: {type(exc).__name__}: {exc}",
            )

        label = self._labels.get(workspace, "revision")
        log_path = self.run_dir / "logs" / label / artifact.probe_id / f"run-{repetition}.log"
        environment = self._probe_environment()
        try:
            with ContainerRunner(
                workspace,
                image=self.execution_image,
                engine=self.engine,
                network="none",
                pull_policy="missing",
                workspace_read_only=True,
                default_timeout_seconds=self.remaining_timeout(),
                allow_remote_endpoint=self.allow_remote_endpoint,
            ) as runner:
                result = runner.run(
                    artifact.command,
                    env=environment,
                    timeout_seconds=self.remaining_timeout(),
                    full_log_path=log_path,
                )
        except TimeoutError as exc:
            return ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=-1,
                stderr=str(exc),
                metadata={
                    "log_path": str(log_path),
                    "probe_path": str(target),
                    **self._environment_metadata(),
                },
            )

        metadata = {
            "log_path": str(log_path),
            "probe_path": str(target),
            **self._environment_metadata(),
        }
        if result.timed_out:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=-1,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        if result.error is not None:
            return ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=result.exit_code if result.exit_code is not None else -1,
                stdout=result.stdout,
                stderr=result.error,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        if result.output_truncated:
            return ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=result.exit_code if result.exit_code is not None else -1,
                stdout=result.stdout,
                stderr=(
                    result.stderr + "\nBounded test output was truncated before it could be trusted"
                ).strip(),
                duration_seconds=result.duration_seconds,
                metadata={**metadata, "result_protocol": "truncated"},
            )
        structured = parse_structured_result(self.adapter.name, result.stdout)
        if structured is None:
            return ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=result.exit_code if result.exit_code is not None else -1,
                stdout=result.stdout,
                stderr=(
                    result.stderr + "\nMachine-readable test result was missing or malformed"
                ).strip(),
                duration_seconds=result.duration_seconds,
                metadata={**metadata, "result_protocol": "missing"},
            )
        metadata.update(structured.metadata())
        if label == "base" and not structured.collection_errors and structured.total >= 1:
            self._collection_validated.add(artifact.probe_id)
        if result.exit_code == 0:
            if (
                structured.failed
                or structured.collection_errors
                or structured.total < 1
                or structured.passed < 1
            ):
                return ExecutionResult(
                    status=ExecutionStatus.INVALID_PROBE,
                    exit_code=0,
                    stdout=result.stdout,
                    stderr=(
                        result.stderr + "\nStructured result contradicted a successful process or "
                        "ran no passing tests"
                    ).strip(),
                    duration_seconds=result.duration_seconds,
                    metadata=metadata,
                )
            return ExecutionResult.passed(
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        output = f"{result.stdout}\n{result.stderr}"
        candidate_collection_failure = (
            label == "candidate"
            and artifact.probe_id in self._collection_validated
            and bool(structured.collection_errors)
        )
        if candidate_collection_failure or structured.failed:
            status = ExecutionStatus.FAILED
        elif (
            structured.collection_errors
            or structured.total < 1
            or _looks_invalid(self.adapter.name, output)
        ):
            status = ExecutionStatus.INVALID_PROBE
        else:
            status = ExecutionStatus.INFRA_ERROR
        return ExecutionResult(
            status=status,
            exit_code=result.exit_code if result.exit_code is not None else 1,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
            failure_signature=(
                structured.failure_signature
                or _stable_failure_signature(self.adapter.name, result.stdout, result.stderr)
                if status == ExecutionStatus.FAILED
                else ""
            ),
            metadata=metadata,
        )

    def _prepare_workspace(self, workspace: Path) -> None:
        commands = [self.setup_argv] if self.setup_argv else self._default_setup(workspace)
        # Warm the declared image, resolve its immutable ID, and pin every
        # meaningful setup/probe execution to that ID. Inspecting only after a
        # setup command would leave a tag-mutation race between execution and
        # attribution.
        self._resolve_probe_environment(workspace)
        if not commands:
            return
        with ContainerRunner(
            workspace,
            image=self.execution_image,
            engine=self.engine,
            network="bridge",
            pull_policy="missing",
            default_timeout_seconds=self.remaining_timeout(),
            allowed_secret_env_names=self.setup_env_names,
            allow_remote_endpoint=self.allow_remote_endpoint,
        ) as runner:
            setup_secrets = self._setup_secret_environment()
            for index, command in enumerate(commands, 1):
                result = runner.run(
                    command,
                    secret_env=setup_secrets,
                    timeout_seconds=self.remaining_timeout(),
                    full_log_path=self.run_dir
                    / "logs"
                    / "setup"
                    / workspace.name
                    / f"step-{index}.log",
                )
                if not result.succeeded:
                    detail = result.error or result.stderr or result.stdout
                    raise RuntimeError(f"{' '.join(command)} failed: {detail.strip()}")

    def preflight(self, revision: Revision, source: str) -> None:
        """Collect/import a generated probe on the passing base without executing it."""

        artifact = self.adapter.build_artifact(source)
        workspace = revision.workspace.resolve()
        try:
            self._materialize(workspace, artifact)
            if self.adapter.name == "pytest":
                self._materialize_pytest_reporter(workspace)
            command = self._collection_command(artifact)
            log_path = self.run_dir / "logs" / "preflight" / f"{artifact.probe_id}.log"
            with ContainerRunner(
                workspace,
                image=self.execution_image,
                engine=self.engine,
                network="none",
                pull_policy="missing",
                workspace_read_only=True,
                default_timeout_seconds=self.remaining_timeout(),
                allow_remote_endpoint=self.allow_remote_endpoint,
            ) as runner:
                result = runner.run(
                    command,
                    env=self._probe_environment(),
                    timeout_seconds=self.remaining_timeout(),
                    full_log_path=log_path,
                )
        except TimeoutError as exc:
            raise ProbeCollectionError("collection_timeout", str(exc)) from exc
        except ProbeCollectionError:
            raise
        except Exception as exc:
            raise ProbeCollectionError(
                "collection_infrastructure", f"{type(exc).__name__}: {exc}"
            ) from exc
        if result.timed_out:
            raise ProbeCollectionError("collection_timeout", "Framework collection timed out")
        if result.error is not None:
            raise ProbeCollectionError(
                "collection_infrastructure",
                "Framework collection infrastructure failed on the base revision.",
            )
        if result.exit_code != 0:
            raise ProbeCollectionError(
                "collection_failed",
                "The probe did not collect or import on the base revision.",
            )
        self._collection_validated.add(artifact.probe_id)

    def _resolve_probe_environment(self, workspace: Path) -> None:
        if workspace in self._attested_workspaces:
            return
        token = secrets.token_bytes(32)
        digest = sha256(token).hexdigest()
        relative = f".evalyn/probes/support/workspace-attestation-{digest[:20]}"
        target = WorkspacePathPolicy(workspace).resolve_write(
            relative,
            protect_sensitive=False,
        )
        RunStore.write_bytes(target, token, mode=0o444)
        container_path = f"/workspace/{relative}"
        if self.adapter.name == "pytest":
            command = (
                "python",
                "-c",
                "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path("
                + repr(container_path)
                + ").read_bytes()).hexdigest())",
            )
        else:
            command = (
                "node",
                "-e",
                "const c=require('crypto'),f=require('fs');console.log(c.createHash('sha256')"
                ".update(f.readFileSync(" + repr(container_path) + ")).digest('hex'))",
            )
        try:
            with ContainerRunner(
                workspace,
                image=self.execution_image,
                engine=self.engine,
                network="none",
                pull_policy="missing",
                workspace_read_only=True,
                default_timeout_seconds=self.remaining_timeout(),
                allow_remote_endpoint=self.allow_remote_endpoint,
            ) as runner:
                result = runner.run(command, timeout_seconds=self.remaining_timeout())
                if (
                    not result.succeeded
                    or result.output_truncated
                    or result.stdout.strip() != digest
                ):
                    detail = result.error or result.stderr or "workspace attestation mismatch"
                    raise RuntimeError("container workspace attestation failed: " + detail.strip())
                if self._environment_fingerprint is None:
                    fingerprint = runner.resolve_environment_fingerprint()
                    self._adopt_probe_fingerprint(fingerprint)
            self._attested_workspaces.add(workspace)
        finally:
            target.unlink(missing_ok=True)

    def _adopt_probe_fingerprint(self, fingerprint: EnvironmentFingerprint) -> None:
        if not fingerprint.complete or fingerprint.resolved_image_identity is None:
            error = fingerprint.image_resolution.error
            raise RuntimeError(
                "container image identity could not be resolved: "
                + (error.value if error is not None else "unknown")
            )
        if (
            self.expected_image_identity is not None
            and fingerprint.resolved_image_identity != self.expected_image_identity
        ):
            raise RuntimeError("resolved container image does not match the saved environment")
        if (
            self.expected_endpoint_scope is not None
            and fingerprint.endpoint_scope != self.expected_endpoint_scope
        ):
            raise RuntimeError("container endpoint scope does not match the saved environment")
        self._environment_fingerprint = fingerprint
        self.execution_image = fingerprint.resolved_image_identity

    def _environment_metadata(self) -> dict[str, str]:
        fingerprint = self._environment_fingerprint
        if fingerprint is None or not fingerprint.complete:
            return {"environment_integrity": "incomplete"}
        assert fingerprint.fingerprint_sha256 is not None
        assert fingerprint.resolved_image_identity is not None
        return {
            "environment_integrity": "verified",
            "environment_fingerprint_sha256": fingerprint.fingerprint_sha256,
            "resolved_image_identity": fingerprint.resolved_image_identity,
        }

    def _collection_command(self, artifact: ProbeArtifact) -> tuple[str, ...]:
        if self.adapter.name == "pytest":
            return (*artifact.command[:-1], "--collect-only", artifact.command[-1])
        return (
            "npx",
            "--no-install",
            "vitest",
            "list",
            artifact.relative_path.as_posix(),
        )

    def remaining_timeout(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Evalyn execution budget was exhausted")
        return min(float(self.command_timeout_seconds), remaining)

    def _setup_secret_environment(self) -> dict[str, str]:
        missing = [name for name in self.setup_env_names if name not in os.environ]
        if missing:
            raise RuntimeError(
                "Configured setup environment variable is unset: " + ", ".join(missing)
            )
        return {name: os.environ[name] for name in self.setup_env_names}

    def _default_setup(self, workspace: Path) -> list[tuple[str, ...]]:
        if self.adapter.name == "pytest":
            commands: list[tuple[str, ...]] = [
                ("python", "-m", "venv", ".evalyn/venv"),
                (
                    ".evalyn/venv/bin/python",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "pytest>=8",
                ),
            ]
            if any(
                (workspace / name).is_file() for name in ("pyproject.toml", "setup.py", "setup.cfg")
            ):
                commands.append(
                    (
                        ".evalyn/venv/bin/python",
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "-e",
                        ".",
                    )
                )
            return commands
        if (workspace / "package-lock.json").is_file():
            return [("npm", "ci", "--ignore-scripts")]
        if (workspace / "package.json").is_file():
            return [("npm", "install", "--ignore-scripts", "--no-package-lock")]
        return []

    def _probe_environment(self) -> dict[str, str]:
        if self.adapter.name == "pytest":
            return {
                "PATH": "/workspace/.evalyn/venv/bin:/usr/local/bin:/usr/bin:/bin",
                "PYTHONPATH": "/workspace/.evalyn/probes/support:/workspace",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        return {"CI": "1"}

    @staticmethod
    def _materialize(workspace: Path, artifact: ProbeArtifact) -> Path:
        policy = WorkspacePathPolicy(workspace)
        target = policy.resolve_write(artifact.relative_path.as_posix(), protect_sensitive=False)
        content = artifact.source.encode("utf-8")
        if target.exists():
            existing = RunStore.read_bytes(target, max_bytes=len(content))
            if existing != content:
                raise ValueError("content-addressed probe path already has different content")
        else:
            RunStore.write_bytes(target, content, mode=0o644)
        return target

    @staticmethod
    def _materialize_pytest_reporter(workspace: Path) -> Path:
        policy = WorkspacePathPolicy(workspace)
        target = policy.resolve_write(
            ".evalyn/probes/support/evalyn_pytest_reporter.py",
            protect_sensitive=False,
        )
        content = PYTEST_REPORTER_SOURCE.encode("utf-8")
        if target.exists():
            existing = RunStore.read_bytes(target, max_bytes=len(content))
            if existing != content:
                raise ValueError("trusted pytest reporter path already has different content")
        else:
            RunStore.write_bytes(target, content, mode=0o444)
        return target


def _looks_invalid(adapter: str, output: str) -> bool:
    folded = output.casefold()
    common = ("syntaxerror", "failed to parse", "no test suite found", "collection error")
    if any(marker in folded for marker in common):
        return True
    if adapter == "pytest" and "error collecting" in folded:
        return True
    return adapter == "vitest" and "failed to load" in folded


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PYTEST_TIMING_SUMMARY = re.compile(
    r"^=*\s*(?=.*\b(?:failed|passed|errors?|skipped|warnings?)\b)"
    r".*\bin\s+\d+(?:\.\d+)?s\s*=*$",
    re.IGNORECASE,
)
_VITEST_TIMING_LINE = re.compile(r"^\s*(?:Start at|Duration)\s+", re.IGNORECASE)
_VITEST_FILE_DURATION = re.compile(r"\s+\d+(?:\.\d+)?m?s(?:\s+\([^)]*\))?\s*$")


def _stable_failure_signature(adapter: str, stdout: str, stderr: str) -> str:
    """Hash failure evidence after removing framework-owned timing noise only."""
    retained: list[str] = []
    for raw_line in _ANSI_ESCAPE.sub("", f"{stdout}\n{stderr}").splitlines():
        line = raw_line.rstrip()
        if adapter == "pytest" and _PYTEST_TIMING_SUMMARY.fullmatch(line.strip()):
            continue
        if adapter == "vitest":
            if _VITEST_TIMING_LINE.match(line):
                continue
            line = _VITEST_FILE_DURATION.sub("", line)
        normalized = " ".join(line.split())
        if normalized:
            retained.append(normalized)
    evidence = "\n".join(retained) or "nonzero exit with no output"
    digest = sha256(evidence.encode()).hexdigest()[:16]
    return f"{adapter}:{digest}"
