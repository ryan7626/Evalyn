"""Container-only materialization and execution of generated probes."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from evalyn.adapters.base import ProbeArtifact, TestAdapter
from evalyn.engine.models import ExecutionResult, ExecutionStatus, Revision
from evalyn.execution import ContainerRunner, WorkspacePathPolicy
from evalyn.run_store import RunStore

DEFAULT_IMAGES = {
    "pytest": "python:3.12-slim",
    "vitest": "node:22-bookworm-slim",
}


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
    ) -> None:
        if adapter.name not in DEFAULT_IMAGES:
            raise ValueError(f"No container image for adapter {adapter.name!r}")
        self.adapter = adapter
        self.engine = engine
        self.run_dir = run_dir.resolve()
        self.image = image or DEFAULT_IMAGES[adapter.name]
        self.setup_argv = setup_argv
        self.setup_env_names = setup_env_names
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.budget_seconds = timeout_seconds
        self.command_timeout_seconds = min(300, timeout_seconds)
        self._deadline = time.monotonic() + timeout_seconds
        self._prepared: set[Path] = set()
        self._setup_errors: dict[Path, str] = {}
        self._labels: dict[Path, str] = {}

    @property
    def setup_errors(self) -> dict[Path, str]:
        return dict(self._setup_errors)

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
                image=self.image,
                engine=self.engine,
                network="none",
                pull_policy="missing",
                workspace_read_only=True,
                default_timeout_seconds=self.remaining_timeout(),
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
                metadata={"log_path": str(log_path), "probe_path": str(target)},
            )

        metadata = {"log_path": str(log_path), "probe_path": str(target)}
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
        if result.exit_code == 0:
            return ExecutionResult.passed(
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        output = f"{result.stdout}\n{result.stderr}"
        status = (
            ExecutionStatus.INVALID_PROBE
            if _looks_invalid(self.adapter.name, output)
            else ExecutionStatus.FAILED
        )
        return ExecutionResult(
            status=status,
            exit_code=result.exit_code if result.exit_code is not None else 1,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
            failure_signature=(
                _stable_failure_signature(self.adapter.name, result.stdout, result.stderr)
                if status == ExecutionStatus.FAILED
                else ""
            ),
            metadata=metadata,
        )

    def _prepare_workspace(self, workspace: Path) -> None:
        commands = [self.setup_argv] if self.setup_argv else self._default_setup(workspace)
        if not commands:
            return
        with ContainerRunner(
            workspace,
            image=self.image,
            engine=self.engine,
            network="bridge",
            pull_policy="missing",
            default_timeout_seconds=self.remaining_timeout(),
            allowed_secret_env_names=self.setup_env_names,
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
                "PYTHONPATH": "/workspace",
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
