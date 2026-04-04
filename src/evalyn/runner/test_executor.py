"""Test executor for Evalyn.

Runs test commands via subprocess and captures structured results.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evalyn.utils.log import get_logger

logger = get_logger("runner.test_executor")


@dataclass
class TestResult:
    """Result of a single test command execution."""

    category: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    status: Literal["passed", "failed", "error", "timeout"]

    @property
    def is_success(self) -> bool:
        return self.status == "passed"

    @property
    def failure_summary(self) -> str:
        """Concise failure summary for logging and reports."""
        if self.status == "passed":
            return ""
        if self.status == "timeout":
            return f"Test timed out after {self.duration_seconds:.0f}s"
        # Combine stderr and tail of stdout for context
        output = self.stderr.strip() or self.stdout.strip()
        lines = output.split("\n")
        if len(lines) > 20:
            output = "\n".join(lines[-20:])
        return output


def run_test_command(
    repo_path: Path,
    command: str,
    category: str,
    *,
    timeout_seconds: int = 300,
    max_output_lines: int = 200,
) -> TestResult:
    """Execute a test command and return structured results.

    Args:
        repo_path: Working directory for the command.
        command: Shell command to execute.
        category: Name of the test category this belongs to.
        timeout_seconds: Maximum time before killing the process.
        max_output_lines: Max lines to keep from stdout/stderr.

    Returns:
        TestResult with captured output and status.
    """
    logger.info(f"Executing: {command}")
    start_time = time.monotonic()

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration = time.monotonic() - start_time

        stdout = _truncate_output(result.stdout, max_output_lines)
        stderr = _truncate_output(result.stderr, max_output_lines)

        if result.returncode == 0:
            status = "passed"
            logger.info(f"PASSED in {duration:.1f}s: {command}")
        else:
            status = "failed"
            logger.warning(
                f"FAILED (exit {result.returncode}) in {duration:.1f}s: {command}"
            )

        return TestResult(
            category=category,
            command=command,
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            status=status,
        )

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start_time
        logger.error(f"TIMEOUT after {duration:.1f}s: {command}")
        return TestResult(
            category=category,
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout_seconds}s",
            duration_seconds=duration,
            status="timeout",
        )

    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(f"ERROR running {command}: {e}")
        return TestResult(
            category=category,
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_seconds=duration,
            status="error",
        )


def _truncate_output(text: str, max_lines: int) -> str:
    """Truncate output, keeping first N/2 and last N/2 lines.

    This preserves both the test setup context (top) and the actual
    failure messages (bottom).
    """
    if not text:
        return ""

    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text

    half = max_lines // 2
    head = lines[:half]
    tail = lines[-half:]
    omitted = len(lines) - max_lines

    return "\n".join(
        head
        + [f"\n... ({omitted} lines omitted) ...\n"]
        + tail
    )
