"""Shared data models for Evalyn.

All core result dataclasses live here to prevent circular imports and enforce
clean layering: runner/* imports from models, never the other way around.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# TestResult — produced by runner/test_executor.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FixAttempt — produced by runner/fixer.py
# ---------------------------------------------------------------------------

@dataclass
class FixAttempt:
    """Represents an attempted fix from the LLM."""

    file_path: str
    original_content: str
    replacement_content: str
    explanation: str
    applied: bool = False
    rejected_reason: Optional[str] = None
    passed: bool = False
    diff_size: int = 0
    is_best_candidate: bool = False


# ---------------------------------------------------------------------------
# CategoryResult / RunResult — produced by runner/runner.py
# ---------------------------------------------------------------------------

@dataclass
class CategoryResult:
    """Aggregated result for a single test category."""

    name: str
    status: Literal["passed", "failed", "uncertain", "skipped"]
    test_results: list[TestResult] = field(default_factory=list)
    fix_attempts: list[FixAttempt] = field(default_factory=list)
    generated_tests: list[str] = field(default_factory=list)
    error_message: str = ""


@dataclass
class RunResult:
    """Complete result of an execution run."""

    timestamp: str
    branch: str
    run_dir: Path
    category_results: list[CategoryResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.category_results if c.status == "passed")

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.category_results if c.status == "failed")

    @property
    def uncertain_count(self) -> int:
        return sum(1 for c in self.category_results if c.status == "uncertain")

    @property
    def skipped_count(self) -> int:
        return sum(1 for c in self.category_results if c.status == "skipped")
