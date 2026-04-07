"""Shared data models for Evalyn.

These are extracted to avoid circular imports between runner and reporter.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from evalyn.runner.test_executor import TestResult
from evalyn.runner.fixer import FixAttempt


@dataclass
class CategoryResult:
    """Aggregated result for a single test category."""

    name: str
    status: Literal["passed", "failed", "uncertain", "skipped"]
    test_results: list[TestResult] = field(default_factory=list)
    fix_attempts: list[FixAttempt] = field(default_factory=list)
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
