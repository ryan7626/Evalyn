"""Three-repeat differential execution classifier."""

from __future__ import annotations

import re
from collections.abc import Sequence
from hashlib import sha256

from evalyn.engine.models import (
    ClassificationKind,
    ClassificationResult,
    ExecutionResult,
    ExecutionStatus,
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_WHITESPACE = re.compile(r"\s+")


def failure_signature(result: ExecutionResult) -> str:
    """Return a stable signature without interpreting framework-specific text."""
    if result.failure_signature.strip():
        return _WHITESPACE.sub(" ", result.failure_signature.strip())
    if result.status == ExecutionStatus.TIMEOUT:
        return "timeout"

    output = result.stderr.strip() or result.stdout.strip()
    if not output:
        return f"{result.status.value}:exit-{result.exit_code}"
    normalized = _WHITESPACE.sub(" ", _ANSI_ESCAPE.sub("", output)).strip()
    return f"{result.status.value}:{sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


class DifferentialClassifier:
    """Classify repeated executions on each side of a change."""

    _FAILURE_STATUSES = {ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT}
    _INVALID_STATUSES = {ExecutionStatus.INVALID_PROBE, ExecutionStatus.INFRA_ERROR}

    def __init__(self, repetitions: int = 3) -> None:
        if isinstance(repetitions, bool) or not isinstance(repetitions, int):
            raise TypeError("repetitions must be an integer")
        if repetitions < 2:
            raise ValueError("repetitions must be at least 2")
        self.repetitions = repetitions

    def classify(
        self,
        base_results: Sequence[ExecutionResult],
        candidate_results: Sequence[ExecutionResult],
    ) -> ClassificationResult:
        self._validate_count("base", base_results)
        self._validate_count("candidate", candidate_results)

        combined = tuple(base_results) + tuple(candidate_results)
        if any(result.status in self._INVALID_STATUSES for result in combined):
            return ClassificationResult(
                ClassificationKind.INVALID,
                "at least one execution was an invalid probe or infrastructure error",
            )

        base_passes = self._all_pass(base_results)
        candidate_passes = self._all_pass(candidate_results)
        base_failures = self._all_fail(base_results)
        candidate_failures = self._all_fail(candidate_results)

        if base_passes and candidate_passes:
            return ClassificationResult(
                ClassificationKind.UNCHANGED,
                f"probe passed all {self.repetitions} base and candidate executions",
            )

        if base_passes and candidate_failures:
            signatures = {failure_signature(result) for result in candidate_results}
            if len(signatures) == 1:
                signature = next(iter(signatures))
                return ClassificationResult(
                    ClassificationKind.STABLE_CHANGE,
                    f"probe passed base {self.repetitions}/{self.repetitions} and failed "
                    f"candidate {self.repetitions}/{self.repetitions} with one signature",
                    failure_signature=signature,
                )
            return ClassificationResult(
                ClassificationKind.UNSTABLE,
                f"candidate failed {self.repetitions}/{self.repetitions} but failure "
                "signatures were inconsistent",
            )

        if base_failures and candidate_passes:
            return ClassificationResult(
                ClassificationKind.IMPROVEMENT,
                f"probe failed base {self.repetitions}/{self.repetitions} and passed "
                f"candidate {self.repetitions}/{self.repetitions}",
            )

        if base_failures:
            return ClassificationResult(
                ClassificationKind.BASELINE_FAILURE,
                "probe did not establish a passing base behavior",
            )

        return ClassificationResult(
            ClassificationKind.UNSTABLE,
            f"base or candidate outcomes varied across {self.repetitions} executions",
        )

    def _validate_count(
        self,
        side: str,
        results: Sequence[ExecutionResult],
    ) -> None:
        if len(results) != self.repetitions:
            raise ValueError(
                f"{side} requires exactly {self.repetitions} execution results; "
                f"received {len(results)}"
            )

    @staticmethod
    def _all_pass(results: Sequence[ExecutionResult]) -> bool:
        return all(result.status == ExecutionStatus.PASSED for result in results)

    def _all_fail(self, results: Sequence[ExecutionResult]) -> bool:
        return all(result.status in self._FAILURE_STATUSES for result in results)
