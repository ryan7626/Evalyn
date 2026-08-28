"""Tests for exact three-repeat differential classification."""

import pytest

from evalyn.engine.classifier import DifferentialClassifier, failure_signature
from evalyn.engine.models import (
    ClassificationKind,
    ExecutionResult,
    ExecutionStatus,
)


def passed():
    return ExecutionResult.passed()


def failed(signature="assertion:x"):
    return ExecutionResult.failed(failure_signature=signature, stderr=signature)


def classify(base, candidate):
    return DifferentialClassifier().classify(base, candidate)


def test_unchanged_requires_three_passes_on_both_sides():
    result = classify([passed(), passed(), passed()], [passed(), passed(), passed()])

    assert result.kind == ClassificationKind.UNCHANGED


def test_stable_change_requires_three_matching_candidate_failures():
    result = classify(
        [passed(), passed(), passed()],
        [failed(), failed(), failed()],
    )

    assert result.kind == ClassificationKind.STABLE_CHANGE
    assert result.failure_signature == "assertion:x"


def test_different_candidate_signatures_are_unstable():
    result = classify(
        [passed(), passed(), passed()],
        [failed("one"), failed("two"), failed("one")],
    )

    assert result.kind == ClassificationKind.UNSTABLE


def test_mixed_candidate_outcomes_are_unstable():
    result = classify(
        [passed(), passed(), passed()],
        [failed(), passed(), failed()],
    )

    assert result.kind == ClassificationKind.UNSTABLE


def test_invalid_probe_or_infrastructure_error_is_invalid():
    invalid = ExecutionResult(
        status=ExecutionStatus.INVALID_PROBE,
        exit_code=2,
    )
    result = classify([passed(), passed(), passed()], [invalid, invalid, invalid])

    assert result.kind == ClassificationKind.INVALID


def test_baseline_failure_is_not_a_regression():
    result = classify(
        [failed(), failed(), failed()],
        [failed(), failed(), failed()],
    )

    assert result.kind == ClassificationKind.BASELINE_FAILURE


def test_base_failure_candidate_pass_is_improvement():
    result = classify(
        [failed(), failed(), failed()],
        [passed(), passed(), passed()],
    )

    assert result.kind == ClassificationKind.IMPROVEMENT


def test_stable_timeouts_form_a_stable_change():
    timeout = ExecutionResult(
        status=ExecutionStatus.TIMEOUT,
        exit_code=-1,
    )
    result = classify(
        [passed(), passed(), passed()],
        [timeout, timeout, timeout],
    )

    assert result.kind == ClassificationKind.STABLE_CHANGE
    assert result.failure_signature == "timeout"


def test_exactly_three_results_are_required():
    with pytest.raises(ValueError, match="exactly 3"):
        classify([passed(), passed()], [passed(), passed(), passed()])


def test_configured_repeat_count_is_supported():
    result = DifferentialClassifier(repetitions=2).classify(
        [passed(), passed()], [failed(), failed()]
    )

    assert result.kind == ClassificationKind.STABLE_CHANGE


def test_repeat_count_must_be_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        DifferentialClassifier(repetitions=1)


def test_failure_signature_collapses_explicit_whitespace():
    result = failed("AssertionError:\n    expected 2")

    assert failure_signature(result) == "AssertionError: expected 2"
