"""Tests separating differential evidence from semantic and human triage."""

from dataclasses import dataclass

import pytest

from evalyn.adapters.pytest import PytestAdapter
from evalyn.engine.assessment import SemanticAssessment, triage_finding
from evalyn.engine.models import (
    ExecutionResult,
    ProbeSource,
    Revision,
    SemanticState,
    TriageStatus,
)
from evalyn.engine.orchestrator import DifferentialEngine


class StableChangeExecutor:
    def execute(self, revision, artifact, *, repetition):
        if revision.sha == "candidate":
            return ExecutionResult.failed(failure_signature="assertion:changed")
        return ExecutionResult.passed()


@dataclass
class StaticAssessor:
    assessment: SemanticAssessment

    def assess(self, finding):
        return self.assessment


def finding_with_assessor(tmp_path, assessor=None):
    engine = DifferentialEngine(
        PytestAdapter(),
        StableChangeExecutor(),
        assessor=assessor,
    )
    result = engine.evaluate(
        Revision("base", tmp_path / "base"),
        Revision("candidate", tmp_path / "candidate"),
        [
            ProbeSource(
                "def test_probe():\n    assert behavior()",
                oracle="Documented behavior",
                oracle_provenance="docs/api.md:10",
            )
        ],
    )
    return result, result.findings[0]


def test_unassessed_stable_change_is_uncertain_and_non_gating(tmp_path):
    result, finding = finding_with_assessor(tmp_path)

    assert finding.semantic_state == SemanticState.UNCERTAIN_CHANGE
    assert finding.triage_status == TriageStatus.UNREVIEWED
    assert finding.should_gate is False
    assert result.gating_findings == ()
    assert result.to_dict()["gating_finding_count"] == 0


def test_likely_regression_requires_grounded_oracle_and_explicit_intent_result(tmp_path):
    assessor = StaticAssessor(
        SemanticAssessment(
            state=SemanticState.LIKELY_REGRESSION,
            reason="documented behavior changed outside stated intent",
            assessor="intent-v1",
            oracle_admissible=True,
            intent_match=False,
        )
    )

    result, finding = finding_with_assessor(tmp_path, assessor)

    assert finding.semantic_state == SemanticState.LIKELY_REGRESSION
    assert finding.should_gate is True
    assert result.gating_findings == (finding,)
    serialized = finding.to_dict()
    assert serialized["semantic_assessor"] == "intent-v1"
    assert serialized["oracle_admissible"] is True
    assert serialized["intent_match"] is False
    assert serialized["should_gate"] is True


def test_intended_change_is_serialized_and_non_gating(tmp_path):
    assessor = StaticAssessor(
        SemanticAssessment(
            state=SemanticState.INTENDED_CHANGE,
            reason="PR intent explicitly changes this response",
            assessor="intent-v1",
            oracle_admissible=True,
            intent_match=True,
        )
    )

    _, finding = finding_with_assessor(tmp_path, assessor)

    assert finding.semantic_state == SemanticState.INTENDED_CHANGE
    assert finding.should_gate is False
    assert finding.to_dict()["semantic_state"] == "intended_change"


def test_confirmed_human_triage_overrides_uncertain_state(tmp_path):
    _, finding = finding_with_assessor(tmp_path)

    confirmed = triage_finding(
        finding,
        TriageStatus.CONFIRMED,
        reason="maintainer reproduced the regression",
    )

    assert confirmed.semantic_state == SemanticState.UNCERTAIN_CHANGE
    assert confirmed.triage_status == TriageStatus.CONFIRMED
    assert confirmed.triage_reason == "maintainer reproduced the regression"
    assert confirmed.should_gate is True
    assert confirmed.to_dict()["triage_status"] == "confirmed"


def test_dismissal_overrides_likely_regression(tmp_path):
    assessor = StaticAssessor(
        SemanticAssessment(
            state=SemanticState.LIKELY_REGRESSION,
            reason="appears outside intent",
            assessor="intent-v1",
            oracle_admissible=True,
            intent_match=False,
        )
    )
    _, finding = finding_with_assessor(tmp_path, assessor)

    dismissed = triage_finding(
        finding,
        TriageStatus.DISMISSED,
        reason="intent file omitted a deliberate compatibility break",
    )

    assert dismissed.should_gate is False


def test_assessor_failure_preserves_evidence_as_uncertain(tmp_path):
    class BrokenAssessor:
        def assess(self, finding):
            raise RuntimeError("provider unavailable")

    _, finding = finding_with_assessor(tmp_path, BrokenAssessor())

    assert finding.semantic_state == SemanticState.UNCERTAIN_CHANGE
    assert finding.should_gate is False
    assert "provider unavailable" in finding.semantic_reason


def test_semantic_invariants_reject_ungrounded_likely_or_unmatched_intended():
    with pytest.raises(ValueError, match="admissible oracle"):
        SemanticAssessment(
            state=SemanticState.LIKELY_REGRESSION,
            reason="guess",
            assessor="test",
            oracle_admissible=False,
            intent_match=False,
        )

    with pytest.raises(ValueError, match="explicit intent match"):
        SemanticAssessment(
            state=SemanticState.LIKELY_REGRESSION,
            reason="guess",
            assessor="test",
            oracle_admissible=True,
            intent_match=None,
        )

    with pytest.raises(ValueError, match="intent_match=True"):
        SemanticAssessment(
            state=SemanticState.INTENDED_CHANGE,
            reason="guess",
            assessor="test",
            oracle_admissible=True,
            intent_match=None,
        )


def test_triage_requires_explicit_nonempty_reason(tmp_path):
    _, finding = finding_with_assessor(tmp_path)

    with pytest.raises(ValueError, match="reason"):
        triage_finding(finding, TriageStatus.CONFIRMED, reason="  ")
