"""Semantic assessment and human triage for stable behavior changes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from evalyn.engine.models import Finding, SemanticState, TriageStatus


@dataclass(frozen=True)
class SemanticAssessment:
    """Narrow assessor output; it cannot replace adapter-owned execution data."""

    state: SemanticState
    reason: str
    assessor: str
    oracle_admissible: bool
    intent_match: bool | None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("semantic assessment reason cannot be empty")
        if not self.assessor.strip():
            raise ValueError("semantic assessor name cannot be empty")
        if self.state == SemanticState.LIKELY_REGRESSION and (
            not self.oracle_admissible or not isinstance(self.intent_match, bool)
        ):
            raise ValueError(
                "likely_regression requires an admissible oracle and an explicit intent match"
            )
        if self.state == SemanticState.INTENDED_CHANGE and self.intent_match is not True:
            raise ValueError("intended_change requires intent_match=True")


@runtime_checkable
class FindingAssessor(Protocol):
    """Injected semantic assessor that returns classification metadata only."""

    def assess(self, finding: Finding) -> SemanticAssessment:
        """Assess one stable behavior change without altering its evidence."""


def apply_semantic_assessment(
    finding: Finding,
    assessment: SemanticAssessment,
) -> Finding:
    """Return a finding with semantic fields populated, preserving evidence."""
    return replace(
        finding,
        semantic_state=assessment.state,
        semantic_reason=assessment.reason.strip(),
        semantic_assessor=assessment.assessor.strip(),
        oracle_admissible=assessment.oracle_admissible,
        intent_match=assessment.intent_match,
    )


def triage_finding(
    finding: Finding,
    status: TriageStatus,
    *,
    reason: str,
) -> Finding:
    """Apply an explicit maintainer disposition to an immutable finding."""
    if status == TriageStatus.UNREVIEWED:
        raise ValueError("triage cannot set a finding back to unreviewed")
    if not reason.strip():
        raise ValueError("triage reason cannot be empty")
    return replace(
        finding,
        triage_status=status,
        triage_reason=reason.strip(),
    )
