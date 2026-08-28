"""Immutable domain models for differential regression evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExecutionStatus(str, Enum):
    """Outcome of one isolated probe execution."""

    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_PROBE = "invalid_probe"
    INFRA_ERROR = "infra_error"


class ClassificationKind(str, Enum):
    """Evidence-only classification across base and candidate repetitions."""

    UNCHANGED = "unchanged"
    STABLE_CHANGE = "stable_change"
    UNSTABLE = "unstable"
    INVALID = "invalid"
    BASELINE_FAILURE = "baseline_failure"
    IMPROVEMENT = "improvement"


class SemanticState(str, Enum):
    """Semantic interpretation applied after differential evidence exists."""

    INTENDED_CHANGE = "intended_change"
    UNCERTAIN_CHANGE = "uncertain_change"
    LIKELY_REGRESSION = "likely_regression"


class TriageStatus(str, Enum):
    """Explicit maintainer disposition for a semantic finding."""

    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Revision:
    """An immutable revision checked out in an executor-owned workspace."""

    sha: str
    workspace: Path

    def __post_init__(self) -> None:
        if not self.sha.strip():
            raise ValueError("revision sha cannot be empty")


@dataclass(frozen=True)
class ProbeSource:
    """Model-authorable probe data.

    Deliberately absent: command, path, framework, workspace, and revision.
    Those trusted values are supplied by adapters and the engine.
    """

    source: str
    title: str = ""
    oracle: str = ""
    oracle_provenance: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("probe source cannot be empty")


@dataclass(frozen=True)
class ExecutionResult:
    """Raw result returned by an injected probe executor."""

    status: ExecutionStatus
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    failure_signature: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        if self.status == ExecutionStatus.PASSED and self.exit_code != 0:
            raise ValueError("passed execution must have exit code 0")

    @classmethod
    def passed(cls, **kwargs: Any) -> ExecutionResult:
        return cls(status=ExecutionStatus.PASSED, exit_code=0, **kwargs)

    @classmethod
    def failed(
        cls,
        *,
        failure_signature: str = "",
        exit_code: int = 1,
        **kwargs: Any,
    ) -> ExecutionResult:
        return cls(
            status=ExecutionStatus.FAILED,
            exit_code=exit_code,
            failure_signature=failure_signature,
            **kwargs,
        )


@dataclass(frozen=True)
class ExecutionRecord:
    """One result anchored to a trusted revision and repetition index."""

    side: str
    revision_sha: str
    repetition: int
    result: ExecutionResult

    def __post_init__(self) -> None:
        if self.side not in {"base", "candidate"}:
            raise ValueError("execution side must be base or candidate")
        if self.repetition < 1:
            raise ValueError("repetition is one-indexed")

    def to_dict(self) -> dict[str, object]:
        return _record_to_dict(self)


@dataclass(frozen=True)
class ArtifactMetadata:
    """Serializable evidence showing adapter-owned probe execution details."""

    probe_id: str
    adapter: str
    relative_path: str
    command: tuple[str, ...]
    source_sha256: str
    byte_size: int
    media_type: str

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "adapter": self.adapter,
            "relative_path": self.relative_path,
            "command": list(self.command),
            "source_sha256": self.source_sha256,
            "byte_size": self.byte_size,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class ClassificationResult:
    """Classification plus concise, auditable supporting metadata."""

    kind: ClassificationKind
    reason: str
    failure_signature: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "failure_signature": self.failure_signature,
        }


@dataclass(frozen=True)
class ProbeAssessment:
    """Complete differential evidence for one generated probe."""

    artifact: ArtifactMetadata
    classification: ClassificationResult
    base_runs: tuple[ExecutionRecord, ...]
    candidate_runs: tuple[ExecutionRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact": self.artifact.to_dict(),
            "classification": self.classification.to_dict(),
            "base_runs": [record.to_dict() for record in self.base_runs],
            "candidate_runs": [record.to_dict() for record in self.candidate_runs],
        }


@dataclass(frozen=True)
class Finding:
    """A stable behavior change surfaced for later semantic triage."""

    finding_id: str
    run_id: str
    title: str
    base_sha: str
    candidate_sha: str
    artifact: ArtifactMetadata
    classification: ClassificationResult
    base_runs: tuple[ExecutionRecord, ...]
    candidate_runs: tuple[ExecutionRecord, ...]
    probe_source: str
    oracle: str = ""
    oracle_provenance: str = ""
    probe_metadata: Mapping[str, str] = field(default_factory=dict)
    semantic_state: SemanticState = SemanticState.UNCERTAIN_CHANGE
    semantic_reason: str = "semantic assessment not provided"
    semantic_assessor: str = "none"
    oracle_admissible: bool = False
    intent_match: bool | None = None
    triage_status: TriageStatus = TriageStatus.UNREVIEWED
    triage_reason: str = ""

    @property
    def should_gate(self) -> bool:
        """Return whether this finding can fail a regression gate.

        Stable differential evidence alone is deliberately insufficient.
        Human triage overrides an automated semantic assessment.
        """
        if self.triage_status == TriageStatus.CONFIRMED:
            return True
        if self.triage_status in {TriageStatus.DISMISSED, TriageStatus.UNCERTAIN}:
            return False
        return self.semantic_state == SemanticState.LIKELY_REGRESSION

    def to_dict(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "run_id": self.run_id,
            "title": self.title,
            "base_sha": self.base_sha,
            "candidate_sha": self.candidate_sha,
            "artifact": self.artifact.to_dict(),
            "classification": self.classification.kind.value,
            "classification_reason": self.classification.reason,
            "failure_signature": self.classification.failure_signature,
            "probe_source": self.probe_source,
            "oracle": self.oracle,
            "oracle_provenance": self.oracle_provenance,
            "probe_metadata": dict(self.probe_metadata),
            "semantic_state": self.semantic_state.value,
            "semantic_reason": self.semantic_reason,
            "semantic_assessor": self.semantic_assessor,
            "oracle_admissible": self.oracle_admissible,
            "intent_match": self.intent_match,
            "triage_status": self.triage_status.value,
            "triage_reason": self.triage_reason,
            "should_gate": self.should_gate,
            "base_runs": [_record_to_dict(record) for record in self.base_runs],
            "candidate_runs": [_record_to_dict(record) for record in self.candidate_runs],
        }


@dataclass(frozen=True)
class DifferentialRunResult:
    """Aggregate result for a base/candidate differential evaluation."""

    run_id: str
    base_sha: str
    candidate_sha: str
    adapter: str
    assessments: tuple[ProbeAssessment, ...]
    findings: tuple[Finding, ...]

    @property
    def stable_change_count(self) -> int:
        return sum(
            assessment.classification.kind == ClassificationKind.STABLE_CHANGE
            for assessment in self.assessments
        )

    @property
    def gating_findings(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.should_gate)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "base_sha": self.base_sha,
            "candidate_sha": self.candidate_sha,
            "adapter": self.adapter,
            "assessment_count": len(self.assessments),
            "stable_change_count": self.stable_change_count,
            "gating_finding_count": len(self.gating_findings),
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _record_to_dict(record: ExecutionRecord) -> dict[str, object]:
    return {
        "side": record.side,
        "revision_sha": record.revision_sha,
        "repetition": record.repetition,
        "status": record.result.status.value,
        "exit_code": record.result.exit_code,
        "duration_seconds": record.result.duration_seconds,
        "failure_signature": record.result.failure_signature,
        "metadata": dict(record.result.metadata),
    }
