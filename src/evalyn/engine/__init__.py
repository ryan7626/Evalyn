"""Differential regression engine independent of CLI and process execution."""

from evalyn.engine.assessment import (
    FindingAssessor,
    SemanticAssessment,
    apply_semantic_assessment,
    triage_finding,
)
from evalyn.engine.classifier import DifferentialClassifier, failure_signature
from evalyn.engine.models import (
    ArtifactMetadata,
    ClassificationKind,
    ClassificationResult,
    DifferentialRunResult,
    EvidenceVector,
    ExecutionRecord,
    ExecutionResult,
    ExecutionStatus,
    Finding,
    ProbeAssessment,
    ProbeSource,
    Revision,
    RunVerdict,
    SemanticState,
    TriageStatus,
)
from evalyn.engine.orchestrator import DifferentialEngine, EngineError, ProbeExecutor
from evalyn.engine.ranges import (
    GitRangeResolver,
    RangeMode,
    RangeResolutionError,
    RequestedRange,
    ResolvedRange,
    parse_range_spec,
)

__all__ = [
    "ArtifactMetadata",
    "ClassificationKind",
    "ClassificationResult",
    "DifferentialClassifier",
    "DifferentialEngine",
    "DifferentialRunResult",
    "EvidenceVector",
    "EngineError",
    "ExecutionRecord",
    "ExecutionResult",
    "ExecutionStatus",
    "Finding",
    "FindingAssessor",
    "GitRangeResolver",
    "ProbeAssessment",
    "ProbeExecutor",
    "ProbeSource",
    "RangeMode",
    "RangeResolutionError",
    "RequestedRange",
    "ResolvedRange",
    "Revision",
    "RunVerdict",
    "SemanticAssessment",
    "SemanticState",
    "TriageStatus",
    "apply_semantic_assessment",
    "failure_signature",
    "parse_range_spec",
    "triage_finding",
]
