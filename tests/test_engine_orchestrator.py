"""Tests for CLI-independent differential orchestration."""

from dataclasses import dataclass, field

import pytest

from evalyn.adapters.pytest import PytestAdapter
from evalyn.engine.models import (
    ClassificationKind,
    ExecutionResult,
    ExecutionStatus,
    ProbeSource,
    Revision,
    SemanticState,
)
from evalyn.engine.orchestrator import DifferentialEngine, EngineError


@dataclass
class RecordingExecutor:
    candidate_fails: bool = True
    calls: list[tuple[Revision, object, int]] = field(default_factory=list)

    def execute(self, revision, artifact, *, repetition):
        self.calls.append((revision, artifact, repetition))
        if self.candidate_fails and revision.sha == "candidate-sha":
            return ExecutionResult.failed(
                failure_signature="AssertionError: expected rejection",
                stderr="AssertionError: expected rejection",
                duration_seconds=0.1,
            )
        return ExecutionResult.passed(duration_seconds=0.05)


def revisions(tmp_path):
    return (
        Revision("base-sha", tmp_path / "base"),
        Revision("candidate-sha", tmp_path / "candidate"),
    )


def test_engine_reuses_identical_artifact_for_all_six_calls(tmp_path):
    executor = RecordingExecutor()
    engine = DifferentialEngine(PytestAdapter(), executor)
    base, candidate = revisions(tmp_path)

    result = engine.evaluate(
        base,
        candidate,
        [ProbeSource("def test_probe():\n    assert True")],
    )

    assert len(executor.calls) == 6
    assert len({id(artifact) for _, artifact, _ in executor.calls}) == 1
    assert [revision.sha for revision, _, _ in executor.calls] == [
        "base-sha",
        "candidate-sha",
        "base-sha",
        "candidate-sha",
        "base-sha",
        "candidate-sha",
    ]
    assert result.stable_change_count == 1


def test_stable_change_creates_finding_with_artifact_metadata(tmp_path):
    engine = DifferentialEngine(PytestAdapter(), RecordingExecutor())
    base, candidate = revisions(tmp_path)
    probe = ProbeSource(
        source="def test_negative_retry():\n    assert reject(-1)",
        title="Negative retries are no longer rejected",
        oracle="Negative values must be rejected",
        oracle_provenance="docs/retries.md:18",
        metadata={"symbol": "reject"},
    )

    result = engine.evaluate(base, candidate, [probe], run_id="run-1")

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.finding_id.startswith("EVL-")
    assert finding.run_id == "run-1"
    assert finding.title == probe.title
    assert finding.classification.kind == ClassificationKind.STABLE_CHANGE
    assert finding.semantic_state == SemanticState.UNCERTAIN_CHANGE
    assert finding.should_gate is False
    assert finding.oracle_provenance == "docs/retries.md:18"
    assert finding.artifact.adapter == "pytest"
    assert finding.artifact.relative_path.startswith(".evalyn/probes/pytest/")
    assert finding.artifact.command[:3] == ("python", "-m", "pytest")
    assert finding.artifact.byte_size > 0
    assert finding.probe_source == probe.source + "\n"
    assert finding.to_dict()["artifact"]["command"][:3] == [
        "python",
        "-m",
        "pytest",
    ]
    assert finding.to_dict()["semantic_state"] == "uncertain_change"
    assert result.to_dict()["assessments"][0]["classification"]["kind"] == "stable_change"
    assert result.to_dict()["verdict"] == "change_observed"
    assert result.gating_findings == ()


def test_finding_id_and_run_id_are_deterministic(tmp_path):
    base, candidate = revisions(tmp_path)
    probe = ProbeSource("def test_probe():\n    assert True")

    first = DifferentialEngine(PytestAdapter(), RecordingExecutor()).evaluate(
        base, candidate, [probe]
    )
    second = DifferentialEngine(PytestAdapter(), RecordingExecutor()).evaluate(
        base, candidate, [probe]
    )

    assert first.run_id == second.run_id
    assert first.findings[0].finding_id == second.findings[0].finding_id


def test_unchanged_probe_has_assessment_but_no_finding(tmp_path):
    executor = RecordingExecutor(candidate_fails=False)
    engine = DifferentialEngine(
        PytestAdapter(),
        executor,
    )
    base, candidate = revisions(tmp_path)

    result = engine.evaluate(
        base,
        candidate,
        [ProbeSource("def test_probe():\n    assert True")],
    )

    assert result.assessments[0].classification.kind == ClassificationKind.UNCHANGED
    assert len(executor.calls) == 2
    assert len(result.assessments[0].base_runs) == 1
    assert len(result.assessments[0].candidate_runs) == 1
    assert result.to_dict()["verdict"] == "no_differential_observed"
    assert result.findings == ()


def test_nonadaptive_mode_preserves_full_repetition_for_clean_probe(tmp_path):
    executor = RecordingExecutor(candidate_fails=False)
    engine = DifferentialEngine(PytestAdapter(), executor, adaptive=False)
    base, candidate = revisions(tmp_path)

    result = engine.evaluate(base, candidate, [ProbeSource("def test_probe():\n    assert True")])

    assert len(executor.calls) == 6
    assert result.assessments[0].classification.kind == ClassificationKind.UNCHANGED


def test_duplicate_probe_source_is_executed_once(tmp_path):
    executor = RecordingExecutor()
    engine = DifferentialEngine(PytestAdapter(), executor)
    base, candidate = revisions(tmp_path)
    probe = ProbeSource("def test_probe():\n    assert True")

    result = engine.evaluate(base, candidate, [probe, probe])

    assert len(executor.calls) == 6
    assert len(result.assessments) == 1


def test_executor_exception_becomes_invalid_evidence(tmp_path):
    class BrokenExecutor:
        def execute(self, revision, artifact, *, repetition):
            raise RuntimeError("sandbox unavailable")

    engine = DifferentialEngine(PytestAdapter(), BrokenExecutor())
    base, candidate = revisions(tmp_path)

    result = engine.evaluate(
        base,
        candidate,
        [ProbeSource("def test_probe():\n    assert True")],
    )

    assessment = result.assessments[0]
    assert assessment.classification.kind == ClassificationKind.INVALID
    assert all(
        record.result.status == ExecutionStatus.INFRA_ERROR
        for record in assessment.base_runs + assessment.candidate_runs
    )
    assert result.findings == ()


def test_rejects_same_revision_or_empty_probes(tmp_path):
    revision = Revision("same", tmp_path)
    engine = DifferentialEngine(PytestAdapter(), RecordingExecutor())

    with pytest.raises(EngineError, match="must differ"):
        engine.evaluate(revision, revision, [ProbeSource("assert True")])

    base, candidate = revisions(tmp_path)
    with pytest.raises(EngineError, match="at least one"):
        engine.evaluate(base, candidate, [])
