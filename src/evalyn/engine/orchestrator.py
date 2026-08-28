"""CLI-independent orchestration for differential probe execution."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from typing import Protocol, runtime_checkable

from evalyn.adapters.base import ProbeArtifact, TestAdapter
from evalyn.engine.assessment import (
    FindingAssessor,
    SemanticAssessment,
    apply_semantic_assessment,
)
from evalyn.engine.classifier import DifferentialClassifier
from evalyn.engine.models import (
    ArtifactMetadata,
    ClassificationKind,
    DifferentialRunResult,
    ExecutionRecord,
    ExecutionResult,
    ExecutionStatus,
    Finding,
    ProbeAssessment,
    ProbeSource,
    Revision,
    SemanticState,
)


class EngineError(ValueError):
    """Raised when a differential evaluation cannot be constructed safely."""


@runtime_checkable
class ProbeExecutor(Protocol):
    """Injected boundary that materializes and runs an adapter-owned artifact."""

    def execute(
        self,
        revision: Revision,
        artifact: ProbeArtifact,
        *,
        repetition: int,
    ) -> ExecutionResult:
        """Execute an artifact in one revision's isolated workspace."""


class DifferentialEngine:
    """Run identical immutable probes repeatedly on base and candidate."""

    def __init__(
        self,
        adapter: TestAdapter,
        executor: ProbeExecutor,
        classifier: DifferentialClassifier | None = None,
        assessor: FindingAssessor | None = None,
    ) -> None:
        self.adapter = adapter
        self.executor = executor
        self.classifier = classifier or DifferentialClassifier()
        self.assessor = assessor

    def evaluate(
        self,
        base: Revision,
        candidate: Revision,
        probes: Iterable[ProbeSource],
        *,
        run_id: str | None = None,
    ) -> DifferentialRunResult:
        if base.sha == candidate.sha:
            raise EngineError("base and candidate revisions must differ")

        prepared = self._prepare_unique_probes(probes)
        actual_run_id = run_id or self._run_id(base.sha, candidate.sha, prepared)
        if not actual_run_id.strip():
            raise EngineError("run_id cannot be empty")

        assessments: list[ProbeAssessment] = []
        findings: list[Finding] = []

        for probe, artifact in prepared:
            base_records: list[ExecutionRecord] = []
            candidate_records: list[ExecutionRecord] = []

            # Interleaving sides limits temporal drift while preserving the exact
            # same frozen artifact object for all six calls.
            for repetition in range(1, self.classifier.repetitions + 1):
                base_records.append(self._execute("base", base, artifact, repetition))
                candidate_records.append(
                    self._execute("candidate", candidate, artifact, repetition)
                )

            classification = self.classifier.classify(
                [record.result for record in base_records],
                [record.result for record in candidate_records],
            )
            artifact_metadata = self._artifact_metadata(artifact)
            assessment = ProbeAssessment(
                artifact=artifact_metadata,
                classification=classification,
                base_runs=tuple(base_records),
                candidate_runs=tuple(candidate_records),
            )
            assessments.append(assessment)

            if classification.kind == ClassificationKind.STABLE_CHANGE:
                finding = Finding(
                    finding_id=self._finding_id(
                        base.sha,
                        candidate.sha,
                        artifact.probe_id,
                        classification.failure_signature,
                    ),
                    run_id=actual_run_id,
                    title=probe.title.strip()
                    or f"Stable behavior change for probe {artifact.probe_id}",
                    base_sha=base.sha,
                    candidate_sha=candidate.sha,
                    artifact=artifact_metadata,
                    classification=classification,
                    base_runs=tuple(base_records),
                    candidate_runs=tuple(candidate_records),
                    probe_source=artifact.source,
                    oracle=probe.oracle,
                    oracle_provenance=probe.oracle_provenance,
                    probe_metadata=dict(probe.metadata),
                )
                findings.append(self._assess_finding(finding))

        return DifferentialRunResult(
            run_id=actual_run_id,
            base_sha=base.sha,
            candidate_sha=candidate.sha,
            adapter=self.adapter.name,
            assessments=tuple(assessments),
            findings=tuple(findings),
        )

    def _assess_finding(self, finding: Finding) -> Finding:
        if self.assessor is None:
            return finding
        try:
            assessment = self.assessor.assess(finding)
            if not isinstance(assessment, SemanticAssessment):
                raise TypeError("assessor must return SemanticAssessment")
            return apply_semantic_assessment(finding, assessment)
        except Exception as exc:
            # Semantic assessment is advisory. Preserve strong differential
            # evidence but keep it non-gating when assessment cannot complete.
            return apply_semantic_assessment(
                finding,
                SemanticAssessment(
                    state=SemanticState.UNCERTAIN_CHANGE,
                    reason=f"semantic assessor failed: {type(exc).__name__}: {exc}",
                    assessor=type(self.assessor).__name__,
                    oracle_admissible=False,
                    intent_match=None,
                ),
            )

    def _prepare_unique_probes(
        self,
        probes: Iterable[ProbeSource],
    ) -> list[tuple[ProbeSource, ProbeArtifact]]:
        prepared: list[tuple[ProbeSource, ProbeArtifact]] = []
        seen: set[str] = set()
        for probe in probes:
            if not isinstance(probe, ProbeSource):
                raise EngineError("probes must be ProbeSource instances")
            artifact = self.adapter.build_artifact(probe.source)
            if artifact.probe_id in seen:
                continue
            seen.add(artifact.probe_id)
            prepared.append((probe, artifact))
        if not prepared:
            raise EngineError("at least one unique probe is required")
        return prepared

    def _execute(
        self,
        side: str,
        revision: Revision,
        artifact: ProbeArtifact,
        repetition: int,
    ) -> ExecutionRecord:
        try:
            result = self.executor.execute(
                revision,
                artifact,
                repetition=repetition,
            )
            if not isinstance(result, ExecutionResult):
                raise TypeError("executor must return ExecutionResult")
        except Exception as exc:  # executor failures are evidence, not engine crashes
            result = ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=-1,
                stderr=f"{type(exc).__name__}: {exc}",
            )
        return ExecutionRecord(
            side=side,
            revision_sha=revision.sha,
            repetition=repetition,
            result=result,
        )

    @staticmethod
    def _artifact_metadata(artifact: ProbeArtifact) -> ArtifactMetadata:
        return ArtifactMetadata(
            probe_id=artifact.probe_id,
            adapter=artifact.adapter,
            relative_path=artifact.relative_path.as_posix(),
            command=artifact.command,
            source_sha256=artifact.source_sha256,
            byte_size=len(artifact.source.encode("utf-8")),
            media_type=artifact.media_type,
        )

    @staticmethod
    def _run_id(
        base_sha: str,
        candidate_sha: str,
        prepared: list[tuple[ProbeSource, ProbeArtifact]],
    ) -> str:
        probe_ids = ":".join(artifact.probe_id for _, artifact in prepared)
        digest = sha256(f"{base_sha}:{candidate_sha}:{probe_ids}".encode()).hexdigest()[:12]
        return f"EVR-{digest.upper()}"

    @staticmethod
    def _finding_id(
        base_sha: str,
        candidate_sha: str,
        probe_id: str,
        signature: str,
    ) -> str:
        digest = sha256(f"{base_sha}:{candidate_sha}:{probe_id}:{signature}".encode()).hexdigest()[
            :8
        ]
        return f"EVL-{digest.upper()}"
