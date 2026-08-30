"""CLI-independent orchestration for differential probe execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
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
        adaptive: bool = True,
        confirm_improvements: bool = False,
    ) -> None:
        self.adapter = adapter
        self.executor = executor
        self.classifier = classifier or DifferentialClassifier()
        self.assessor = assessor
        self.adaptive = adaptive
        self.confirm_improvements = confirm_improvements

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
            base_records, candidate_records, classification = self._evaluate_artifact(
                base,
                candidate,
                artifact,
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
                    evidence=self._evidence_vector(
                        probe,
                        tuple(base_records),
                        tuple(candidate_records),
                    ),
                )
                findings.append(finding)

        findings = [
            self._assess_finding(finding) for finding in self._with_corroboration(tuple(findings))
        ]

        return DifferentialRunResult(
            run_id=actual_run_id,
            base_sha=base.sha,
            candidate_sha=candidate.sha,
            adapter=self.adapter.name,
            assessments=tuple(assessments),
            findings=tuple(findings),
        )

    @staticmethod
    def _with_corroboration(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
        """Count independent causal probes supporting the same target contract."""

        groups: dict[tuple[str, str, str], set[str]] = {}
        for finding in findings:
            evidence = finding.evidence
            if not (
                evidence is not None
                and evidence.environment_integrity
                and evidence.target_binding_verified
                and evidence.oracle_binding_verified
                and evidence.target_intervention_verified
            ):
                continue
            metadata = finding.probe_metadata
            target = metadata.get("change_target_id") or "::".join(
                (
                    metadata.get("changed_file", ""),
                    metadata.get("changed_symbol", ""),
                )
            )
            key = (
                target,
                metadata.get("oracle_grounding_source", ""),
                finding.oracle,
            )
            groups.setdefault(key, set()).add(finding.artifact.probe_id)

        enriched: list[Finding] = []
        for finding in findings:
            metadata = dict(finding.probe_metadata)
            target = metadata.get("change_target_id") or "::".join(
                (metadata.get("changed_file", ""), metadata.get("changed_symbol", ""))
            )
            key = (
                target,
                metadata.get("oracle_grounding_source", ""),
                finding.oracle,
            )
            count = max(1, len(groups.get(key, set())))
            metadata["corroboration_count"] = str(count)
            evidence = (
                replace(finding.evidence, corroboration_count=count)
                if finding.evidence is not None
                else None
            )
            enriched.append(replace(finding, probe_metadata=metadata, evidence=evidence))
        return tuple(enriched)

    def _evaluate_artifact(
        self,
        base: Revision,
        candidate: Revision,
        artifact: ProbeArtifact,
    ) -> tuple[list[ExecutionRecord], list[ExecutionRecord], ClassificationResult]:
        """Run a cheap discovery pair and confirm only interesting transitions."""

        base_records = [self._execute("base", base, artifact, 1)]
        candidate_records = [self._execute("candidate", candidate, artifact, 1)]
        if not self.adaptive:
            self._append_confirmations(base, candidate, artifact, base_records, candidate_records)
            return (
                base_records,
                candidate_records,
                self.classifier.classify(
                    [record.result for record in base_records],
                    [record.result for record in candidate_records],
                ),
            )

        base_result = base_records[0].result
        candidate_result = candidate_records[0].result
        invalid = {ExecutionStatus.INVALID_PROBE, ExecutionStatus.INFRA_ERROR}
        failures = {ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT}
        if base_result.status in invalid or candidate_result.status in invalid:
            return (
                base_records,
                candidate_records,
                ClassificationResult(
                    ClassificationKind.INVALID,
                    "discovery execution was an invalid probe or infrastructure error",
                ),
            )
        if base_result.status == ExecutionStatus.PASSED and candidate_result.status in failures:
            self._append_confirmations(base, candidate, artifact, base_records, candidate_records)
            return (
                base_records,
                candidate_records,
                self.classifier.classify(
                    [record.result for record in base_records],
                    [record.result for record in candidate_records],
                ),
            )
        if (
            self.confirm_improvements
            and base_result.status in failures
            and candidate_result.status == ExecutionStatus.PASSED
        ):
            self._append_confirmations(base, candidate, artifact, base_records, candidate_records)
            return (
                base_records,
                candidate_records,
                self.classifier.classify(
                    [record.result for record in base_records],
                    [record.result for record in candidate_records],
                ),
            )
        if (
            base_result.status == ExecutionStatus.PASSED
            and candidate_result.status == ExecutionStatus.PASSED
        ):
            return (
                base_records,
                candidate_records,
                ClassificationResult(
                    ClassificationKind.UNCHANGED,
                    "probe passed the base and candidate discovery executions; "
                    "no differential was observed",
                ),
            )
        if base_result.status in failures and candidate_result.status == ExecutionStatus.PASSED:
            return (
                base_records,
                candidate_records,
                ClassificationResult(
                    ClassificationKind.IMPROVEMENT,
                    "probe failed base and passed candidate during discovery",
                ),
            )
        if base_result.status in failures:
            return (
                base_records,
                candidate_records,
                ClassificationResult(
                    ClassificationKind.BASELINE_FAILURE,
                    "probe did not establish passing base behavior during discovery",
                ),
            )
        return (
            base_records,
            candidate_records,
            ClassificationResult(
                ClassificationKind.UNSTABLE,
                "base or candidate discovery outcome could not establish a causal transition",
            ),
        )

    def _append_confirmations(
        self,
        base: Revision,
        candidate: Revision,
        artifact: ProbeArtifact,
        base_records: list[ExecutionRecord],
        candidate_records: list[ExecutionRecord],
    ) -> None:
        # Interleaving sides limits temporal drift while preserving the exact
        # same frozen artifact object for every confirmation call.
        for repetition in range(2, self.classifier.repetitions + 1):
            base_records.append(self._execute("base", base, artifact, repetition))
            candidate_records.append(self._execute("candidate", candidate, artifact, repetition))

    @staticmethod
    def _evidence_vector(
        probe: ProbeSource,
        base_records: tuple[ExecutionRecord, ...],
        candidate_records: tuple[ExecutionRecord, ...],
    ) -> EvidenceVector:
        metadata = probe.metadata
        try:
            corroboration = max(1, int(metadata.get("corroboration_count", "1")))
        except (TypeError, ValueError):
            corroboration = 1
        try:
            intervention_repetitions = max(
                0, int(metadata.get("target_intervention_repetitions", "0"))
            )
        except (TypeError, ValueError):
            intervention_repetitions = 0
        executions = (*base_records, *candidate_records)
        environment_integrity = bool(executions) and all(
            record.result.metadata.get("environment_integrity") == "verified"
            for record in executions
        )
        return EvidenceVector(
            causal_transition="base_pass_candidate_fail",
            base_repetitions=len(base_records),
            candidate_repetitions=len(candidate_records),
            repeatable=True,
            oracle_kind=probe.oracle_provenance or "none",
            oracle_source=metadata.get("oracle_grounding_source", ""),
            target_file=metadata.get("changed_file", ""),
            target_symbol=metadata.get("changed_symbol", ""),
            target_binding_verified=metadata.get("target_binding") == "static_reference",
            oracle_binding_verified=metadata.get("oracle_binding") == "static_assertion_match",
            target_intervention_verified=(
                metadata.get("target_intervention") == "candidate_target_revert_passed"
                and intervention_repetitions >= 3
            ),
            target_intervention_repetitions=intervention_repetitions,
            corroboration_count=corroboration,
            environment_integrity=environment_integrity,
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
