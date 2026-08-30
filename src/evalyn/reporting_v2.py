"""Proof-first Markdown, JSON, JUnit, and SARIF reports."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from evalyn.engine.models import (
    ClassificationKind,
    DifferentialRunResult,
    Finding,
    RunVerdict,
    SemanticState,
)
from evalyn.run_store import RunStore


def write_reports(
    run_path: Path,
    results: Iterable[DifferentialRunResult],
    *,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    runs = tuple(results)
    findings = tuple(finding for result in runs for finding in result.findings)
    verdict = _aggregate_verdict(runs)
    payload = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "base_sha": metadata["base_sha"],
        "candidate_sha": metadata["candidate_sha"],
        "intent": list(metadata.get("intent", ())),
        "environments": metadata.get("environments", {}),
        "adapters": [result.adapter for result in runs],
        "verdict": verdict.value,
        "results": [result.to_dict() for result in runs],
        "findings": [finding.to_dict() for finding in findings],
    }
    paths = {
        "json": run_path / "findings.json",
        "markdown": run_path / "report.md",
        "junit": run_path / "junit.xml",
        "sarif": run_path / "evalyn.sarif",
    }
    RunStore.write_json(paths["json"], payload)
    RunStore.write_text(paths["markdown"], render_markdown(runs, metadata=metadata))
    RunStore.write_bytes(
        paths["junit"],
        ElementTree.tostring(
            _junit(runs, metadata=metadata),
            encoding="utf-8",
            xml_declaration=True,
        ),
    )
    RunStore.write_json(
        paths["sarif"],
        _sarif(
            findings,
            verdict=verdict,
            environments=metadata.get("environments", {}),
        ),
    )
    for finding in findings:
        _write_finding_artifacts(run_path, finding)
    return paths


def render_markdown(results: Iterable[DifferentialRunResult], *, metadata: dict[str, Any]) -> str:
    runs = tuple(results)
    findings = tuple(finding for result in runs for finding in result.findings)
    verdict = _aggregate_verdict(runs)
    gating = tuple(finding for finding in findings if finding.should_gate)
    assessments = tuple(item for result in runs for item in result.assessments)
    counts: dict[str, int] = {}
    for assessment in assessments:
        key = assessment.classification.kind.value
        counts[key] = counts.get(key, 0) + 1
    lines = [
        f"# Evalyn diff report — {metadata['run_id']}",
        "",
        f"- Base: `{metadata['base_sha']}`",
        f"- Candidate: `{metadata['candidate_sha']}`",
        f"- Adapters: {', '.join(result.adapter for result in runs) or 'none'}",
        f"- Probes: {len(assessments)}",
        f"- Verdict: `{verdict.value}`",
        f"- Stable changes: {counts.get('stable_change', 0)}",
        f"- Gating findings: {len(gating)}",
        f"- No differential observed: {counts.get('unchanged', 0)}",
        f"- Baseline failures: {counts.get('baseline_failure', 0)}",
        f"- Unstable: {counts.get('unstable', 0)}",
        f"- Invalid/infrastructure: {counts.get('invalid', 0)}",
        "",
    ]
    environments = metadata.get("environments", {})
    if isinstance(environments, dict):
        for adapter, fingerprint in sorted(environments.items()):
            if not isinstance(fingerprint, dict):
                continue
            digest = fingerprint.get("fingerprint_sha256") or "incomplete"
            lines.insert(-1, f"- {adapter} environment: `{digest}`")
    if not findings:
        if verdict == RunVerdict.NO_DIFFERENTIAL_OBSERVED:
            lines.extend(
                (
                    "No differential was observed by the generated probes. "
                    "This is not proof that the change is regression-free.",
                    "",
                )
            )
        else:
            lines.extend(
                (
                    "Evalyn did not obtain enough conclusive evidence to make a clean claim.",
                    "",
                )
            )
    for finding in findings:
        label = "LIKELY REGRESSION" if finding.should_gate else finding.semantic_state.value.upper()
        base_passes = sum(record.result.status.value == "passed" for record in finding.base_runs)
        candidate_failures = sum(
            record.result.status.value in {"failed", "timeout"} for record in finding.candidate_runs
        )
        evidence = finding.evidence
        environment_integrity = (
            "verified" if evidence is not None and evidence.environment_integrity else "incomplete"
        )
        target_binding = (
            "verified"
            if evidence is not None and evidence.target_binding_verified
            else "unverified"
        )
        oracle_binding = (
            "verified"
            if evidence is not None and evidence.oracle_binding_verified
            else "unverified"
        )
        target_intervention = (
            "verified"
            if evidence is not None and evidence.target_intervention_verified
            else "unverified"
        )
        lines.extend(
            (
                f"## {label} — {finding.finding_id}",
                "",
                finding.title,
                "",
                f"- Oracle: {finding.oracle or 'Not grounded'}",
                f"- Provenance: `{finding.oracle_provenance or 'none'}`",
                f"- Oracle source: "
                f"`{finding.probe_metadata.get('oracle_grounding_source', 'unverified')}` "
                f"({finding.probe_metadata.get('oracle_source_side', 'unknown')})",
                f"- Assessment: `{finding.semantic_state.value}` — {finding.semantic_reason}",
                f"- Base: PASS {base_passes}/{len(finding.base_runs)}",
                f"- Candidate: FAIL {candidate_failures}/{len(finding.candidate_runs)}",
                f"- Failure signature: `{finding.classification.failure_signature}`",
                f"- Change target: "
                f"`{finding.probe_metadata.get('change_target_id', 'unverified')}`",
                f"- Changed symbol: `{finding.probe_metadata.get('changed_symbol', 'unknown')}`",
                f"- Static target binding: {target_binding}",
                f"- Static oracle binding: {oracle_binding}",
                f"- Target-revert intervention: {target_intervention}",
                f"- Confirmed executions: base {len(finding.base_runs)}, "
                f"candidate {len(finding.candidate_runs)}",
                f"- Corroborating probes: "
                f"{evidence.corroboration_count if evidence is not None else 0}",
                f"- Environment integrity: {environment_integrity}",
                f"- Reproduce: `evalyn reproduce {finding.finding_id}`",
                "",
            )
        )
    lines.extend(
        (
            "## Interpretation",
            "",
            "Findings are repeatable behavior changes. Only entries labeled `LIKELY REGRESSION` "
            "satisfied the automatic evidence gate, and they still require maintainer review. "
            "Use `evalyn triage` to record a decision.",
            "",
        )
    )
    return "\n".join(lines)


def _write_finding_artifacts(run_path: Path, finding: Finding) -> None:
    root = run_path / "findings" / finding.finding_id
    suffix = ".py" if finding.artifact.adapter == "pytest" else ".test.ts"
    source_path = root / f"test-source{suffix}"
    RunStore.write_text(source_path, finding.probe_source)
    patch = "".join(
        difflib.unified_diff(
            [],
            finding.probe_source.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile=f"b/{finding.artifact.relative_path}",
        )
    )
    RunStore.write_text(root / "test.patch", patch)
    RunStore.write_json(
        root / "oracle.json",
        {
            "oracle": finding.oracle,
            "provenance": finding.oracle_provenance,
            "admissible": finding.oracle_admissible,
            "semantic_state": finding.semantic_state.value,
            "semantic_reason": finding.semantic_reason,
        },
    )
    RunStore.write_json(
        root / "reproduce.json",
        {
            "finding_id": finding.finding_id,
            "base_sha": finding.base_sha,
            "candidate_sha": finding.candidate_sha,
            "adapter": finding.artifact.adapter,
            "command": list(finding.artifact.command),
            "source_sha256": finding.artifact.source_sha256,
        },
    )


def _junit(
    results: tuple[DifferentialRunResult, ...],
    *,
    metadata: dict[str, Any],
) -> ElementTree.Element:
    assessments = tuple(item for result in results for item in result.assessments)
    findings_by_probe = {
        finding.artifact.probe_id: finding for result in results for finding in result.findings
    }
    inconclusive = {
        ClassificationKind.BASELINE_FAILURE,
        ClassificationKind.UNSTABLE,
        ClassificationKind.IMPROVEMENT,
    }
    skipped_count = sum(
        assessment.classification.kind == ClassificationKind.STABLE_CHANGE
        and not (
            (finding := findings_by_probe.get(assessment.artifact.probe_id)) and finding.should_gate
        )
        for assessment in assessments
    )
    error_count = sum(
        assessment.classification.kind in ({ClassificationKind.INVALID} | inconclusive)
        for assessment in assessments
    )
    verdict = _aggregate_verdict(results)
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": "evalyn",
            "tests": str(len(assessments)),
            "failures": str(sum(finding.should_gate for finding in findings_by_probe.values())),
            "errors": str(error_count),
            "skipped": str(skipped_count),
        },
    )
    properties = ElementTree.SubElement(suite, "properties")
    ElementTree.SubElement(
        properties,
        "property",
        {"name": "evalyn.verdict", "value": verdict.value},
    )
    environments = metadata.get("environments", {})
    if isinstance(environments, dict):
        for adapter, fingerprint in sorted(environments.items()):
            if not isinstance(adapter, str) or not isinstance(fingerprint, dict):
                continue
            digest = fingerprint.get("fingerprint_sha256")
            if isinstance(digest, str) and digest:
                ElementTree.SubElement(
                    properties,
                    "property",
                    {"name": f"evalyn.environment.{adapter}", "value": digest},
                )
    for assessment in assessments:
        case = ElementTree.SubElement(
            suite,
            "testcase",
            {"classname": assessment.artifact.adapter, "name": assessment.artifact.probe_id},
        )
        finding = findings_by_probe.get(assessment.artifact.probe_id)
        if finding and finding.should_gate:
            failure = ElementTree.SubElement(case, "failure", {"type": "likely_regression"})
            failure.text = finding.title
        elif assessment.classification.kind == ClassificationKind.INVALID:
            error = ElementTree.SubElement(case, "error", {"type": "invalid_probe"})
            error.text = assessment.classification.reason
        elif assessment.classification.kind in inconclusive:
            error = ElementTree.SubElement(case, "error", {"type": "insufficient_evidence"})
            error.text = assessment.classification.reason
        elif assessment.classification.kind == ClassificationKind.STABLE_CHANGE:
            skipped = ElementTree.SubElement(
                case,
                "skipped",
                {"type": assessment.classification.kind.value},
            )
            skipped.text = assessment.classification.reason
    return suite


def _sarif(
    findings: tuple[Finding, ...],
    *,
    verdict: RunVerdict,
    environments: object,
) -> dict[str, Any]:
    results = []
    for finding in findings:
        level = (
            "error"
            if finding.should_gate
            else ("note" if finding.semantic_state == SemanticState.INTENDED_CHANGE else "warning")
        )
        item: dict[str, Any] = {
            "ruleId": (
                "evalyn/likely-regression" if finding.should_gate else "evalyn/behavior-change"
            ),
            "level": level,
            "message": {"text": f"{finding.title} ({finding.finding_id})"},
            "properties": {
                "findingId": finding.finding_id,
                "semanticState": finding.semantic_state.value,
                "oracleProvenance": finding.oracle_provenance,
                "evidence": finding.evidence.to_dict() if finding.evidence is not None else None,
            },
        }
        changed_file = finding.probe_metadata.get("changed_file", "")
        if changed_file:
            item["locations"] = [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": changed_file},
                    }
                }
            ]
        results.append(item)
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Evalyn",
                        "informationUri": "https://github.com/ryan7626/Evalyn",
                        "rules": [
                            {
                                "id": "evalyn/likely-regression",
                                "shortDescription": {
                                    "text": "Automatic or maintainer-confirmed regression gate"
                                },
                            },
                            {
                                "id": "evalyn/behavior-change",
                                "shortDescription": {
                                    "text": "Repeatable behavior change requiring review"
                                },
                            },
                        ],
                    }
                },
                "invocations": [
                    {
                        "executionSuccessful": verdict
                        not in {RunVerdict.INCOMPLETE, RunVerdict.INSUFFICIENT_EVIDENCE},
                        "properties": {"verdict": verdict.value},
                    }
                ],
                "properties": {
                    "verdict": verdict.value,
                    "environments": environments if isinstance(environments, dict) else {},
                },
                "results": results,
            }
        ],
    }


def _aggregate_verdict(results: tuple[DifferentialRunResult, ...]) -> RunVerdict:
    verdicts = {result.verdict for result in results}
    if RunVerdict.INCOMPLETE in verdicts:
        return RunVerdict.INCOMPLETE
    if RunVerdict.REGRESSION_FOUND in verdicts:
        return RunVerdict.REGRESSION_FOUND
    if RunVerdict.CHANGE_OBSERVED in verdicts:
        return RunVerdict.CHANGE_OBSERVED
    if verdicts and verdicts == {RunVerdict.NO_DIFFERENTIAL_OBSERVED}:
        return RunVerdict.NO_DIFFERENTIAL_OBSERVED
    return RunVerdict.INSUFFICIENT_EVIDENCE
