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
    payload = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "base_sha": metadata["base_sha"],
        "candidate_sha": metadata["candidate_sha"],
        "intent": list(metadata.get("intent", ())),
        "adapters": [result.adapter for result in runs],
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
        ElementTree.tostring(_junit(runs), encoding="utf-8", xml_declaration=True),
    )
    RunStore.write_json(paths["sarif"], _sarif(findings))
    for finding in findings:
        _write_finding_artifacts(run_path, finding)
    return paths


def render_markdown(results: Iterable[DifferentialRunResult], *, metadata: dict[str, Any]) -> str:
    runs = tuple(results)
    findings = tuple(finding for result in runs for finding in result.findings)
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
        f"- Stable changes: {counts.get('stable_change', 0)}",
        f"- Gating findings: {len(gating)}",
        "",
    ]
    if not findings:
        lines.extend(("No stable behavior changes were found.", ""))
    for finding in findings:
        label = "LIKELY REGRESSION" if finding.should_gate else finding.semantic_state.value.upper()
        base_passes = sum(record.result.status.value == "passed" for record in finding.base_runs)
        candidate_failures = sum(
            record.result.status.value in {"failed", "timeout"} for record in finding.candidate_runs
        )
        lines.extend(
            (
                f"## {label} — {finding.finding_id}",
                "",
                finding.title,
                "",
                f"- Oracle: {finding.oracle or 'Not grounded'}",
                f"- Provenance: `{finding.oracle_provenance or 'none'}`",
                f"- Base: PASS {base_passes}/{len(finding.base_runs)}",
                f"- Candidate: FAIL {candidate_failures}/{len(finding.candidate_runs)}",
                f"- Failure signature: `{finding.classification.failure_signature}`",
                f"- Changed symbol: `{finding.probe_metadata.get('changed_symbol', 'unknown')}`",
                f"- Reproduce: `evalyn reproduce {finding.finding_id}`",
                "",
            )
        )
    lines.extend(
        (
            "## Interpretation",
            "",
            "Automated findings are evidence-backed likely regressions, not verified regressions. "
            "Use `evalyn triage` to record a maintainer decision.",
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


def _junit(results: tuple[DifferentialRunResult, ...]) -> ElementTree.Element:
    assessments = tuple(item for result in results for item in result.assessments)
    findings_by_probe = {
        finding.artifact.probe_id: finding for result in results for finding in result.findings
    }
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": "evalyn",
            "tests": str(len(assessments)),
            "failures": str(sum(finding.should_gate for finding in findings_by_probe.values())),
            "errors": str(
                sum(item.classification.kind == ClassificationKind.INVALID for item in assessments)
            ),
        },
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
        elif assessment.classification.kind == ClassificationKind.UNSTABLE:
            skipped = ElementTree.SubElement(case, "skipped", {"type": "unstable"})
            skipped.text = assessment.classification.reason
    return suite


def _sarif(findings: tuple[Finding, ...]) -> dict[str, Any]:
    results = []
    for finding in findings:
        level = (
            "error"
            if finding.should_gate
            else ("note" if finding.semantic_state == SemanticState.INTENDED_CHANGE else "warning")
        )
        item: dict[str, Any] = {
            "ruleId": "evalyn/likely-regression",
            "level": level,
            "message": {"text": f"{finding.title} ({finding.finding_id})"},
            "properties": {
                "findingId": finding.finding_id,
                "semanticState": finding.semantic_state.value,
                "oracleProvenance": finding.oracle_provenance,
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
                                    "text": "Stable behavior change with an admissible oracle"
                                },
                            }
                        ],
                    }
                },
                "results": results,
            }
        ],
    }
