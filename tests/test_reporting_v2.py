import json
import os
from pathlib import Path

import pytest

from evalyn.adapters.pytest import PytestAdapter
from evalyn.engine.models import ExecutionResult, ProbeSource, Revision
from evalyn.engine.orchestrator import DifferentialEngine
from evalyn.engine.rules import IntentOracleAssessor
from evalyn.reporting_v2 import write_reports


class Executor:
    def execute(self, revision, artifact, *, repetition):
        if revision.sha == "base":
            return ExecutionResult.passed(metadata={"environment_integrity": "verified"})
        return ExecutionResult.failed(
            failure_signature="AssertionError:value",
            metadata={"environment_integrity": "verified"},
        )


def result(tmp_path: Path):
    probe = ProbeSource(
        source="def test_value():\n    assert value() == 1",
        title="value remains one",
        oracle="the existing test guarantees value one",
        oracle_provenance="existing_test",
        metadata={
            "changed_file": "src/value.py",
            "changed_symbol": "value",
            "change_target_id": "target_0123456789abcdef",
            "target_binding": "static_reference",
            "oracle_binding": "static_assertion_match",
            "target_intervention": "candidate_target_revert_passed",
            "target_intervention_repetitions": "3",
            "oracle_grounding": "repository_context",
            "oracle_grounding_category": "existing_test",
            "oracle_grounding_source": "tests/test_value.py",
            "oracle_source_side": "both",
        },
    )
    return DifferentialEngine(
        PytestAdapter(), Executor(), assessor=IntentOracleAssessor()
    ).evaluate(
        Revision("base", tmp_path / "base"),
        Revision("candidate", tmp_path / "candidate"),
        [probe],
        run_id="run-1",
    )


def test_writes_all_report_formats_and_finding_artifacts(tmp_path: Path) -> None:
    run = result(tmp_path)

    paths = write_reports(
        tmp_path,
        [run],
        metadata={
            "run_id": "run-1",
            "base_sha": "base",
            "candidate_sha": "candidate",
            "intent": (),
            "environments": {"pytest": {"fingerprint_sha256": "sha256:" + "a" * 64}},
        },
    )

    assert set(paths) == {"json", "markdown", "junit", "sarif"}
    assert all(path.is_file() for path in paths.values())
    payload = json.loads(paths["json"].read_text())
    assert payload["findings"][0]["should_gate"] is True
    assert payload["verdict"] == "regression_found"
    assessment = payload["results"][0]["assessments"][0]
    assert assessment["classification"]["kind"] == "stable_change"
    assert [item["status"] for item in assessment["base_runs"]] == [
        "passed",
        "passed",
        "passed",
    ]
    finding_id = payload["findings"][0]["finding_id"]
    assert (tmp_path / "findings" / finding_id / "test.patch").is_file()
    assert "LIKELY REGRESSION" in paths["markdown"].read_text()
    junit = paths["junit"].read_text(encoding="utf-8")
    assert 'name="evalyn.verdict" value="regression_found"' in junit
    assert 'name="evalyn.environment.pytest" value="sha256:' in junit


def test_sarif_points_to_changed_file(tmp_path: Path) -> None:
    paths = write_reports(
        tmp_path,
        [result(tmp_path)],
        metadata={
            "run_id": "run-1",
            "base_sha": "base",
            "candidate_sha": "candidate",
        },
    )

    sarif = json.loads(paths["sarif"].read_text())
    assert sarif["runs"][0]["properties"]["verdict"] == "regression_found"
    assert sarif["runs"][0]["invocations"][0]["executionSuccessful"] is True
    location = sarif["runs"][0]["results"][0]["locations"][0]
    assert location["physicalLocation"]["artifactLocation"]["uri"] == "src/value.py"


def test_reports_do_not_follow_a_preplanted_output_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "victim.md"
    victim.write_text("keep", encoding="utf-8")
    (tmp_path / "report.md").symlink_to(victim)

    with pytest.raises(ValueError, match="linked/non-file"):
        write_reports(
            tmp_path,
            [result(tmp_path)],
            metadata={
                "run_id": "run-1",
                "base_sha": "base",
                "candidate_sha": "candidate",
            },
        )

    assert victim.read_text(encoding="utf-8") == "keep"


def test_reports_atomically_replace_hardlinks_without_mutating_their_peer(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim.xml"
    victim.write_text("keep", encoding="utf-8")
    junit = tmp_path / "junit.xml"
    try:
        os.link(victim, junit)
    except OSError:
        pytest.skip("hardlinks are unavailable")

    paths = write_reports(
        tmp_path,
        [result(tmp_path)],
        metadata={
            "run_id": "run-1",
            "base_sha": "base",
            "candidate_sha": "candidate",
        },
    )

    assert victim.read_text(encoding="utf-8") == "keep"
    assert paths["junit"].read_text(encoding="utf-8").startswith("<?xml")
    assert not os.path.samefile(victim, paths["junit"])
