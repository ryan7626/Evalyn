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
            return ExecutionResult.passed()
        return ExecutionResult.failed(failure_signature="AssertionError:value")


def result(tmp_path: Path):
    probe = ProbeSource(
        source="def test_value():\n    assert value() == 1",
        title="value remains one",
        oracle="the public documentation guarantees value one",
        oracle_provenance="documentation",
        metadata={
            "changed_file": "src/value.py",
            "changed_symbol": "value",
            "oracle_grounding": "repository_context",
            "oracle_grounding_category": "documentation",
            "oracle_grounding_source": "docs/api.md",
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
        },
    )

    assert set(paths) == {"json", "markdown", "junit", "sarif"}
    assert all(path.is_file() for path in paths.values())
    payload = json.loads(paths["json"].read_text())
    assert payload["findings"][0]["should_gate"] is True
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
