from dataclasses import replace

import pytest

from evalyn.engine.models import (
    ArtifactMetadata,
    ClassificationKind,
    ClassificationResult,
    EvidenceVector,
    ExecutionRecord,
    ExecutionResult,
    Finding,
    SemanticState,
)
from evalyn.engine.rules import IntentOracleAssessor


def finding(**overrides) -> Finding:
    base_runs = tuple(
        ExecutionRecord("base", "base", index, ExecutionResult.passed()) for index in range(1, 4)
    )
    candidate_runs = tuple(
        ExecutionRecord(
            "candidate",
            "candidate",
            index,
            ExecutionResult.failed(failure_signature="signature"),
        )
        for index in range(1, 4)
    )
    value = Finding(
        finding_id="EVL-12345678",
        run_id="run",
        title="negative retry values are rejected",
        base_sha="base",
        candidate_sha="candidate",
        artifact=ArtifactMetadata(
            probe_id="a" * 16,
            adapter="pytest",
            relative_path=".evalyn/probe.py",
            command=("python", "-m", "pytest", ".evalyn/probe.py"),
            source_sha256="b" * 64,
            byte_size=10,
            media_type="text/x-python",
        ),
        classification=ClassificationResult(
            ClassificationKind.STABLE_CHANGE, "stable", "signature"
        ),
        base_runs=base_runs,
        candidate_runs=candidate_runs,
        probe_source="def test_value(): assert True",
        oracle="the API documentation says negative retry values are rejected",
        oracle_provenance="documentation",
        probe_metadata={
            "oracle_grounding": "repository_context",
            "oracle_grounding_category": "documentation",
            "oracle_grounding_source": "docs/api.md",
            "oracle_source_side": "both",
            "change_target_id": "target_0123456789abcdef",
            "changed_file": "retry.py",
            "changed_symbol": "retry",
            "target_binding": "static_reference",
            "oracle_binding": "static_assertion_match",
        },
        evidence=EvidenceVector(
            causal_transition="base_pass_candidate_fail",
            base_repetitions=3,
            candidate_repetitions=3,
            repeatable=True,
            oracle_kind="documentation",
            oracle_source="docs/api.md",
            target_file="retry.py",
            target_symbol="retry",
            target_binding_verified=True,
            oracle_binding_verified=True,
            target_intervention_verified=True,
            target_intervention_repetitions=3,
            corroboration_count=2,
        ),
    )
    return replace(value, **overrides)


def test_grounded_documentation_remains_advisory_without_executable_binding() -> None:
    result = IntentOracleAssessor(("add retry logging",)).assess(finding())

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert result.oracle_admissible
    assert result.intent_match is False


def test_prose_oracle_remains_advisory_without_executable_binding() -> None:
    result = IntentOracleAssessor().assess(finding())

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert result.oracle_admissible
    assert "no trusted executable oracle-binding verifier" in result.reason


def test_unchanged_machine_contract_can_gate_one_three_repeat_probe() -> None:
    item = finding(
        oracle_provenance="existing_test",
        probe_metadata={
            **finding().probe_metadata,
            "oracle_grounding_category": "existing_test",
            "oracle_grounding_source": "tests/test_retry.py",
        },
        evidence=replace(
            finding().evidence,
            oracle_kind="existing_test",
            oracle_source="tests/test_retry.py",
            corroboration_count=1,
        ),
    )

    result = IntentOracleAssessor().assess(item)

    assert result.state == SemanticState.LIKELY_REGRESSION


@pytest.mark.parametrize("category", ("api_schema", "type_contract"))
def test_non_executable_machine_contracts_remain_advisory(category: str) -> None:
    item = finding(
        oracle_provenance=category,
        probe_metadata={
            **finding().probe_metadata,
            "oracle_grounding_category": category,
            "oracle_grounding_source": "contracts/api.schema.json",
        },
        evidence=replace(
            finding().evidence,
            oracle_kind=category,
            oracle_source="contracts/api.schema.json",
        ),
    )

    result = IntentOracleAssessor().assess(item)

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert "no trusted executable oracle-binding verifier" in result.reason


def test_changed_or_candidate_only_contract_remains_advisory() -> None:
    item = finding(
        probe_metadata={**finding().probe_metadata, "oracle_source_side": "base"},
    )

    result = IntentOracleAssessor().assess(item)

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert "not verified as unchanged" in result.reason


def test_fewer_than_three_repetitions_cannot_auto_gate() -> None:
    item = finding(base_runs=finding().base_runs[:2])

    result = IntentOracleAssessor().assess(item)

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert "three-repeat" in result.reason


def test_unverified_or_missing_environment_evidence_cannot_auto_gate() -> None:
    unverified = finding(evidence=replace(finding().evidence, environment_integrity=False))
    missing = finding(evidence=None)

    assert IntentOracleAssessor().assess(unverified).state == SemanticState.UNCERTAIN_CHANGE
    assert IntentOracleAssessor().assess(missing).state == SemanticState.UNCERTAIN_CHANGE


def test_unverified_target_or_oracle_binding_cannot_auto_gate() -> None:
    unbound_target = finding(evidence=replace(finding().evidence, target_binding_verified=False))
    unbound_oracle = finding(evidence=replace(finding().evidence, oracle_binding_verified=False))

    assert IntentOracleAssessor().assess(unbound_target).state == SemanticState.UNCERTAIN_CHANGE
    assert IntentOracleAssessor().assess(unbound_oracle).state == SemanticState.UNCERTAIN_CHANGE


def test_missing_target_revert_intervention_cannot_auto_gate() -> None:
    item = finding(
        evidence=replace(
            finding().evidence,
            target_intervention_verified=False,
            target_intervention_repetitions=0,
        )
    )

    assert IntentOracleAssessor().assess(item).state == SemanticState.UNCERTAIN_CHANGE


def test_topical_overlap_alone_remains_uncertain() -> None:
    result = IntentOracleAssessor(("negative retry values are rejected",)).assess(finding())

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert result.intent_match is None


def test_verbatim_preservation_invariant_requires_maintainer_review() -> None:
    result = IntentOracleAssessor(
        ("preserve: the API documentation says negative retry values are rejected",)
    ).assess(finding())

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert result.intent_match is None


def test_explicit_opposite_change_is_non_gating_intended_change() -> None:
    result = IntentOracleAssessor(
        ("change: allow negative retry values instead of rejecting them",)
    ).assess(finding())

    assert result.state == SemanticState.INTENDED_CHANGE
    assert result.intent_match is True


def test_unverified_model_oracle_cannot_self_suppress_as_an_intended_change() -> None:
    result = IntentOracleAssessor(
        ("change: allow negative retry values instead of rejecting them",)
    ).assess(
        finding(
            oracle_provenance="none",
            probe_metadata={"oracle_grounding": "unverified"},
        )
    )

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert not result.oracle_admissible
    assert result.intent_match is None


def test_ambiguous_change_description_is_uncertain_even_with_repository_oracle() -> None:
    result = IntentOracleAssessor(("change negative retry handling",)).assess(finding())

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert result.oracle_admissible
    assert result.intent_match is None


def test_model_controlled_title_cannot_make_unrelated_intent_suppress_a_finding() -> None:
    result = IntentOracleAssessor(("change payment timeout handling",)).assess(
        finding(title="change payment timeout handling")
    )

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert result.oracle_admissible
    assert result.intent_match is False


def test_model_opinion_remains_uncertain() -> None:
    result = IntentOracleAssessor().assess(
        finding(oracle_provenance="none", oracle="the model expects this behavior")
    )

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert not result.oracle_admissible


def test_user_invariant_requires_supplied_intent() -> None:
    result = IntentOracleAssessor().assess(finding(oracle_provenance="user_invariant"))

    assert result.state == SemanticState.UNCERTAIN_CHANGE


def test_model_claimed_documentation_without_verified_excerpt_is_uncertain() -> None:
    result = IntentOracleAssessor().assess(
        finding(probe_metadata={"oracle_grounding": "unverified"})
    )

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert not result.oracle_admissible


def test_grounding_category_must_match_model_provenance() -> None:
    result = IntentOracleAssessor().assess(
        finding(
            probe_metadata={
                "oracle_grounding": "repository_context",
                "oracle_grounding_category": "existing_test",
                "oracle_grounding_source": "tests/test_api.py",
            }
        )
    )

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert not result.oracle_admissible
