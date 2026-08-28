from dataclasses import replace

from evalyn.engine.models import (
    ArtifactMetadata,
    ClassificationKind,
    ClassificationResult,
    Finding,
    SemanticState,
)
from evalyn.engine.rules import IntentOracleAssessor


def finding(**overrides) -> Finding:
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
        base_runs=(),
        candidate_runs=(),
        probe_source="def test_value(): assert True",
        oracle="the API documentation says negative retry values are rejected",
        oracle_provenance="documentation",
        probe_metadata={
            "oracle_grounding": "repository_context",
            "oracle_grounding_category": "documentation",
            "oracle_grounding_source": "docs/api.md",
        },
    )
    return replace(value, **overrides)


def test_grounded_unintended_change_is_likely() -> None:
    result = IntentOracleAssessor(("add retry logging",)).assess(finding())

    assert result.state == SemanticState.LIKELY_REGRESSION
    assert result.oracle_admissible
    assert result.intent_match is False


def test_topical_overlap_alone_remains_uncertain() -> None:
    result = IntentOracleAssessor(("negative retry values are rejected",)).assess(finding())

    assert result.state == SemanticState.UNCERTAIN_CHANGE
    assert result.intent_match is None


def test_verbatim_preservation_invariant_is_a_grounded_regression() -> None:
    result = IntentOracleAssessor(
        ("preserve: the API documentation says negative retry values are rejected",)
    ).assess(finding())

    assert result.state == SemanticState.LIKELY_REGRESSION
    assert result.intent_match is True


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

    assert result.state == SemanticState.LIKELY_REGRESSION
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
