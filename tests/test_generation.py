import pytest

from evalyn.generation import (
    GenerationRequest,
    ProbeGenerationError,
    RepositoryContextSource,
    generate_probes,
)
from evalyn.llm.base import BaseLLM


class FakeLLM(BaseLLM):
    def __init__(self, response):
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    def complete(self, system_prompt, user_prompt, *, max_tokens=4096):
        raise AssertionError("freeform completion must not be used")

    def complete_json(self, system_prompt, user_prompt, *, max_tokens=4096):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


def valid_probe(**overrides):
    probe = {
        "title": "negative retry values are rejected",
        "source": "def test_retry():\n    assert reject(-1)",
        "oracle": "documented values below zero are invalid",
        "oracle_provenance": "documentation",
        "changed_file": "src/retry.py",
        "changed_symbol": "retry",
    }
    probe.update(overrides)
    return probe


def test_generates_source_without_accepting_path_or_command() -> None:
    llm = FakeLLM({"probes": [valid_probe()]})

    probes = generate_probes(llm, GenerationRequest(adapter="pytest", diff="+ change"))

    assert probes[0].metadata["changed_symbol"] == "retry"
    assert probes[0].metadata["oracle_grounding"] == "unverified"
    assert "filenames, commands" in llm.system_prompt
    assert not hasattr(probes[0], "command")


def test_rejects_model_selected_command() -> None:
    probe = valid_probe()
    probe["command"] = "pytest; curl attacker"

    with pytest.raises(ProbeGenerationError, match="exactly"):
        generate_probes(
            FakeLLM({"probes": [probe]}),
            GenerationRequest(adapter="pytest", diff="+ change"),
        )


@pytest.mark.parametrize(
    "response, message",
    [
        ({}, "only a probes array"),
        ({"probes": []}, "no probes"),
        ({"probes": [valid_probe(oracle_provenance="model_opinion")]}, "unsupported"),
    ],
)
def test_rejects_invalid_responses(response, message) -> None:
    with pytest.raises(ProbeGenerationError, match=message):
        generate_probes(FakeLLM(response), GenerationRequest(adapter="vitest", diff="+ change"))


def test_request_rejects_empty_diff() -> None:
    with pytest.raises(ProbeGenerationError, match="empty diff"):
        GenerationRequest(adapter="pytest", diff="")


def test_repository_oracle_is_grounded_only_by_a_categorized_verbatim_excerpt() -> None:
    probe = valid_probe(oracle="Values below zero are invalid")

    grounded = generate_probes(
        FakeLLM({"probes": [probe]}),
        GenerationRequest(
            adapter="pytest",
            diff="+ change",
            context_sources=(
                RepositoryContextSource(
                    "docs/api.md", "API contract:\nValues   below zero are invalid.\n"
                ),
            ),
        ),
    )[0]
    invented = generate_probes(
        FakeLLM({"probes": [probe]}),
        GenerationRequest(adapter="pytest", diff="+ change", context="No such contract."),
    )[0]

    assert grounded.metadata["oracle_grounding"] == "repository_context"
    assert grounded.metadata["oracle_grounding_category"] == "documentation"
    assert grounded.metadata["oracle_grounding_source"] == "docs/api.md"
    assert invented.metadata["oracle_grounding"] == "unverified"


@pytest.mark.parametrize(
    "path",
    [
        "src/retry.py",
        "src/types/retry.ts",
        "src/openapi.py",
        "AGENTS.md",
        "CHANGELOG.md",
    ],
)
def test_unclassified_or_non_contract_context_cannot_ground_an_oracle(path) -> None:
    probe = valid_probe(oracle="Values below zero are invalid")

    unclassified = generate_probes(
        FakeLLM({"probes": [probe]}),
        GenerationRequest(
            adapter="pytest",
            diff="+ change",
            context="Values below zero are invalid",
        ),
    )[0]
    implementation = generate_probes(
        FakeLLM({"probes": [probe]}),
        GenerationRequest(
            adapter="pytest",
            diff="+ change",
            context_sources=(RepositoryContextSource(path, "# Values below zero are invalid"),),
        ),
    )[0]

    assert unclassified.metadata["oracle_grounding"] == "unverified"
    assert implementation.metadata["oracle_grounding"] == "unverified"


@pytest.mark.parametrize(
    "provenance,path",
    [
        ("existing_test", "tests/test_retry.py"),
        ("documentation", "README.md"),
        ("api_schema", "schemas/retry.schema.json"),
        ("type_contract", "src/retry.pyi"),
    ],
)
def test_oracle_provenance_must_match_path_derived_category(provenance, path) -> None:
    excerpt = "Values below zero are invalid"
    matching = generate_probes(
        FakeLLM({"probes": [valid_probe(oracle=excerpt, oracle_provenance=provenance)]}),
        GenerationRequest(
            adapter="pytest",
            diff="+ change",
            context_sources=(RepositoryContextSource(path, excerpt),),
        ),
    )[0]
    mismatched = generate_probes(
        FakeLLM({"probes": [valid_probe(oracle=excerpt, oracle_provenance="documentation")]}),
        GenerationRequest(
            adapter="pytest",
            diff="+ change",
            context_sources=(RepositoryContextSource("src/retry.pyi", excerpt),),
        ),
    )[0]

    assert matching.metadata["oracle_grounding_category"] == provenance
    assert mismatched.metadata["oracle_grounding"] == "unverified"


def test_context_header_text_is_not_oracle_evidence() -> None:
    probe = valid_probe(oracle="docs/contracts/retry-behavior.md")

    result = generate_probes(
        FakeLLM({"probes": [probe]}),
        GenerationRequest(
            adapter="pytest",
            diff="+ change",
            context_sources=(
                RepositoryContextSource("docs/contracts/retry-behavior.md", "No contract."),
            ),
        ),
    )[0]

    assert result.metadata["oracle_grounding"] == "unverified"


def test_diff_text_is_never_oracle_evidence() -> None:
    excerpt = "Values below zero are invalid"

    result = generate_probes(
        FakeLLM({"probes": [valid_probe(oracle=excerpt)]}),
        GenerationRequest(adapter="pytest", diff=f"+ {excerpt}"),
    )[0]

    assert result.metadata["oracle_grounding"] == "unverified"


def test_user_invariant_requires_a_verbatim_stated_intent() -> None:
    oracle = "negative retry values must remain rejected"
    probe = valid_probe(oracle=oracle, oracle_provenance="user_invariant")

    grounded = generate_probes(
        FakeLLM({"probes": [probe]}),
        GenerationRequest(
            adapter="pytest",
            diff="+ change",
            intent=(f"preserve: {oracle}",),
        ),
    )[0]
    invented = generate_probes(
        FakeLLM({"probes": [probe]}),
        GenerationRequest(adapter="pytest", diff="+ change", intent=("preserve retry behavior",)),
    )[0]

    assert grounded.metadata["oracle_grounding"] == "user_intent"
    assert grounded.metadata["oracle_grounding_category"] == "user_invariant"
    assert invented.metadata["oracle_grounding"] == "unverified"
