import re

import pytest

from evalyn.analysis import build_change_graph
from evalyn.generation import (
    GenerationRequest,
    ProbeGenerationError,
    ProbeInfrastructureError,
    ProbeValidationError,
    _probe_binds_target,
    generate_probe_portfolio,
    generate_probes,
)
from evalyn.llm.base import BaseLLM


class SequenceLLM(BaseLLM):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system_prompt, user_prompt, *, max_tokens=4096):
        raise AssertionError("freeform completion must not be used")

    def complete_json(self, system_prompt, user_prompt, *, max_tokens=4096):
        self.calls.append((system_prompt, user_prompt, max_tokens))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(system_prompt, user_prompt)
        return response


def _graph(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    for root, multiplier in ((base, "2"), (candidate, "3")):
        source = (
            "def delay(attempt):\n"
            "    if attempt < 0:\n"
            "        raise ValueError('attempt must be non-negative')\n"
            f"    return attempt * {multiplier}\n"
        )
        path = root / "retry.py"
        path.write_text(source, encoding="utf-8")
        tests = root / "tests"
        tests.mkdir()
        (tests / "test_retry.py").write_text(
            "from retry import delay\n\n"
            "def test_negative():\n"
            "    try:\n"
            "        delay(-1)\n"
            "        assert False, 'negative attempts must be rejected'\n"
            "    except ValueError:\n"
            "        assert True\n",
            encoding="utf-8",
        )
    diff = """\
diff --git a/retry.py b/retry.py
--- a/retry.py
+++ b/retry.py
@@ -1,4 +1,4 @@
 def delay(attempt):
     if attempt < 0:
         raise ValueError('attempt must be non-negative')
-    return attempt * 2
+    return attempt * 3
"""
    return build_change_graph(base, candidate, diff=diff), diff


def _hypothesis(title, category, target_id, oracle_id=""):
    return {
        "title": title,
        "category": category,
        "target_id": target_id,
        "behavior": f"probe {title}",
        "oracle_id": oracle_id,
    }


def test_portfolio_salvages_candidates_and_corrects_each_at_most_once(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    oracle = next(oracle for oracle in graph.oracles if oracle.category == "existing_test")
    plan = {
        "hypotheses": [
            _hypothesis("negative input", "error", target.target_id, oracle.oracle_id),
            _hypothesis("zero boundary", "boundary", target.target_id),
        ]
    }
    llm = SequenceLLM(
        [
            plan,
            {"source": "def test_bad():\n    value = 1"},
            {"source": "def test_negative():\n    assert True"},
            {"source": "def test_zero():\n    assert 0 == 0"},
        ]
    )
    preflight_calls = []

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph, max_tests=8),
        preflight=lambda adapter, source: preflight_calls.append((adapter, source)),
    )

    assert len(portfolio.probes) == 2
    assert [attempt.status for attempt in portfolio.attempts] == [
        "rejected",
        "accepted",
        "accepted",
    ]
    assert portfolio.attempts[0].rejection_code == "adapter_validation_failed"
    assert len(llm.calls) == 4
    assert len(preflight_calls) == 2
    first = portfolio.probes[0]
    assert first.metadata["changed_file"] == "retry.py"
    assert first.metadata["changed_symbol"] == "delay"
    assert first.metadata["change_target_id"] == target.target_id
    assert first.metadata["oracle_id"] == oracle.oracle_id
    assert first.metadata["oracle_grounding_category"] == "existing_test"
    assert first.metadata["oracle_source_side"] in {"base", "both"}
    assert first.metadata["target_binding"] == "unverified"
    assert first.metadata["oracle_binding"] == "unverified"
    assert f"compatible_targets={target.target_id}" in llm.calls[0][1]


def test_planning_salvages_only_known_targets_and_oracles(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    plan = {
        "hypotheses": [
            _hypothesis("invented location", "boundary", "target_0000000000000000"),
            _hypothesis(
                "invented oracle",
                "error",
                target.target_id,
                "oracle_0000000000000000",
            ),
            _hypothesis("valid", "boundary", target.target_id),
        ]
    }
    llm = SequenceLLM([plan, {"source": "def test_valid():\n    assert True"}])

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
    )

    assert len(portfolio.hypotheses) == 1
    assert len(portfolio.planning_errors) == 2
    assert portfolio.probes[0].metadata["changed_file"] == "retry.py"
    assert portfolio.probes[0].metadata["oracle_grounding"] == "unverified"


def test_schema_failure_isolated_and_extra_model_locations_are_never_trusted(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    plan = {
        "hypotheses": [
            _hypothesis("first", "boundary", target.target_id),
            _hypothesis("second", "error", target.target_id),
        ]
    }
    llm = SequenceLLM(
        [
            plan,
            {
                "source": "def test_bad():\n    assert True",
                "changed_file": "attacker/choice.py",
            },
            {"still": "invalid"},
            {"source": "def test_second():\n    assert True"},
        ]
    )

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
    )

    assert len(portfolio.probes) == 1
    assert len(portfolio.attempts) == 3
    assert portfolio.attempts[0].rejection_code == "schema_invalid"
    assert portfolio.attempts[1].rejection_code == "schema_invalid"
    assert portfolio.probes[0].metadata["changed_file"] == "retry.py"


def test_preflight_feedback_is_sanitized_before_single_correction(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    plan = {"hypotheses": [_hypothesis("collection", "boundary", target.target_id)]}
    llm = SequenceLLM(
        [
            plan,
            {"source": "def test_first():\n    assert True"},
            {"source": "def test_corrected():\n    assert True"},
        ]
    )

    class Preflight:
        calls = 0

        def __call__(self, adapter, source):
            self.calls += 1
            if self.calls == 1:
                raise ProbeValidationError(
                    "collection_failed",
                    "The probe did not collect at /Users/private/repository/file.py",
                )

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
        preflight=Preflight(),
    )

    correction_prompt = llm.calls[2][1]
    assert "/Users/private" not in correction_prompt
    assert "The probe did not collect at <path>" in correction_prompt
    assert len(portfolio.probes) == 1
    assert [attempt.attempt_number for attempt in portfolio.attempts] == [1, 2]


def test_preflight_infrastructure_does_not_trigger_model_correction(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    llm = SequenceLLM(
        [
            {"hypotheses": [_hypothesis("collection", "boundary", target.target_id)]},
            {"source": "def test_first():\n    assert True"},
        ]
    )

    with pytest.raises(ProbeInfrastructureError, match="infrastructure"):
        generate_probe_portfolio(
            llm,
            GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
            preflight=lambda *_args: (_ for _ in ()).throw(RuntimeError("daemon unavailable")),
        )

    assert len(llm.calls) == 2


def test_static_bindings_require_executable_target_and_exact_trusted_assertion(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    oracle = next(oracle for oracle in graph.oracles if oracle.category == "existing_test")
    plan = {
        "hypotheses": [
            _hypothesis("bound", "error", target.target_id, oracle.oracle_id),
            _hypothesis("unbound", "error", target.target_id, oracle.oracle_id),
        ]
    }
    llm = SequenceLLM(
        [
            plan,
            {
                "source": (
                    "from retry import delay\n\n"
                    "def test_negative():\n"
                    "    try:\n"
                    "        delay(-1)\n"
                    "        assert False, 'negative attempts must be rejected'\n"
                    "    except ValueError:\n"
                    "        pass\n"
                )
            },
            {"source": "def test_unbound():\n    assert True"},
        ]
    )

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
    )

    assert portfolio.probes[0].metadata["target_binding"] == "static_reference"
    assert portfolio.probes[0].metadata["oracle_binding"] == "static_assertion_match"
    assert portfolio.probes[1].metadata["target_binding"] == "unverified"
    assert portfolio.probes[1].metadata["oracle_binding"] == "unverified"


def test_static_target_binding_correlates_imported_module_symbol_and_alias(tmp_path):
    graph, _diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")

    assert _probe_binds_target(
        "pytest", "from retry import delay as d\ndef test_x(): assert d(0) == 0", target
    )
    assert not _probe_binds_target(
        "pytest", "from evil.retry import delay\ndef test_x(): assert delay(0) == 0", target
    )
    assert not _probe_binds_target(
        "pytest",
        "from retry import other\ndef test_x(): assert object().delay() == 0",
        target,
    )
    assert _probe_binds_target(
        "vitest",
        "import { delay as d } from './retry';\ntest('x', () => expect(d(0)).toBe(0));",
        target,
    )
    assert not _probe_binds_target(
        "vitest",
        "import { delay } from './evil/retry';\ntest('x', () => expect(delay(0)).toBe(0));",
        target,
    )


def test_provider_failure_is_audited_separately_from_invalid_probe_source(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    plan = {
        "hypotheses": [
            _hypothesis("provider failure", "boundary", target.target_id),
            _hypothesis("survivor", "error", target.target_id),
        ]
    }
    llm = SequenceLLM(
        [
            plan,
            RuntimeError("remote payload with a secret"),
            {"bad": "response"},
            {"source": "def test_survivor():\n    assert True"},
        ]
    )

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
    )

    assert len(portfolio.probes) == 1
    assert [attempt.rejection_code for attempt in portfolio.attempts[:2]] == [
        "provider_infrastructure",
        "schema_invalid",
    ]
    assert portfolio.infrastructure_error_count == 1
    assert "secret" not in portfolio.attempts[0].feedback


def test_compatibility_wrapper_uses_portfolio_when_graph_is_present(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")
    llm = SequenceLLM(
        [
            {"hypotheses": [_hypothesis("wrapper", "boundary", target.target_id)]},
            {"source": "def test_wrapper():\n    assert True"},
        ]
    )

    probes = generate_probes(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
    )

    assert len(probes) == 1
    assert probes[0].metadata["change_target_id"] == target.target_id


def test_portfolio_rejects_a_plan_without_valid_hypotheses(tmp_path):
    graph, diff = _graph(tmp_path)
    llm = SequenceLLM([{"hypotheses": [{"untrusted": "shape"}]}])

    with pytest.raises(ProbeGenerationError, match="no valid graph-bound hypotheses"):
        generate_probe_portfolio(
            llm,
            GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
        )


def test_legacy_batch_is_rebound_without_trusting_location_or_provenance(tmp_path):
    graph, diff = _graph(tmp_path)
    llm = SequenceLLM(
        [
            {
                "probes": [
                    {
                        "title": "recorded probe",
                        "source": "def test_recorded():\n    assert True",
                        "oracle": "an invented expectation",
                        "oracle_provenance": "documentation",
                        "changed_file": "attacker/selected.py",
                        "changed_symbol": "attacker_symbol",
                    }
                ]
            }
        ]
    )

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(adapter="pytest", diff=diff, change_graph=graph),
    )

    assert len(llm.calls) == 1
    assert len(portfolio.probes) == 1
    assert portfolio.probes[0].metadata["changed_file"] == "retry.py"
    assert portfolio.probes[0].metadata["changed_symbol"] == "delay"
    assert portfolio.probes[0].oracle_provenance == "none"
    assert portfolio.probes[0].metadata["oracle_grounding"] == "unverified"
    assert "legacy batch response" in portfolio.planning_errors[0]


def test_user_intent_oracle_records_operator_source_side(tmp_path):
    graph, diff = _graph(tmp_path)
    target = next(target for target in graph.targets if target.symbol == "delay")

    def plan_from_prompt(system_prompt, user_prompt):
        match = re.search(r"\[(oracle_[a-f0-9]{16})\] source=stated_intent", user_prompt)
        assert match is not None
        return {
            "hypotheses": [_hypothesis("intent-bound", "error", target.target_id, match.group(1))]
        }

    llm = SequenceLLM([plan_from_prompt, {"source": "def test_intent():\n    assert True"}])

    portfolio = generate_probe_portfolio(
        llm,
        GenerationRequest(
            adapter="pytest",
            diff=diff,
            change_graph=graph,
            intent=("negative attempts must remain rejected",),
        ),
    )

    assert portfolio.probes[0].metadata["oracle_source_side"] == "operator"
    assert portfolio.probes[0].metadata["oracle_grounding"] == "user_intent"
