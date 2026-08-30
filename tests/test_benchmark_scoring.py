import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from bench.runner import (
    DEFAULT_MANIFEST,
    CorpusError,
    ScoringCase,
    execute_gold_case,
    load_benchmark_manifest,
)
from bench.runner import main as benchmark_main
from bench.scoring import (
    RESULT_SCHEMA_VERSION,
    EvaluationFingerprint,
    RecordedExecutorFactory,
    RecordedProbeExecution,
    RecordedProvider,
    ScoringError,
    TrialObservation,
    current_prompt_sha256,
    environment_sha256,
    evaluation_fingerprint_sha256,
    load_observation_bundle,
    normalized_probe_sha256,
    run_recorded_trial,
    score_observations,
    scoring_case_revision_sha256,
)


def _write_snapshot(path: Path, value: int) -> None:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text(
        "[project]\nname='benchmark-fixture'\nversion='1'\n[tool.pytest.ini_options]\n",
        encoding="utf-8",
    )
    (path / "value.py").write_text(f"def value(): return {value}\n", encoding="utf-8")
    (path / "docs").mkdir()
    (path / "docs" / "api.md").write_text(
        "The public documentation guarantees the value remains one.\n",
        encoding="utf-8",
    )


def _case_fixture(root: Path, *, label: str) -> dict[str, object]:
    framework = "pytest" if label == "positive" else "vitest"
    _write_snapshot(root / "base", 1)
    _write_snapshot(root / "candidate", 2 if label == "positive" else 1)
    revisions = {"base": "base", "candidate": "candidate"}
    expected = {"base": "pass", "candidate": "pass"}
    if label == "positive":
        _write_snapshot(root / "fixed", 1)
        revisions["fixed"] = "fixed"
        expected = {"base": "pass", "candidate": "fail", "fixed": "pass"}
    (root / "gold").mkdir()
    if framework == "pytest":
        gold_name = "test_gold.py"
        gold_target = ".evalyn/gold/test_gold.py"
        gold_command = [sys.executable, "-m", "pytest", gold_target]
        gold_availability = [sys.executable, "-m", "pytest", "--version"]
        gold_source = "from value import value\ndef test_gold(): assert value() == 1\n"
    else:
        gold_name = "gold.test.ts"
        gold_target = ".evalyn/gold/gold.test.ts"
        gold_command = ["npx", "--no-install", "vitest", "run", gold_target]
        gold_availability = ["npx", "--no-install", "vitest", "--version"]
        gold_source = (
            "import { expect, test } from 'vitest';\ntest('gold', () => expect(1).toBe(1));\n"
        )
    (root / "gold" / gold_name).write_text(gold_source, encoding="utf-8")
    return {
        "id": f"{label}-case",
        "framework": framework,
        "label": label,
        "control_kind": None if label == "positive" else "unchanged_behavior",
        "summary": f"reviewed {label} fixture",
        "fixture": root.name,
        "revisions": revisions,
        "hidden_gold_probe": f"gold/{gold_name}",
        "gold_probe_target": gold_target,
        "gold_command": gold_command,
        "gold_availability": gold_availability,
        "gold_expected": expected,
        "intent": [],
    }


def _manifest_payload(root: Path) -> dict[str, object]:
    smoke = root / "smoke"
    for revision in ("base", "buggy", "fixed"):
        (smoke / revision).mkdir(parents=True)
    (smoke / "probe").mkdir()
    (smoke / "probe" / "test_probe.py").write_text("assert True\n", encoding="utf-8")
    scoring_cases = [
        _case_fixture(root / "positive", label="positive"),
        _case_fixture(root / "negative", label="negative"),
    ]
    return {
        "schema_version": 2,
        "description": "test detector corpus",
        "scoring_protocol": {
            "trials_per_case": 3,
            "portfolio_k": 8,
            "repetitions_per_finding": 3,
            "required_fingerprints": [
                "provider",
                "model",
                "prompt_sha256",
                "environment_sha256",
            ],
            "thresholds": {
                "probe_validity_rate_min": 0.85,
                "positive_discovery_pass_at_k_min": 0.4,
                "false_gate_rate_max": 0.05,
                "automatic_gate_precision_min": 0.9,
                "reproduction_rate_min": 0.95,
                "latency_p50_seconds_max": 300,
                "latency_p95_seconds_max": 720,
            },
        },
        "promotion_gate": {
            "target_scoring_cases": 2,
            "target_positive_cases": 1,
            "target_negative_cases": 1,
            "target_pytest_cases": 1,
            "target_vitest_cases": 1,
            "registered_scoring_cases": 2,
            "registered_positive_cases": 1,
            "registered_negative_cases": 1,
            "registered_pytest_cases": 1,
            "registered_vitest_cases": 1,
            "met": True,
        },
        "smoke_cases": [
            {
                "id": "smoke-case",
                "framework": "pytest",
                "summary": "execution smoke",
                "fixture": "smoke",
                "probe": "probe/test_probe.py",
                "probe_target": ".evalyn/probe.py",
                "command": [sys.executable, "-c", "pass"],
                "availability": [sys.executable, "--version"],
                "expected": {"base": "pass", "buggy": "fail", "fixed": "pass"},
                "repetitions": 3,
            }
        ],
        "scoring_cases": scoring_cases,
    }


def _load_test_manifest(tmp_path: Path):
    payload = _manifest_payload(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_benchmark_manifest(path)


def _fingerprint() -> EvaluationFingerprint:
    return EvaluationFingerprint(
        provider="recorded",
        model="fixture-v1",
        prompt_sha256=current_prompt_sha256(),
        environment_sha256=environment_sha256(
            {"runtime": "recorded", "python": sys.version_info[:2]}
        ),
    )


def _observation(
    case_id: str,
    trial: int,
    seed: str,
    *,
    evaluation_fingerprint_digest: str | None = None,
    case_revision_sha256: str = "c" * 64,
    status: str = "complete",
    workflow_verdict: str = "no_differential_observed",
    portfolio_k: int = 8,
    repetitions_per_finding: int = 3,
    generated: int = 1,
    valid: int = 1,
    generation_rejections: int = 0,
    generation_infrastructure: int = 0,
    runtime_invalid: int = 0,
    verification_invalid: int = 0,
    infrastructure_errors: int = 0,
    stable_rank: int | None = None,
    fixed_rank: int | None = None,
    automatic_gate_fixed_verified: bool | None = None,
    duration: float = 1,
    reproduction_successes: int | None = None,
    attest_transcript: bool = True,
) -> TrialObservation:
    reproduction_required = int(fixed_rank is not None)
    if reproduction_successes is None:
        reproduction_successes = reproduction_required
    if evaluation_fingerprint_digest is None:
        evaluation_fingerprint_digest = evaluation_fingerprint_sha256(_fingerprint())
    automatic_gate = workflow_verdict == "regression_found"
    if automatic_gate_fixed_verified is None:
        automatic_gate_fixed_verified = automatic_gate and fixed_rank is not None
    return TrialObservation(
        case_id=case_id,
        trial=trial,
        effective_seed=seed,
        evaluation_fingerprint_sha256=evaluation_fingerprint_digest,
        case_revision_sha256=case_revision_sha256,
        status=status,
        workflow_verdict=workflow_verdict,
        portfolio_k=portfolio_k,
        repetitions_per_finding=repetitions_per_finding,
        generated_probes=generated,
        valid_probes=valid,
        generation_rejection_count=generation_rejections,
        generation_infrastructure_count=generation_infrastructure,
        runtime_invalid_probe_count=runtime_invalid,
        verification_invalid_count=verification_invalid,
        infrastructure_error_count=infrastructure_errors,
        first_stable_change_rank=stable_rank,
        first_fixed_verified_rank=fixed_rank,
        automatic_gate=automatic_gate,
        automatic_gate_fixed_verified=automatic_gate_fixed_verified,
        duration_seconds=duration,
        fixed_replay_attempts=3 * reproduction_required,
        fixed_replay_passes=3 * reproduction_required,
        reproduction_required=reproduction_required,
        reproduction_base_attempts=3 * reproduction_required,
        reproduction_candidate_attempts=3 * reproduction_required,
        reproduction_successes=reproduction_successes,
        request_sha256=(
            hashlib.sha256(f"request:{seed}".encode()).hexdigest() if attest_transcript else ""
        ),
        response_transcript_sha256=(
            hashlib.sha256(f"response:{seed}".encode()).hexdigest() if attest_transcript else ""
        ),
    )


def test_checked_manifest_keeps_smoke_separate_from_scoring_gate() -> None:
    manifest = load_benchmark_manifest(DEFAULT_MANIFEST)
    report = score_observations(manifest, _fingerprint(), ())

    assert len(manifest.smoke_cases) == 2
    assert manifest.scoring_cases == ()
    assert manifest.promotion_gate.registered_scoring_cases == 0
    assert manifest.promotion_gate.registered_positive_cases == 0
    assert manifest.promotion_gate.registered_negative_cases == 0
    assert manifest.promotion_gate.registered_pytest_cases == 0
    assert manifest.promotion_gate.registered_vitest_cases == 0
    assert manifest.promotion_gate.met is False
    assert report["corpus"]["scoring_cases"] == 0
    assert report["metrics"]["probe_validity_rate"] is None
    assert report["confidence_intervals_95"]["probe_validity_rate"] is None
    assert report["promotion_gate"]["target_scoring_cases"] == 60
    assert report["promotion_gate"]["registered_scoring_cases"] == 0
    assert report["promotion_gate"]["evaluation_complete"] is False
    assert report["promotion_gate"]["met"] is False


def test_gold_cli_truthfully_reports_that_no_scoring_cases_exist(capsys) -> None:
    assert benchmark_main(["--execute-gold"]) == 0
    assert capsys.readouterr().out.strip() == "No reviewed scoring cases are registered."


def test_scoring_cli_returns_nonzero_for_the_unmet_zero_case_gate(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = tmp_path / "empty-observations.json"
    bundle.write_text(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "fingerprints": _fingerprint().to_dict(),
                "trials": [],
            }
        ),
        encoding="utf-8",
    )

    assert benchmark_main(["--score-observations", str(bundle)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["promotion_gate"]["registered_scoring_cases"] == 0
    assert report["promotion_gate"]["met"] is False


def test_scoring_schema_supports_reviewed_positive_and_negative_cases(tmp_path: Path) -> None:
    manifest = _load_test_manifest(tmp_path)

    assert [case.label for case in manifest.scoring_cases] == ["positive", "negative"]
    assert manifest.protocol.trials_per_case == 3
    assert manifest.protocol.portfolio_k == 8
    assert manifest.protocol.repetitions_per_finding == 3
    assert manifest.scoring_cases[0].hidden_gold_probe.name == "test_gold.py"
    assert manifest.promotion_gate.met is True


def test_negative_flaky_control_can_require_unstable_gold(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    negative = payload["scoring_cases"][1]
    negative["control_kind"] = "flaky_baseline"
    negative["gold_expected"]["base"] = "unstable"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    case = load_benchmark_manifest(path).scoring_cases[1]

    assert case.control_kind == "flaky_baseline"
    assert case.gold_expected["base"] == "unstable"


def test_scoring_gate_counts_cannot_include_smoke_cases(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    payload["promotion_gate"]["registered_scoring_cases"] = 0
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusError, match="registered_scoring_cases must equal 2"):
        load_benchmark_manifest(path)


def test_hidden_gold_probe_cannot_be_part_of_a_revision_snapshot(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    positive = payload["scoring_cases"][0]
    positive["hidden_gold_probe"] = "base/hidden_gold.py"
    (tmp_path / "positive" / "base" / "hidden_gold.py").write_text(
        "assert True\n", encoding="utf-8"
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusError, match="hidden gold probe must be outside"):
        load_benchmark_manifest(path)


def test_hidden_gold_command_is_adapter_owned_and_offline(tmp_path: Path) -> None:
    payload = _manifest_payload(tmp_path)
    payload["scoring_cases"][0]["gold_command"] = ["curl", "https://example.invalid"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusError, match="adapter-owned probe"):
        load_benchmark_manifest(path)


def test_positive_hidden_gold_establishes_base_candidate_fixed_causality(
    tmp_path: Path,
) -> None:
    case = _load_test_manifest(tmp_path).scoring_cases[0]

    result = execute_gold_case(case, strict_tools=True)

    assert result["status"] == "passed"
    assert result["classification"] == "stable_change"
    assert result["causal"] is True
    assert result["mismatches"] == []


def test_recorded_trial_exercises_public_workflow_without_exposing_gold(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    source = "from value import value\ndef test_value():\n    assert value() == 1"
    _write_snapshot(fixture / "base", 1)
    _write_snapshot(fixture / "candidate", 2)
    _write_snapshot(fixture / "fixed", 1)
    for revision in ("base", "candidate", "fixed"):
        tests_dir = fixture / revision / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_contract.py").write_text(source + "\n", encoding="utf-8")
    (fixture / "gold").mkdir()
    hidden_gold = fixture / "gold" / "test_gold.py"
    hidden_gold.write_text("TOP SECRET GOLD PROBE\n", encoding="utf-8")
    case = ScoringCase(
        case_id="recorded-positive",
        framework="pytest",
        label="positive",
        control_kind=None,
        summary="recorded workflow",
        fixture_root=fixture,
        base_root=fixture / "base",
        candidate_root=fixture / "candidate",
        fixed_root=fixture / "fixed",
        hidden_gold_probe=hidden_gold,
        gold_probe_target=Path(".evalyn/gold/test_gold.py"),
        gold_command=(sys.executable, "-m", "pytest", ".evalyn/gold/test_gold.py"),
        gold_availability=(sys.executable, "-m", "pytest", "--version"),
        gold_expected={"base": "pass", "candidate": "fail", "fixed": "pass"},
        intent=("The value must remain one.",),
    )

    def replies(_system_prompt: str, user_prompt: str, index: int, effective_seed: str):
        assert effective_seed == "seed-1"
        if index == 0:
            target_match = re.search(r"\[(target_[a-f0-9]{16})\] value\.py::value ", user_prompt)
            assert target_match
            target_id = target_match.group(1)
            oracle_matches = re.findall(
                r"\[(oracle_[a-f0-9]{16})\].*category=existing_test "
                r"compatible_targets=([^\n]+)",
                user_prompt,
            )
            oracle_id = next(
                oracle_id
                for oracle_id, compatible_targets in oracle_matches
                if target_id in compatible_targets.split(",")
            )
            return {
                "hypotheses": [
                    {
                        "title": "value remains one",
                        "category": "defaults_compatibility",
                        "target_id": target_id,
                        "behavior": "The public value remains one.",
                        "oracle_id": oracle_id,
                    }
                ]
            }
        if index == 1:
            return {"source": "def broken("}
        if index == 2:
            return {"source": source}
        raise AssertionError(f"unexpected provider call {index + 1}")

    provider = RecordedProvider(replies, effective_seed="seed-1")
    factory = RecordedExecutorFactory(
        [
            RecordedProbeExecution(
                source_sha256=normalized_probe_sha256(source),
                base=("passed",) * 3,
                candidate=("failed",) * 3,
                intervention_candidate=("passed",) * 3,
                fixed=("passed",) * 3,
                reproduction_base=("passed",) * 3,
                reproduction_candidate=("failed",) * 3,
            )
        ]
    )

    observation = run_recorded_trial(
        case,
        trial=1,
        seed="seed-1",
        provider=provider,
        executor_factory=factory,
        fingerprint=_fingerprint(),
        work_root=tmp_path / "runs",
    )

    assert observation.status == "complete"
    assert observation.generated_probes == 2
    assert observation.valid_probes == 1
    assert observation.generation_rejection_count == 1
    assert observation.runtime_invalid_probe_count == 0
    assert observation.first_stable_change_rank == 1
    assert observation.first_fixed_verified_rank == 1
    assert observation.fixed_replay_attempts == 3
    assert observation.fixed_replay_passes == 3
    assert observation.reproduction_required == 1
    assert observation.reproduction_base_attempts == 3
    assert observation.reproduction_candidate_attempts == 3
    assert observation.reproduction_successes == 1
    assert observation.effective_seed == "seed-1"
    assert observation.automatic_gate == (observation.workflow_verdict == "regression_found")
    assert observation.request_sha256
    assert observation.response_transcript_sha256
    assert len(provider.user_prompts) == 3
    assert all("TOP SECRET GOLD PROBE" not in prompt for prompt in provider.user_prompts)
    intervention_events = [event for event in factory.events if event.side == "target_intervention"]
    assert len(intervention_events) == 3
    assert all(event.status.value == "passed" for event in intervention_events)

    missing_replay_provider = RecordedProvider(replies, effective_seed="seed-1")
    missing_replay_observation = run_recorded_trial(
        case,
        trial=1,
        seed="seed-1",
        provider=missing_replay_provider,
        executor_factory=RecordedExecutorFactory(
            [
                RecordedProbeExecution(
                    source_sha256=normalized_probe_sha256(source),
                    base=("passed",) * 3,
                    candidate=("failed",) * 3,
                )
            ]
        ),
        fingerprint=_fingerprint(),
        work_root=tmp_path / "missing-replay-runs",
    )

    assert missing_replay_observation.status == "infrastructure_error"
    assert missing_replay_observation.first_stable_change_rank == 1
    assert missing_replay_observation.first_fixed_verified_rank is None
    assert missing_replay_observation.infrastructure_error_count >= 3

    def infrastructure_replies(
        system_prompt: str,
        user_prompt: str,
        index: int,
        effective_seed: str,
    ):
        if index == 1:
            raise TimeoutError("recorded provider transport failed")
        return replies(system_prompt, user_prompt, index, effective_seed)

    infrastructure_provider = RecordedProvider(
        infrastructure_replies,
        effective_seed="seed-1",
    )
    infrastructure_observation = run_recorded_trial(
        case,
        trial=1,
        seed="seed-1",
        provider=infrastructure_provider,
        executor_factory=RecordedExecutorFactory(
            [
                RecordedProbeExecution(
                    source_sha256=normalized_probe_sha256(source),
                    base=("passed",) * 3,
                    candidate=("failed",) * 3,
                    intervention_candidate=("passed",) * 3,
                    fixed=("passed",) * 3,
                    reproduction_base=("passed",) * 3,
                    reproduction_candidate=("failed",) * 3,
                )
            ]
        ),
        fingerprint=_fingerprint(),
        work_root=tmp_path / "provider-infrastructure-runs",
    )

    assert infrastructure_observation.status == "infrastructure_error"
    assert infrastructure_observation.generation_rejection_count == 0
    assert infrastructure_observation.generation_infrastructure_count == 1
    assert infrastructure_observation.runtime_invalid_probe_count == 0
    assert infrastructure_observation.infrastructure_error_count == 1

    insufficient_provider = RecordedProvider(replies, effective_seed="seed-1")
    insufficient_observation = run_recorded_trial(
        case,
        trial=1,
        seed="seed-1",
        provider=insufficient_provider,
        executor_factory=RecordedExecutorFactory(
            [
                RecordedProbeExecution(
                    source_sha256=normalized_probe_sha256(source),
                    base=("failed",) * 3,
                    candidate=("failed",) * 3,
                )
            ]
        ),
        fingerprint=_fingerprint(),
        work_root=tmp_path / "insufficient-runs",
    )

    assert insufficient_observation.status == "insufficient_evidence"
    assert insufficient_observation.workflow_verdict == "insufficient_evidence"
    assert insufficient_observation.automatic_gate is False

    signature_mismatch_provider = RecordedProvider(replies, effective_seed="seed-1")
    signature_mismatch_observation = run_recorded_trial(
        case,
        trial=1,
        seed="seed-1",
        provider=signature_mismatch_provider,
        executor_factory=RecordedExecutorFactory(
            [
                RecordedProbeExecution(
                    source_sha256=normalized_probe_sha256(source),
                    base=("passed",) * 3,
                    candidate=("timeout",) * 3,
                    intervention_candidate=("passed",) * 3,
                    fixed=("passed",) * 3,
                    reproduction_base=("passed",) * 3,
                    reproduction_candidate=("failed",) * 3,
                )
            ]
        ),
        fingerprint=_fingerprint(),
        work_root=tmp_path / "signature-mismatch-runs",
    )

    assert signature_mismatch_observation.status == "complete"
    assert signature_mismatch_observation.first_fixed_verified_rank == 1
    assert signature_mismatch_observation.reproduction_required == 1
    assert signature_mismatch_observation.reproduction_successes == 0


def test_recorded_trial_rejects_a_seed_not_bound_to_the_provider(tmp_path: Path) -> None:
    case = _load_test_manifest(tmp_path).scoring_cases[0]
    provider = RecordedProvider({"hypotheses": []}, effective_seed="provider-seed")

    with pytest.raises(ScoringError, match="does not match"):
        run_recorded_trial(
            case,
            trial=1,
            seed="different-seed",
            provider=provider,
            executor_factory=RecordedExecutorFactory(()),
            fingerprint=_fingerprint(),
            work_root=tmp_path / "runs",
        )


def test_credited_discovery_requires_its_own_reproduction_coverage() -> None:
    observation = _observation(
        "positive-case",
        1,
        "seed",
        workflow_verdict="regression_found",
        stable_rank=1,
        fixed_rank=1,
    )

    with pytest.raises(ScoringError, match="requires its own reproduction"):
        replace(observation, reproduction_required=0, reproduction_successes=0)


def test_trial_status_and_gate_are_derived_from_workflow_evidence() -> None:
    observation = _observation("negative-case", 1, "seed")

    with pytest.raises(ScoringError, match="status must be derived"):
        replace(observation, status="invalid")
    with pytest.raises(ScoringError, match="automatic_gate must be derived"):
        replace(observation, automatic_gate=True)


def test_scoring_rejects_revision_rebinding_and_off_portfolio_credit(
    tmp_path: Path,
) -> None:
    manifest = _load_test_manifest(tmp_path)
    positive = manifest.scoring_cases[0]
    case_digest = scoring_case_revision_sha256(positive)

    with pytest.raises(ScoringError, match="bundle fingerprint"):
        score_observations(
            manifest,
            _fingerprint(),
            (
                _observation(
                    positive.case_id,
                    1,
                    "seed",
                    evaluation_fingerprint_digest="f" * 64,
                    case_revision_sha256=case_digest,
                ),
            ),
        )

    with pytest.raises(ScoringError, match="different revision snapshots"):
        score_observations(
            manifest,
            _fingerprint(),
            (
                _observation(
                    positive.case_id,
                    1,
                    "seed",
                    case_revision_sha256="d" * 64,
                ),
            ),
        )

    with pytest.raises(ScoringError, match="portfolio_k does not match"):
        score_observations(
            manifest,
            _fingerprint(),
            (
                _observation(
                    positive.case_id,
                    1,
                    "seed",
                    case_revision_sha256=case_digest,
                    portfolio_k=1,
                ),
            ),
        )

    with pytest.raises(ScoringError, match="exceeds configured portfolio_k"):
        score_observations(
            manifest,
            _fingerprint(),
            (
                _observation(
                    positive.case_id,
                    1,
                    "seed",
                    case_revision_sha256=case_digest,
                    workflow_verdict="regression_found",
                    generated=9,
                    valid=9,
                    stable_rank=9,
                    fixed_rank=9,
                ),
            ),
        )

    original_digest = scoring_case_revision_sha256(positive)
    positive.hidden_gold_probe.write_text("changed hidden oracle\n", encoding="utf-8")
    assert scoring_case_revision_sha256(positive) != original_digest


def test_scoring_rejects_transcripts_that_do_not_attest_distinct_seeds(
    tmp_path: Path,
) -> None:
    manifest = _load_test_manifest(tmp_path)
    negative = manifest.scoring_cases[1]
    case_digest = scoring_case_revision_sha256(negative)
    first = _observation(
        negative.case_id,
        1,
        "seed-1",
        case_revision_sha256=case_digest,
    )
    second = _observation(
        negative.case_id,
        2,
        "seed-2",
        case_revision_sha256=case_digest,
    )
    second = replace(second, request_sha256=first.request_sha256)

    with pytest.raises(ScoringError, match="request transcripts"):
        score_observations(manifest, _fingerprint(), (first, second))


def test_positive_gate_is_true_only_when_credited_finding_is_fixed_verified(
    tmp_path: Path,
) -> None:
    manifest = _load_test_manifest(tmp_path)
    positive = manifest.scoring_cases[0]
    observation = _observation(
        positive.case_id,
        1,
        "seed",
        case_revision_sha256=scoring_case_revision_sha256(positive),
        workflow_verdict="regression_found",
        stable_rank=1,
        fixed_rank=1,
        automatic_gate_fixed_verified=False,
    )

    metrics = score_observations(manifest, _fingerprint(), (observation,))["metrics"]

    assert metrics["true_gate_count"] == 0
    assert metrics["unverified_positive_gate_count"] == 1
    assert metrics["automatic_gate_precision"] == 0


def test_scoring_reports_discovery_false_gates_failures_latency_and_reproduction(
    tmp_path: Path,
) -> None:
    manifest = _load_test_manifest(tmp_path)
    case_digests = {
        case.case_id: scoring_case_revision_sha256(case) for case in manifest.scoring_cases
    }
    observations = (
        _observation(
            "positive-case",
            1,
            "p1",
            case_revision_sha256=case_digests["positive-case"],
            workflow_verdict="regression_found",
            stable_rank=1,
            fixed_rank=1,
            duration=1,
        ),
        _observation(
            "positive-case",
            2,
            "p2",
            case_revision_sha256=case_digests["positive-case"],
            workflow_verdict="regression_found",
            generated=8,
            valid=8,
            stable_rank=8,
            fixed_rank=8,
            duration=2,
            reproduction_successes=0,
        ),
        _observation(
            "positive-case",
            3,
            "p3",
            case_revision_sha256=case_digests["positive-case"],
            status="invalid",
            workflow_verdict="error",
            generated=0,
            valid=0,
            duration=3,
            attest_transcript=False,
        ),
        _observation(
            "negative-case",
            1,
            "n1",
            case_revision_sha256=case_digests["negative-case"],
            workflow_verdict="regression_found",
            stable_rank=1,
            duration=4,
        ),
        _observation(
            "negative-case",
            2,
            "n2",
            case_revision_sha256=case_digests["negative-case"],
            duration=5,
        ),
        _observation(
            "negative-case",
            3,
            "n3",
            case_revision_sha256=case_digests["negative-case"],
            status="infrastructure_error",
            workflow_verdict="error",
            infrastructure_errors=1,
            duration=6,
            attest_transcript=False,
        ),
    )

    report = score_observations(manifest, _fingerprint(), observations)

    metrics = report["metrics"]
    assert metrics["probe_validity_rate"] == 1
    assert metrics["positive_discovery_pass_at_k"] == pytest.approx(2 / 3)
    assert metrics["false_gate_rate"] == pytest.approx(1 / 3)
    assert metrics["automatic_gate_precision"] == pytest.approx(2 / 3)
    assert metrics["invalid_trial_count"] == 1
    assert metrics["infrastructure_trial_count"] == 1
    assert metrics["latency_p50_seconds"] == 3.5
    assert metrics["latency_p95_seconds"] == 5.75
    assert metrics["reproduction_rate"] == pytest.approx(1 / 2)
    assert metrics["reproduction_required_count"] == 2
    lower, upper = report["confidence_intervals_95"]["false_gate_rate"]
    assert 0 <= lower < metrics["false_gate_rate"] < upper <= 1
    assert report["promotion_gate"]["trial_matrix_complete"] is True
    assert report["promotion_gate"]["evaluation_complete"] is False
    assert report["promotion_gate"]["met"] is False


def test_observation_bundle_is_versioned_and_rescorable(tmp_path: Path) -> None:
    observation = _observation(
        "positive-case",
        1,
        "seed",
        workflow_verdict="regression_found",
        stable_rank=1,
        fixed_rank=1,
        duration=0.5,
    )
    path = tmp_path / "observations.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "fingerprints": _fingerprint().to_dict(),
                "trials": [observation.to_dict()],
            }
        ),
        encoding="utf-8",
    )

    fingerprint, observations = load_observation_bundle(path)

    assert fingerprint == _fingerprint()
    assert observations == (observation,)


def test_observation_bundle_rejects_schema_v1(tmp_path: Path) -> None:
    path = tmp_path / "observations-v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprints": _fingerprint().to_dict(),
                "trials": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ScoringError, match="schema_version 2"):
        load_observation_bundle(path)
