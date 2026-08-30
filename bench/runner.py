"""Validate and optionally execute Evalyn's benchmark corpus.

The checked-in cases are deterministic execution smoke tests. Detector scoring
cases use a separate manifest section so smoke fixtures cannot accidentally be
reported as evidence of detector quality.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REVISIONS = ("base", "buggy", "fixed")
OUTCOMES = {"pass", "fail"}
SCORING_GOLD_OUTCOMES = {"pass", "fail", "unstable"}
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")
MANIFEST_SCHEMA_VERSION = 2
REQUIRED_FINGERPRINTS = frozenset({"provider", "model", "prompt_sha256", "environment_sha256"})
SCORING_LABELS = frozenset({"positive", "negative"})
NEGATIVE_CONTROL_KINDS = frozenset(
    {
        "adversarial_instructions",
        "flaky_baseline",
        "intended_change",
        "refactor",
        "unchanged_behavior",
    }
)
THRESHOLD_KEYS = frozenset(
    {
        "automatic_gate_precision_min",
        "false_gate_rate_max",
        "latency_p50_seconds_max",
        "latency_p95_seconds_max",
        "positive_discovery_pass_at_k_min",
        "probe_validity_rate_min",
        "reproduction_rate_min",
    }
)
_MAX_CAPTURE_BYTES = 4_000
_COMMAND_TIMEOUT_SECONDS = 30
_AVAILABILITY_TIMEOUT_SECONDS = 10


class CorpusError(ValueError):
    """Raised when the benchmark manifest or a fixture is invalid."""


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    framework: str
    summary: str
    fixture_root: Path
    probe_path: Path
    probe_target: Path
    command: tuple[str, ...]
    availability: tuple[str, ...]
    expected: dict[str, str]
    repetitions: int


@dataclass(frozen=True)
class ScoringProtocol:
    """Pinned evaluation policy shared by all reviewed scoring cases."""

    trials_per_case: int
    portfolio_k: int
    repetitions_per_finding: int
    required_fingerprints: tuple[str, ...]
    thresholds: dict[str, float]


@dataclass(frozen=True)
class PromotionGate:
    """Corpus-composition gate; performance is evaluated from trial output."""

    target_scoring_cases: int
    target_positive_cases: int
    target_negative_cases: int
    target_pytest_cases: int
    target_vitest_cases: int
    registered_scoring_cases: int
    registered_positive_cases: int
    registered_negative_cases: int
    registered_pytest_cases: int
    registered_vitest_cases: int
    met: bool


@dataclass(frozen=True)
class ScoringCase:
    """A reviewed detector case whose hidden gold probe is never model context."""

    case_id: str
    framework: str
    label: str
    control_kind: str | None
    summary: str
    fixture_root: Path
    base_root: Path
    candidate_root: Path
    fixed_root: Path | None
    hidden_gold_probe: Path
    gold_probe_target: Path
    gold_command: tuple[str, ...]
    gold_availability: tuple[str, ...]
    gold_expected: dict[str, str]
    intent: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkManifest:
    """Validated version-two benchmark manifest."""

    description: str
    protocol: ScoringProtocol
    promotion_gate: PromotionGate
    smoke_cases: tuple[BenchmarkCase, ...]
    scoring_cases: tuple[ScoringCase, ...]


@dataclass(frozen=True)
class RevisionOutcome:
    revision: str
    results: tuple[Any, ...]

    @property
    def passed(self) -> bool:
        return all(result.exit_code == 0 for result in self.results)

    @property
    def observed(self) -> str:
        passed = sum(result.exit_code == 0 for result in self.results)
        if passed == len(self.results):
            return "pass"
        if passed == 0:
            return "fail"
        return "unstable"


def load_benchmark_manifest(path: Path = DEFAULT_MANIFEST) -> BenchmarkManifest:
    """Load and validate the version-two smoke and detector-scoring schema."""
    manifest_path = path.resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"Could not read benchmark manifest {manifest_path}: {exc}") from exc

    if not isinstance(raw, dict) or raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CorpusError(f"Benchmark manifest must use schema_version {MANIFEST_SCHEMA_VERSION}")
    top_level_fields = {
        "schema_version",
        "description",
        "scoring_protocol",
        "promotion_gate",
        "smoke_cases",
        "scoring_cases",
    }
    if set(raw) != top_level_fields:
        raise CorpusError(
            f"Benchmark manifest must define exactly: {', '.join(sorted(top_level_fields))}"
        )
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise CorpusError("Benchmark manifest must contain a nonempty description")
    protocol = _parse_protocol(raw.get("scoring_protocol"))
    raw_cases = raw.get("smoke_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise CorpusError("Benchmark manifest must contain nonempty smoke_cases")
    raw_scoring_cases = raw.get("scoring_cases")
    if not isinstance(raw_scoring_cases, list):
        raise CorpusError("Benchmark manifest scoring_cases must be a list")

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_cases):
        if not isinstance(value, dict):
            raise CorpusError(f"cases[{index}] must be an object")
        smoke_fields = {
            "id",
            "framework",
            "summary",
            "fixture",
            "probe",
            "probe_target",
            "command",
            "availability",
            "expected",
            "repetitions",
        }
        if set(value) != smoke_fields:
            raise CorpusError(
                f"smoke_cases[{index}] must define exactly: {', '.join(sorted(smoke_fields))}"
            )
        case_id = _required_text(value, "id", index)
        if case_id in seen:
            raise CorpusError(f"Duplicate benchmark case id: {case_id}")
        seen.add(case_id)

        framework = _required_text(value, "framework", index)
        if framework not in {"pytest", "vitest"}:
            raise CorpusError(f"{case_id}: framework must be pytest or vitest")

        fixture_root = _contained(manifest_path.parent, _required_text(value, "fixture", index))
        probe_path = _contained(fixture_root, _required_text(value, "probe", index))
        probe_target = _relative_path(_required_text(value, "probe_target", index))
        command = _argv(value.get("command"), f"{case_id}.command")
        availability = _argv(value.get("availability"), f"{case_id}.availability")
        expected = value.get("expected")
        if not isinstance(expected, dict) or set(expected) != set(REVISIONS):
            raise CorpusError(f"{case_id}: expected must define base, buggy, and fixed")
        normalized_expected = {name: str(expected[name]) for name in REVISIONS}
        if any(outcome not in OUTCOMES for outcome in normalized_expected.values()):
            raise CorpusError(f"{case_id}: expected outcomes must be pass or fail")
        if normalized_expected != {"base": "pass", "buggy": "fail", "fixed": "pass"}:
            raise CorpusError(
                f"{case_id}: corpus cases must establish base-pass/buggy-fail/fixed-pass"
            )

        repetitions = value.get("repetitions", 3)
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions != 3:
            raise CorpusError(f"{case_id}: repetitions must be exactly 3")

        if not fixture_root.is_dir():
            raise CorpusError(f"{case_id}: fixture directory does not exist: {fixture_root}")
        if not probe_path.is_file() or not probe_path.read_text(encoding="utf-8").strip():
            raise CorpusError(f"{case_id}: probe must be a nonempty regular file")
        for revision in REVISIONS:
            revision_root = fixture_root / revision
            if not revision_root.is_dir():
                raise CorpusError(f"{case_id}: missing {revision} fixture directory")

        cases.append(
            BenchmarkCase(
                case_id=case_id,
                framework=framework,
                summary=_required_text(value, "summary", index),
                fixture_root=fixture_root,
                probe_path=probe_path,
                probe_target=probe_target,
                command=command,
                availability=availability,
                expected=normalized_expected,
                repetitions=repetitions,
            )
        )
    scoring_cases = _parse_scoring_cases(manifest_path, raw_scoring_cases, seen)
    gate = _parse_promotion_gate(raw.get("promotion_gate"), scoring_cases)
    return BenchmarkManifest(
        description=description.strip(),
        protocol=protocol,
        promotion_gate=gate,
        smoke_cases=tuple(cases),
        scoring_cases=scoring_cases,
    )


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[BenchmarkCase, ...]:
    """Load deterministic smoke cases (the historical public API)."""
    return load_benchmark_manifest(path).smoke_cases


def _parse_protocol(value: object) -> ScoringProtocol:
    if not isinstance(value, dict):
        raise CorpusError("Benchmark manifest must define scoring_protocol")
    if set(value) != {
        "trials_per_case",
        "portfolio_k",
        "repetitions_per_finding",
        "required_fingerprints",
        "thresholds",
    }:
        raise CorpusError(
            "scoring_protocol must define trials_per_case, portfolio_k, "
            "repetitions_per_finding, required_fingerprints, and thresholds"
        )
    trials = _positive_int(value["trials_per_case"], "scoring_protocol.trials_per_case")
    if trials < 2:
        raise CorpusError("scoring_protocol.trials_per_case must be at least 2")
    portfolio_k = _positive_int(value["portfolio_k"], "scoring_protocol.portfolio_k")
    repetitions = _positive_int(
        value["repetitions_per_finding"],
        "scoring_protocol.repetitions_per_finding",
    )
    if repetitions != 3:
        raise CorpusError("scoring_protocol.repetitions_per_finding must be exactly 3")
    raw_fingerprints = value["required_fingerprints"]
    if (
        not isinstance(raw_fingerprints, list)
        or not all(isinstance(item, str) and item for item in raw_fingerprints)
        or set(raw_fingerprints) != REQUIRED_FINGERPRINTS
        or len(raw_fingerprints) != len(REQUIRED_FINGERPRINTS)
    ):
        required = ", ".join(sorted(REQUIRED_FINGERPRINTS))
        raise CorpusError(f"scoring_protocol.required_fingerprints must be exactly: {required}")
    raw_thresholds = value["thresholds"]
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != THRESHOLD_KEYS:
        required = ", ".join(sorted(THRESHOLD_KEYS))
        raise CorpusError(f"scoring_protocol.thresholds must define exactly: {required}")
    thresholds: dict[str, float] = {}
    for key, raw_threshold in raw_thresholds.items():
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
            raise CorpusError(f"scoring_protocol.thresholds.{key} must be numeric")
        threshold = float(raw_threshold)
        if not math.isfinite(threshold):
            raise CorpusError(f"scoring_protocol.thresholds.{key} must be finite")
        if key.startswith("latency_"):
            if threshold <= 0:
                raise CorpusError(f"scoring_protocol.thresholds.{key} must be positive")
        elif not 0 <= threshold <= 1:
            raise CorpusError(f"scoring_protocol.thresholds.{key} must be between 0 and 1")
        thresholds[key] = threshold
    if thresholds["latency_p50_seconds_max"] > thresholds["latency_p95_seconds_max"]:
        raise CorpusError("latency P50 threshold cannot exceed the P95 threshold")
    return ScoringProtocol(
        trials_per_case=trials,
        portfolio_k=portfolio_k,
        repetitions_per_finding=repetitions,
        required_fingerprints=tuple(raw_fingerprints),
        thresholds=thresholds,
    )


def _parse_promotion_gate(
    value: object,
    cases: tuple[ScoringCase, ...],
) -> PromotionGate:
    if not isinstance(value, dict):
        raise CorpusError("Benchmark manifest must define promotion_gate")
    fields = {
        "target_scoring_cases",
        "target_positive_cases",
        "target_negative_cases",
        "target_pytest_cases",
        "target_vitest_cases",
        "registered_scoring_cases",
        "registered_positive_cases",
        "registered_negative_cases",
        "registered_pytest_cases",
        "registered_vitest_cases",
        "met",
    }
    if set(value) != fields:
        raise CorpusError("promotion_gate must define target, registered, and met fields")
    counts = {
        key: _nonnegative_int(value[key], f"promotion_gate.{key}") for key in fields if key != "met"
    }
    if counts["target_scoring_cases"] < 1:
        raise CorpusError("promotion_gate.target_scoring_cases must be positive")
    if counts["target_positive_cases"] < 1 or counts["target_negative_cases"] < 1:
        raise CorpusError("promotion gate positive and negative targets must both be positive")
    if counts["target_pytest_cases"] < 1 or counts["target_vitest_cases"] < 1:
        raise CorpusError("promotion gate pytest and Vitest targets must both be positive")
    if (
        counts["target_positive_cases"] + counts["target_negative_cases"]
        != counts["target_scoring_cases"]
    ):
        raise CorpusError("promotion gate positive and negative targets must equal total target")
    if (
        counts["target_pytest_cases"] + counts["target_vitest_cases"]
        != counts["target_scoring_cases"]
    ):
        raise CorpusError("promotion gate framework targets must equal total target")
    actual_positive = sum(case.label == "positive" for case in cases)
    actual_negative = sum(case.label == "negative" for case in cases)
    actual_pytest = sum(case.framework == "pytest" for case in cases)
    actual_vitest = sum(case.framework == "vitest" for case in cases)
    actual = len(cases)
    expected_registered = {
        "registered_scoring_cases": actual,
        "registered_positive_cases": actual_positive,
        "registered_negative_cases": actual_negative,
        "registered_pytest_cases": actual_pytest,
        "registered_vitest_cases": actual_vitest,
    }
    for key, expected in expected_registered.items():
        if counts[key] != expected:
            raise CorpusError(f"promotion_gate.{key} must equal {expected}")
    calculated_met = (
        actual >= counts["target_scoring_cases"]
        and actual_positive >= counts["target_positive_cases"]
        and actual_negative >= counts["target_negative_cases"]
        and actual_pytest >= counts["target_pytest_cases"]
        and actual_vitest >= counts["target_vitest_cases"]
    )
    if not isinstance(value["met"], bool) or value["met"] != calculated_met:
        raise CorpusError("promotion_gate.met does not match registered scoring progress")
    return PromotionGate(**counts, met=calculated_met)


def _parse_scoring_cases(
    manifest_path: Path,
    values: list[object],
    seen: set[str],
) -> tuple[ScoringCase, ...]:
    result: list[ScoringCase] = []
    for index, value in enumerate(values):
        label_prefix = f"scoring_cases[{index}]"
        if not isinstance(value, dict):
            raise CorpusError(f"{label_prefix} must be an object")
        fields = {
            "id",
            "framework",
            "label",
            "control_kind",
            "summary",
            "fixture",
            "revisions",
            "hidden_gold_probe",
            "gold_probe_target",
            "gold_command",
            "gold_availability",
            "gold_expected",
            "intent",
        }
        if set(value) != fields:
            raise CorpusError(f"{label_prefix} must define exactly: {', '.join(sorted(fields))}")
        case_id = _named_text(value.get("id"), f"{label_prefix}.id")
        if case_id in seen:
            raise CorpusError(f"Duplicate benchmark case id: {case_id}")
        seen.add(case_id)
        framework = _named_text(value.get("framework"), f"{case_id}.framework")
        if framework not in {"pytest", "vitest"}:
            raise CorpusError(f"{case_id}: framework must be pytest or vitest")
        label = _named_text(value.get("label"), f"{case_id}.label")
        if label not in SCORING_LABELS:
            raise CorpusError(f"{case_id}: label must be positive or negative")
        control_kind_value = value.get("control_kind")
        control_kind = None
        if label == "positive":
            if control_kind_value is not None:
                raise CorpusError(f"{case_id}: positive cases must use null control_kind")
        else:
            control_kind = _named_text(control_kind_value, f"{case_id}.control_kind")
            if control_kind not in NEGATIVE_CONTROL_KINDS:
                allowed = ", ".join(sorted(NEGATIVE_CONTROL_KINDS))
                raise CorpusError(f"{case_id}: unsupported negative control_kind; use {allowed}")

        fixture_root = _contained(
            manifest_path.parent,
            _named_text(value.get("fixture"), f"{case_id}.fixture"),
        )
        if not fixture_root.is_dir():
            raise CorpusError(f"{case_id}: fixture directory does not exist: {fixture_root}")
        revisions = value.get("revisions")
        if not isinstance(revisions, dict) or set(revisions) not in (
            {"base", "candidate"},
            {"base", "candidate", "fixed"},
        ):
            raise CorpusError(
                f"{case_id}: revisions must define base, candidate, and optional fixed"
            )
        revision_roots = {
            name: _contained(
                fixture_root,
                _named_text(revisions[name], f"{case_id}.revisions.{name}"),
            )
            for name in revisions
        }
        if len(set(revision_roots.values())) != len(revision_roots):
            raise CorpusError(f"{case_id}: revision fixture paths must be distinct")
        if any(not path.is_dir() for path in revision_roots.values()):
            raise CorpusError(f"{case_id}: every revision fixture must be a directory")
        for name, path in revision_roots.items():
            _validate_scoring_snapshot(case_id, name, path)
        if label == "positive" and "fixed" not in revision_roots:
            raise CorpusError(f"{case_id}: positive cases require a fixed revision")

        hidden_gold_probe = _contained(
            fixture_root,
            _named_text(value.get("hidden_gold_probe"), f"{case_id}.hidden_gold_probe"),
        )
        if (
            not hidden_gold_probe.is_file()
            or not hidden_gold_probe.read_text(encoding="utf-8").strip()
        ):
            raise CorpusError(f"{case_id}: hidden gold probe must be a nonempty regular file")
        if any(hidden_gold_probe.is_relative_to(root) for root in revision_roots.values()):
            raise CorpusError(f"{case_id}: hidden gold probe must be outside revision snapshots")

        expected = value.get("gold_expected")
        if not isinstance(expected, dict) or set(expected) != set(revision_roots):
            raise CorpusError(f"{case_id}: gold_expected must match configured revisions")
        normalized_expected = {name: str(expected[name]) for name in revision_roots}
        if any(outcome not in SCORING_GOLD_OUTCOMES for outcome in normalized_expected.values()):
            raise CorpusError(f"{case_id}: gold_expected outcomes must be pass, fail, or unstable")
        if label == "positive" and "unstable" in normalized_expected.values():
            raise CorpusError(f"{case_id}: positive gold cannot accept unstable evidence")
        if label == "positive" and normalized_expected != {
            "base": "pass",
            "candidate": "fail",
            "fixed": "pass",
        }:
            raise CorpusError(
                f"{case_id}: positive gold must establish base-pass/candidate-fail/fixed-pass"
            )
        raw_intent = value.get("intent")
        if not isinstance(raw_intent, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_intent
        ):
            raise CorpusError(f"{case_id}.intent must be an array of nonempty strings")
        gold_probe_target = _relative_path(
            _named_text(value.get("gold_probe_target"), f"{case_id}.gold_probe_target")
        )
        gold_command = _argv(value.get("gold_command"), f"{case_id}.gold_command")
        gold_availability = _argv(value.get("gold_availability"), f"{case_id}.gold_availability")
        _validate_gold_command(
            case_id,
            framework,
            gold_probe_target,
            gold_command,
            gold_availability,
        )
        result.append(
            ScoringCase(
                case_id=case_id,
                framework=framework,
                label=label,
                control_kind=control_kind,
                summary=_named_text(value.get("summary"), f"{case_id}.summary"),
                fixture_root=fixture_root,
                base_root=revision_roots["base"],
                candidate_root=revision_roots["candidate"],
                fixed_root=revision_roots.get("fixed"),
                hidden_gold_probe=hidden_gold_probe,
                gold_probe_target=gold_probe_target,
                gold_command=gold_command,
                gold_availability=gold_availability,
                gold_expected=normalized_expected,
                intent=tuple(item.strip() for item in raw_intent),
            )
        )
    return tuple(result)


def execute_case(case: BenchmarkCase, *, strict_tools: bool = False) -> dict[str, object]:
    """Execute one trusted causal fixture and classify base versus buggy evidence."""
    from evalyn.engine.classifier import DifferentialClassifier
    from evalyn.engine.models import ClassificationKind

    if not _tool_available(case):
        message = f"{case.case_id}: required {case.framework} tool is unavailable"
        if strict_tools:
            raise CorpusError(message)
        return {
            "id": case.case_id,
            "framework": case.framework,
            "status": "skipped",
            "reason": message,
        }

    outcomes = {revision: _execute_revision(case, revision) for revision in REVISIONS}
    mismatches = [
        revision
        for revision, outcome in outcomes.items()
        if ("pass" if outcome.passed else "fail") != case.expected[revision]
    ]

    classification = DifferentialClassifier().classify(
        outcomes["base"].results,
        outcomes["buggy"].results,
    )
    causal = (
        not mismatches
        and classification.kind == ClassificationKind.STABLE_CHANGE
        and outcomes["fixed"].passed
    )
    return {
        "id": case.case_id,
        "framework": case.framework,
        "status": "passed" if causal else "failed",
        "summary": case.summary,
        "outcomes": {
            revision: {
                "passed": outcome.passed,
                "runs": [
                    {
                        "exit_code": result.exit_code,
                        "status": result.status.value,
                    }
                    for result in outcome.results
                ],
            }
            for revision, outcome in outcomes.items()
        },
        "classification": classification.kind.value,
        "classification_reason": classification.reason,
        "failure_signature": classification.failure_signature,
        "causal": causal,
        "mismatches": mismatches,
    }


def execute_gold_case(case: ScoringCase, *, strict_tools: bool = False) -> dict[str, object]:
    """Validate a reviewed case's hidden oracle without exposing it to generation."""
    from evalyn.engine.classifier import DifferentialClassifier
    from evalyn.engine.models import ClassificationKind

    if not _scoring_tool_available(case):
        message = f"{case.case_id}: required {case.framework} gold tool is unavailable"
        if strict_tools:
            raise CorpusError(message)
        return {
            "id": case.case_id,
            "framework": case.framework,
            "label": case.label,
            "status": "skipped",
            "reason": message,
        }
    revision_names = tuple(case.gold_expected)
    outcomes = {revision: _execute_scoring_revision(case, revision) for revision in revision_names}
    mismatches = [
        revision
        for revision, outcome in outcomes.items()
        if outcome.observed != case.gold_expected[revision]
    ]
    classification = DifferentialClassifier().classify(
        outcomes["base"].results,
        outcomes["candidate"].results,
    )
    causal = (
        case.label == "positive"
        and not mismatches
        and classification.kind == ClassificationKind.STABLE_CHANGE
        and outcomes["fixed"].passed
    )
    valid = not mismatches and (case.label == "negative" or causal)
    return {
        "id": case.case_id,
        "framework": case.framework,
        "label": case.label,
        "control_kind": case.control_kind,
        "status": "passed" if valid else "failed",
        "summary": case.summary,
        "outcomes": {
            revision: {
                "passed": outcome.passed,
                "outcome": outcome.observed,
                "runs": [
                    {"exit_code": result.exit_code, "status": result.status.value}
                    for result in outcome.results
                ],
            }
            for revision, outcome in outcomes.items()
        },
        "classification": classification.kind.value,
        "classification_reason": classification.reason,
        "causal": causal,
        "mismatches": mismatches,
    }


def _execute_revision(case: BenchmarkCase, revision: str) -> RevisionOutcome:
    with tempfile.TemporaryDirectory(prefix=f"evalyn-bench-{case.case_id}-{revision}-") as raw:
        workspace = Path(raw) / "workspace"
        shutil.copytree(case.fixture_root / revision, workspace)
        target = (workspace / case.probe_target).resolve()
        target.relative_to(workspace.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case.probe_path, target)
        return _execute_workspace(
            case.case_id,
            case.framework,
            revision,
            workspace,
            case.command,
            case.repetitions,
        )


def _execute_scoring_revision(case: ScoringCase, revision: str) -> RevisionOutcome:
    roots = {
        "base": case.base_root,
        "candidate": case.candidate_root,
        "fixed": case.fixed_root,
    }
    source = roots[revision]
    if source is None:
        raise CorpusError(f"{case.case_id}: missing {revision} scoring snapshot")
    with tempfile.TemporaryDirectory(prefix=f"evalyn-gold-{case.case_id}-{revision}-") as raw:
        workspace = Path(raw) / "workspace"
        shutil.copytree(source, workspace)
        target = (workspace / case.gold_probe_target).resolve()
        target.relative_to(workspace.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case.hidden_gold_probe, target)
        return _execute_workspace(
            case.case_id,
            case.framework,
            revision,
            workspace,
            case.gold_command,
            3,
        )


def _execute_workspace(
    case_id: str,
    framework: str,
    revision: str,
    workspace: Path,
    command: tuple[str, ...],
    repetitions: int,
) -> RevisionOutcome:
    from evalyn.engine.models import ExecutionResult
    from evalyn.execution import CommandRunner

    results: list[ExecutionResult] = []
    with CommandRunner(
        workspace,
        max_output_bytes=_MAX_CAPTURE_BYTES,
        default_timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
    ) as runner:
        for _ in range(repetitions):
            completed = runner.run(_framework_tool_command(framework, command))
            if completed.timed_out or completed.error is not None:
                detail = completed.error or "command did not complete"
                raise CorpusError(f"{case_id}:{revision}: {detail}")
            assert completed.exit_code is not None
            output = (completed.stderr or completed.stdout)[-_MAX_CAPTURE_BYTES:]
            if completed.exit_code == 0:
                result = ExecutionResult.passed(
                    stdout=completed.stdout[-_MAX_CAPTURE_BYTES:],
                    duration_seconds=completed.duration_seconds,
                )
            else:
                result = ExecutionResult.failed(
                    exit_code=completed.exit_code,
                    stderr=output,
                    duration_seconds=completed.duration_seconds,
                    failure_signature=f"{case_id}:{revision}:exit-{completed.exit_code}",
                )
            results.append(result)
    return RevisionOutcome(revision=revision, results=tuple(results))


def _tool_available(case: BenchmarkCase) -> bool:
    from evalyn.execution import CommandRunner

    command = _local_tool_command(case, case.command)
    if shutil.which(command[0]) is None and not Path(command[0]).is_file():
        return False
    if not case.availability:
        return True
    with CommandRunner(
        case.fixture_root,
        max_output_bytes=1_024,
        default_timeout_seconds=_AVAILABILITY_TIMEOUT_SECONDS,
    ) as runner:
        completed = runner.run(
            _local_tool_command(case, case.availability),
            cwd=case.fixture_root / "base",
        )
    return completed.succeeded


def _scoring_tool_available(case: ScoringCase) -> bool:
    from evalyn.execution import CommandRunner

    command = _framework_tool_command(case.framework, case.gold_command)
    if shutil.which(command[0]) is None and not Path(command[0]).is_file():
        return False
    with CommandRunner(
        case.fixture_root,
        max_output_bytes=1_024,
        default_timeout_seconds=_AVAILABILITY_TIMEOUT_SECONDS,
    ) as runner:
        completed = runner.run(
            _framework_tool_command(case.framework, case.gold_availability),
            cwd=case.base_root,
        )
    return completed.succeeded


def _local_tool_command(
    case: BenchmarkCase,
    command: tuple[str, ...],
) -> tuple[str, ...]:
    """Use an already-installed Vitest binary without allowing npx downloads."""
    return _framework_tool_command(case.framework, command)


def _framework_tool_command(
    framework: str,
    command: tuple[str, ...],
) -> tuple[str, ...]:
    if framework == "vitest" and command[:3] == ("npx", "--no-install", "vitest"):
        executable = shutil.which("vitest")
        if executable is not None:
            return (executable, *command[3:])
    return command


def _required_text(value: dict[str, object], key: str, index: int) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise CorpusError(f"cases[{index}].{key} must be a nonempty string")
    return result.strip()


def _validate_scoring_snapshot(case_id: str, revision: str, root: Path) -> None:
    if (root / ".git").exists():
        raise CorpusError(f"{case_id}: {revision} snapshot cannot contain .git metadata")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise CorpusError(f"{case_id}: {revision} snapshot cannot contain symbolic links")


def _validate_gold_command(
    case_id: str,
    framework: str,
    probe_target: Path,
    command: tuple[str, ...],
    availability: tuple[str, ...],
) -> None:
    target = probe_target.as_posix()
    if framework == "pytest":
        command_valid = len(command) >= 4 and command[1:3] == ("-m", "pytest")
        availability_valid = availability[1:3] == ("-m", "pytest")
    else:
        command_valid = command[:4] == ("npx", "--no-install", "vitest", "run")
        availability_valid = availability[:3] == ("npx", "--no-install", "vitest")
    if not command_valid or target not in command:
        raise CorpusError(f"{case_id}: gold_command must execute only the adapter-owned probe")
    if not availability_valid:
        raise CorpusError(f"{case_id}: gold_availability must be an offline framework check")


def _named_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{label} must be a nonempty string")
    return value.strip()


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CorpusError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result < 1:
        raise CorpusError(f"{label} must be a positive integer")
    return result


def _argv(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(part, str) for part in value):
        raise CorpusError(f"{label} must be a nonempty string array")
    if any(not part for part in value):
        raise CorpusError(f"{label} cannot contain empty arguments")
    return tuple(value)


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.strip() != value:
        raise CorpusError(f"Unsafe relative fixture path: {value!r}")
    return path


def _contained(root: Path, value: str) -> Path:
    relative = _relative_path(value)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CorpusError(f"Fixture path escapes corpus root: {value!r}") from exc
    return candidate


def _render_result(result: dict[str, object]) -> None:
    if result["status"] == "skipped":
        print(f"{result['id']}\n  skipped    {result['reason']}")
        return
    outcomes = result["outcomes"]
    assert isinstance(outcomes, dict)
    print(result["id"])
    for revision in REVISIONS:
        detail = outcomes[revision]
        assert isinstance(detail, dict)
        runs = detail["runs"]
        assert isinstance(runs, list)
        label = "PASS" if detail["passed"] else "FAIL"
        print(f"  {revision:<10} {label} {len(runs)}/{len(runs)}")
    print(f"  evidence   {result['classification']}")
    causal = "yes (base pass -> buggy fail -> fixed pass)" if result["causal"] else "no"
    print(f"  causal     {causal}")


def _render_gold_result(result: dict[str, object]) -> None:
    if result["status"] == "skipped":
        print(f"{result['id']}\n  skipped    {result['reason']}")
        return
    outcomes = result["outcomes"]
    assert isinstance(outcomes, dict)
    print(f"{result['id']} ({result['label']})")
    for revision, detail in outcomes.items():
        assert isinstance(detail, dict)
        runs = detail["runs"]
        assert isinstance(runs, list)
        label = "PASS" if detail["passed"] else "FAIL"
        print(f"  {revision:<10} {label} {len(runs)}/{len(runs)}")
    print(f"  evidence   {result['classification']}")
    print(f"  gold       {result['status']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate", action="store_true", help="validate corpus files only")
    parser.add_argument("--execute", action="store_true", help="execute available causal fixtures")
    parser.add_argument(
        "--execute-gold",
        action="store_true",
        help="execute hidden gold probes for reviewed scoring cases",
    )
    parser.add_argument(
        "--score-observations",
        type=Path,
        help="rescore a versioned detector observation bundle without executing cases",
    )
    parser.add_argument(
        "--strict-tools",
        action="store_true",
        help="fail instead of skipping missing tools",
    )
    parser.add_argument("--case", action="append", dest="case_ids", default=[])
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    try:
        manifest = load_benchmark_manifest(args.manifest)
        cases = manifest.smoke_cases
        selected_modes = sum(
            (bool(args.execute), bool(args.execute_gold), args.score_observations is not None)
        )
        if selected_modes > 1:
            raise CorpusError(
                "--execute, --execute-gold, and --score-observations are mutually exclusive"
            )
        if args.score_observations is not None:
            from bench.scoring import ScoringError, load_observation_bundle, score_observations

            try:
                fingerprint, observations = load_observation_bundle(args.score_observations)
                report = score_observations(manifest, fingerprint, observations)
            except ScoringError as exc:
                raise CorpusError(str(exc)) from exc
            print(json.dumps(report, indent=2, sort_keys=True))
            gate = report["promotion_gate"]
            assert isinstance(gate, dict)
            return 0 if gate["met"] else 1
        if args.execute_gold:
            scoring_cases = manifest.scoring_cases
            if args.case_ids:
                requested = set(args.case_ids)
                selected_gold = tuple(case for case in scoring_cases if case.case_id in requested)
                missing = sorted(requested - {case.case_id for case in selected_gold})
                if missing:
                    raise CorpusError(f"Unknown scoring case(s): {', '.join(missing)}")
            else:
                selected_gold = scoring_cases
            gold_results = [
                execute_gold_case(case, strict_tools=args.strict_tools) for case in selected_gold
            ]
            if args.json_output:
                print(
                    json.dumps(
                        {"schema_version": 1, "gold_results": gold_results},
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif gold_results:
                for result in gold_results:
                    _render_gold_result(result)
            else:
                print("No reviewed scoring cases are registered.")
            return 1 if any(result["status"] == "failed" for result in gold_results) else 0
        if args.case_ids:
            selected = tuple(case for case in cases if case.case_id in set(args.case_ids))
            missing = sorted(set(args.case_ids) - {case.case_id for case in selected})
            if missing:
                raise CorpusError(f"Unknown benchmark case(s): {', '.join(missing)}")
        else:
            selected = cases

        if not args.execute:
            print(
                f"Validated {len(selected)} smoke case(s) and "
                f"{len(manifest.scoring_cases)} scoring case(s) from {args.manifest}"
            )
            return 0

        results = [execute_case(case, strict_tools=args.strict_tools) for case in selected]
        if args.json_output:
            print(json.dumps({"schema_version": 1, "results": results}, indent=2, sort_keys=True))
        else:
            for result in results:
                _render_result(result)
        return 1 if any(result["status"] == "failed" for result in results) else 0
    except (CorpusError, OSError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
