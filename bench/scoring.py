"""End-to-end detector scoring, including a deterministic recorded test path.

The recorded path deliberately exercises :func:`evalyn.workflow.run_check`
with its provider and executor seams. It is suitable for harness tests and
replays, but recorded execution outcomes are never accepted as real benchmark
evidence unless the surrounding evaluation process separately attests them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import evalyn.generation as generation_module
from bench.runner import MANIFEST_SCHEMA_VERSION, BenchmarkManifest, ScoringCase
from evalyn.adapters.base import AdapterError, ProbeArtifact
from evalyn.adapters.registry import default_registry
from evalyn.engine.classifier import DifferentialClassifier
from evalyn.engine.models import (
    ClassificationKind,
    ExecutionResult,
    ExecutionStatus,
    Finding,
    ProbeAssessment,
    Revision,
    RunVerdict,
)
from evalyn.engine.orchestrator import EngineError
from evalyn.generation import (
    PORTFOLIO_CANDIDATE_SYSTEM_PROMPT,
    PORTFOLIO_CORRECTION_SYSTEM_PROMPT,
    PORTFOLIO_PLANNING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ProbeGenerationError,
)
from evalyn.llm.base import BaseLLM
from evalyn.workflow import CheckError, CheckOptions, run_check

RESULT_SCHEMA_VERSION = 2
TRIAL_STATUSES = frozenset({"complete", "infrastructure_error", "invalid", "insufficient_evidence"})
_SHA256_LENGTH = 64
_REQUIRED_REPETITIONS = 3


class ScoringError(ValueError):
    """A scoring input, recording, or evaluation result is inconsistent."""


@dataclass(frozen=True)
class EvaluationFingerprint:
    """Identity of the model and immutable evaluation inputs."""

    provider: str
    model: str
    prompt_sha256: str
    environment_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider, str)
            or not isinstance(self.model, str)
            or not self.provider.strip()
            or not self.model.strip()
        ):
            raise ScoringError("provider and model fingerprints cannot be empty")
        _validate_sha256(self.prompt_sha256, "prompt_sha256")
        _validate_sha256(self.environment_sha256, "environment_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "environment_sha256": self.environment_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> EvaluationFingerprint:
        if not isinstance(value, dict) or set(value) != {
            "provider",
            "model",
            "prompt_sha256",
            "environment_sha256",
        }:
            raise ScoringError("fingerprints must define provider, model, prompt, and environment")
        if not all(isinstance(item, str) for item in value.values()):
            raise ScoringError("fingerprint values must be strings")
        return cls(**value)


@dataclass(frozen=True)
class RecordedProbeExecution:
    """Recorded outcomes for one normalized generated probe source."""

    source_sha256: str
    base: tuple[str, ...]
    candidate: tuple[str, ...]
    intervention_candidate: tuple[str, ...] = ()
    fixed: tuple[str, ...] = ()
    reproduction_base: tuple[str, ...] = ()
    reproduction_candidate: tuple[str, ...] = ()
    failure_signature: str = "recorded-assertion"
    duration_seconds: float = 0.001

    def __post_init__(self) -> None:
        _validate_sha256(self.source_sha256, "source_sha256")
        if (
            not isinstance(self.base, tuple)
            or not isinstance(self.candidate, tuple)
            or not self.base
            or not self.candidate
        ):
            raise ScoringError("recorded execution requires base and candidate outcomes")
        if bool(self.reproduction_base) != bool(self.reproduction_candidate):
            raise ScoringError("recorded reproduction requires both base and candidate outcomes")
        for side, statuses in (
            ("base", self.base),
            ("candidate", self.candidate),
            ("intervention_candidate", self.intervention_candidate),
            ("fixed", self.fixed),
            ("reproduction_base", self.reproduction_base),
            ("reproduction_candidate", self.reproduction_candidate),
        ):
            if not isinstance(statuses, tuple):
                raise ScoringError(f"recorded {side} outcomes must be a tuple")
            if statuses and len(statuses) != _REQUIRED_REPETITIONS:
                raise ScoringError(
                    f"recorded {side} requires exactly {_REQUIRED_REPETITIONS} outcomes"
                )
        if self.reproduction_base and not self.fixed:
            raise ScoringError("recorded reproduction requires fixed-revision outcomes")
        for status in (
            *self.base,
            *self.candidate,
            *self.intervention_candidate,
            *self.fixed,
            *self.reproduction_base,
            *self.reproduction_candidate,
        ):
            if status not in {item.value for item in ExecutionStatus}:
                raise ScoringError(f"unsupported recorded execution status: {status}")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or self.duration_seconds < 0
        ):
            raise ScoringError("recorded execution duration cannot be negative")


@dataclass(frozen=True)
class TrialObservation:
    """One independently seeded detector trial, ready for aggregate scoring."""

    case_id: str
    trial: int
    effective_seed: str
    evaluation_fingerprint_sha256: str
    case_revision_sha256: str
    status: str
    workflow_verdict: str
    portfolio_k: int
    repetitions_per_finding: int
    generated_probes: int
    valid_probes: int
    generation_rejection_count: int
    generation_infrastructure_count: int
    runtime_invalid_probe_count: int
    verification_invalid_count: int
    infrastructure_error_count: int
    first_stable_change_rank: int | None
    first_fixed_verified_rank: int | None
    automatic_gate: bool
    automatic_gate_fixed_verified: bool
    duration_seconds: float
    fixed_replay_attempts: int = 0
    fixed_replay_passes: int = 0
    reproduction_required: int = 0
    reproduction_base_attempts: int = 0
    reproduction_candidate_attempts: int = 0
    reproduction_successes: int = 0
    request_sha256: str = ""
    response_transcript_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or not isinstance(self.effective_seed, str)
            or not self.case_id.strip()
            or not self.effective_seed.strip()
        ):
            raise ScoringError("trial case_id and effective_seed cannot be empty")
        if self.effective_seed != self.effective_seed.strip():
            raise ScoringError("effective_seed cannot have surrounding whitespace")
        _validate_sha256(
            self.evaluation_fingerprint_sha256,
            "evaluation_fingerprint_sha256",
        )
        _validate_sha256(self.case_revision_sha256, "case_revision_sha256")
        if not isinstance(self.trial, int) or isinstance(self.trial, bool) or self.trial < 1:
            raise ScoringError("trial must be one-indexed")
        for field_name, value in (
            ("portfolio_k", self.portfolio_k),
            ("repetitions_per_finding", self.repetitions_per_finding),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ScoringError(f"{field_name} must be a positive integer")
        if not isinstance(self.status, str) or self.status not in TRIAL_STATUSES:
            raise ScoringError(f"unsupported trial status: {self.status}")
        workflow_verdicts = {item.value for item in RunVerdict} | {"error"}
        if self.workflow_verdict not in workflow_verdicts:
            raise ScoringError(f"unsupported workflow verdict: {self.workflow_verdict}")
        counts = (
            self.generated_probes,
            self.valid_probes,
            self.generation_rejection_count,
            self.generation_infrastructure_count,
            self.runtime_invalid_probe_count,
            self.verification_invalid_count,
            self.infrastructure_error_count,
            self.fixed_replay_attempts,
            self.fixed_replay_passes,
            self.reproduction_required,
            self.reproduction_base_attempts,
            self.reproduction_candidate_attempts,
            self.reproduction_successes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts
        ):
            raise ScoringError("trial counts cannot be negative")
        if (
            self.valid_probes
            + self.generation_rejection_count
            + self.generation_infrastructure_count
            + self.runtime_invalid_probe_count
            != self.generated_probes
        ):
            raise ScoringError(
                "valid, rejected, infrastructure, and runtime-invalid counts must equal generated"
            )
        for label, rank in (
            ("first stable-change", self.first_stable_change_rank),
            ("first fixed-verified", self.first_fixed_verified_rank),
        ):
            if rank is not None and (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or not 1 <= rank <= self.generated_probes
            ):
                raise ScoringError(f"{label} rank must identify a generated probe")
        if self.first_fixed_verified_rank is not None and (
            self.first_stable_change_rank is None
            or self.first_fixed_verified_rank < self.first_stable_change_rank
        ):
            raise ScoringError("fixed verification must identify a stable-change probe")
        if not isinstance(self.automatic_gate, bool):
            raise ScoringError("automatic_gate must be boolean")
        if not isinstance(self.automatic_gate_fixed_verified, bool):
            raise ScoringError("automatic_gate_fixed_verified must be boolean")
        if self.automatic_gate != (self.workflow_verdict == RunVerdict.REGRESSION_FOUND.value):
            raise ScoringError("automatic_gate must be derived from the workflow verdict")
        if self.automatic_gate_fixed_verified and (
            not self.automatic_gate or self.first_fixed_verified_rank is None
        ):
            raise ScoringError(
                "a fixed-verified gate requires a gate and a fixed-verified discovery"
            )
        expected_status = (
            "infrastructure_error"
            if self.workflow_verdict == "error" and self.infrastructure_error_count
            else "invalid"
            if self.workflow_verdict == "error"
            else _status_from_workflow(
                RunVerdict(self.workflow_verdict),
                runtime_invalid_count=self.runtime_invalid_probe_count,
                verification_invalid_count=self.verification_invalid_count,
                infrastructure_error_count=self.infrastructure_error_count,
            )
        )
        if self.status != expected_status:
            raise ScoringError(
                "trial status must be derived from workflow verdict and observed errors"
            )
        stable_verdicts = {
            RunVerdict.CHANGE_OBSERVED.value,
            RunVerdict.REGRESSION_FOUND.value,
        }
        if self.first_stable_change_rank is not None and self.workflow_verdict not in (
            stable_verdicts
        ):
            raise ScoringError("stable-change rank requires a stable workflow verdict")
        if self.workflow_verdict in stable_verdicts and self.first_stable_change_rank is None:
            raise ScoringError("stable workflow verdict requires a ranked finding")
        if self.status == "complete" and not self.valid_probes:
            raise ScoringError("complete trials require at least one valid generated probe")
        if self.reproduction_required not in {0, 1} or self.reproduction_successes not in {0, 1}:
            raise ScoringError("reproduction coverage is binary per credited discovery")
        expected_reproduction = int(self.first_fixed_verified_rank is not None)
        if self.reproduction_required != expected_reproduction:
            raise ScoringError(
                "every fixed-verified discovery requires its own reproduction attempt"
            )
        if self.reproduction_successes > self.reproduction_required:
            raise ScoringError("reproduction successes cannot exceed required coverage")
        expected_replay_attempts = _REQUIRED_REPETITIONS * expected_reproduction
        if (
            self.fixed_replay_attempts != expected_replay_attempts
            or self.fixed_replay_passes != expected_replay_attempts
        ):
            raise ScoringError("credited discovery requires three passing fixed-revision replays")
        if (
            self.reproduction_base_attempts != expected_replay_attempts
            or self.reproduction_candidate_attempts != expected_replay_attempts
        ):
            raise ScoringError(
                "credited discovery requires three reproduction replays per revision"
            )
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or self.duration_seconds < 0
        ):
            raise ScoringError("trial duration cannot be negative")
        if bool(self.request_sha256) != bool(self.response_transcript_sha256):
            raise ScoringError("request and response transcript digests must be supplied together")
        if self.status == "complete" and not self.request_sha256:
            raise ScoringError("complete trials require request and response transcript digests")
        for field_name, digest in (
            ("request_sha256", self.request_sha256),
            ("response_transcript_sha256", self.response_transcript_sha256),
        ):
            if digest:
                _validate_sha256(digest, field_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "trial": self.trial,
            "effective_seed": self.effective_seed,
            "evaluation_fingerprint_sha256": self.evaluation_fingerprint_sha256,
            "case_revision_sha256": self.case_revision_sha256,
            "status": self.status,
            "workflow_verdict": self.workflow_verdict,
            "portfolio_k": self.portfolio_k,
            "repetitions_per_finding": self.repetitions_per_finding,
            "generated_probes": self.generated_probes,
            "valid_probes": self.valid_probes,
            "generation_rejection_count": self.generation_rejection_count,
            "generation_infrastructure_count": self.generation_infrastructure_count,
            "runtime_invalid_probe_count": self.runtime_invalid_probe_count,
            "verification_invalid_count": self.verification_invalid_count,
            "infrastructure_error_count": self.infrastructure_error_count,
            "first_stable_change_rank": self.first_stable_change_rank,
            "first_fixed_verified_rank": self.first_fixed_verified_rank,
            "automatic_gate": self.automatic_gate,
            "automatic_gate_fixed_verified": self.automatic_gate_fixed_verified,
            "duration_seconds": self.duration_seconds,
            "fixed_replay_attempts": self.fixed_replay_attempts,
            "fixed_replay_passes": self.fixed_replay_passes,
            "reproduction_required": self.reproduction_required,
            "reproduction_base_attempts": self.reproduction_base_attempts,
            "reproduction_candidate_attempts": self.reproduction_candidate_attempts,
            "reproduction_successes": self.reproduction_successes,
            "request_sha256": self.request_sha256,
            "response_transcript_sha256": self.response_transcript_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrialObservation:
        fields = {
            "case_id",
            "trial",
            "effective_seed",
            "evaluation_fingerprint_sha256",
            "case_revision_sha256",
            "status",
            "workflow_verdict",
            "portfolio_k",
            "repetitions_per_finding",
            "generated_probes",
            "valid_probes",
            "generation_rejection_count",
            "generation_infrastructure_count",
            "runtime_invalid_probe_count",
            "verification_invalid_count",
            "infrastructure_error_count",
            "first_stable_change_rank",
            "first_fixed_verified_rank",
            "automatic_gate",
            "automatic_gate_fixed_verified",
            "duration_seconds",
            "fixed_replay_attempts",
            "fixed_replay_passes",
            "reproduction_required",
            "reproduction_base_attempts",
            "reproduction_candidate_attempts",
            "reproduction_successes",
            "request_sha256",
            "response_transcript_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ScoringError("trial observations must be objects")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ScoringError(f"trial observation has invalid fields: {exc}") from exc


@dataclass(frozen=True)
class _PortfolioSummary:
    generated: int
    accepted: int
    rejected: int
    infrastructure_errors: int
    hypothesis_ranks: dict[str, int]


RecordedReply = Mapping[str, object]
RecordedReplyFactory = Callable[[str, str, int, str], RecordedReply]


class RecordedProvider(BaseLLM):
    """Replay pinned or scripted JSON replies while recording exact prompts."""

    def __init__(
        self,
        responses: RecordedReply | Sequence[RecordedReply] | RecordedReplyFactory,
        *,
        effective_seed: str,
    ) -> None:
        if not isinstance(effective_seed, str) or not effective_seed.strip():
            raise ScoringError("recorded provider effective_seed cannot be empty")
        if effective_seed != effective_seed.strip():
            raise ScoringError(
                "recorded provider effective_seed cannot have surrounding whitespace"
            )
        self.effective_seed = effective_seed
        self._factory: RecordedReplyFactory | None = None
        self._serialized_responses: tuple[str, ...] = ()
        if callable(responses):
            self._factory = responses
        elif isinstance(responses, Mapping):
            self._serialized_responses = (json.dumps(responses, sort_keys=True),)
        else:
            self._serialized_responses = tuple(
                json.dumps(response, sort_keys=True) for response in responses
            )
            if not self._serialized_responses:
                raise ScoringError("recorded provider requires at least one response")
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []
        self.response_sha256: list[str] = []

    @property
    def prompt_sha256(self) -> str:
        if not self.system_prompts:
            raise ScoringError("recorded provider did not receive a structured completion")
        return _sha256_text("\0".join(self.system_prompts))

    @property
    def request_sha256(self) -> str:
        if not self.system_prompts or len(self.system_prompts) != len(self.user_prompts):
            raise ScoringError("recorded provider has an incomplete request transcript")
        transcript = f"effective-seed\0{self.effective_seed}\0\0" + "\0\0".join(
            f"{system}\0{user}"
            for system, user in zip(self.system_prompts, self.user_prompts, strict=True)
        )
        return _sha256_text(transcript)

    @property
    def response_transcript_sha256(self) -> str:
        if not self.system_prompts:
            raise ScoringError("recorded provider did not receive a structured completion")
        return _sha256_text(
            f"request-sha256\0{self.request_sha256}\0\0" + "\0".join(self.response_sha256)
        )

    @property
    def transcript_complete(self) -> bool:
        return bool(self.system_prompts) and len(self.system_prompts) == len(self.response_sha256)

    def complete(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 4096) -> str:
        raise AssertionError("recorded benchmark provider only supports JSON completions")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
    ) -> dict | list:
        index = len(self.system_prompts)
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if self._factory is not None:
            response = self._factory(system_prompt, user_prompt, index, self.effective_seed)
            serialized = json.dumps(response, sort_keys=True)
        else:
            if index >= len(self._serialized_responses):
                raise ScoringError(f"recorded provider has no response for call {index + 1}")
            serialized = self._serialized_responses[index]
        self.response_sha256.append(_sha256_text(serialized))
        return json.loads(serialized)


@dataclass(frozen=True)
class _RecordedExecutionEvent:
    side: str
    repetition: int
    status: ExecutionStatus


class RecordedExecutorFactory:
    """Workflow executor factory backed by content-addressed result recordings."""

    def __init__(self, executions: Sequence[RecordedProbeExecution]) -> None:
        self.executions = {item.source_sha256: item for item in executions}
        self.events: list[_RecordedExecutionEvent] = []
        if len(self.executions) != len(executions):
            raise ScoringError("recorded probe source digests must be unique")

    def __call__(self, *_args: object) -> _RecordedExecutor:
        return _RecordedExecutor(self.executions, self.events)

    @property
    def intervention_error_counts(self) -> tuple[int, int]:
        events = tuple(event for event in self.events if event.side == "target_intervention")
        return (
            sum(event.status == ExecutionStatus.INVALID_PROBE for event in events),
            sum(event.status == ExecutionStatus.INFRA_ERROR for event in events),
        )


class _RecordedExecutor:
    def __init__(
        self,
        executions: Mapping[str, RecordedProbeExecution],
        events: list[_RecordedExecutionEvent],
    ) -> None:
        self.executions = executions
        self.events = events
        self.setup_errors: dict[Path, str] = {}
        self._sides: dict[str, str] = {}

    def prepare(self, revisions: tuple[tuple[str, Revision], ...]) -> None:
        self._sides = {revision.sha: side for side, revision in revisions}

    def execute(
        self,
        revision: Revision,
        artifact: ProbeArtifact,
        *,
        repetition: int,
    ) -> ExecutionResult:
        recording = self.executions.get(artifact.source_sha256)
        side = self._sides.get(revision.sha)
        if recording is None or side not in {
            "base",
            "candidate",
            "fixed",
            "reproduction_base",
            "reproduction_candidate",
        }:
            result = ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=-1,
                stderr="recorded benchmark outcome is missing",
            )
            return self._record(side or "unknown", repetition, result)
        if side == "candidate" and repetition >= 10_001:
            event_side = "target_intervention"
            sequence = recording.intervention_candidate
            sequence_index = repetition - 10_001
        else:
            event_side = side
            sequence = {
                "base": recording.base,
                "candidate": recording.candidate,
                "fixed": recording.fixed,
                "reproduction_base": recording.reproduction_base,
                "reproduction_candidate": recording.reproduction_candidate,
            }[side]
            sequence_index = repetition - 1
        if sequence_index < 0 or sequence_index >= len(sequence):
            result = ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=-1,
                stderr=f"recording has no outcome for repetition {repetition}",
            )
            return self._record(event_side, repetition, result)
        status = ExecutionStatus(sequence[sequence_index])
        if status == ExecutionStatus.PASSED:
            result = ExecutionResult.passed(
                duration_seconds=recording.duration_seconds,
                metadata={"environment_integrity": "verified"},
            )
        elif status == ExecutionStatus.FAILED:
            result = ExecutionResult.failed(
                failure_signature=recording.failure_signature,
                stderr=recording.failure_signature,
                duration_seconds=recording.duration_seconds,
                metadata={"environment_integrity": "verified"},
            )
        else:
            exit_codes = {
                ExecutionStatus.TIMEOUT: 124,
                ExecutionStatus.INVALID_PROBE: 2,
                ExecutionStatus.INFRA_ERROR: -1,
            }
            result = ExecutionResult(
                status=status,
                exit_code=exit_codes[status],
                stderr=status.value,
                duration_seconds=recording.duration_seconds,
                metadata={"environment_integrity": "verified"},
            )
        return self._record(event_side, repetition, result)

    def _record(
        self,
        side: str,
        repetition: int,
        result: ExecutionResult,
    ) -> ExecutionResult:
        self.events.append(_RecordedExecutionEvent(side, repetition, result.status))
        return result


def normalized_probe_sha256(source: str) -> str:
    """Return the digest adapters use after normalizing a generated probe."""
    return _sha256_text(source.rstrip() + "\n")


def current_prompt_sha256() -> str:
    """Fingerprint the exact generation prompts and renderer used by this build."""
    implementation = inspect.getsource(generation_module)
    return _sha256_text(implementation)


def environment_sha256(value: Mapping[str, object]) -> str:
    """Fingerprint a caller-provided, secret-free environment description."""
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _sha256_text(serialized)


def evaluation_fingerprint_sha256(fingerprint: EvaluationFingerprint) -> str:
    """Return a canonical digest suitable for binding every trial to a bundle."""
    serialized = json.dumps(fingerprint.to_dict(), sort_keys=True, separators=(",", ":"))
    return _sha256_text(serialized)


def scoring_case_revision_sha256(case: ScoringCase) -> str:
    """Bind observations to the exact reviewed case and revision snapshots."""
    payload = {
        "schema_version": 1,
        "case_id": case.case_id,
        "framework": case.framework,
        "label": case.label,
        "control_kind": case.control_kind,
        "summary": case.summary,
        "intent": list(case.intent),
        "base_sha256": _snapshot_sha256(case.base_root),
        "candidate_sha256": _snapshot_sha256(case.candidate_root),
        "fixed_sha256": (
            _snapshot_sha256(case.fixed_root) if case.fixed_root is not None else None
        ),
        "hidden_gold_relative_path": _relative_case_path(case.fixture_root, case.hidden_gold_probe),
        "hidden_gold_sha256": _regular_file_sha256(case.hidden_gold_probe),
        "gold_probe_target": case.gold_probe_target.as_posix(),
        "gold_command": list(case.gold_command),
        "gold_availability": list(case.gold_availability),
        "gold_expected": dict(sorted(case.gold_expected.items())),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _snapshot_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ScoringError(f"benchmark revision root must be a real directory: {root}")
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ScoringError(f"benchmark revision snapshots cannot contain links: {path}")
        if path.is_dir():
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "kind": "directory",
                }
            )
            continue
        if not path.is_file():
            raise ScoringError(f"benchmark revision contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_bytes()
            executable = bool(path.stat().st_mode & 0o111)
        except OSError as exc:
            raise ScoringError(
                f"could not fingerprint benchmark revision file {path}: {exc}"
            ) from exc
        entries.append(
            {
                "path": relative,
                "kind": "file",
                "sha256": hashlib.sha256(content).hexdigest(),
                "executable": executable,
            }
        )
    return _sha256_text(json.dumps(entries, sort_keys=True, separators=(",", ":")))


def _regular_file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ScoringError(f"benchmark gold probe must be a real file: {path}")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ScoringError(f"could not fingerprint benchmark gold probe {path}: {exc}") from exc


def _relative_case_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ScoringError(f"benchmark case path escapes its fixture root: {path}") from exc


def run_recorded_trial(
    case: ScoringCase,
    *,
    trial: int,
    seed: str,
    provider: RecordedProvider,
    executor_factory: RecordedExecutorFactory,
    fingerprint: EvaluationFingerprint,
    work_root: Path,
    portfolio_k: int = 8,
    repetitions: int = 3,
) -> TrialObservation:
    """Run one no-key replay through the public workflow and derive an observation."""
    if trial < 1 or portfolio_k < 1:
        raise ScoringError("trial and portfolio_k must be positive")
    if repetitions != _REQUIRED_REPETITIONS:
        raise ScoringError("recorded scoring trials require exactly three repetitions")
    if not isinstance(seed, str) or not seed.strip() or seed != seed.strip():
        raise ScoringError("effective trial seed must be a non-empty normalized string")
    if seed != provider.effective_seed:
        raise ScoringError("trial seed does not match the recorded provider effective_seed")
    if provider.system_prompts or provider.user_prompts or provider.response_sha256:
        raise ScoringError("each recorded trial requires a fresh provider transcript")
    if executor_factory.events:
        raise ScoringError("each recorded trial requires a fresh executor transcript")
    if fingerprint.prompt_sha256 != current_prompt_sha256():
        raise ScoringError("prompt_sha256 does not identify this Evalyn generation prompt")
    fingerprint_sha256 = evaluation_fingerprint_sha256(fingerprint)
    case_revision_sha256 = scoring_case_revision_sha256(case)
    work_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"{case.case_id}-{trial}-", dir=work_root) as raw:
        root = Path(raw)
        repo = root / "repo"
        _materialize_repository(case, repo)
        options = CheckOptions(
            repo_path=repo,
            range_spec="main...candidate",
            intent=case.intent,
            adapters=(case.framework,),
            max_tests=portfolio_k,
            repeat=repetitions,
            budget_seconds=300,
            state_root=root / "state",
        )
        try:
            outcome = run_check(
                options,
                llm=provider,
                executor_factory=executor_factory,
            )
        except CheckError as exc:
            if provider.system_prompts:
                _verify_prompt_fingerprint(provider)
            portfolio = _load_portfolio_summary(exc.run_path, case.framework)
            status = _classify_check_error(exc, portfolio, provider)
            generated = portfolio.generated if portfolio is not None else 0
            rejected = portfolio.rejected if portfolio is not None else 0
            provider_infrastructure = (
                portfolio.infrastructure_errors if portfolio is not None else 0
            )
            request_sha256, response_sha256 = _transcript_digests(provider)
            return TrialObservation(
                case_id=case.case_id,
                trial=trial,
                effective_seed=provider.effective_seed,
                evaluation_fingerprint_sha256=fingerprint_sha256,
                case_revision_sha256=case_revision_sha256,
                status=status,
                workflow_verdict="error",
                portfolio_k=portfolio_k,
                repetitions_per_finding=repetitions,
                generated_probes=generated,
                valid_probes=generated - rejected,
                generation_rejection_count=rejected - provider_infrastructure,
                generation_infrastructure_count=provider_infrastructure,
                runtime_invalid_probe_count=0,
                verification_invalid_count=0,
                infrastructure_error_count=(
                    max(1, provider_infrastructure) if status == "infrastructure_error" else 0
                ),
                first_stable_change_rank=None,
                first_fixed_verified_rank=None,
                automatic_gate=False,
                automatic_gate_fixed_verified=False,
                duration_seconds=time.monotonic() - started,
                reproduction_required=0,
                reproduction_successes=0,
                request_sha256=request_sha256,
                response_transcript_sha256=response_sha256,
            )

        portfolio = _load_portfolio_summary(outcome.run.path, case.framework)

    _verify_prompt_fingerprint(provider)
    assessments = tuple(
        assessment for result in outcome.results for assessment in result.assessments
    )
    runtime_invalid_count = sum(
        any(
            record.result.status == ExecutionStatus.INVALID_PROBE
            for record in (*assessment.base_runs, *assessment.candidate_runs)
        )
        for assessment in assessments
    )
    infrastructure_count = sum(
        record.result.status == ExecutionStatus.INFRA_ERROR
        for assessment in assessments
        for record in (*assessment.base_runs, *assessment.candidate_runs)
    )
    intervention_invalid_count, intervention_infrastructure_count = (
        executor_factory.intervention_error_counts
    )
    infrastructure_count += intervention_infrastructure_count
    intervention_incomplete_count = sum(
        finding.probe_metadata.get("target_intervention") == "candidate_target_revert_incomplete"
        for finding in outcome.findings
    )
    if intervention_incomplete_count and not (
        intervention_invalid_count or intervention_infrastructure_count
    ):
        infrastructure_count += intervention_incomplete_count
    if portfolio is None:
        infrastructure_count += 1
    ranked_findings = _ranked_findings(outcome.findings, assessments, portfolio)
    unranked_finding_count = len(outcome.findings) - len(ranked_findings)
    if portfolio is not None:
        generated_count = portfolio.generated
        generation_infrastructure_count = portfolio.infrastructure_errors
        generation_rejection_count = portfolio.rejected - generation_infrastructure_count
    else:
        generated_count = len(assessments)
        generation_rejection_count = 0
        generation_infrastructure_count = 0
    valid_count = (
        generated_count
        - generation_rejection_count
        - generation_infrastructure_count
        - runtime_invalid_count
    )
    if valid_count < 0:
        raise ScoringError("workflow evidence reports impossible probe validity counts")

    provider_infrastructure_count = generation_infrastructure_count
    infrastructure_count += provider_infrastructure_count
    if not provider.transcript_complete and not provider_infrastructure_count:
        infrastructure_count += 1
    verification = _DiscoveryVerification()
    if (
        case.label == "positive"
        and portfolio is not None
        and outcome.verdict not in {RunVerdict.INCOMPLETE, RunVerdict.INSUFFICIENT_EVIDENCE}
        and ranked_findings
    ):
        verification = _verify_recorded_discovery(
            case,
            ranked_findings,
            executor_factory,
            repetitions=repetitions,
        )
        infrastructure_count += verification.infrastructure_error_count
    verification_invalid_count = (
        verification.invalid_count + unranked_finding_count + intervention_invalid_count
    )
    status = _status_from_workflow(
        outcome.verdict,
        runtime_invalid_count=runtime_invalid_count,
        verification_invalid_count=verification_invalid_count,
        infrastructure_error_count=infrastructure_count,
    )
    request_sha256, response_sha256 = _transcript_digests(provider)
    return TrialObservation(
        case_id=case.case_id,
        trial=trial,
        effective_seed=provider.effective_seed,
        evaluation_fingerprint_sha256=fingerprint_sha256,
        case_revision_sha256=case_revision_sha256,
        status=status,
        workflow_verdict=outcome.verdict.value,
        portfolio_k=portfolio_k,
        repetitions_per_finding=repetitions,
        generated_probes=generated_count,
        valid_probes=valid_count,
        generation_rejection_count=generation_rejection_count,
        generation_infrastructure_count=generation_infrastructure_count,
        runtime_invalid_probe_count=runtime_invalid_count,
        verification_invalid_count=verification_invalid_count,
        infrastructure_error_count=infrastructure_count,
        first_stable_change_rank=ranked_findings[0][0] if ranked_findings else None,
        first_fixed_verified_rank=verification.first_fixed_verified_rank,
        automatic_gate=outcome.verdict == RunVerdict.REGRESSION_FOUND,
        automatic_gate_fixed_verified=(
            outcome.verdict == RunVerdict.REGRESSION_FOUND and verification.verified_finding_gates
        ),
        duration_seconds=time.monotonic() - started,
        fixed_replay_attempts=verification.fixed_replay_attempts,
        fixed_replay_passes=verification.fixed_replay_passes,
        reproduction_required=verification.reproduction_required,
        reproduction_base_attempts=verification.reproduction_base_attempts,
        reproduction_candidate_attempts=verification.reproduction_candidate_attempts,
        reproduction_successes=verification.reproduction_successes,
        request_sha256=request_sha256,
        response_transcript_sha256=response_sha256,
    )


@dataclass(frozen=True)
class _DiscoveryVerification:
    first_fixed_verified_rank: int | None = None
    verified_finding_gates: bool = False
    fixed_replay_attempts: int = 0
    fixed_replay_passes: int = 0
    reproduction_required: int = 0
    reproduction_base_attempts: int = 0
    reproduction_candidate_attempts: int = 0
    reproduction_successes: int = 0
    invalid_count: int = 0
    infrastructure_error_count: int = 0


def _ranked_findings(
    findings: Sequence[Finding],
    assessments: Sequence[ProbeAssessment],
    portfolio: _PortfolioSummary | None,
) -> tuple[tuple[int, Finding], ...]:
    assessment_ranks = {
        assessment.artifact.source_sha256: index
        for index, assessment in enumerate(assessments, start=1)
    }
    ranked: list[tuple[int, Finding]] = []
    for finding in findings:
        hypothesis_id = finding.probe_metadata.get("hypothesis_id")
        rank = (
            portfolio.hypothesis_ranks.get(hypothesis_id)
            if portfolio is not None and hypothesis_id is not None
            else assessment_ranks.get(finding.artifact.source_sha256)
        )
        if rank is not None:
            ranked.append((rank, finding))
    return tuple(sorted(ranked, key=lambda item: item[0]))


def _verify_recorded_discovery(
    case: ScoringCase,
    ranked_findings: Sequence[tuple[int, Finding]],
    executor_factory: RecordedExecutorFactory,
    *,
    repetitions: int,
) -> _DiscoveryVerification:
    if case.fixed_root is None:
        return _DiscoveryVerification(infrastructure_error_count=1)

    adapter = default_registry().get(case.framework)
    classifier = DifferentialClassifier(repetitions)
    for rank, finding in ranked_findings:
        try:
            artifact = adapter.build_artifact(finding.probe_source)
        except AdapterError:
            return _DiscoveryVerification(invalid_count=1)
        digest = artifact.source_sha256[:16]
        fixed_revision = Revision(f"benchmark-fixed-{digest}", case.fixed_root)
        reproduction_base = Revision(f"benchmark-reproduction-base-{digest}", case.base_root)
        reproduction_candidate = Revision(
            f"benchmark-reproduction-candidate-{digest}", case.candidate_root
        )
        executor = executor_factory()
        executor.prepare(
            (
                ("fixed", fixed_revision),
                ("reproduction_base", reproduction_base),
                ("reproduction_candidate", reproduction_candidate),
            )
        )
        fixed_results = _replay_side(executor, fixed_revision, artifact, repetitions)
        fixed_invalid, fixed_infrastructure = _replay_error_counts(fixed_results)
        if fixed_invalid or fixed_infrastructure:
            return _DiscoveryVerification(
                invalid_count=fixed_invalid,
                infrastructure_error_count=fixed_infrastructure,
            )
        if not all(result.status == ExecutionStatus.PASSED for result in fixed_results):
            continue

        reproduction_base_results = _replay_side(executor, reproduction_base, artifact, repetitions)
        reproduction_candidate_results = _replay_side(
            executor, reproduction_candidate, artifact, repetitions
        )
        reproduction_results = (*reproduction_base_results, *reproduction_candidate_results)
        reproduction_invalid, reproduction_infrastructure = _replay_error_counts(
            reproduction_results
        )
        reproduced = False
        if not reproduction_invalid and not reproduction_infrastructure:
            reproduction_classification = classifier.classify(
                reproduction_base_results,
                reproduction_candidate_results,
            )
            reproduced = (
                reproduction_classification.kind == ClassificationKind.STABLE_CHANGE
                and reproduction_classification.failure_signature
                == finding.classification.failure_signature
            )
        return _DiscoveryVerification(
            first_fixed_verified_rank=rank,
            verified_finding_gates=finding.should_gate,
            fixed_replay_attempts=len(fixed_results),
            fixed_replay_passes=sum(
                result.status == ExecutionStatus.PASSED for result in fixed_results
            ),
            reproduction_required=1,
            reproduction_base_attempts=len(reproduction_base_results),
            reproduction_candidate_attempts=len(reproduction_candidate_results),
            reproduction_successes=int(reproduced),
            invalid_count=reproduction_invalid,
            infrastructure_error_count=reproduction_infrastructure,
        )
    return _DiscoveryVerification()


def _replay_side(
    executor: _RecordedExecutor,
    revision: Revision,
    artifact: ProbeArtifact,
    repetitions: int,
) -> tuple[ExecutionResult, ...]:
    results: list[ExecutionResult] = []
    for repetition in range(1, repetitions + 1):
        try:
            result = executor.execute(revision, artifact, repetition=repetition)
        except Exception:
            result = ExecutionResult(
                status=ExecutionStatus.INFRA_ERROR,
                exit_code=-1,
                stderr="recorded replay execution raised an exception",
            )
        results.append(result)
    return tuple(results)


def _replay_error_counts(results: Sequence[ExecutionResult]) -> tuple[int, int]:
    return (
        sum(result.status == ExecutionStatus.INVALID_PROBE for result in results),
        sum(result.status == ExecutionStatus.INFRA_ERROR for result in results),
    )


def _status_from_workflow(
    verdict: RunVerdict,
    *,
    runtime_invalid_count: int,
    verification_invalid_count: int,
    infrastructure_error_count: int,
) -> str:
    if infrastructure_error_count:
        return "infrastructure_error"
    if runtime_invalid_count or verification_invalid_count or verdict == RunVerdict.INCOMPLETE:
        return "invalid"
    if verdict == RunVerdict.INSUFFICIENT_EVIDENCE:
        return "insufficient_evidence"
    return "complete"


def score_observations(
    manifest: BenchmarkManifest,
    fingerprint: EvaluationFingerprint,
    observations: Sequence[TrialObservation],
) -> dict[str, object]:
    """Aggregate detector trials without rerunning repositories or providers."""
    case_by_id = {case.case_id: case for case in manifest.scoring_cases}
    fingerprint_sha256 = evaluation_fingerprint_sha256(fingerprint)
    case_revision_digests = {
        case.case_id: scoring_case_revision_sha256(case) for case in manifest.scoring_cases
    }
    observed: dict[tuple[str, int], TrialObservation] = {}
    for observation in observations:
        case = case_by_id.get(observation.case_id)
        if case is None:
            raise ScoringError(
                f"observation references unknown scoring case: {observation.case_id}"
            )
        if observation.trial > manifest.protocol.trials_per_case:
            raise ScoringError(f"{observation.case_id}: trial exceeds configured trials_per_case")
        if observation.evaluation_fingerprint_sha256 != fingerprint_sha256:
            raise ScoringError(
                f"{observation.case_id}: trial does not match the bundle fingerprint"
            )
        if observation.portfolio_k != manifest.protocol.portfolio_k:
            raise ScoringError(
                f"{observation.case_id}: trial portfolio_k does not match the manifest"
            )
        if observation.repetitions_per_finding != manifest.protocol.repetitions_per_finding:
            raise ScoringError(
                f"{observation.case_id}: trial repetitions do not match the manifest"
            )
        if observation.case_revision_sha256 != case_revision_digests[observation.case_id]:
            raise ScoringError(
                f"{observation.case_id}: observation is bound to different revision snapshots"
            )
        for rank in (
            observation.first_stable_change_rank,
            observation.first_fixed_verified_rank,
        ):
            if rank is not None and rank > manifest.protocol.portfolio_k:
                raise ScoringError(
                    f"{observation.case_id}: finding rank exceeds configured portfolio_k"
                )
        if case.label == "negative" and observation.first_fixed_verified_rank is not None:
            raise ScoringError(
                f"{observation.case_id}: negative controls cannot claim fixed-verified discovery"
            )
        if observation.first_fixed_verified_rank is not None:
            if observation.workflow_verdict in {
                "error",
                RunVerdict.INCOMPLETE.value,
                RunVerdict.INSUFFICIENT_EVIDENCE.value,
            }:
                raise ScoringError(
                    f"{observation.case_id}: incomplete workflows cannot claim discovery"
                )
            if not observation.request_sha256:
                raise ScoringError(
                    f"{observation.case_id}: credited discovery lacks transcript attestations"
                )
        key = (observation.case_id, observation.trial)
        if key in observed:
            raise ScoringError(
                f"duplicate trial observation: {observation.case_id}#{observation.trial}"
            )
        observed[key] = observation

    for case in manifest.scoring_cases:
        case_observations = [item for item in observed.values() if item.case_id == case.case_id]
        seeds = [item.effective_seed for item in case_observations]
        if len(seeds) != len(set(seeds)):
            raise ScoringError(f"{case.case_id}: every trial must use an independent seed")
        for label, digests in (
            (
                "request",
                [item.request_sha256 for item in case_observations if item.request_sha256],
            ),
            (
                "response",
                [
                    item.response_transcript_sha256
                    for item in case_observations
                    if item.response_transcript_sha256
                ],
            ),
        ):
            if len(digests) != len(set(digests)):
                raise ScoringError(
                    f"{case.case_id}: {label} transcripts do not attest independent seeds"
                )

    expected_keys = {
        (case.case_id, trial)
        for case in manifest.scoring_cases
        for trial in range(1, manifest.protocol.trials_per_case + 1)
    }
    missing = sorted(expected_keys - set(observed))
    ordered = tuple(observed[key] for key in sorted(observed))
    positive = tuple(item for item in ordered if case_by_id[item.case_id].label == "positive")
    negative = tuple(item for item in ordered if case_by_id[item.case_id].label == "negative")
    generated = sum(item.generated_probes for item in ordered)
    valid = sum(item.valid_probes for item in ordered)
    discovered = sum(
        item.first_fixed_verified_rank is not None
        and item.first_fixed_verified_rank <= manifest.protocol.portfolio_k
        for item in positive
    )
    fixed_verified = sum(item.first_fixed_verified_rank is not None for item in positive)
    false_gates = sum(item.automatic_gate for item in negative)
    positive_gates = sum(item.automatic_gate for item in positive)
    true_gates = sum(item.automatic_gate_fixed_verified for item in positive)
    all_gates = positive_gates + false_gates
    reproduction_required = sum(item.reproduction_required for item in ordered)
    reproduction_successes = sum(item.reproduction_successes for item in ordered)
    if reproduction_required != fixed_verified:
        raise ScoringError(
            "reproduction coverage must be recorded for every credited positive discovery"
        )
    durations = [item.duration_seconds for item in ordered]
    generation_rejections = sum(item.generation_rejection_count for item in ordered)
    generation_infrastructure = sum(item.generation_infrastructure_count for item in ordered)
    runtime_invalid = sum(item.runtime_invalid_probe_count for item in ordered)
    verification_invalid = sum(item.verification_invalid_count for item in ordered)
    invalid_probes = generation_rejections + runtime_invalid
    evaluable_probes = generated - generation_infrastructure
    metrics: dict[str, int | float | None] = {
        "generated_probe_count": generated,
        "evaluable_probe_count": evaluable_probes,
        "valid_probe_count": valid,
        "probe_validity_rate": _ratio(valid, evaluable_probes),
        "positive_trial_count": len(positive),
        "positive_discovery_pass_at_k": _ratio(discovered, len(positive)),
        "positive_discovery_count": discovered,
        "positive_fixed_verified_count": fixed_verified,
        "negative_trial_count": len(negative),
        "false_gate_rate": _ratio(false_gates, len(negative)),
        "false_gate_count": false_gates,
        "true_gate_count": true_gates,
        "unverified_positive_gate_count": positive_gates - true_gates,
        "automatic_gate_precision": _ratio(true_gates, all_gates),
        "invalid_trial_count": sum(item.status == "invalid" for item in ordered),
        "invalid_trial_rate": _ratio(
            sum(item.status == "invalid" for item in ordered), len(ordered)
        ),
        "generation_rejection_count": generation_rejections,
        "generation_infrastructure_count": generation_infrastructure,
        "runtime_invalid_probe_count": runtime_invalid,
        "verification_invalid_count": verification_invalid,
        "invalid_probe_count": invalid_probes,
        "invalid_probe_rate": _ratio(invalid_probes, evaluable_probes),
        "insufficient_evidence_trial_count": sum(
            item.status == "insufficient_evidence" for item in ordered
        ),
        "insufficient_evidence_trial_rate": _ratio(
            sum(item.status == "insufficient_evidence" for item in ordered), len(ordered)
        ),
        "infrastructure_trial_count": sum(
            item.status == "infrastructure_error" for item in ordered
        ),
        "infrastructure_trial_rate": _ratio(
            sum(item.status == "infrastructure_error" for item in ordered), len(ordered)
        ),
        "infrastructure_error_count": sum(item.infrastructure_error_count for item in ordered),
        "latency_p50_seconds": _percentile(durations, 0.5),
        "latency_p95_seconds": _percentile(durations, 0.95),
        "reproduction_rate": _ratio(reproduction_successes, reproduction_required),
        "reproduction_required_count": reproduction_required,
        "reproduction_success_count": reproduction_successes,
    }
    confidence_intervals = {
        "probe_validity_rate": _wilson_interval(valid, evaluable_probes),
        "positive_discovery_pass_at_k": _wilson_interval(discovered, len(positive)),
        "false_gate_rate": _wilson_interval(false_gates, len(negative)),
        "automatic_gate_precision": _wilson_interval(true_gates, all_gates),
        "reproduction_rate": _wilson_interval(reproduction_successes, reproduction_required),
    }
    threshold_results = _threshold_results(metrics, manifest.protocol.thresholds)
    trial_matrix_complete = bool(manifest.scoring_cases) and not missing
    evaluation_complete = trial_matrix_complete and all(
        item.status == "complete" for item in ordered
    )
    performance_met = bool(threshold_results) and all(threshold_results.values())
    corpus_gate = manifest.promotion_gate.met
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "fingerprints": fingerprint.to_dict(),
        "protocol": {
            "trials_per_case": manifest.protocol.trials_per_case,
            "portfolio_k": manifest.protocol.portfolio_k,
            "repetitions_per_finding": manifest.protocol.repetitions_per_finding,
        },
        "corpus": {
            "smoke_cases": len(manifest.smoke_cases),
            "scoring_cases": len(manifest.scoring_cases),
            "positive_cases": sum(case.label == "positive" for case in manifest.scoring_cases),
            "negative_cases": sum(case.label == "negative" for case in manifest.scoring_cases),
            "pytest_cases": sum(case.framework == "pytest" for case in manifest.scoring_cases),
            "vitest_cases": sum(case.framework == "vitest" for case in manifest.scoring_cases),
            "expected_trials": len(expected_keys),
            "observed_trials": len(ordered),
            "missing_trials": [f"{case_id}#{trial}" for case_id, trial in missing],
        },
        "metrics": metrics,
        "confidence_intervals_95": confidence_intervals,
        "promotion_gate": {
            "target_scoring_cases": manifest.promotion_gate.target_scoring_cases,
            "target_positive_cases": manifest.promotion_gate.target_positive_cases,
            "target_negative_cases": manifest.promotion_gate.target_negative_cases,
            "target_pytest_cases": manifest.promotion_gate.target_pytest_cases,
            "target_vitest_cases": manifest.promotion_gate.target_vitest_cases,
            "registered_scoring_cases": manifest.promotion_gate.registered_scoring_cases,
            "registered_positive_cases": manifest.promotion_gate.registered_positive_cases,
            "registered_negative_cases": manifest.promotion_gate.registered_negative_cases,
            "registered_pytest_cases": manifest.promotion_gate.registered_pytest_cases,
            "registered_vitest_cases": manifest.promotion_gate.registered_vitest_cases,
            "corpus_composition_met": corpus_gate,
            "trial_matrix_complete": trial_matrix_complete,
            "evaluation_complete": evaluation_complete,
            "performance_thresholds": threshold_results,
            "performance_met": performance_met,
            "met": corpus_gate and evaluation_complete and performance_met,
        },
        "trials": [item.to_dict() for item in ordered],
    }


def load_observation_bundle(
    path: Path,
) -> tuple[EvaluationFingerprint, tuple[TrialObservation, ...]]:
    """Load the stable, rescorable JSON observation format."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError(f"could not read observation bundle {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ScoringError(f"observation bundle must use schema_version {RESULT_SCHEMA_VERSION}")
    if set(value) != {"schema_version", "fingerprints", "trials"}:
        raise ScoringError("observation bundle must contain fingerprints and trials only")
    raw_trials = value["trials"]
    if not isinstance(raw_trials, list):
        raise ScoringError("observation bundle trials must be a list")
    return (
        EvaluationFingerprint.from_dict(value["fingerprints"]),
        tuple(TrialObservation.from_dict(item) for item in raw_trials),
    )


def _materialize_repository(case: ScoringCase, target: Path) -> None:
    target.mkdir(parents=True)
    _copy_snapshot(case.base_root, target)
    _git(target, "init", "-b", "main")
    _git(target, "config", "user.email", "benchmark@evalyn.invalid")
    _git(target, "config", "user.name", "Evalyn Benchmark")
    _git(target, "add", "-A")
    _git(target, "commit", "-m", "benchmark base")
    _git(target, "checkout", "-b", "candidate")

    for child in tuple(target.iterdir()):
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    _copy_snapshot(case.candidate_root, target)
    _git(target, "add", "-A")
    _git(target, "commit", "-m", "benchmark candidate")


def _load_portfolio_summary(
    run_path: Path | None,
    adapter: str,
) -> _PortfolioSummary | None:
    if run_path is None:
        return None
    path = run_path / "analysis" / f"{adapter}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError(f"could not read workflow portfolio evidence: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("portfolio"), dict):
        raise ScoringError("workflow portfolio evidence has an invalid schema")
    portfolio = value["portfolio"]
    hypotheses = portfolio.get("hypotheses")
    attempts = portfolio.get("attempts")
    infrastructure_errors = portfolio.get("infrastructure_error_count")
    if not isinstance(hypotheses, list) or not isinstance(attempts, list):
        raise ScoringError("workflow portfolio evidence is missing hypotheses or attempts")
    if (
        not isinstance(infrastructure_errors, int)
        or isinstance(infrastructure_errors, bool)
        or infrastructure_errors < 0
    ):
        raise ScoringError("workflow portfolio has an invalid infrastructure error count")
    ranks: dict[str, int] = {}
    for index, hypothesis in enumerate(hypotheses, start=1):
        if not isinstance(hypothesis, dict) or not isinstance(hypothesis.get("hypothesis_id"), str):
            raise ScoringError("workflow portfolio contains an invalid hypothesis")
        hypothesis_id = hypothesis["hypothesis_id"]
        if hypothesis_id in ranks:
            raise ScoringError("workflow portfolio contains duplicate hypotheses")
        ranks[hypothesis_id] = index
    statuses: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("status") not in {"accepted", "rejected"}:
            raise ScoringError("workflow portfolio contains an invalid attempt")
        statuses.append(attempt["status"])
    if infrastructure_errors > statuses.count("rejected"):
        raise ScoringError("workflow portfolio infrastructure count exceeds rejected attempts")
    return _PortfolioSummary(
        generated=len(statuses),
        accepted=statuses.count("accepted"),
        rejected=statuses.count("rejected"),
        infrastructure_errors=infrastructure_errors,
        hypothesis_ranks=ranks,
    )


def _copy_snapshot(source: Path, target: Path) -> None:
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ScoringError(f"could not materialize benchmark repository: {detail}")
    return completed.stdout.strip()


def _verify_prompt_fingerprint(provider: RecordedProvider) -> None:
    allowed = {
        SYSTEM_PROMPT,
        PORTFOLIO_PLANNING_SYSTEM_PROMPT,
        PORTFOLIO_CANDIDATE_SYSTEM_PROMPT,
        PORTFOLIO_CORRECTION_SYSTEM_PROMPT,
    }
    if (
        not provider.system_prompts
        or provider.system_prompts[0] != PORTFOLIO_PLANNING_SYSTEM_PROMPT
        or any(prompt not in allowed for prompt in provider.system_prompts)
    ):
        raise ScoringError("recorded provider received an unexpected generation prompt sequence")


def _classify_check_error(
    error: BaseException,
    portfolio: _PortfolioSummary | None,
    provider: RecordedProvider,
) -> str:
    if portfolio is not None and portfolio.infrastructure_errors:
        return "infrastructure_error"
    if provider.system_prompts and not provider.transcript_complete:
        return "infrastructure_error"
    if portfolio is not None and portfolio.generated and not portfolio.accepted:
        return "invalid"
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, (ProbeGenerationError, AdapterError, EngineError)):
            return "invalid"
        current = current.__cause__ or current.__context__
    return "infrastructure_error"


def _transcript_digests(provider: RecordedProvider) -> tuple[str, str]:
    if not provider.system_prompts:
        return "", ""
    return provider.request_sha256, provider.response_transcript_sha256


def _threshold_results(
    metrics: Mapping[str, int | float | None],
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    mapping = {
        "probe_validity_rate_min": ("probe_validity_rate", "min"),
        "positive_discovery_pass_at_k_min": ("positive_discovery_pass_at_k", "min"),
        "false_gate_rate_max": ("false_gate_rate", "max"),
        "automatic_gate_precision_min": ("automatic_gate_precision", "min"),
        "reproduction_rate_min": ("reproduction_rate", "min"),
        "latency_p50_seconds_max": ("latency_p50_seconds", "max"),
        "latency_p95_seconds_max": ("latency_p95_seconds", "max"),
    }
    result: dict[str, bool] = {}
    for threshold_name, (metric_name, direction) in mapping.items():
        metric = metrics[metric_name]
        threshold = thresholds[threshold_name]
        result[threshold_name] = bool(
            metric is not None
            and (metric >= threshold if direction == "min" else metric <= threshold)
        )
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    if not total:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + (z * z / total)
    center = (proportion + (z * z / (2 * total))) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1 - proportion) / total) + (z * z / (4 * total * total)))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ScoringError(f"{label} must be a lowercase SHA-256 digest")
