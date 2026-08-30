"""Immutable, serializable models for change-aware probe generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_IDENTIFIER = re.compile(r"^[a-z]+_[a-f0-9]{16}$")
_CHANGE_KINDS = frozenset({"added", "modified", "deleted", "renamed"})
_CONTEXT_SIDES = frozenset({"base", "candidate", "both"})
_CONTEXT_CATEGORIES = frozenset(
    {
        "implementation",
        "existing_test",
        "documentation",
        "api_schema",
        "type_contract",
        "manifest",
    }
)
_ORACLE_CATEGORIES = frozenset(
    {
        "existing_test",
        "documentation",
        "api_schema",
        "type_contract",
        "user_invariant",
    }
)
_HYPOTHESIS_CATEGORIES = frozenset(
    {
        "boundary",
        "error",
        "defaults_compatibility",
        "state_transition",
        "serialization_round_trip",
        "metamorphic",
    }
)
_ATTEMPT_STATUSES = frozenset({"accepted", "rejected"})


def _validate_identifier(value: str, prefix: str) -> None:
    if not _IDENTIFIER.fullmatch(value) or not value.startswith(f"{prefix}_"):
        raise ValueError(f"{prefix} identifier is invalid: {value!r}")


def _validate_relative_path(value: str, field_name: str) -> None:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise ValueError(f"{field_name} must be a safe repository-relative POSIX path")


@dataclass(frozen=True)
class ChangeTarget:
    """A model-independent code location affected by the supplied diff."""

    target_id: str
    path: str
    change_kind: str
    symbol: str = ""
    symbol_kind: str = "file"
    base_path: str = ""
    candidate_path: str = ""
    base_start_line: int | None = None
    base_end_line: int | None = None
    candidate_start_line: int | None = None
    candidate_end_line: int | None = None
    base_definition: str = ""
    candidate_definition: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.target_id, "target")
        _validate_relative_path(self.path, "path")
        if self.base_path:
            _validate_relative_path(self.base_path, "base_path")
        if self.candidate_path:
            _validate_relative_path(self.candidate_path, "candidate_path")
        if self.change_kind not in _CHANGE_KINDS:
            raise ValueError(f"unsupported change kind: {self.change_kind!r}")
        if not self.symbol_kind.strip():
            raise ValueError("symbol_kind cannot be empty")
        for start, end, side in (
            (self.base_start_line, self.base_end_line, "base"),
            (self.candidate_start_line, self.candidate_end_line, "candidate"),
        ):
            if (start is None) != (end is None):
                raise ValueError(f"{side} line range must provide both start and end")
            if start is not None and (start < 1 or end is None or end < start):
                raise ValueError(f"{side} line range is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "path": self.path,
            "change_kind": self.change_kind,
            "symbol": self.symbol,
            "symbol_kind": self.symbol_kind,
            "base_path": self.base_path,
            "candidate_path": self.candidate_path,
            "base_start_line": self.base_start_line,
            "base_end_line": self.base_end_line,
            "candidate_start_line": self.candidate_start_line,
            "candidate_end_line": self.candidate_end_line,
            "base_definition": self.base_definition,
            "candidate_definition": self.candidate_definition,
        }


@dataclass(frozen=True)
class ContextEvidence:
    """Ranked context with complete source-side and content-hash provenance."""

    evidence_id: str
    path: str
    side: str
    category: str
    content: str
    content_sha256: str
    score: int
    rank: int
    target_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.evidence_id, "evidence")
        _validate_relative_path(self.path, "path")
        if self.side not in _CONTEXT_SIDES:
            raise ValueError(f"unsupported context side: {self.side!r}")
        if self.category not in _CONTEXT_CATEGORIES:
            raise ValueError(f"unsupported context category: {self.category!r}")
        if not re.fullmatch(r"[a-f0-9]{64}", self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if self.rank < 1:
            raise ValueError("context rank is one-indexed")
        if tuple(sorted(set(self.target_ids))) != self.target_ids:
            raise ValueError("target_ids must be sorted and unique")
        for target_id in self.target_ids:
            _validate_identifier(target_id, "target")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("context reasons must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "path": self.path,
            "side": self.side,
            "category": self.category,
            "content": self.content,
            "content_sha256": self.content_sha256,
            "score": self.score,
            "rank": self.rank,
            "target_ids": list(self.target_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class OracleReference:
    """A trusted oracle identifier derived from repository or operator input."""

    oracle_id: str
    category: str
    excerpt: str
    excerpt_sha256: str
    source_path: str
    source_side: str
    evidence_id: str = ""
    target_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.oracle_id, "oracle")
        if self.category not in _ORACLE_CATEGORIES:
            raise ValueError(f"unsupported oracle category: {self.category!r}")
        if not self.excerpt.strip():
            raise ValueError("oracle excerpt cannot be empty")
        if not re.fullmatch(r"[a-f0-9]{64}", self.excerpt_sha256):
            raise ValueError("excerpt_sha256 must be a lowercase SHA-256 digest")
        if self.category == "user_invariant":
            if self.source_path != "stated_intent" or self.source_side != "operator":
                raise ValueError("user invariant provenance must be stated_intent/operator")
            if self.evidence_id:
                raise ValueError("user invariant cannot claim repository evidence")
        else:
            _validate_relative_path(self.source_path, "source_path")
            if self.source_side not in {"base", "both"}:
                raise ValueError("repository oracles must come from base-visible content")
            _validate_identifier(self.evidence_id, "evidence")
        if tuple(sorted(set(self.target_ids))) != self.target_ids:
            raise ValueError("target_ids must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_id": self.oracle_id,
            "category": self.category,
            "excerpt": self.excerpt,
            "excerpt_sha256": self.excerpt_sha256,
            "source_path": self.source_path,
            "source_side": self.source_side,
            "evidence_id": self.evidence_id,
            "target_ids": list(self.target_ids),
        }


@dataclass(frozen=True)
class ProbeHypothesis:
    """A bounded probe idea bound to deterministic target/oracle identifiers."""

    hypothesis_id: str
    title: str
    category: str
    target_id: str
    behavior: str
    oracle_id: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.hypothesis_id, "hypothesis")
        _validate_identifier(self.target_id, "target")
        if self.oracle_id:
            _validate_identifier(self.oracle_id, "oracle")
        if not self.title.strip() or not self.behavior.strip():
            raise ValueError("hypothesis title and behavior cannot be empty")
        if self.category not in _HYPOTHESIS_CATEGORIES:
            raise ValueError(f"unsupported hypothesis category: {self.category!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "category": self.category,
            "target_id": self.target_id,
            "behavior": self.behavior,
            "oracle_id": self.oracle_id,
        }


@dataclass(frozen=True)
class ProbeAttempt:
    """One independently generated probe candidate or bounded correction."""

    attempt_id: str
    hypothesis_id: str
    attempt_number: int
    status: str
    source: str = ""
    rejection_code: str = ""
    feedback: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.attempt_id, "attempt")
        _validate_identifier(self.hypothesis_id, "hypothesis")
        if self.attempt_number not in {1, 2}:
            raise ValueError("attempt_number must be 1 or 2")
        if self.status not in _ATTEMPT_STATUSES:
            raise ValueError(f"unsupported attempt status: {self.status!r}")
        if self.status == "accepted" and (not self.source.strip() or self.rejection_code):
            raise ValueError("accepted attempts require source and no rejection code")
        if self.status == "rejected" and not self.rejection_code:
            raise ValueError("rejected attempts require a rejection code")

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "hypothesis_id": self.hypothesis_id,
            "attempt_number": self.attempt_number,
            "status": self.status,
            "source": self.source,
            "rejection_code": self.rejection_code,
            "feedback": self.feedback,
        }


@dataclass(frozen=True)
class ChangeGraph:
    """Deterministic targets, ranked context, and verified oracle references."""

    targets: tuple[ChangeTarget, ...]
    context: tuple[ContextEvidence, ...]
    oracles: tuple[OracleReference, ...]
    changed_paths: tuple[str, ...]
    diff_sha256: str

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("change graph requires at least one target")
        if tuple(sorted(set(self.changed_paths))) != self.changed_paths:
            raise ValueError("changed_paths must be sorted and unique")
        for path in self.changed_paths:
            _validate_relative_path(path, "changed_path")
        if not re.fullmatch(r"[a-f0-9]{64}", self.diff_sha256):
            raise ValueError("diff_sha256 must be a lowercase SHA-256 digest")
        target_ids = [target.target_id for target in self.targets]
        evidence_ids = [item.evidence_id for item in self.context]
        oracle_ids = [oracle.oracle_id for oracle in self.oracles]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("change target identifiers must be unique")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("context evidence identifiers must be unique")
        if len(oracle_ids) != len(set(oracle_ids)):
            raise ValueError("oracle identifiers must be unique")
        if tuple(item.rank for item in self.context) != tuple(range(1, len(self.context) + 1)):
            raise ValueError("context must be stored in contiguous rank order")

    def target(self, target_id: str) -> ChangeTarget | None:
        return next((target for target in self.targets if target.target_id == target_id), None)

    def oracle(self, oracle_id: str) -> OracleReference | None:
        return next((oracle for oracle in self.oracles if oracle.oracle_id == oracle_id), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "targets": [target.to_dict() for target in self.targets],
            "context": [item.to_dict() for item in self.context],
            "oracles": [oracle.to_dict() for oracle in self.oracles],
            "changed_paths": list(self.changed_paths),
            "diff_sha256": self.diff_sha256,
        }
