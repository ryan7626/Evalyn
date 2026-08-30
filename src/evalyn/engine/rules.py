"""Conservative deterministic semantic assessment rules."""

from __future__ import annotations

import re
from enum import Enum

from evalyn.engine.assessment import SemanticAssessment
from evalyn.engine.models import (
    ClassificationKind,
    ExecutionStatus,
    Finding,
    SemanticState,
)

_ADMISSIBLE = {
    "existing_test",
    "documentation",
    "api_schema",
    "type_contract",
}
_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_OPPOSITES = (
    ({"allow", "accept", "enable"}, {"reject", "deny", "disable"}),
    ({"add", "create"}, {"remove", "delete"}),
    ({"increase", "raise"}, {"decrease", "lower"}),
)
_TOKEN_NORMALIZATION = {
    "accepted": "accept",
    "accepting": "accept",
    "allowed": "allow",
    "allowing": "allow",
    "created": "create",
    "creating": "create",
    "deleted": "delete",
    "deleting": "delete",
    "denied": "deny",
    "denying": "deny",
    "disabled": "disable",
    "disabling": "disable",
    "enabled": "enable",
    "enabling": "enable",
    "increased": "increase",
    "increasing": "increase",
    "lowered": "lower",
    "lowering": "lower",
    "raised": "raise",
    "raising": "raise",
    "rejected": "reject",
    "rejecting": "reject",
    "removed": "remove",
    "removing": "remove",
}
_PRESERVATION = re.compile(
    r"\b(?:invariant|keep|must|preserve|remain(?:s|ed)?|unchanged|continue(?:s|d)?)\b",
    re.IGNORECASE,
)
_EXPLICIT_CHANGE = re.compile(
    r"(?:^\s*(?:change|breaking-change)\s*:|\b(?:instead|no longer)\b)",
    re.IGNORECASE,
)


class _IntentRelation(Enum):
    ABSENT = "absent"
    UNRELATED = "unrelated"
    AMBIGUOUS = "ambiguous"
    CONTRADICTED_INVARIANT = "contradicted_invariant"
    EXPLAINS_CHANGE = "explains_change"


class IntentOracleAssessor:
    """Gate only trusted oracles with an unambiguous intent relationship.

    Free-form topical overlap is not semantic evidence. A supplied intent can
    gate a finding only when a preservation statement contains the oracle
    verbatim. An explicit change description with opposite behavior marks an
    intended change. Other related text keeps the finding uncertain.
    """

    name = "deterministic-intent-oracle-v3"

    def __init__(self, intent: tuple[str, ...] = ()) -> None:
        self.intent = tuple(item.strip() for item in intent if item.strip())

    def assess(self, finding: Finding) -> SemanticAssessment:
        relation = self._intent_relation(finding)
        repository_oracle = self._repository_oracle_is_admissible(finding)
        user_invariant = self._user_invariant_is_admissible(finding)
        causal_evidence = self._causal_evidence_is_gateable(finding)
        if relation == _IntentRelation.CONTRADICTED_INVARIANT and user_invariant:
            return SemanticAssessment(
                state=SemanticState.UNCERTAIN_CHANGE,
                reason=(
                    "the user supplied a preservation invariant, but operator prose is "
                    "maintainer evidence rather than an automatic executable oracle"
                ),
                assessor=self.name,
                oracle_admissible=True,
                intent_match=None,
            )
        if relation == _IntentRelation.EXPLAINS_CHANGE and repository_oracle:
            return SemanticAssessment(
                state=SemanticState.INTENDED_CHANGE,
                reason=(
                    "stable differential result is explained by an explicit intended "
                    "change with opposite behavior"
                ),
                assessor=self.name,
                oracle_admissible=repository_oracle,
                intent_match=True,
            )
        if relation == _IntentRelation.AMBIGUOUS:
            return SemanticAssessment(
                state=SemanticState.UNCERTAIN_CHANGE,
                reason=(
                    "supplied intent is related to the finding but does not provide "
                    "reliable directional evidence"
                ),
                assessor=self.name,
                oracle_admissible=repository_oracle,
                intent_match=None,
            )
        if repository_oracle and not causal_evidence:
            return SemanticAssessment(
                state=SemanticState.UNCERTAIN_CHANGE,
                reason=(
                    "the repository oracle is grounded, but the finding lacks three-repeat "
                    "causal evidence bound to a verified change target"
                ),
                assessor=self.name,
                oracle_admissible=True,
                intent_match=False,
            )
        if repository_oracle and finding.probe_metadata.get("oracle_source_side") != "both":
            return SemanticAssessment(
                state=SemanticState.UNCERTAIN_CHANGE,
                reason=(
                    "the contract was not verified as unchanged across base and candidate; "
                    "candidate-authored or changed contracts remain advisory"
                ),
                assessor=self.name,
                oracle_admissible=True,
                intent_match=False,
            )
        if repository_oracle and finding.oracle_provenance != "existing_test":
            return SemanticAssessment(
                state=SemanticState.UNCERTAIN_CHANGE,
                reason=(
                    f"the grounded {finding.oracle_provenance} contract is useful for review, "
                    "but Evalyn has no trusted executable oracle-binding verifier for it"
                ),
                assessor=self.name,
                oracle_admissible=True,
                intent_match=False,
            )
        if repository_oracle and finding.oracle_provenance == "existing_test":
            return SemanticAssessment(
                state=SemanticState.LIKELY_REGRESSION,
                reason=(
                    "stable differential result exactly transplants an unchanged test assertion, "
                    "binds its graph target, and passes the target-revert intervention"
                ),
                assessor=self.name,
                oracle_admissible=True,
                intent_match=False,
            )
        return SemanticAssessment(
            state=SemanticState.UNCERTAIN_CHANGE,
            reason=(
                "stable change lacks a category-matched repository oracle or "
                "verbatim user-supplied preservation invariant"
            ),
            assessor=self.name,
            oracle_admissible=False,
            intent_match=False if relation == _IntentRelation.UNRELATED else None,
        )

    @staticmethod
    def _repository_oracle_is_admissible(finding: Finding) -> bool:
        if len(finding.oracle.strip()) < 10:
            return False
        return (
            finding.oracle_provenance in _ADMISSIBLE
            and finding.probe_metadata.get("oracle_grounding") == "repository_context"
            and finding.probe_metadata.get("oracle_grounding_category") == finding.oracle_provenance
            and bool(finding.probe_metadata.get("oracle_grounding_source"))
            and finding.probe_metadata.get("oracle_source_side") in {"base", "both"}
        )

    @staticmethod
    def _causal_evidence_is_gateable(finding: Finding) -> bool:
        if finding.classification.kind != ClassificationKind.STABLE_CHANGE:
            return False
        if len(finding.base_runs) < 3 or len(finding.candidate_runs) < 3:
            return False
        if any(record.result.status != ExecutionStatus.PASSED for record in finding.base_runs):
            return False
        failure_statuses = {ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT}
        if any(record.result.status not in failure_statuses for record in finding.candidate_runs):
            return False
        target_id = finding.probe_metadata.get("change_target_id", "")
        if re.fullmatch(r"target_[a-f0-9]{16}", target_id) is None:
            return False
        if not finding.probe_metadata.get("changed_file"):
            return False
        return (
            finding.evidence is not None
            and finding.evidence.environment_integrity
            and finding.evidence.target_binding_verified
            and finding.evidence.oracle_binding_verified
            and finding.evidence.target_intervention_verified
            and bool(finding.evidence.target_symbol)
        )

    def _user_invariant_is_admissible(self, finding: Finding) -> bool:
        if len(finding.oracle.strip()) < 10:
            return False
        return any(
            _PRESERVATION.search(statement) and _contains_verbatim(statement, finding.oracle)
            for statement in self.intent
        )

    def _intent_relation(self, finding: Finding) -> _IntentRelation:
        if not self.intent:
            return _IntentRelation.ABSENT
        related = False
        for statement in self.intent:
            if not _is_related(statement, finding):
                continue
            related = True
            if _PRESERVATION.search(statement) and _contains_verbatim(statement, finding.oracle):
                return _IntentRelation.CONTRADICTED_INVARIANT
            if _EXPLICIT_CHANGE.search(statement) and _has_opposite_behavior(statement, finding):
                return _IntentRelation.EXPLAINS_CHANGE
        return _IntentRelation.AMBIGUOUS if related else _IntentRelation.UNRELATED


def _is_related(intent: str, finding: Finding) -> bool:
    intent_tokens = _tokens(intent)
    finding_tokens = _tokens(finding.oracle)
    if len(intent_tokens) < 2 or len(finding_tokens) < 2:
        return False
    overlap = intent_tokens & finding_tokens
    denominator = min(len(intent_tokens), len(finding_tokens))
    return len(overlap) >= 2 and len(overlap) / denominator >= 0.5


def _has_opposite_behavior(intent: str, finding: Finding) -> bool:
    intent_tokens = _tokens(intent)
    finding_tokens = _tokens(finding.oracle)
    return any(
        (intent_tokens & left and finding_tokens & right)
        or (intent_tokens & right and finding_tokens & left)
        for left, right in _OPPOSITES
    )


def _contains_verbatim(container: str, excerpt: str) -> bool:
    normalized_excerpt = " ".join(excerpt.casefold().split())
    normalized_container = " ".join(container.casefold().split())
    return len(normalized_excerpt) >= 10 and normalized_excerpt in normalized_container


def _tokens(value: str) -> set[str]:
    return {
        _TOKEN_NORMALIZATION.get(token, token)
        for token in _TOKEN.findall(value.casefold())
        if token not in _STOP_WORDS and len(token) > 1
    }
