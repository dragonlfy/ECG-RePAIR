"""Typed records for the direct ECG Agent environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DirectActionType(str, Enum):
    PROPOSE = "propose_claims"
    INSPECT = "inspect"
    CROSS_CHECK = "cross_check"
    RETRIEVE_Q = "retrieve_action_value"
    COMMIT_ADD = "commit_add"
    COMMIT_REVISE_EVIDENCE = "commit_revise_evidence"
    COMMIT_WITHHOLD = "commit_withhold"
    COMMIT_KEEP = "commit_keep"
    ABSTAIN = "abstain"
    STOP = "stop"


class DirectClaimAction(str, Enum):
    ADD = "add"
    REVISE_EVIDENCE = "revise_evidence"
    WITHHOLD = "withhold"
    KEEP = "keep"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Proposal:
    claim_id: str
    label: str
    evidence_text: str
    measurement_text: str


@dataclass(frozen=True)
class ReportVariant:
    selected_claims: frozenset[str]
    report: str
    labels: tuple[str, ...]
    diagnosis_score: float
    reward_source: str


@dataclass
class DirectEpisode:
    record_id: str
    fold: int
    base_report: str
    base_labels: tuple[str, ...]
    proposals: tuple[Proposal, ...]
    variants: dict[frozenset[str], ReportVariant]

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        return tuple(proposal.claim_id for proposal in self.proposals)

    @property
    def base_score(self) -> float:
        return self.variants[frozenset()].diagnosis_score


@dataclass(frozen=True)
class ReplayTransition:
    record_id: str
    fold: int
    selected_claims: frozenset[str]
    action_claim: str
    next_claims: frozenset[str]
    reward: float
    reward_observed: bool = True
    edit_available: bool = True

    @property
    def key(self) -> tuple[str, frozenset[str], str]:
        return (self.record_id, self.selected_claims, self.action_claim)


@dataclass(frozen=True)
class ActionValue:
    record_id: str
    selected_claims: frozenset[str]
    action_claim: str
    mean_return: float
    uncertainty: float
    safe_return: float
    query_fold: int
    memory_folds: tuple[int, ...]
    model: str

    def json(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_claims"] = sorted(self.selected_claims)
        return value


@dataclass
class DirectAgentState:
    record_id: str
    selected_claims: frozenset[str] = frozenset()
    rejected_claims: set[str] = field(default_factory=set)
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    expert_calls: int = 0
    memory_queries: int = 0
    remaining_budget: int = 4
    stopped: bool = False
