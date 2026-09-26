"""Typed state carried through an ECG Agent trajectory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ClaimStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class ActionType(str, Enum):
    INSPECT = "inspect"
    CROSS_CHECK = "cross_check"
    RETRIEVE_OUTCOME = "retrieve_outcome"
    RETRIEVE_STRATEGY = "retrieve_strategy"
    RETRIEVE_CASE = "retrieve_case"
    FINALIZE_CLAIM = "finalize_claim"
    STOP = "stop"


class ClaimAction(str, Enum):
    KEEP = "keep"
    WITHHOLD = "withhold"
    ADD = "add"
    ABSTAIN = "abstain"


@dataclass
class Claim:
    claim_id: str
    protocol_step: str
    source: str
    r1_present: bool
    r1_value: bool | None
    r1_text: str | None = None
    present_in_behavior_report: bool = False
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    support_score: float = 0.0
    contradiction_score: float = 0.0
    uncertainty: float = 1.0
    evidence_ids: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    finalized: bool = False


@dataclass
class Evidence:
    evidence_id: str
    claim_id: str
    expert: str
    measurement: str
    value: Any
    unit: str | None
    lead_set: tuple[str, ...] = ()
    start_sample: int | None = None
    end_sample: int | None = None
    relation: VerificationStatus = VerificationStatus.UNCERTAIN
    reliability: float = 0.0
    backend_agreement: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    claim_id: str | None = None
    expert: str | None = None
    leads: tuple[str, ...] = ()
    window: tuple[int, int] | None = None
    cost: float = 0.0
    planner_score: float = 0.0


@dataclass
class ClaimDecision:
    claim_id: str
    action: ClaimAction
    confidence: float
    reason: str
    evidence_ids: tuple[str, ...] = ()
    safe_outcome_advantage: float | None = None


@dataclass
class AgentState:
    record_id: str
    claims: dict[str, Claim]
    remaining_budget: int
    tool_calls: int = 0
    step: int = 0
    current_claim_id: str | None = None
    outcome_estimates: dict[str, Any] = field(default_factory=dict)
    strategy_results: dict[str, Any] = field(default_factory=dict)
    case_results: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, ClaimDecision] = field(default_factory=dict)
    stopped: bool = False


def jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and enums to JSON-native values."""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
