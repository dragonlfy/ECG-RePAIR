"""Stable, model-agnostic data contracts for ECG-Agent Framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceRelation(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class ReportAction(str, Enum):
    KEEP = "keep"
    ADD = "add"
    REVISE = "revise_evidence"
    WITHHOLD = "withhold"
    ABSTAIN = "abstain"


class TracePhase(str, Enum):
    BASE_MODEL = "base_model"
    SCAN = "scan"
    PLAN = "plan"
    INSPECT = "inspect"
    VERIFY = "verify"
    VALUE = "value"
    COMMIT = "commit"
    PRESENT = "present"
    STOP = "stop"
    ERROR = "error"


@dataclass(frozen=True)
class ECGCase:
    """One ECG input; waveform/image storage is deliberately backend-neutral."""

    record_id: str
    signal: Any | None = None
    sampling_rate_hz: float | None = None
    lead_names: tuple[str, ...] = ()
    image: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Report:
    text: str
    labels: tuple[str, ...] = ()
    model_id: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    clinical_group: str
    expert: str
    aliases: tuple[str, ...] = ()
    legal_actions: tuple[ReportAction, ...] = (
        ReportAction.ADD,
        ReportAction.REVISE,
        ReportAction.ABSTAIN,
    )
    priority: float = 0.5
    description: str = ""


@dataclass(frozen=True)
class ClaimCandidate:
    claim_id: str
    asserted: bool
    source: str
    source_text: str | None = None
    priority: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    claim_id: str
    expert: str
    measurement: str
    value: Any
    unit: str | None = None
    leads: tuple[str, ...] = ()
    relation: EvidenceRelation = EvidenceRelation.UNCERTAIN
    reliability: float = 0.0
    backend_agreement: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValueEstimate:
    mean_return: float
    uncertainty: float
    safe_return: float
    policy_id: str
    memory_folds: tuple[int, ...] = ()
    query_fold: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    claim_id: str
    action: ReportAction
    reason: str
    evidence: Evidence | None = None
    value: ValueEstimate | None = None


@dataclass(frozen=True)
class RunContext:
    case: ECGCase
    report: Report
    step: int
    committed_claims: tuple[str, ...] = ()
    inspected_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceEvent:
    step: int
    phase: TracePhase
    component: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    record_id: str
    base_report: Report
    final_report: Report
    decisions: tuple[Decision, ...]
    evidence: tuple[Evidence, ...]
    trace: tuple[TraceEvent, ...]
    graph: dict[str, Any]
    inspections: int
    committed_actions: int
    stopped_reason: str
    framework_version: str

    @property
    def changed(self) -> bool:
        return self.base_report != self.final_report

    def as_dict(self) -> dict[str, Any]:
        value = jsonable(self)
        value["changed"] = self.changed
        return value


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value
