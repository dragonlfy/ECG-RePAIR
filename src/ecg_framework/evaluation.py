"""Framework-level audit and paired-evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .config import AgentConfig
from .schema import AgentResult, ReportAction, TracePhase


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    issues: tuple[str, ...]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_result(
    result: AgentResult,
    *,
    config: AgentConfig | None = None,
    query_fold: int | None = None,
) -> AuditReport:
    """Check lifecycle, provenance, budgets, and cross-fold isolation."""

    issues = []
    config = config or AgentConfig()
    terminal_trace = bool(result.trace and result.trace[-1].phase is TracePhase.STOP)
    if not terminal_trace:
        issues.append("trajectory does not end with STOP")
    within_inspection_budget = result.inspections <= config.max_inspections
    if not within_inspection_budget:
        issues.append("inspection budget exceeded")
    within_action_budget = result.committed_actions <= config.max_committed_actions
    if not within_action_budget:
        issues.append("committed-action budget exceeded")
    evidence_ids = [row.evidence_id for row in result.evidence]
    unique_evidence_ids = len(evidence_ids) == len(set(evidence_ids))
    if not unique_evidence_ids:
        issues.append("duplicate evidence IDs")
    decision_evidence_consistent = all(
        decision.evidence is None or decision.evidence.claim_id == decision.claim_id
        for decision in result.decisions
    )
    if not decision_evidence_consistent:
        issues.append("decision is linked to evidence from another claim")
    committed = sum(
        decision.action
        in (ReportAction.ADD, ReportAction.REVISE, ReportAction.WITHHOLD)
        for decision in result.decisions
    )
    committed_count_consistent = committed == result.committed_actions
    if not committed_count_consistent:
        issues.append("committed-action count disagrees with decisions")
    fold_isolation = True
    if query_fold is not None:
        fold_isolation = all(
            decision.value is None or query_fold not in decision.value.memory_folds
            for decision in result.decisions
        )
        if not fold_isolation:
            issues.append("query fold appears in outcome memory folds")
    presentation_events = [
        event for event in result.trace if event.phase is TracePhase.PRESENT
    ]
    presentation_lock = all(
        event.details.get("diagnosis_locked") is True for event in presentation_events
    )
    if not presentation_lock:
        issues.append("presentation phase did not record a diagnosis lock")
    checks = {
        "terminal_trace": terminal_trace,
        "within_inspection_budget": within_inspection_budget,
        "within_action_budget": within_action_budget,
        "unique_evidence_ids": unique_evidence_ids,
        "decision_evidence_consistent": decision_evidence_consistent,
        "committed_count_consistent": committed_count_consistent,
        "fold_isolation": fold_isolation,
        "presentation_lock": presentation_lock,
    }
    return AuditReport(not issues, tuple(issues), checks)


def paired_summary(
    base: Mapping[str, float], agent: Mapping[str, float]
) -> dict[str, float | int]:
    """Summarize paired scores on an identical record denominator."""

    if set(base) != set(agent):
        missing_agent = len(set(base) - set(agent))
        missing_base = len(set(agent) - set(base))
        raise ValueError(
            "paired score keys differ: "
            f"missing_agent={missing_agent}, missing_base={missing_base}"
        )
    keys: Sequence[str] = tuple(sorted(base))
    differences = [float(agent[key]) - float(base[key]) for key in keys]
    return {
        "n": len(keys),
        "base": sum(float(base[key]) for key in keys) / len(keys),
        "agent": sum(float(agent[key]) for key in keys) / len(keys),
        "delta": sum(differences) / len(differences),
        "wins": sum(value > 0.0 for value in differences),
        "ties": sum(value == 0.0 for value in differences),
        "losses": sum(value < 0.0 for value in differences),
    }
