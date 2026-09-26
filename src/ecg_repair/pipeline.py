"""File-independent repair entry point using the existing research components.

Only historical episodes carry rewards. Query reports and instruments are passed
through separate arguments and never supply reference text to the agent.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ecg_agent.registry import CLAIM_REGISTRY
from ecg_agent.schema import Evidence, VerificationStatus
from ecg_agent.stage3_direct.evidence_advantage import (
    EvidenceActionRow, EvidenceAdvantageConfig, EvidenceConditionedAdvantagePolicy,
)
from ecg_agent.stage3_direct.schema import DirectClaimAction
from ecg_framework import AgentConfig, ECGCase, ReportAction, audit_result
from ecg_framework.adapters import build_stage5_framework
from ecg_framework.adapters.common import FrozenMappingBaseModel

DEFAULT_CLAIMS = (
    "axis_left", "axis_right", "pr_prolonged", "low_voltage",
    "bradycardia", "poor_r_progression",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(row)
    return rows


def index_records(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = str(row["record_id"])
        if key in result:
            raise ValueError(f"duplicate record_id: {key}")
        result[key] = row
    return result


def load_memory(path: str | Path) -> tuple[EvidenceActionRow, ...]:
    """Load user-owned historical action outcomes; no bundled patient episodes."""
    result = []
    for row in read_jsonl(path):
        claim_id = str(row["claim_id"])
        if claim_id not in CLAIM_REGISTRY:
            raise ValueError(f"unknown memory claim: {claim_id}")
        reward = float(row["reward"])
        if not math.isfinite(reward):
            raise ValueError("memory reward must be finite")
        ev = dict(row["evidence"])
        ev["relation"] = VerificationStatus(ev["relation"])
        ev["lead_set"] = tuple(ev.get("lead_set", ()))
        evidence = Evidence(**ev)
        if evidence.claim_id != claim_id:
            raise ValueError("memory claim and evidence claim do not match")
        result.append(EvidenceActionRow(
            record_id=str(row["record_id"]), fold=int(row["fold"]),
            report="", labels=tuple(row.get("labels", ())),
            selected_claims=frozenset(row.get("selected_claims", ())),
            claim_id=claim_id, action=DirectClaimAction(row["action"]),
            evidence=evidence, reward=reward,
        ))
    return tuple(result)


def build_repair_agent(
    reports: Mapping[str, Mapping[str, Any]],
    instruments: Mapping[str, Mapping[str, Any]],
    memory: Sequence[EvidenceActionRow],
    *, query_fold: int, alpha: float = 100.0, margin: float = 2.0,
    candidate_claims: Sequence[str] = DEFAULT_CLAIMS,
    max_inspections: int = 6, base_model_id: str = "base-ecg-model",
):
    """Fit on other folds, then build a diagnosis-locked repair runtime.

    Hyperparameters are explicit fixed settings here. This entry point does not
    reproduce the experiment's nested model selection or cached core/expansion
    orchestration. It exposes the reusable verification/repair components.
    """
    if alpha <= 0 or margin < 0 or not all(map(math.isfinite, (alpha, margin))):
        raise ValueError("alpha must be positive and margin nonnegative and finite")
    if not candidate_claims or set(candidate_claims) - CLAIM_REGISTRY.keys():
        raise ValueError("candidate_claims must contain registered findings")
    usable = [row for row in memory if row.fold != query_fold]
    query_ids = {key for key, row in reports.items() if int(row["fold"]) == query_fold}
    if any(row.record_id in query_ids for row in usable):
        raise ValueError("a query record occurs in the usable historical memory")
    policies = {}
    if usable:
        policies[query_fold] = EvidenceConditionedAdvantagePolicy.fit(
            usable, EvidenceAdvantageConfig(alpha=alpha, margin=margin),
        )
    config = AgentConfig(
        framework_id="ecg-repair", max_inspections=max_inspections,
        max_committed_actions=1, min_safe_return=0.0,
    )
    agent = build_stage5_framework(
        base_rows=reports, instruments=instruments, policies=policies,
        candidate_claims=candidate_claims, config=config,
    )
    agent.base_model = FrozenMappingBaseModel(reports, model_id=base_model_id)
    # Historical adapters expose WITHHOLD for older studies. The public default
    # uses ADD / REVISE and preserves the current report when no edit qualifies.
    for spec in tuple(agent.registry.claims.values()):
        from dataclasses import replace
        agent.registry.register_claim(replace(
            spec, legal_actions=(ReportAction.ADD, ReportAction.REVISE, ReportAction.ABSTAIN),
        ), replace=True)
    return agent


def repair_reports(
    reports: Sequence[Mapping[str, Any]], instruments: Sequence[Mapping[str, Any]],
    memory: Sequence[EvidenceActionRow] = (), **settings: Any,
) -> list[dict[str, Any]]:
    """Repair precomputed reports, producing a complete decision/evidence trace."""
    indexed = index_records(reports)
    measured = index_records(instruments)
    absent = indexed.keys() - measured.keys()
    if absent:
        raise ValueError(f"missing instruments for {len(absent)} query records")
    output = []
    agents = {}
    for key, row in indexed.items():
        fold = int(row["fold"])
        if fold not in agents:
            agents[fold] = build_repair_agent(
                indexed, measured, memory, query_fold=fold, **settings,
            )
        agent = agents[fold]
        result = agent.run(ECGCase(key, metadata={"fold": fold}))
        audit = audit_result(result, config=agent.config, query_fold=fold)
        output.append({**result.as_dict(), "release_audit": audit.as_dict()})
    return output
