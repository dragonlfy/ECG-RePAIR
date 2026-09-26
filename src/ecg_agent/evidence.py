"""Claim-evidence graph and a verifier separate from measurement tools."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .schema import Claim, ClaimStatus, Evidence, VerificationStatus


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _rv_pattern(value: Any) -> bool | None:
    if not isinstance(value, dict):
        return None
    v1, v2 = value.get("V1"), value.get("V2")
    if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
        return None
    return bool(v1 >= 0.1 and v1 - v2 >= 0.05)


PREDICATES: dict[str, Callable[[Any], bool | None]] = {
    "axis_left": lambda value: (
        None if _number(value) is None else _number(value) < -30.0
    ),
    "axis_right": lambda value: (
        None if _number(value) is None else _number(value) > 90.0
    ),
    "axis_extreme": lambda value: (
        None
        if _number(value) is None
        else (_number(value) <= -90.0 or _number(value) > 180.0)
    ),
    "pr_prolonged": lambda value: (
        None if _number(value) is None else _number(value) >= 200.0
    ),
    "pr_short": lambda value: (
        None if _number(value) is None else _number(value) < 120.0
    ),
    "qrs_prolonged": lambda value: (
        None if _number(value) is None else _number(value) >= 120.0
    ),
    "qrs_intermediate": lambda value: (
        None if _number(value) is None else 100.0 <= _number(value) < 120.0
    ),
    "bradycardia": lambda value: (
        None if _number(value) is None else _number(value) < 60.0
    ),
    "tachycardia": lambda value: (
        None if _number(value) is None else _number(value) > 100.0
    ),
    "rhythm_irregular": lambda value: (
        None if _number(value) is None else _number(value) >= 0.12
    ),
    "low_voltage": lambda value: value if isinstance(value, bool) else None,
    "poor_r_progression": lambda value: value if isinstance(value, bool) else None,
    "pathologic_q": lambda value: value if isinstance(value, bool) else None,
    "t_wave_inversion": lambda value: value if isinstance(value, bool) else None,
    "rv_infarction": _rv_pattern,
}


class EvidenceVerifier:
    """Translate an observation into clinical support/conflict/unknown."""

    reliability_floor = 0.55

    def verify(self, claim: Claim, evidence: Evidence) -> VerificationStatus:
        predicate = PREDICATES.get(claim.claim_id)
        result = predicate(evidence.value) if predicate else None
        if result is None or evidence.reliability < self.reliability_floor:
            evidence.relation = VerificationStatus.UNCERTAIN
        elif result:
            evidence.relation = VerificationStatus.SUPPORTED
        else:
            evidence.relation = VerificationStatus.CONTRADICTED
        return evidence.relation


class ClaimEvidenceGraph:
    """Small in-memory graph with explicit claim and evidence provenance."""

    def __init__(self, claims: list[Claim]) -> None:
        self.claims = {claim.claim_id: claim for claim in claims}
        self.evidence: dict[str, Evidence] = {}
        self.by_claim: dict[str, list[str]] = defaultdict(list)

    def add_evidence(self, evidence: Evidence) -> dict[str, Any]:
        if evidence.claim_id not in self.claims:
            raise KeyError(f"unknown claim: {evidence.claim_id}")
        before = self.snapshot(evidence.claim_id)
        self.evidence[evidence.evidence_id] = evidence
        if evidence.evidence_id not in self.by_claim[evidence.claim_id]:
            self.by_claim[evidence.claim_id].append(evidence.evidence_id)
        claim = self.claims[evidence.claim_id]
        claim.evidence_ids = list(self.by_claim[evidence.claim_id])
        self._update(claim)
        return {"before": before, "after": self.snapshot(evidence.claim_id)}

    def _update(self, claim: Claim) -> None:
        rows = [self.evidence[eid] for eid in self.by_claim[claim.claim_id]]
        support = sum(
            row.reliability
            for row in rows
            if row.relation is VerificationStatus.SUPPORTED
        )
        conflict = sum(
            row.reliability
            for row in rows
            if row.relation is VerificationStatus.CONTRADICTED
        )
        uncertain = sum(
            row.reliability
            for row in rows
            if row.relation is VerificationStatus.UNCERTAIN
        )
        total = support + conflict + uncertain
        claim.support_score = support / total if total else 0.0
        claim.contradiction_score = conflict / total if total else 0.0
        claim.uncertainty = 1.0 - max(claim.support_score, claim.contradiction_score)
        if not rows:
            claim.status = ClaimStatus.UNVERIFIED
        elif claim.support_score >= 0.60:
            claim.status = ClaimStatus.SUPPORTED
        elif claim.contradiction_score >= 0.60:
            claim.status = ClaimStatus.CONTRADICTED
        else:
            claim.status = ClaimStatus.UNCERTAIN

    def snapshot(self, claim_id: str) -> dict[str, Any]:
        claim = self.claims[claim_id]
        return {
            "status": claim.status.value,
            "support_score": claim.support_score,
            "contradiction_score": claim.contradiction_score,
            "uncertainty": claim.uncertainty,
            "evidence_ids": list(claim.evidence_ids),
            "finalized": claim.finalized,
        }
