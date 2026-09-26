"""Claim-evidence graph shared across framework components and audit tools."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .schema import ClaimSpec, Decision, Evidence, EvidenceRelation, jsonable


class EvidenceGraph:
    def __init__(self, claims: Mapping[str, ClaimSpec]) -> None:
        self.claims = dict(claims)
        self.evidence: dict[str, Evidence] = {}
        self.by_claim: dict[str, list[str]] = defaultdict(list)
        self.decisions: dict[str, Decision] = {}

    def add_evidence(self, evidence: Evidence) -> dict[str, object]:
        if evidence.claim_id not in self.claims:
            raise KeyError(f"evidence references unknown claim: {evidence.claim_id}")
        before = self.claim_snapshot(evidence.claim_id)
        existing = self.evidence.get(evidence.evidence_id)
        if existing is not None and existing != evidence:
            raise ValueError(f"evidence ID collision: {evidence.evidence_id}")
        self.evidence[evidence.evidence_id] = evidence
        if evidence.evidence_id not in self.by_claim[evidence.claim_id]:
            self.by_claim[evidence.claim_id].append(evidence.evidence_id)
        return {
            "before": before,
            "after": self.claim_snapshot(evidence.claim_id),
        }

    def set_decision(self, decision: Decision) -> None:
        if decision.claim_id not in self.claims:
            raise KeyError(f"decision references unknown claim: {decision.claim_id}")
        if decision.evidence and decision.evidence.claim_id != decision.claim_id:
            raise ValueError("decision evidence belongs to a different claim")
        self.decisions[decision.claim_id] = decision

    def claim_snapshot(self, claim_id: str) -> dict[str, object]:
        rows = [self.evidence[key] for key in self.by_claim.get(claim_id, ())]
        weights = {relation: 0.0 for relation in EvidenceRelation}
        for row in rows:
            weights[row.relation] += max(0.0, float(row.reliability))
        total = sum(weights.values())
        normalized = {
            relation.value: (weights[relation] / total if total else 0.0)
            for relation in EvidenceRelation
        }
        if not rows:
            status = "unverified"
        elif total == 0.0:
            status = EvidenceRelation.UNCERTAIN.value
        elif (
            weights[EvidenceRelation.SUPPORTED] > weights[EvidenceRelation.CONTRADICTED]
            and normalized[EvidenceRelation.SUPPORTED.value] >= 0.60
        ):
            status = EvidenceRelation.SUPPORTED.value
        elif (
            weights[EvidenceRelation.CONTRADICTED] > weights[EvidenceRelation.SUPPORTED]
            and normalized[EvidenceRelation.CONTRADICTED.value] >= 0.60
        ):
            status = EvidenceRelation.CONTRADICTED.value
        else:
            status = EvidenceRelation.UNCERTAIN.value
        return {
            "claim_id": claim_id,
            "status": status,
            "relation_weights": normalized,
            "evidence_ids": list(self.by_claim.get(claim_id, ())),
            "decision": jsonable(self.decisions.get(claim_id)),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "claims": {
                claim_id: self.claim_snapshot(claim_id)
                for claim_id in sorted(self.claims)
            },
            "evidence_count": len(self.evidence),
            "decision_count": len(self.decisions),
        }
