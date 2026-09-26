"""Stage-3 verifier with mutually exclusive clinical categories."""

from __future__ import annotations

from typing import Any

from ecg_agent.evidence import EvidenceVerifier
from ecg_agent.schema import Claim, Evidence, VerificationStatus


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


class ClinicalEvidenceVerifier(EvidenceVerifier):
    """Refine broad cache predicates into clinically exclusive axis bins."""

    def verify(self, claim: Claim, evidence: Evidence) -> VerificationStatus:
        axis = _numeric(evidence.value)
        if not claim.claim_id.startswith("axis_"):
            return super().verify(claim, evidence)
        if axis is None or evidence.reliability < self.reliability_floor:
            relation = VerificationStatus.UNCERTAIN
        elif claim.claim_id == "axis_left":
            relation = (
                VerificationStatus.SUPPORTED
                if -90.0 < axis < -30.0
                else VerificationStatus.CONTRADICTED
            )
        elif claim.claim_id == "axis_right":
            relation = (
                VerificationStatus.SUPPORTED
                if 90.0 < axis <= 180.0
                else VerificationStatus.CONTRADICTED
            )
        elif claim.claim_id == "axis_extreme":
            relation = (
                VerificationStatus.SUPPORTED
                if axis <= -90.0 or axis > 180.0
                else VerificationStatus.CONTRADICTED
            )
        else:
            relation = VerificationStatus.UNCERTAIN
        evidence.relation = relation
        return relation
