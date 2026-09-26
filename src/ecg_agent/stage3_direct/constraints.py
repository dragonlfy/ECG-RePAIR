"""Hard clinical constraints over claim-level report actions."""

from __future__ import annotations

from collections.abc import Iterable

EXCLUSIVE_GROUPS = (
    frozenset({"axis_left", "axis_right", "axis_extreme"}),
    frozenset({"pr_prolonged", "pr_short"}),
    frozenset({"qrs_prolonged", "qrs_intermediate"}),
    frozenset({"bradycardia", "tachycardia"}),
)

CONFLICTS = {
    claim_id: frozenset(group - {claim_id})
    for group in EXCLUSIVE_GROUPS
    for claim_id in group
}


def conflicting_claims(claim_id: str, active_claims: Iterable[str]) -> frozenset[str]:
    """Return active diagnoses that cannot coexist with ``claim_id``."""

    return frozenset(active_claims) & CONFLICTS.get(claim_id, frozenset())


def is_legal_add(claim_id: str, active_claims: Iterable[str]) -> bool:
    """Whether adding a positive claim preserves registry exclusivity."""

    return not conflicting_claims(claim_id, active_claims)
