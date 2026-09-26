"""Offline invariants for Stage-5 reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .renderer import RenderedReport
from .schema import ClinicalEvidenceGraph


def audit_rendered_report(
    rendered: RenderedReport,
    graph: ClinicalEvidenceGraph,
    instrument: Mapping[str, Any],
    *,
    require_ledger: bool = True,
    require_old_correction_removed: bool = True,
    require_selected_claims: bool = True,
    evidence_heading: str = "Measurement-grounded evidence used by the ECG Agent",
    allow_empty_evidence: bool = False,
    allow_invalid_instrument: bool = False,
) -> dict[str, Any]:
    """Check diagnosis locking, provenance, and dynamic-card coverage."""

    source_paths = [
        measurement.path for card in graph.cards for measurement in card.measurements
    ]
    selected_rendered = {
        card.claim_id for card in graph.cards if card.claim_id is not None
    }
    missing_claims = sorted(set(graph.selected_claims) - selected_rendered)
    checks = {
        "diagnosis_locked": rendered.diagnosis_locked,
        "has_expected_evidence_section": (
            allow_empty_evidence and not rendered.evidence_card_ids
        )
        or (
            "<ecg_evidence>" in rendered.report and "</ecg_evidence>" in rendered.report
            if require_ledger
            else evidence_heading in rendered.report
        ),
        "old_correction_policy_satisfied": (
            "Verified correction" not in rendered.report
            if require_old_correction_removed
            else True
        ),
        "all_cards_have_measurement_provenance": all(
            card.measurements for card in graph.cards
        ),
        "selected_claims_rendered": not require_selected_claims or not missing_claims,
        "instrument_valid": allow_invalid_instrument
        or instrument.get("instrument_valid") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "n_cards": len(graph.cards),
        "source_paths": source_paths,
        "missing_selected_claims": missing_claims,
    }
