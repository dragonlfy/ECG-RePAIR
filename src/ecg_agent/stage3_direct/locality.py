"""Locality audits for deterministic ECG report actions."""

from __future__ import annotations

import re

NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")

MEASUREMENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "axis": re.compile(r"\b(?:qrs\s+|frontal(?:\s+plane)?\s+|electrical\s+)?axis\b"),
    "pr": re.compile(r"\bpr\s+(?:interval|duration)\b"),
    "qrs": re.compile(r"\bqrs(?:\s+(?:interval|duration|complex))?\b"),
    "rate": re.compile(r"\b(?:heart|ventricular)\s+rate\b|\bbpm\b"),
    "rr": re.compile(r"\brr\s+(?:interval|variability|cv)\b|\br-r\s+interval\b"),
    "voltage": re.compile(r"\b(?:qrs\s+)?(?:voltage|amplitude)s?\b"),
    "t_wave": re.compile(r"\bt[- ]?waves?\b"),
    "st": re.compile(r"\bst[- ]?segments?\b"),
}


def _target_measurement_groups(selected_claims: frozenset[str]) -> set[str]:
    groups: set[str] = set()
    if any(claim_id.startswith("pr_") for claim_id in selected_claims):
        groups.add("pr")
    if any(claim_id.startswith("axis_") for claim_id in selected_claims):
        groups.add("axis")
    if "low_voltage" in selected_claims:
        groups.add("voltage")
    if any(claim_id.startswith("qrs_") for claim_id in selected_claims):
        groups.add("qrs")
    if selected_claims & {"bradycardia", "tachycardia"}:
        groups.add("rate")
    if "rhythm_irregular" in selected_claims:
        groups.add("rr")
    if "t_wave_inversion" in selected_claims:
        groups.add("t_wave")
    if "rv_infarction" in selected_claims:
        groups.add("st")
    return groups


def target_numeric_correction_allowed(
    report: str, token: str, selected_claims: frozenset[str]
) -> bool:
    """Allow a number to change only inside a sentence targeted by an edit."""

    target_groups = _target_measurement_groups(selected_claims)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", report.lower()):
        if token not in NUMBER.findall(sentence):
            continue
        number_positions = [match.start() for match in NUMBER.finditer(sentence) if match.group() == token]
        measurements = [
            (group, match.start())
            for group, pattern in MEASUREMENT_PATTERNS.items()
            for match in pattern.finditer(sentence)
        ]
        for number_position in number_positions:
            if not measurements:
                continue
            preceding = [
                (group, position)
                for group, position in measurements
                if position <= number_position
            ]
            candidates = preceding or measurements
            nearest_distance = min(
                abs(position - number_position) for _, position in candidates
            )
            nearest_groups = {
                group
                for group, position in candidates
                if abs(position - number_position) == nearest_distance
            }
            if nearest_groups & target_groups:
                return True
    return False


def audit_numeric_locality(
    base_report: str, revised_report: str, selected_claims: frozenset[str]
) -> dict[str, object]:
    """Return target corrections and any deleted non-target numeric evidence."""

    missing = set(NUMBER.findall(base_report)) - set(NUMBER.findall(revised_report))
    target = sorted(
        token
        for token in missing
        if target_numeric_correction_allowed(base_report, token, selected_claims)
    )
    unprotected = sorted(missing - set(target))
    return {
        "passed": not unprotected,
        "target_numeric_corrections": target,
        "unprotected_missing_numbers": unprotected,
    }
