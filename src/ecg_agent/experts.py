"""Deterministic ECG evidence experts backed by the frozen tool cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from .registry import ClaimSpec
from .schema import Evidence


def _agreement(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _reliability(
    row: Mapping[str, Any], agreed: Any, *, measurable: bool = True
) -> float:
    if not measurable or not row.get("instrument_valid", False):
        return 0.0
    global_agreement = float(row.get("oracle_agreement_rate") or 0.0)
    score = 0.55 + 0.45 * max(0.0, min(1.0, global_agreement))
    if agreed is False:
        score *= 0.45
    return float(max(0.0, min(1.0, score)))


class EvidenceExpert(ABC):
    name: str

    def can_handle(self, spec: ClaimSpec) -> bool:
        return self.name in spec.allowed_experts

    @abstractmethod
    def inspect(
        self,
        record_id: str,
        claim_id: str,
        instrument: Mapping[str, Any],
        *,
        cross_check: bool = False,
        leads: tuple[str, ...] = (),
    ) -> Evidence:
        raise NotImplementedError

    def _evidence(
        self,
        record_id: str,
        claim_id: str,
        measurement: str,
        value: Any,
        unit: str | None,
        instrument: Mapping[str, Any],
        *,
        agreed: Any,
        cross_check: bool,
        leads: tuple[str, ...] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> Evidence:
        suffix = "cross" if cross_check else "primary"
        measurable = value is not None
        return Evidence(
            evidence_id=f"{record_id}:{claim_id}:{self.name}:{suffix}",
            claim_id=claim_id,
            expert=self.name,
            measurement=measurement,
            value=value,
            unit=unit,
            lead_set=leads,
            reliability=_reliability(instrument, agreed, measurable=measurable),
            backend_agreement=_agreement(agreed),
            provenance={
                "source": instrument.get("source", "provided-instrument"),
                "oracle_backends": dict(instrument.get("oracle_backends") or {}),
                "mode": suffix,
                **dict(provenance or {}),
            },
        )


class QualityExpert(EvidenceExpert):
    name = "quality"

    def inspect(self, record_id, claim_id, instrument, *, cross_check=False, leads=()):
        value = float(instrument.get("oracle_agreement_rate") or 0.0)
        return self._evidence(
            record_id,
            claim_id,
            "tool_reliability",
            value,
            None,
            instrument,
            agreed=instrument.get("instrument_valid", False),
            cross_check=cross_check,
            leads=leads,
            provenance={
                "instrument_valid": bool(instrument.get("instrument_valid", False))
            },
        )


class IntervalExpert(EvidenceExpert):
    name = "interval"
    KEYS: ClassVar[dict[str, str]] = {
        "pr_prolonged": "pr_ms",
        "pr_short": "pr_ms",
        "qrs_prolonged": "qrs_ms",
        "qrs_intermediate": "qrs_ms",
    }

    def inspect(self, record_id, claim_id, instrument, *, cross_check=False, leads=()):
        key = self.KEYS.get(claim_id, "pr_ms")
        block = dict((instrument.get("scalar") or {}).get(key) or {})
        morphology = dict(instrument.get("morphology") or {})
        p_wave = dict(instrument.get("p_wave") or {})
        return self._evidence(
            record_id,
            claim_id,
            key,
            block.get("value"),
            "ms",
            instrument,
            agreed=block.get("agree"),
            cross_check=cross_check,
            leads=leads,
            provenance={
                "spread": block.get("spread"),
                "cross_check_scope": "backend",
                # PR is meaningful only when a discrete atrial deflection can be
                # measured.  Keeping this separate prevents a noisy baseline from
                # looking like a highly confident conduction abnormality.
                "p_wave_measurable": morphology.get("p_wave_measurable"),
                "p_duration_ms": morphology.get("p_duration_ms"),
                "p_wave_score": p_wave.get("value"),
                "p_wave_agree": p_wave.get("agree"),
            },
        )


class RhythmExpert(EvidenceExpert):
    name = "rhythm"

    def inspect(self, record_id, claim_id, instrument, *, cross_check=False, leads=()):
        if claim_id == "rhythm_irregular":
            block = dict(instrument.get("rr_cv") or {})
            measurement, unit = "rr_cv", None
        else:
            block = dict((instrument.get("scalar") or {}).get("heart_rate_bpm") or {})
            measurement, unit = "heart_rate_bpm", "bpm"
        p_wave = dict(instrument.get("p_wave") or {})
        morphology = dict(instrument.get("morphology") or {})
        heart_rate = dict((instrument.get("scalar") or {}).get("heart_rate_bpm") or {})
        provenance = {
            "backend_values": block.get("backend_values"),
            "spread": block.get("spread"),
            "heart_rate_bpm": heart_rate.get("value"),
            "p_wave_score": p_wave.get("value"),
            "p_wave_agree": p_wave.get("agree"),
            "p_wave_measurable": morphology.get("p_wave_measurable"),
        }
        return self._evidence(
            record_id,
            claim_id,
            measurement,
            block.get("value"),
            unit,
            instrument,
            agreed=block.get("agree"),
            cross_check=cross_check,
            leads=leads,
            provenance=provenance,
        )


class AxisExpert(EvidenceExpert):
    name = "axis"

    def inspect(self, record_id, claim_id, instrument, *, cross_check=False, leads=()):
        block = dict(instrument.get("frontal_axis") or {})
        provenance = {
            "category": block.get("category"),
            "method": block.get("method"),
            "spread_degrees": block.get("spread_degrees"),
            "n_backends": block.get("n_backends"),
            "beats_by_backend": block.get("beats_by_backend"),
        }
        if cross_check:
            provenance["backend_values_degrees"] = block.get("backend_values_degrees")
        return self._evidence(
            record_id,
            claim_id,
            "qrs_axis",
            block.get("value_degrees"),
            "degrees",
            instrument,
            agreed=block.get("agree"),
            cross_check=cross_check,
            leads=leads or ("I", "aVF"),
            provenance=provenance,
        )


class MorphologyExpert(EvidenceExpert):
    name = "morphology"
    KEYS: ClassVar[dict[str, str]] = {
        "low_voltage": "low_voltage",
        "poor_r_progression": "poor_r_progression",
        "pathologic_q": "pathologic_q",
    }

    def inspect(self, record_id, claim_id, instrument, *, cross_check=False, leads=()):
        block = dict(instrument.get("morphology") or {})
        if claim_id in self.KEYS:
            legacy_key = self.KEYS[claim_id]
            strict_key = f"{legacy_key}_strict"
            key = strict_key if strict_key in block else legacy_key
            value = block.get(key)
            agreed = block.get(f"{key}_agree")
            provenance = {
                "territory": block.get(f"{key}_territory")
                or block.get(f"{legacy_key}_territory"),
                "n_backends": block.get("n_backends"),
                "criterion_profile": block.get("criterion_profile")
                if key == strict_key
                else "legacy_textbook_v1",
            }
            lead_set = tuple(leads)
            if claim_id == "low_voltage":
                limb_max = block.get("max_limb_pp_mv")
                precordial_max = block.get("max_precordial_pp_mv")
                limb_support = isinstance(limb_max, (int, float)) and limb_max < 0.5
                precordial_support = (
                    isinstance(precordial_max, (int, float))
                    and precordial_max < 1.0
                )
                provenance.update(
                    {
                        "min_limb_pp_mv": block.get("min_limb_pp_mv"),
                        "max_limb_pp_mv": limb_max,
                        "max_precordial_pp_mv": precordial_max,
                        "limb_criterion_met": limb_support,
                        "precordial_criterion_met": precordial_support,
                        "criterion_basis": (
                            "limb_and_precordial"
                            if limb_support and precordial_support
                            else "limb"
                            if limb_support
                            else "precordial"
                            if precordial_support
                            else "neither"
                        ),
                    }
                )
                if not lead_set:
                    lead_set = (
                        ("I", "II", "III", "aVR", "aVL", "aVF")
                        if limb_support
                        else ("V1", "V2", "V3", "V4", "V5", "V6")
                        if precordial_support
                        else ()
                    )
            elif claim_id == "poor_r_progression":
                provenance["r_v3_mv"] = block.get("r_v3_mv")
                lead_set = lead_set or ("V1", "V2", "V3", "V4")
            elif claim_id == "pathologic_q":
                provenance.update(
                    {
                        "pathologic_q_n_leads": block.get(f"{key}_n_leads")
                        or block.get("pathologic_q_n_leads"),
                        "pathologic_q_territory": block.get(
                            f"{key}_territory"
                        ),
                    }
                )
                lead_set = lead_set or tuple(block.get(f"{key}_leads") or ())
            if cross_check:
                provenance["backend_findings"] = block.get("backend_findings")
        else:
            voltage = dict(instrument.get("voltage") or {})
            key = "lvh_by_voltage"
            value = voltage.get(key)
            agreed = voltage.get("agree")
            provenance = {
                "sokolow_lyon_mv": voltage.get("sokolow_lyon_mv"),
                "r_avl_mv": voltage.get("r_avl_mv"),
                "r_lead_i_mv": voltage.get("r_lead_i_mv"),
                "spread_mv": voltage.get("spread_mv"),
                "n_backends": voltage.get("n_backends"),
            }
            lead_set = tuple(leads) or ("I", "aVL", "V1", "V5", "V6")
        return self._evidence(
            record_id,
            claim_id,
            key,
            value,
            None,
            instrument,
            agreed=agreed,
            cross_check=cross_check,
            leads=lead_set,
            provenance=provenance,
        )


class STTExpert(EvidenceExpert):
    name = "st_t"

    def inspect(self, record_id, claim_id, instrument, *, cross_check=False, leads=()):
        morphology = dict(instrument.get("morphology") or {})
        if claim_id == "t_wave_inversion":
            measurement = (
                "t_wave_inversion_strict"
                if "t_wave_inversion_strict" in morphology
                else "t_wave_inversion"
            )
            value = morphology.get(measurement)
            agreed = morphology.get(f"{measurement}_agree")
            lead_key = (
                "t_inversion_strict_leads"
                if measurement.endswith("_strict")
                else "t_inversion_leads"
            )
            territory_key = (
                "t_inversion_strict_territory"
                if measurement.endswith("_strict")
                else "t_inversion_territory"
            )
            lead_set = tuple(morphology.get(lead_key) or leads)
            provenance = {
                "depth_mv": morphology.get("t_inversion_depth_mv"),
                "territory": morphology.get(territory_key),
                "n_backends": morphology.get("n_backends"),
                "criterion_profile": morphology.get("criterion_profile")
                if measurement.endswith("_strict")
                else "legacy_textbook_v1",
            }
        else:
            block = dict(instrument.get("st_mv") or {})
            values = dict(block.get("value") or {})
            value = {"V1": values.get("V1"), "V2": values.get("V2")}
            agreed = block.get("agree")
            measurement = "st_v1_v2"
            lead_set = ("V1", "V2")
            provenance = {"spread": block.get("spread") if cross_check else None}
        return self._evidence(
            record_id,
            claim_id,
            measurement,
            value,
            None,
            instrument,
            agreed=agreed,
            cross_check=cross_check,
            leads=lead_set,
            provenance=provenance,
        )


EXPERTS: dict[str, EvidenceExpert] = {
    expert.name: expert
    for expert in (
        QualityExpert(),
        IntervalExpert(),
        RhythmExpert(),
        AxisExpert(),
        MorphologyExpert(),
        STTExpert(),
    )
}


class ToolSimulator:
    """Offline executor: expert calls read only the precomputed instrument cache."""

    def __init__(self, instruments: Mapping[str, Mapping[str, Any]]) -> None:
        self.instruments = instruments

    def call(
        self,
        action_expert: str,
        record_id: str,
        claim_id: str,
        *,
        cross_check: bool = False,
        leads: tuple[str, ...] = (),
    ) -> Evidence:
        if action_expert not in EXPERTS:
            raise ValueError(f"unknown expert: {action_expert}")
        if record_id not in self.instruments:
            raise KeyError(f"no cached instrument row for {record_id}")
        return EXPERTS[action_expert].inspect(
            record_id,
            claim_id,
            self.instruments[record_id],
            cross_check=cross_check,
            leads=leads,
        )
