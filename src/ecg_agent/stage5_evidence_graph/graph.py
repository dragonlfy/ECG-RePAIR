"""Construct a conservative, lead-aware ECG evidence graph.

The graph is deliberately not the ECG-R1 six-step prose template.  A domain is
rendered only when the frozen measurement cache contains a usable value and the
independent detector paths agree, or when Stage-4 has already committed the
corresponding finding.  Missing evidence remains an explicit abstention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import ClinicalEvidenceGraph, EvidenceCard, MeasurementRef

_CLAIM_LABELS = {
    "axis_left": "left axis deviation",
    "axis_right": "right axis deviation",
    "axis_extreme": "extreme axis deviation",
    "pr_prolonged": "first-degree AV block",
    "pr_short": "short PR interval",
    "qrs_prolonged": "prolonged QRS duration",
    "qrs_intermediate": "intraventricular conduction delay",
    "low_voltage": "low QRS voltage",
    "poor_r_progression": "poor R-wave progression",
    "pathologic_q": "pathologic Q waves",
    "t_wave_inversion": "T-wave inversion",
    "rv_infarction": "right ventricular involvement",
    "rhythm_irregular": "irregular rhythm",
    "bradycardia": "bradycardia",
    "tachycardia": "tachycardia",
}

_DOMAINS = (
    "rate_rhythm",
    "conduction_intervals",
    "axis",
    "voltage_morphology",
    "st_t",
    "repolarization",
)

_TERRITORY_LEADS = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anteroseptal": ("V1", "V2", "V3", "V4"),
}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _agreed(block: Mapping[str, Any], value_key: str = "value") -> float | None:
    value = _number(block.get(value_key))
    if value is None or block.get("agree") is not True:
        return None
    return value


def _label_text(diagnoses: Sequence[str]) -> str:
    return " | ".join(str(label).lower() for label in diagnoses)


def _territory_matches(label: str, territory: str | None) -> bool:
    """Require an anatomic match when both detector and diagnosis name a wall."""

    if not territory:
        return True
    lowered = label.lower()
    aliases = {
        "inferior": ("inferior",),
        "anterior": ("anterior", "anteroseptal", "ant/septal"),
        "anteroseptal": ("anterior", "anteroseptal", "ant/septal", "septal"),
        "septal": ("anteroseptal", "ant/septal", "septal"),
        "lateral": ("lateral", "anterolateral"),
        "anterolateral": ("anterior", "lateral", "anterolateral"),
    }
    detector_terms = aliases.get(str(territory).lower(), (str(territory).lower(),))
    named_terms = tuple(
        term
        for values in aliases.values()
        for term in values
        if term in lowered
    )
    return not named_terms or any(term in lowered for term in detector_terms)


def _r_progression_evidence(
    block: Mapping[str, Any],
) -> tuple[str, tuple[MeasurementRef, ...], tuple[str, ...]] | None:
    """Expose the actual precordial R amplitudes behind the strict PRWP flag."""

    per_lead = block.get("per_lead") or {}
    values: list[tuple[str, float]] = []
    for lead in ("V1", "V2", "V3", "V4"):
        value = _number((per_lead.get(lead) or {}).get("r_mv"))
        if value is not None:
            values.append((lead, value))
    r_v3 = _number(block.get("r_v3_mv"))
    if r_v3 is None:
        return None
    if not any(lead == "V3" for lead, _ in values):
        values.append(("V3", r_v3))
        values.sort(key=lambda item: int(item[0][1:]))
    measured = ", ".join(f"R({lead}) {value:.2f} mV" for lead, value in values)
    n_backends = int(block.get("n_backends") or 2)
    statement = (
        f"Median precordial amplitudes are {measured}; R(V3) is at or below "
        f"0.10 mV and the V2-V4 rise is non-monotonic, with {n_backends} "
        "independent beat-detector paths agreeing on the strict poor-progression "
        "criterion."
    )
    refs = tuple(
        MeasurementRef(
            f"morphology.per_lead.{lead}.r_mv",
            value,
            "mV",
            "V3 <=0.10 mV" if lead == "V3" else None,
        )
        for lead, value in values
    )
    return statement, refs, tuple(lead for lead, _ in values)


class ClinicalEvidenceGraphBuilder:
    """Build dynamic evidence cards from the frozen dual-detector cache."""

    def build(
        self,
        *,
        record_id: str,
        instrument: Mapping[str, Any],
        diagnoses: Sequence[str],
        selected_claims: Sequence[str],
    ) -> ClinicalEvidenceGraph:
        selected = tuple(dict.fromkeys(str(value) for value in selected_claims))
        selected_set = frozenset(selected)
        labels = tuple(str(value) for value in diagnoses)
        cards: list[EvidenceCard] = []
        abstentions: dict[str, str] = {}

        if instrument.get("instrument_valid") is not True:
            return ClinicalEvidenceGraph(
                record_id=record_id,
                diagnoses=labels,
                selected_claims=selected,
                cards=(),
                abstentions={domain: "instrument_invalid" for domain in _DOMAINS},
                provenance={"instrument_valid": False},
            )

        rate = self._rate_rhythm(instrument, labels, selected_set)
        self._admit(cards, abstentions, "rate_rhythm", rate)

        intervals = self._intervals(instrument, labels, selected_set)
        self._admit(cards, abstentions, "conduction_intervals", intervals)

        axis = self._axis(instrument, labels, selected_set)
        self._admit(cards, abstentions, "axis", axis)

        morphology = self._morphology(instrument, labels, selected_set)
        self._admit(cards, abstentions, "voltage_morphology", morphology)

        st_t = self._st_t(instrument, labels, selected_set)
        self._admit(cards, abstentions, "st_t", st_t)

        repolarization = self._repolarization(instrument, labels)
        self._admit(cards, abstentions, "repolarization", repolarization)

        represented = {card.claim_id for card in cards if card.claim_id}
        if "poor_r_progression" in selected_set - represented:
            extra_progression = self._poor_r_progression(instrument)
            if extra_progression is not None:
                cards.append(extra_progression)
                represented.add("poor_r_progression")
        missing_selected = selected_set - represented
        if missing_selected:
            abstentions["selected_claims"] = "unrenderable:" + ",".join(
                sorted(missing_selected)
            )

        return ClinicalEvidenceGraph(
            record_id=record_id,
            diagnoses=labels,
            selected_claims=selected,
            cards=tuple(cards),
            abstentions=abstentions,
            provenance={
                "instrument_valid": True,
                "oracle_agreement_rate": instrument.get("oracle_agreement_rate"),
                "measurement_policy": "dual_detector_agreement_or_committed_claim",
                "reference_used": False,
            },
        )

    @staticmethod
    def _admit(
        cards: list[EvidenceCard],
        abstentions: dict[str, str],
        domain: str,
        card: EvidenceCard | None,
    ) -> None:
        if card is None:
            abstentions[domain] = "no_agreed_measurement"
        else:
            cards.append(card)

    @staticmethod
    def _rate_rhythm(
        instrument: Mapping[str, Any],
        diagnoses: Sequence[str],
        selected: frozenset[str],
    ) -> EvidenceCard | None:
        scalar = instrument.get("scalar") or {}
        heart_rate = scalar.get("heart_rate_bpm") or {}
        rate = _agreed(heart_rate)
        rr_block = instrument.get("rr_cv") or {}
        rr_cv = _agreed(rr_block)
        p_block = instrument.get("p_wave") or {}
        p_score = _agreed(p_block)
        p_measurable = (instrument.get("morphology") or {}).get(
            "p_wave_measurable"
        ) is True
        if rate is None:
            return None

        parts = [f"ventricular rate {rate:.0f} bpm"]
        measurements = [
            MeasurementRef(
                "scalar.heart_rate_bpm.value",
                rate,
                "bpm",
                "bradycardia <60; tachycardia >100",
            )
        ]
        claim_id: str | None = None
        criterion: str | None = None
        rhythm_label = _label_text(diagnoses)
        diagnosed_bradycardia = "brady" in rhythm_label and rate < 60.0
        diagnosed_tachycardia = "tachy" in rhythm_label and rate > 100.0
        if "bradycardia" in selected or diagnosed_bradycardia:
            parts.append("below the 60 bpm bradycardia threshold")
            claim_id = "bradycardia" if "bradycardia" in selected else None
            criterion = "ventricular rate <60 bpm"
        elif "tachycardia" in selected or diagnosed_tachycardia:
            parts.append("above the 100 bpm tachycardia threshold")
            claim_id = "tachycardia" if "tachycardia" in selected else None
            criterion = "ventricular rate >100 bpm"
        report_irregularity = "rhythm_irregular" in selected or any(
            token in rhythm_label for token in ("irregular", "fibrillation", "flutter")
        )
        if rr_cv is not None and (rr_cv < 0.12 or report_irregularity):
            rr_interpretation = (
                "above the 0.12 marked-irregularity gate"
                if rr_cv >= 0.12
                else "below the 0.12 marked-irregularity gate"
            )
            parts.append(
                f"lead-II RR coefficient of variation {rr_cv:.3f} ({rr_interpretation})"
            )
            measurements.append(
                MeasurementRef("rr_cv.value", rr_cv, None, "marked irregularity >=0.12")
            )
            if "rhythm_irregular" in selected:
                claim_id = "rhythm_irregular"
                criterion = "RR coefficient of variation >=0.12"
        if (
            p_score is not None
            and p_score >= 0.5
            and p_measurable
            and "sinus" in rhythm_label
        ):
            parts.append(
                f"lead-II pre-QRS P-wave presence fraction {p_score:.2f} with "
                "measurable atrial morphology"
            )
            measurements.append(
                MeasurementRef("p_wave.value", p_score, None, ">=0.50")
            )
        elif (
            p_score is not None
            and p_score < 0.5
            and "fibrillation" in rhythm_label
        ):
            parts.append(f"lead-II pre-QRS P-wave presence fraction {p_score:.2f}")
            measurements.append(
                MeasurementRef("p_wave.value", p_score, None, "<0.50")
            )
        supports: list[str] = []
        for label in diagnoses:
            lowered = label.lower()
            sinus_supported = (
                "sinus" in lowered
                and p_score is not None
                and p_score >= 0.5
                and p_measurable
                and (rr_cv is None or rr_cv < 0.12)
            )
            rate_supported = (
                "brady" in lowered
                and rate < 60.0
                and ("sinus" not in lowered or sinus_supported)
            ) or (
                "tachy" in lowered
                and rate > 100.0
                and not any(
                    token in lowered
                    for token in ("atrial tachy", "supraventricular", "ventricular")
                )
                and ("sinus" not in lowered or sinus_supported)
            )
            irregular_supported = (
                any(token in lowered for token in ("fibrillation", "irregular"))
                and rr_cv is not None
                and rr_cv >= 0.12
                and p_score is not None
                and p_score < 0.5
            )
            if rate_supported or sinus_supported or irregular_supported:
                supports.append(label)
        return EvidenceCard(
            card_id="rate-rhythm",
            domain="rate_rhythm",
            statement="; ".join(parts) + ".",
            measurements=tuple(measurements),
            lead_set=("II",),
            supports=tuple(supports),
            claim_id=claim_id,
            criterion=criterion,
        )

    @staticmethod
    def _intervals(
        instrument: Mapping[str, Any],
        diagnoses: Sequence[str],
        selected: frozenset[str],
    ) -> EvidenceCard | None:
        scalar = instrument.get("scalar") or {}
        morphology = instrument.get("morphology") or {}
        pr = _agreed(scalar.get("pr_ms") or {})
        qrs = _agreed(scalar.get("qrs_ms") or {})
        parts: list[str] = []
        measurements: list[MeasurementRef] = []
        claim_id: str | None = None
        criterion: str | None = None
        pr_targeted = bool({"pr_prolonged", "pr_short"} & selected)
        qrs_targeted = bool({"qrs_prolonged", "qrs_intermediate"} & selected)
        action_targeted = pr_targeted or qrs_targeted

        p_measurable = morphology.get("p_wave_measurable") is True
        if (
            pr is not None
            and (not action_targeted or pr_targeted)
            and (p_measurable or "pr_prolonged" in selected)
        ):
            parts.append(f"PR {pr:.0f} ms")
            measurements.append(
                MeasurementRef(
                    "scalar.pr_ms.value", pr, "ms", "short <120; prolonged >=200"
                )
            )
            if "pr_prolonged" in selected:
                claim_id = "pr_prolonged"
                if p_measurable:
                    parts.append("meeting the >=200 ms first-degree AV-block criterion")
                    criterion = "PR >=200 ms with measurable atrial activity"
                else:
                    parts.append(
                        "above 200 ms, but the separate atrial-morphology gate was "
                        "not measurable; confirm AV conduction manually"
                    )
                    criterion = "PR >=200 ms; atrial measurability caveat"
            elif "pr_short" in selected:
                parts.append("below the 120 ms short-PR threshold")
                claim_id = "pr_short"
                criterion = "PR <120 ms"
        elif pr is not None and not action_targeted:
            parts.append(
                "PR estimate withheld because atrial activity was not measurable"
            )

        if qrs is not None and (not action_targeted or qrs_targeted):
            parts.append(f"QRS {qrs:.0f} ms")
            measurements.append(
                MeasurementRef(
                    "scalar.qrs_ms.value",
                    qrs,
                    "ms",
                    "intermediate 100-119; prolonged >=120",
                )
            )
            if "qrs_prolonged" in selected:
                claim_id = "qrs_prolonged"
                criterion = "QRS >=120 ms"
            elif "qrs_intermediate" in selected:
                claim_id = "qrs_intermediate"
                criterion = "QRS 100-119 ms"
        if not measurements:
            return None
        supports: list[str] = []
        for label in diagnoses:
            lowered = label.lower()
            first_degree = any(
                token in lowered
                for token in ("first-degree", "first degree", "1st degree")
            )
            pr_supported = (
                (first_degree or "prolonged pr" in lowered)
                and pr is not None
                and pr >= 200.0
                and p_measurable
            ) or ("short pr" in lowered and pr is not None and pr < 120.0)
            ivcd_supported = (
                "intraventricular conduction delay" in lowered
                and qrs is not None
                and 100.0 <= qrs < 120.0
            )
            wide_qrs_supported = (
                any(token in lowered for token in ("prolonged qrs", "wide qrs"))
                and qrs is not None
                and qrs >= 120.0
            )
            if pr_supported or ivcd_supported or wide_qrs_supported:
                supports.append(label)
        return EvidenceCard(
            card_id="conduction-intervals",
            domain="conduction_intervals",
            statement="; ".join(parts) + ".",
            measurements=tuple(measurements),
            lead_set=("II",),
            supports=tuple(dict.fromkeys(supports)),
            claim_id=claim_id,
            criterion=criterion,
        )

    @staticmethod
    def _axis(
        instrument: Mapping[str, Any],
        diagnoses: Sequence[str],
        selected: frozenset[str],
    ) -> EvidenceCard | None:
        block = instrument.get("frontal_axis") or {}
        angle = _agreed(block, "value_degrees")
        if angle is None:
            return None
        category = str(block.get("category") or "unclassified").replace("_", " ")
        claim_id = next(
            (
                claim
                for claim in ("axis_left", "axis_right", "axis_extreme")
                if claim in selected
            ),
            None,
        )
        parts = [
            f"frontal QRS axis {angle:+.0f} degrees",
            f"{category} by the median net-QRS vector in leads I and aVF",
        ]
        n_backends = block.get("n_backends")
        spread = _number(block.get("spread_degrees"))
        if n_backends and spread is not None:
            parts.append(
                f"{int(n_backends)} detector paths, {spread:.1f}-degree spread"
            )
        supports: list[str] = []
        for label in diagnoses:
            lowered = label.lower()
            matched = (
                "left axis" in lowered and category == "left axis deviation"
            ) or (
                "right axis" in lowered and category == "right axis deviation"
            ) or (
                any(token in lowered for token in ("extreme axis", "northwest axis"))
                and category == "extreme axis deviation"
            )
            if matched:
                supports.append(label)
        return EvidenceCard(
            card_id="frontal-axis",
            domain="axis",
            statement="; ".join(parts) + ".",
            measurements=(
                MeasurementRef(
                    "frontal_axis.value_degrees",
                    angle,
                    "degrees",
                    "left -90 to <-30; right >90 to 180",
                ),
            ),
            lead_set=("I", "aVF"),
            supports=tuple(supports),
            claim_id=claim_id,
            criterion=(
                "frontal-axis mutually exclusive clinical bins" if claim_id else None
            ),
        )

    @staticmethod
    def _morphology(
        instrument: Mapping[str, Any],
        diagnoses: Sequence[str],
        selected: frozenset[str],
    ) -> EvidenceCard | None:
        block = instrument.get("morphology") or {}
        voltage = instrument.get("voltage") or {}
        label_text = _label_text(diagnoses)

        low_voltage_labels = tuple(
            label
            for label in diagnoses
            if "low" in label.lower() and "voltage" in label.lower()
        )
        if "low_voltage" in selected or low_voltage_labels:
            limb = _number(block.get("max_limb_pp_mv"))
            chest = _number(block.get("max_precordial_pp_mv"))
            parts: list[str] = []
            refs: list[MeasurementRef] = []
            leads: tuple[str, ...] = ()
            if limb is not None and limb < 0.5:
                parts.append(
                    f"maximum limb-lead peak-to-peak QRS {limb:.2f} mV (<0.50 mV)"
                )
                refs.append(
                    MeasurementRef("morphology.max_limb_pp_mv", limb, "mV", "<0.50")
                )
                leads = ("I", "II", "III", "aVR", "aVL", "aVF")
            if chest is not None and chest < 1.0:
                parts.append(
                    f"maximum precordial peak-to-peak QRS {chest:.2f} mV (<1.00 mV)"
                )
                refs.append(
                    MeasurementRef(
                        "morphology.max_precordial_pp_mv", chest, "mV", "<1.00"
                    )
                )
                leads += ("V1", "V2", "V3", "V4", "V5", "V6")
            if parts:
                return EvidenceCard(
                    card_id="low-qrs-voltage",
                    domain="voltage_morphology",
                    statement="; ".join(parts) + "; supports low QRS voltage.",
                    measurements=tuple(refs),
                    lead_set=leads,
                    supports=low_voltage_labels
                    or (("low QRS voltage",) if "low_voltage" in selected else ()),
                    claim_id="low_voltage",
                    criterion="all leads in either limb or precordial set below threshold",
                )

        progression_labels = tuple(
            label
            for label in diagnoses
            if "poor" in label.lower()
            and "progression" in label.lower()
            and "r" in label.lower()
        )
        if "poor_r_progression" in selected or progression_labels:
            r_v3 = _number(block.get("r_v3_mv"))
            if (
                r_v3 is not None
                and block.get("poor_r_progression_strict") is True
                and block.get("poor_r_progression_strict_agree") is True
            ):
                evidence = _r_progression_evidence(block)
                if evidence is None:
                    return None
                statement, refs, leads = evidence
                return EvidenceCard(
                    card_id="r-wave-progression",
                    domain="voltage_morphology",
                    statement=statement,
                    measurements=refs,
                    lead_set=leads,
                    supports=progression_labels
                    or (
                        ("poor R-wave progression",)
                        if "poor_r_progression" in selected
                        else ()
                    ),
                    claim_id="poor_r_progression",
                    criterion="externally frozen high-specificity precordial progression profile",
                )

        if (
            voltage.get("agree") is True
            and voltage.get("lvh_by_voltage") is True
            and "ventricular hypertrophy" in label_text
        ):
            sokolow = _number(voltage.get("sokolow_lyon_mv"))
            avl = _number(voltage.get("r_avl_mv"))
            lead_i = _number(voltage.get("r_lead_i_mv"))
            if sokolow is not None and avl is not None:
                values = [f"Sokolow-Lyon {sokolow:.2f} mV", f"R(aVL) {avl:.2f} mV"]
                refs = [
                    MeasurementRef("voltage.sokolow_lyon_mv", sokolow, "mV", ">3.50"),
                    MeasurementRef("voltage.r_avl_mv", avl, "mV", ">1.10"),
                ]
                if lead_i is not None:
                    values.append(f"R(I) {lead_i:.2f} mV")
                    refs.append(
                        MeasurementRef("voltage.r_lead_i_mv", lead_i, "mV", ">1.50")
                    )
                positive = []
                if sokolow > 3.5:
                    positive.append(f"Sokolow-Lyon {sokolow:.2f} mV >3.50 mV")
                if avl > 1.1:
                    positive.append(f"R(aVL) {avl:.2f} mV >1.10 mV")
                if lead_i is not None and lead_i > 1.5:
                    positive.append(f"R(I) {lead_i:.2f} mV >1.50 mV")
                conclusion = (
                    "; " + " and ".join(positive) + " support LVH by voltage."
                    if positive
                    else "."
                )
                return EvidenceCard(
                    card_id="lvh-voltage",
                    domain="voltage_morphology",
                    statement="; ".join(values) + conclusion,
                    measurements=tuple(refs),
                    lead_set=("I", "aVL", "V1", "V5", "V6"),
                    supports=tuple(
                        label for label in diagnoses if "hypertrophy" in label.lower()
                    ),
                    criterion="Sokolow-Lyon, R(aVL), and R(I) voltage criteria",
                )
        return None

    @staticmethod
    def _poor_r_progression(
        instrument: Mapping[str, Any],
    ) -> EvidenceCard | None:
        block = instrument.get("morphology") or {}
        r_v3 = _number(block.get("r_v3_mv"))
        if (
            r_v3 is None
            or block.get("poor_r_progression_strict") is not True
            or block.get("poor_r_progression_strict_agree") is not True
        ):
            return None
        evidence = _r_progression_evidence(block)
        if evidence is None:
            return None
        statement, refs, leads = evidence
        return EvidenceCard(
            card_id="r-wave-progression",
            domain="voltage_morphology",
            statement=statement,
            measurements=refs,
            lead_set=leads,
            supports=("poor R-wave progression",),
            claim_id="poor_r_progression",
            criterion="externally frozen high-specificity precordial progression profile",
        )

    @staticmethod
    def _st_t(
        instrument: Mapping[str, Any],
        diagnoses: Sequence[str],
        selected: frozenset[str],
    ) -> EvidenceCard | None:
        morphology = instrument.get("morphology") or {}
        t_labels = tuple(
            label
            for label in diagnoses
            if (
                "t-wave inversion" in label.lower()
                or "t wave inversion" in label.lower()
                or "inverted t" in label.lower()
                or "t wave change" in label.lower()
                or "t-wave change" in label.lower()
                or "t wave abnormal" in label.lower()
                or "t abnormal" in label.lower()
                or "st-t change" in label.lower()
            )
        )
        if "t_wave_inversion" in selected or t_labels:
            use_strict = (
                morphology.get("t_wave_inversion_strict") is True
                and morphology.get("t_wave_inversion_strict_agree") is True
            )
            use_standard = (
                morphology.get("t_wave_inversion") is True
                and morphology.get("t_wave_inversion_agree") is True
            )
            lead_key = (
                "t_inversion_strict_leads" if use_strict else "t_inversion_leads"
            )
            territory_key = (
                "t_inversion_strict_territory"
                if use_strict
                else "t_inversion_territory"
            )
            leads = tuple(morphology.get(lead_key) or ())
            territory = morphology.get(territory_key)
            if territory in _TERRITORY_LEADS:
                leads = tuple(
                    lead for lead in leads if lead in _TERRITORY_LEADS[territory]
                )
            depth = _number(morphology.get("t_inversion_depth_mv"))
            matched_t_labels = tuple(
                label for label in t_labels if _territory_matches(label, territory)
            )
            if (
                leads
                and (use_strict or use_standard)
                and (matched_t_labels or not t_labels)
            ):
                parts = [f"T-wave inversion in contiguous leads {', '.join(leads)}"]
                refs: list[MeasurementRef] = []
                if territory:
                    parts.append(f"{territory} territory")
                if depth is not None:
                    parts.append(f"second-deepest T amplitude {depth:.2f} mV")
                    refs.append(
                        MeasurementRef("morphology.t_inversion_depth_mv", depth, "mV")
                    )
                if not refs:
                    refs.append(
                        MeasurementRef(
                            "morphology.t_wave_inversion_strict",
                            True,
                            None,
                            (
                                "contiguous-territory strict profile"
                                if use_strict
                                else "contiguous leads <=-0.10 mV"
                            ),
                        )
                    )
                return EvidenceCard(
                    card_id="t-wave-inversion",
                    domain="st_t",
                    statement="; ".join(parts) + "; supports T-wave inversion.",
                    measurements=tuple(refs),
                    lead_set=leads,
                    supports=matched_t_labels
                    or (
                        ("T-wave inversion",)
                        if "t_wave_inversion" in selected
                        else ()
                    ),
                    claim_id="t_wave_inversion",
                    criterion=(
                        "contiguous-territory high-specificity T-wave profile"
                        if use_strict
                        else "contiguous leads with T amplitude <=-0.10 mV"
                    ),
                )

        infarct_labels = tuple(
            label
            for label in diagnoses
            if any(
                token in label.lower()
                for token in ("infarct", "myocardial infarction")
            )
        )
        q_targeted = "pathologic_q" in selected
        q_strict = (
            morphology.get("pathologic_q_strict") is True
            and morphology.get("pathologic_q_strict_agree") is True
        )
        q_standard = (
            morphology.get("pathologic_q") is True
            and morphology.get("pathologic_q_agree") is True
        )
        q_lead_key = "pathologic_q_strict_leads" if q_strict else "pathologic_q_leads"
        q_territory_key = (
            "pathologic_q_strict_territory" if q_strict else "pathologic_q_territory"
        )
        q_leads = tuple(morphology.get(q_lead_key) or ())
        q_territory = morphology.get(q_territory_key)
        if q_territory in _TERRITORY_LEADS:
            q_leads = tuple(
                lead for lead in q_leads if lead in _TERRITORY_LEADS[q_territory]
            )
        matched_infarct_labels = tuple(
            label
            for label in infarct_labels
            if _territory_matches(label, q_territory)
        )
        if (
            (q_targeted or matched_infarct_labels)
            and (q_strict or q_standard)
            and q_leads
        ):
            per_lead = morphology.get("per_lead") or {}
            refs: list[MeasurementRef] = []
            details: list[str] = []
            for lead in q_leads:
                values = per_lead.get(lead) or {}
                duration = _number(values.get("q_ms"))
                amplitude = _number(values.get("q_mv"))
                if duration is not None:
                    refs.append(
                        MeasurementRef(
                            f"morphology.per_lead.{lead}.q_ms",
                            duration,
                            "ms",
                            ">=30 ms with sufficient depth",
                        )
                    )
                if amplitude is not None:
                    refs.append(
                        MeasurementRef(
                            f"morphology.per_lead.{lead}.q_mv",
                            amplitude,
                            "mV",
                        )
                    )
                if duration is not None and amplitude is not None:
                    details.append(f"{lead}: Q {duration:.0f} ms/{amplitude:.2f} mV")
            if not refs:
                refs.append(
                    MeasurementRef(
                        "morphology.pathologic_q_strict", True, None, "strict profile"
                    )
                )
            territory_text = f" in the {q_territory} territory" if q_territory else ""
            detail_text = f" ({'; '.join(details)})" if details else ""
            return EvidenceCard(
                card_id="pathologic-q-territory",
                domain="st_t",
                statement=(
                    f"Pathologic Q-wave morphology is present in contiguous leads "
                    f"{', '.join(q_leads)}{territory_text}{detail_text}; this is "
                    "infarction-compatible evidence, interpreted with the final "
                    "diagnostic context."
                ),
                measurements=tuple(refs),
                lead_set=q_leads,
                supports=matched_infarct_labels
                or (("pathologic Q waves",) if q_targeted else ()),
                claim_id="pathologic_q" if q_targeted else None,
                criterion=(
                    "contiguous-territory high-specificity pathologic-Q profile"
                    if q_strict
                    else "contiguous pathologic Q waves >=30 ms or >=0.10 mV deep"
                ),
            )

        st_block = instrument.get("st_mv") or {}
        values = st_block.get("value") or {}
        if st_block.get("agree") is not True or not values:
            return None
        numeric = {
            str(lead): float(value)
            for lead, value in values.items()
            if _number(value) is not None
        }
        if not numeric:
            return None
        lead, value = max(numeric.items(), key=lambda item: abs(item[1]))
        all_refs = tuple(
            MeasurementRef(f"st_mv.value.{name}", measured, "mV")
            for name, measured in sorted(numeric.items())
        )
        # This is a measurement statement, not an ischemia exclusion: V2/V3
        # thresholds depend on age and sex, and reciprocal morphology matters.
        return EvidenceCard(
            card_id="st-territory-review",
            domain="st_t",
            statement=(
                "J+80-ms ST levels were checked across inferior (II, III, aVF), "
                "anterior (V1-V4), and lateral (I, aVL, V5-V6) territories; the "
                f"largest absolute measured displacement was {abs(value):.2f} mV in {lead}. "
                "Amplitude alone is not used to infer ischemia without contiguous "
                "morphology and demographic thresholds."
            ),
            measurements=all_refs,
            lead_set=(
                "II",
                "III",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "I",
                "aVL",
                "V5",
                "V6",
            ),
            supports=(),
            criterion="territory-organised J+80-ms ST measurement; no diagnosis inferred alone",
        )

    @staticmethod
    def _repolarization(
        instrument: Mapping[str, Any], diagnoses: Sequence[str]
    ) -> EvidenceCard | None:
        scalar = instrument.get("scalar") or {}
        qt = _agreed(scalar.get("qt_ms") or {})
        qtc = _agreed(scalar.get("qtc_ms") or {})
        if qt is None and qtc is None:
            return None
        parts: list[str] = []
        refs: list[MeasurementRef] = []
        if qt is not None:
            parts.append(f"QT {qt:.0f} ms")
            refs.append(MeasurementRef("scalar.qt_ms.value", qt, "ms"))
        if qtc is not None:
            parts.append(f"detector-reported QTc {qtc:.0f} ms")
            refs.append(MeasurementRef("scalar.qtc_ms.value", qtc, "ms"))
        prolonged_labels = tuple(
            label
            for label in diagnoses
            if "qt" in label.lower()
            and any(token in label.lower() for token in ("prolong", "long"))
        )
        supports = prolonged_labels if qtc is not None and qtc >= 480.0 else ()
        if supports:
            parts.append(
                "at or above the conservative sex-independent 480 ms prolonged-QTc gate"
            )
        else:
            parts.append(
                "sex-specific QTc classification was not inferred without metadata"
            )
        return EvidenceCard(
            card_id="qt-repolarization",
            domain="repolarization",
            statement="; ".join(parts) + ".",
            measurements=tuple(refs),
            lead_set=("II",),
            supports=supports,
            criterion=(
                "conservative sex-independent QTc >=480 ms"
                if supports
                else "measurement only; metadata-dependent threshold abstention"
            ),
        )
