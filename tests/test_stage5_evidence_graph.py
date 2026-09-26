from __future__ import annotations

from ecg_agent.stage5_evidence_graph.audit import audit_rendered_report
from ecg_agent.stage5_evidence_graph.graph import ClinicalEvidenceGraphBuilder
from ecg_agent.stage5_evidence_graph.renderer import (
    DiagnosisEvidenceRenderer,
    DiagnosisLockedRenderer,
    SelectedEvidenceRenderer,
)


def _instrument() -> dict:
    return {
        "instrument_valid": True,
        "oracle_agreement_rate": 1.0,
        "scalar": {
            "heart_rate_bpm": {"value": 52.0, "agree": True, "spread": 1.0},
            "pr_ms": {"value": 220.0, "agree": True},
            "qrs_ms": {"value": 92.0, "agree": True},
            "qt_ms": {"value": 420.0, "agree": True},
            "qtc_ms": {"value": 410.0, "agree": True},
        },
        "rr_cv": {"value": 0.02, "agree": True},
        "p_wave": {"value": 1.0, "agree": True},
        "frontal_axis": {
            "value_degrees": -45.0,
            "category": "left_axis_deviation",
            "agree": True,
            "n_backends": 2,
            "spread_degrees": 3.0,
        },
        "morphology": {
            "p_wave_measurable": True,
            "max_limb_pp_mv": 0.40,
            "max_precordial_pp_mv": 1.20,
        },
        "voltage": {"agree": True, "lvh_by_voltage": False},
        "st_mv": {
            "agree": True,
            "value": {"I": 0.02, "II": 0.01, "V2": 0.04, "V5": -0.03},
        },
    }


def test_stage5_locks_diagnosis_and_renders_dynamic_measured_cards() -> None:
    instrument = _instrument()
    graph = ClinicalEvidenceGraphBuilder().build(
        record_id="r0",
        instrument=instrument,
        diagnoses=("sinus bradycardia", "left axis deviation", "low qrs voltage"),
        selected_claims=("bradycardia", "axis_left", "low_voltage"),
    )
    source = (
        "<think>Six-step source reasoning.</think>\n\n"
        "Verified correction (superseding any conflicting template statement): "
        "the registered ECG measurement supports bradycardia.\n\n"
        "<answer>Sinus bradycardia; left axis deviation; low QRS voltage</answer>"
    )
    rendered = DiagnosisLockedRenderer().render(source, graph)
    audit = audit_rendered_report(rendered, graph, instrument)

    assert rendered.diagnosis_locked
    assert rendered.answer_after == (
        "Sinus bradycardia; left axis deviation; low QRS voltage"
    )
    assert "52 bpm" in rendered.report
    assert "-45 degrees" in rendered.report
    assert "0.40 mV" in rendered.report
    assert "Verified correction" not in rendered.report
    assert audit["passed"]


def test_pr_card_marks_unmeasurable_atrial_activity_for_confirmation() -> None:
    instrument = _instrument()
    instrument["morphology"]["p_wave_measurable"] = False
    graph = ClinicalEvidenceGraphBuilder().build(
        record_id="r1",
        instrument=instrument,
        diagnoses=("first-degree AV block",),
        selected_claims=("pr_prolonged",),
    )
    conduction = next(
        card for card in graph.cards if card.domain == "conduction_intervals"
    )

    assert conduction.claim_id == "pr_prolonged"
    assert "confirm AV conduction manually" in conduction.statement


def test_invalid_instrument_abstains_instead_of_inventing_evidence() -> None:
    graph = ClinicalEvidenceGraphBuilder().build(
        record_id="r2",
        instrument={"instrument_valid": False},
        diagnoses=("sinus rhythm",),
        selected_claims=(),
    )

    assert not graph.cards
    assert set(graph.abstentions) == {
        "rate_rhythm",
        "conduction_intervals",
        "axis",
        "voltage_morphology",
        "st_t",
        "repolarization",
    }


def test_selected_renderer_keeps_action_evidence_and_old_correction() -> None:
    instrument = _instrument()
    graph = ClinicalEvidenceGraphBuilder().build(
        record_id="r3",
        instrument=instrument,
        diagnoses=("sinus bradycardia", "left axis deviation"),
        selected_claims=("axis_left",),
    )
    source = (
        "Narrative.\n\nVerified correction (superseding any conflicting template statement): "
        "axis evidence supports left axis deviation.\n\n"
        "<answer>Sinus bradycardia; left axis deviation</answer>"
    )
    rendered = SelectedEvidenceRenderer().render(source, graph)
    audit = audit_rendered_report(
        rendered,
        graph,
        instrument,
        require_ledger=False,
        require_old_correction_removed=False,
    )

    assert rendered.diagnosis_locked
    assert rendered.evidence_card_ids == ("frontal-axis",)
    assert "QT and repolarization" not in rendered.report
    assert "Verified correction" in rendered.report
    assert audit["passed"]


def test_diagnosis_evidence_renderer_adds_only_supported_cards() -> None:
    instrument = _instrument()
    graph = ClinicalEvidenceGraphBuilder().build(
        record_id="r4",
        instrument=instrument,
        diagnoses=("sinus bradycardia", "left axis deviation", "abnormal ECG"),
        selected_claims=(),
    )
    source = "Narrative.\n\n<answer>Sinus bradycardia; left axis deviation; abnormal ECG</answer>"
    rendered = DiagnosisEvidenceRenderer().render(source, graph)

    assert rendered.diagnosis_locked
    assert rendered.evidence_card_ids == ("rate-rhythm", "frontal-axis")
    assert "Leads assessed: II" in rendered.report
    assert "Leads assessed: I, aVF" in rendered.report
    assert "abnormal ECG.**" not in rendered.report
