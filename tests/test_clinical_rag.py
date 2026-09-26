from __future__ import annotations

import json
from pathlib import Path

from ecg_agent.clinical_rag import ClinicalCriterionRAG, ClinicalKnowledgeIndex
from ecg_agent.stage5_evidence_graph.graph import ClinicalEvidenceGraphBuilder


def _index(tmp_path: Path) -> ClinicalKnowledgeIndex:
    rows = [
        {
            "chunk_id": "openrn:p10:c1",
            "content_sha256": "abc",
            "text": (
                "Atrial fibrillation has irregularly irregular QRS intervals "
                "with no distinct P waves."
            ),
            "metadata": {
                "source_id": "openrn_basic_ecg_cc_by_4",
                "title": "Chapter 7: Interpret Basic ECG",
                "page": 10,
                "license": "CC BY 4.0",
                "catalog_url": "https://example.test",
                "authority_tier": "institutional_educational_source",
            },
        }
    ]
    path = tmp_path / "knowledge.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return ClinicalKnowledgeIndex.from_jsonl([path])


def test_rag_adds_provenance_only_after_patient_measurement(tmp_path: Path) -> None:
    instrument = {
        "instrument_valid": True,
        "scalar": {"heart_rate_bpm": {"value": 72.0, "agree": True}},
        "rr_cv": {"value": 0.20, "agree": True},
        "p_wave": {"value": 0.10, "agree": True},
        "morphology": {"p_wave_measurable": True},
    }
    graph = ClinicalEvidenceGraphBuilder().build(
        record_id="r1",
        instrument=instrument,
        diagnoses=("atrial fibrillation",),
        selected_claims=(),
    )
    rag = ClinicalCriterionRAG(_index(tmp_path), min_score=0.0)
    retrieved = rag.retrieve_for_graph(graph)
    augmented = rag.augment_graph(graph, retrieved, expose_in_report=True)

    assert "rate-rhythm" in retrieved
    assert retrieved["rate-rhythm"].page == 10
    assert "Criterion check: atrial fibrillation" in augmented.cards[0].statement
    assert (
        augmented.provenance["clinical_criterion_rag"]["patient_evidence_from_rag"]
        is False
    )


def test_rag_does_not_turn_knowledge_into_patient_evidence(tmp_path: Path) -> None:
    graph = ClinicalEvidenceGraphBuilder().build(
        record_id="r2",
        instrument={"instrument_valid": False},
        diagnoses=("atrial fibrillation",),
        selected_claims=(),
    )
    rag = ClinicalCriterionRAG(_index(tmp_path), min_score=0.0)

    assert rag.retrieve_for_graph(graph) == {}
