"""Bind retrieved ECG criteria to measurement-grounded evidence cards.

Retrieval supplies the verification contract, not the patient observation.  A
criterion is rendered only when the existing ECG tools already measured all
patient-side quantities required by the corresponding conservative rule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ecg_agent.stage5_evidence_graph.schema import (
    ClinicalEvidenceGraph,
    EvidenceCard,
)

from .index import ClinicalKnowledgeIndex, SearchHit


@dataclass(frozen=True)
class RetrievedCriterion:
    card_id: str
    query: str
    summary: str
    chunk_id: str
    source_id: str
    title: str
    page: int
    score: float
    content_sha256: str
    license: str
    catalog_url: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "query": self.query,
            "summary": self.summary,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "title": self.title,
            "page": self.page,
            "score": self.score,
            "content_sha256": self.content_sha256,
            "license": self.license,
            "catalog_url": self.catalog_url,
        }


@dataclass(frozen=True)
class _Request:
    query: str
    summary: str
    anchor_groups: tuple[tuple[str, ...], ...]


def _measurement(card: EvidenceCard, suffix: str) -> float | None:
    for item in card.measurements:
        if item.path.endswith(suffix) and isinstance(item.value, (int, float)):
            return float(item.value)
    return None


def _request(card: EvidenceCard) -> _Request | None:
    labels = " | ".join(card.supports).lower()
    if card.card_id == "rate-rhythm":
        rate = _measurement(card, "heart_rate_bpm.value")
        rr_cv = _measurement(card, "rr_cv.value")
        p_fraction = _measurement(card, "p_wave.value")
        if (
            "atrial fibrillation" in labels
            and rr_cv is not None
            and rr_cv >= 0.12
            and p_fraction is not None
            and p_fraction < 0.5
        ):
            return _Request(
                query=(
                    "atrial fibrillation ECG criterion irregularly irregular "
                    "QRS intervals absent distinct P waves"
                ),
                summary=(
                    "Criterion check: atrial fibrillation calls for "
                    "irregularly irregular ventricular timing and absent distinct "
                    "P waves; the RR-variability and pre-QRS P-wave measurements "
                    "above evaluate those two features."
                ),
                anchor_groups=(("atrial fibrillation",), ("irregular",), ("p wave",)),
            )
        if "sinus brady" in labels and rate is not None and rate < 60.0:
            return _Request(
                query=(
                    "sinus bradycardia ECG criterion rate less than 60 P wave "
                    "before every QRS regular rhythm"
                ),
                summary=(
                    "Criterion check: sinus bradycardia requires sinus "
                    "activation with a ventricular rate below 60 bpm; rate, "
                    "regularity, and pre-QRS atrial activity were checked above."
                ),
                anchor_groups=(("sinus bradycardia",), ("60",), ("p wave", "sinus")),
            )
        if "sinus tachy" in labels and rate is not None and rate > 100.0:
            return _Request(
                query=(
                    "sinus tachycardia ECG criterion rate greater than 100 P wave "
                    "before every QRS regular rhythm"
                ),
                summary=(
                    "Criterion check: sinus tachycardia requires sinus "
                    "activation with a ventricular rate above 100 bpm; rate, "
                    "regularity, and pre-QRS atrial activity were checked above."
                ),
                anchor_groups=(("sinus tachycardia",), ("100",), ("p wave", "sinus")),
            )
        if "sinus" in labels and rr_cv is not None and p_fraction is not None:
            return _Request(
                query=(
                    "normal sinus rhythm ECG criterion regular P wave before every "
                    "QRS PR interval"
                ),
                summary=(
                    "Criterion check: sinus rhythm requires regular "
                    "atrial-to-ventricular activation; RR regularity and pre-QRS "
                    "atrial activity were checked above."
                ),
                anchor_groups=(("sinus",), ("p wave",), ("qrs",)),
            )

    if card.card_id == "conduction-intervals" and any(
        term in labels
        for term in ("first-degree", "first degree", "1st degree", "prolonged pr")
    ):
        pr = _measurement(card, "pr_ms.value")
        if pr is not None and pr >= 200.0 and "caveat" not in (card.criterion or ""):
            return _Request(
                query=(
                    "first degree atrioventricular AV block ECG criterion PR "
                    "interval greater than 0.20 seconds P wave QRS"
                ),
                summary=(
                    "Criterion check: first-degree AV block is defined by a "
                    "consistently prolonged PR interval above 0.20 s with preserved "
                    "atrial-to-ventricular conduction; the measured PR interval and "
                    "atrial measurability satisfy the available checks."
                ),
                anchor_groups=(
                    ("first degree", "first-degree"),
                    ("pr interval",),
                    ("0.20", "0.2", "200"),
                ),
            )
    return None


def _anchors_match(hit: SearchHit, groups: tuple[tuple[str, ...], ...]) -> bool:
    text = hit.passage.text.lower()
    return all(any(anchor in text for anchor in group) for group in groups)


class ClinicalCriterionRAG:
    """Retrieve and conservatively attach knowledge to supported ECG cards."""

    def __init__(
        self,
        index: ClinicalKnowledgeIndex,
        *,
        primary_source_id: str = "openrn_basic_ecg_cc_by_4",
        min_score: float = 0.08,
    ) -> None:
        self.index = index
        self.primary_source_id = primary_source_id
        self.min_score = min_score

    def retrieve_for_graph(
        self, graph: ClinicalEvidenceGraph
    ) -> dict[str, RetrievedCriterion]:
        retrieved: dict[str, RetrievedCriterion] = {}
        for card in graph.cards:
            if not card.supports:
                continue
            request = _request(card)
            if request is None:
                continue
            hits = self.index.retrieve(
                request.query,
                k=5,
                source_ids=frozenset({self.primary_source_id}),
            )
            hit = next(
                (
                    candidate
                    for candidate in hits
                    if candidate.score >= self.min_score
                    and _anchors_match(candidate, request.anchor_groups)
                ),
                None,
            )
            if hit is None:
                continue
            passage = hit.passage
            retrieved[card.card_id] = RetrievedCriterion(
                card_id=card.card_id,
                query=request.query,
                summary=request.summary,
                chunk_id=passage.chunk_id,
                source_id=passage.source_id,
                title=passage.title,
                page=passage.page,
                score=hit.score,
                content_sha256=passage.content_sha256,
                license=passage.license,
                catalog_url=passage.catalog_url,
            )
        return retrieved

    def augment_graph(
        self,
        graph: ClinicalEvidenceGraph,
        retrieved: dict[str, RetrievedCriterion],
        *,
        expose_in_report: bool,
    ) -> ClinicalEvidenceGraph:
        cards = tuple(
            replace(
                card,
                statement=(
                    f"{card.statement} {retrieved[card.card_id].summary}"
                    if expose_in_report and card.card_id in retrieved
                    else card.statement
                ),
            )
            for card in graph.cards
        )
        return replace(
            graph,
            cards=cards,
            provenance={
                **graph.provenance,
                "clinical_criterion_rag": {
                    "enabled": True,
                    "patient_evidence_from_rag": False,
                    "retrieved": [
                        item.as_dict()
                        for item in sorted(
                            retrieved.values(), key=lambda value: value.card_id
                        )
                    ],
                },
            },
        )
