"""Diagnosis-locked renderer for auditable ECG evidence cards."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ecg_agent.stage3_direct.composer import ANSWER

from .schema import ClinicalEvidenceGraph

_OLD_LEDGER = re.compile(
    r"\n*<ecg_evidence>.*?</ecg_evidence>\n*", re.IGNORECASE | re.DOTALL
)
_CORRECTION = re.compile(
    r"(?:\n\s*)?Verified correction \(superseding any conflicting template statement\):"
    r".*?(?:This claim is withheld from the final diagnosis\.)?(?=\n|<answer>)",
    re.IGNORECASE,
)

_DOMAIN_TITLES = {
    "rate_rhythm": "Rate and rhythm",
    "conduction_intervals": "Conduction and intervals",
    "axis": "Frontal plane axis",
    "voltage_morphology": "Voltage and QRS morphology",
    "st_t": "ST-T and lead territories",
    "repolarization": "QT and repolarization",
}


def _compact_support_heading(labels: tuple[str, ...], domain: str) -> str:
    """Avoid echoing malformed paragraph-long labels from legacy generations."""

    clean = [label.strip().rstrip(".") for label in labels if label.strip()]
    if clean and all(len(label) <= 100 and "\n" not in label for label in clean):
        return "; ".join(clean)
    text = " | ".join(clean).lower()
    domain_terms = {
        "rate_rhythm": (
            ("atrial fibrillation", "Atrial fibrillation"),
            ("sinus bradycardia", "Sinus bradycardia"),
            ("sinus tachycardia", "Sinus tachycardia"),
            ("sinus rhythm", "Sinus rhythm"),
        ),
        "conduction_intervals": (
            ("first-degree", "First-degree AV block"),
            ("first degree", "First-degree AV block"),
            ("1st degree", "First-degree AV block"),
            ("prolonged pr", "Prolonged PR interval"),
            ("short pr", "Short PR interval"),
            ("intraventricular", "Intraventricular conduction delay"),
        ),
        "axis": (
            ("left axis", "Left axis deviation"),
            ("right axis", "Right axis deviation"),
            ("extreme axis", "Extreme axis deviation"),
        ),
        "voltage_morphology": (
            ("low qrs", "Low QRS voltage"),
            ("low voltage", "Low QRS voltage"),
            ("poor r", "Poor R-wave progression"),
            ("ventricular hypertrophy", "Left ventricular hypertrophy"),
        ),
        "st_t": (
            ("infarct", "Infarction-compatible diagnosis"),
            ("t wave", "T-wave abnormality"),
            ("t-wave", "T-wave abnormality"),
            ("st-t", "ST-T abnormality"),
        ),
        "repolarization": (("qt", "Prolonged QT interval"),),
    }
    found = [display for needle, display in domain_terms.get(domain, ()) if needle in text]
    return "; ".join(dict.fromkeys(found)) or "Diagnosis-consistent evidence"


@dataclass(frozen=True)
class RenderedReport:
    report: str
    answer_before: str
    answer_after: str
    evidence_card_ids: tuple[str, ...]
    diagnosis_locked: bool


class DiagnosisLockedRenderer:
    """Add a dynamic clinical evidence ledger without changing the answer."""

    def render(self, report: str, graph: ClinicalEvidenceGraph) -> RenderedReport:
        match = ANSWER.search(report)
        if match is None:
            answer = "; ".join(graph.diagnoses).strip()
            if not answer:
                raise ValueError("Stage-5 requires an answer or frozen diagnoses")
            narrative = report
        else:
            answer = match.group(1).strip()
            narrative = report[: match.start()] + report[match.end() :]
        narrative = _OLD_LEDGER.sub("\n", narrative)
        narrative = _CORRECTION.sub("", narrative)
        narrative = re.sub(r"\n{3,}", "\n\n", narrative).rstrip()

        lines = [
            "<ecg_evidence>",
            "**Measurement-grounded ECG evidence ledger**",
        ]
        for index, card in enumerate(graph.cards, start=1):
            title = _DOMAIN_TITLES.get(card.domain, card.domain.replace("_", " "))
            lines.append(f"E{index}. **{title}.** {card.statement}")
        if not graph.cards:
            lines.append(
                "No measurement card met the reliability rule; the frozen diagnosis is retained."
            )
        lines.append("</ecg_evidence>")
        ledger = "\n".join(lines)
        rendered = f"{narrative}\n\n{ledger}\n\n<answer>{answer}</answer>"
        after = ANSWER.search(rendered)
        answer_after = after.group(1).strip() if after is not None else ""
        return RenderedReport(
            report=rendered,
            answer_before=answer,
            answer_after=answer_after,
            evidence_card_ids=tuple(card.card_id for card in graph.cards),
            diagnosis_locked=answer == answer_after,
        )


class SelectedEvidenceRenderer:
    """Expose only action-facing evidence while preserving Stage-4's narrative."""

    def render(self, report: str, graph: ClinicalEvidenceGraph) -> RenderedReport:
        match = ANSWER.search(report)
        if match is None:
            answer = "; ".join(graph.diagnoses).strip()
            if not answer:
                raise ValueError("Stage-5 requires an answer or frozen diagnoses")
            narrative = report
        else:
            answer = match.group(1).strip()
            narrative = report[: match.start()] + report[match.end() :]
        narrative = _OLD_LEDGER.sub("\n", narrative)
        narrative = re.sub(r"\n{3,}", "\n\n", narrative).rstrip()

        cards = tuple(
            card
            for card in graph.cards
            if card.claim_id is not None and card.claim_id in graph.selected_claims
        )
        lines = ["**Measurement-grounded evidence used by the ECG Agent:**"]
        lines.extend(f"- {card.statement}" for card in cards)
        evidence = "\n".join(lines)
        rendered = f"{narrative}\n\n{evidence}\n\n<answer>{answer}</answer>"
        after = ANSWER.search(rendered)
        answer_after = after.group(1).strip() if after is not None else ""
        return RenderedReport(
            report=rendered,
            answer_before=answer,
            answer_after=answer_after,
            evidence_card_ids=tuple(card.card_id for card in cards),
            diagnosis_locked=answer == answer_after,
        )


class DiagnosisEvidenceRenderer:
    """Append only criterion-verified evidence linked to a frozen diagnosis.

    Unlike the full ledger, this head does not expose measurements that merely
    happened to be available.  Each rendered card must support at least one
    diagnosis already present in ``<answer>``.  The head therefore improves the
    report's lead-level audit trail while leaving both diagnosis selection and
    the original ECG-R1/Stage-4 reasoning untouched.
    """

    heading = "**Diagnosis-linked ECG evidence verified by the ECG Agent:**"

    def render(self, report: str, graph: ClinicalEvidenceGraph) -> RenderedReport:
        match = ANSWER.search(report)
        if match is None:
            answer = "; ".join(graph.diagnoses).strip()
            if not answer:
                raise ValueError("Stage-5 requires an answer or frozen diagnoses")
            narrative = report.rstrip()
        else:
            answer = match.group(1).strip()
            narrative = (report[: match.start()] + report[match.end() :]).rstrip()
        narrative = _OLD_LEDGER.sub("\n", narrative)
        narrative = re.sub(r"\n{3,}", "\n\n", narrative).rstrip()

        cards = tuple(card for card in graph.cards if card.supports)
        if not cards and match is not None:
            return RenderedReport(
                report=report,
                answer_before=answer,
                answer_after=answer,
                evidence_card_ids=(),
                diagnosis_locked=True,
            )
        if cards:
            lines = [self.heading]
            for index, card in enumerate(cards, start=1):
                diagnoses = _compact_support_heading(card.supports, card.domain)
                lead_clause = (
                    f" Leads assessed: {', '.join(card.lead_set)}."
                    if card.lead_set
                    else ""
                )
                lines.append(
                    f"{index}. **{diagnoses}.** {card.statement}{lead_clause}"
                )
            evidence = "\n".join(lines)
            rendered = f"{narrative}\n\n{evidence}\n\n<answer>{answer}</answer>"
        else:
            rendered = f"{narrative}\n\n<answer>{answer}</answer>"

        after = ANSWER.search(rendered)
        answer_after = after.group(1).strip() if after is not None else ""
        return RenderedReport(
            report=rendered,
            answer_before=answer,
            answer_after=answer_after,
            evidence_card_ids=tuple(card.card_id for card in cards),
            diagnosis_locked=answer == answer_after,
        )
