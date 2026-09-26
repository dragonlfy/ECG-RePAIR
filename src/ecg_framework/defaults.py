"""Small reference components useful for examples, tests, and new plugins."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .schema import (
    ClaimCandidate,
    ClaimSpec,
    ECGCase,
    Evidence,
    EvidenceRelation,
    Report,
    ReportAction,
    RunContext,
    ValueEstimate,
)


class SeedReportModel:
    """Use a report supplied in ``ECGCase.metadata`` as the frozen base output."""

    def __init__(
        self,
        report_key: str = "base_report",
        labels_key: str = "base_labels",
        model_id: str = "frozen-base-model",
    ) -> None:
        self.report_key = report_key
        self.labels_key = labels_key
        self.model_id = model_id

    def infer(self, case: ECGCase) -> Report:
        if self.report_key not in case.metadata:
            raise KeyError(f"ECGCase.metadata lacks {self.report_key!r}")
        return Report(
            text=str(case.metadata[self.report_key]),
            labels=tuple(
                str(value) for value in case.metadata.get(self.labels_key, ())
            ),
            model_id=self.model_id,
        )


class RegistryClaimScanner:
    """Scan the registered ontology without treating local negation as positive."""

    NEGATION = re.compile(r"\b(?:no|not|without|absent|neither|nor)\b", re.I)

    def scan(
        self, context: RunContext, claims: Mapping[str, ClaimSpec]
    ) -> Sequence[ClaimCandidate]:
        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            " ".join((context.report.text, *context.report.labels)),
        )
        output = []
        for claim_id, spec in claims.items():
            asserted = False
            source_text = None
            for sentence in sentences:
                lower = sentence.lower()
                positions = [lower.find(alias.lower()) for alias in spec.aliases]
                positions = [position for position in positions if position >= 0]
                if not positions:
                    continue
                position = min(positions)
                asserted = not bool(
                    self.NEGATION.search(lower[max(0, position - 40) : position])
                )
                source_text = sentence.strip()
                break
            output.append(
                ClaimCandidate(
                    claim_id=claim_id,
                    asserted=asserted,
                    source="base_report" if source_text else "ontology_scan",
                    source_text=source_text,
                    priority=spec.priority,
                )
            )
        return output


class PriorityInspectionPlanner:
    def rank(
        self, context: RunContext, candidates: Sequence[ClaimCandidate]
    ) -> Sequence[ClaimCandidate]:
        del context
        return tuple(
            sorted(candidates, key=lambda item: (-item.priority, item.claim_id))
        )


class PredicateVerifier:
    """Verify evidence with registered claim predicates and a reliability floor."""

    def __init__(
        self,
        predicates: Mapping[str, Callable[[Any], bool | None]],
        reliability_floor: float = 0.55,
    ) -> None:
        self.predicates = dict(predicates)
        self.reliability_floor = float(reliability_floor)

    def verify(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        spec: ClaimSpec,
        evidence: Evidence,
    ) -> Evidence:
        del context, spec
        predicate = self.predicates.get(candidate.claim_id)
        result = predicate(evidence.value) if predicate else None
        if result is None or evidence.reliability < self.reliability_floor:
            relation = EvidenceRelation.UNCERTAIN
        elif result:
            relation = EvidenceRelation.SUPPORTED
        else:
            relation = EvidenceRelation.CONTRADICTED
        return replace(evidence, relation=relation)


class FixedOutcomePolicy:
    """Deterministic policy for smoke tests; real work should learn from outcomes."""

    def __init__(self, safe_return: float = 1.0) -> None:
        self.safe_return = float(safe_return)

    def estimate(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        action: ReportAction,
        evidence: Evidence,
    ) -> ValueEstimate:
        del context, candidate, action, evidence
        return ValueEstimate(
            mean_return=self.safe_return,
            uncertainty=0.0,
            safe_return=self.safe_return,
            policy_id="fixed-reference-policy",
        )


class EvidenceAppendComposer:
    """Minimal immutable composer intended as an extension starting point."""

    def apply(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        action: ReportAction,
        evidence: Evidence,
    ) -> Report:
        label = candidate.claim_id.replace("_", " ")
        labels = list(context.report.labels)
        if action is ReportAction.WITHHOLD:
            labels = [value for value in labels if label not in value.lower()]
            suffix = f"Verified correction: {label} is not supported."
        else:
            if action is ReportAction.ADD and not any(
                label in value.lower() for value in labels
            ):
                labels.append(label)
            lead_text = f" in {', '.join(evidence.leads)}" if evidence.leads else ""
            unit = f" {evidence.unit}" if evidence.unit else ""
            suffix = (
                f"Verified ECG evidence: {evidence.measurement}="
                f"{evidence.value}{unit}{lead_text} supports {label}."
            )
        text = f"{context.report.text.rstrip()}\n\n{suffix}"
        return Report(
            text=text,
            labels=tuple(labels),
            model_id=context.report.model_id,
            metadata={**context.report.metadata, "last_action": action.value},
        )
