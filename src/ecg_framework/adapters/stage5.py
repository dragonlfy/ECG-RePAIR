"""Public-SDK adapter for the diagnosis-locked Stage-5 presentation head."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..config import AgentConfig
from ..schema import Evidence, Report, RunContext


class DiagnosisLockedEvidencePresenter:
    """Render criterion-verified cards while preserving final labels exactly."""

    def __init__(self, instruments: Mapping[str, Mapping[str, Any]]) -> None:
        self.instruments = instruments

    def present(
        self,
        context: RunContext,
        evidence: Sequence[Evidence],
        graph: Mapping[str, object],
    ) -> Report:
        del evidence
        from ecg_agent.stage5_evidence_graph.graph import (
            ClinicalEvidenceGraphBuilder,
        )
        from ecg_agent.stage5_evidence_graph.renderer import (
            DiagnosisEvidenceRenderer,
        )

        instrument = self.instruments.get(context.case.record_id)
        if instrument is None:
            return context.report
        clinical_graph = ClinicalEvidenceGraphBuilder().build(
            record_id=context.case.record_id,
            instrument=instrument,
            diagnoses=context.report.labels,
            selected_claims=context.committed_claims,
        )
        rendered = DiagnosisEvidenceRenderer().render(
            context.report.text, clinical_graph
        )
        if not rendered.diagnosis_locked:
            raise ValueError("Stage-5 renderer changed the frozen answer")
        return Report(
            text=rendered.report,
            labels=context.report.labels,
            model_id=context.report.model_id,
            metadata={
                **context.report.metadata,
                "presenter": "stage5-diagnosis-locked-evidence-graph",
                "presented_evidence_card_ids": rendered.evidence_card_ids,
                "clinical_evidence_graph": clinical_graph.as_dict(),
                "decision_graph": dict(graph),
            },
        )


def build_stage5_framework(
    *,
    base_rows: Mapping[str, Mapping[str, Any]],
    instruments: Mapping[str, Mapping[str, Any]],
    policies: Mapping[int, Any],
    candidate_claims: Sequence[str],
    config: AgentConfig | None = None,
) -> Any:
    """Build the current two-head Stage-5 framework through the stable SDK."""

    from .stage4 import build_stage4_framework

    return build_stage4_framework(
        base_rows=base_rows,
        instruments=instruments,
        policies=policies,
        candidate_claims=candidate_claims,
        config=config,
        presenter=DiagnosisLockedEvidencePresenter(instruments),
    )
