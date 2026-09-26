"""Compatibility bridge from the research Stage-4 code to the public SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

from ..config import AgentConfig
from ..defaults import PriorityInspectionPlanner
from ..protocols import EvidencePresenter
from ..registry import FrameworkRegistry
from ..runtime import ECGFramework
from ..schema import (
    ClaimCandidate,
    ClaimSpec,
    Evidence,
    EvidenceRelation,
    Report,
    ReportAction,
    RunContext,
    ValueEstimate,
)
from .common import FrozenMappingBaseModel


def _evidence_from_legacy(value: Any) -> Evidence:
    relation = EvidenceRelation(str(value.relation.value))
    return Evidence(
        evidence_id=value.evidence_id,
        claim_id=value.claim_id,
        expert=value.expert,
        measurement=value.measurement,
        value=value.value,
        unit=value.unit,
        leads=tuple(value.lead_set),
        relation=relation,
        reliability=float(value.reliability),
        backend_agreement=value.backend_agreement,
        provenance=dict(value.provenance),
    )


def _evidence_to_legacy(value: Evidence) -> Any:
    from ecg_agent.schema import Evidence as LegacyEvidence
    from ecg_agent.schema import VerificationStatus

    return LegacyEvidence(
        evidence_id=value.evidence_id,
        claim_id=value.claim_id,
        expert=value.expert,
        measurement=value.measurement,
        value=value.value,
        unit=value.unit,
        lead_set=value.leads,
        relation=VerificationStatus(value.relation.value),
        reliability=value.reliability,
        backend_agreement=value.backend_agreement,
        provenance=dict(value.provenance),
    )


class LegacyStage4Scanner:
    def __init__(self, candidate_claims: Sequence[str]) -> None:
        self.candidate_claims = tuple(candidate_claims)

    def scan(
        self, context: RunContext, claims: Mapping[str, ClaimSpec]
    ) -> Sequence[ClaimCandidate]:
        from ecg_agent.stage3_direct.claim_parser import DiagnosticClaimParser

        parsed = {
            claim.claim_id: claim
            for claim in DiagnosticClaimParser().parse(
                context.report.text, context.report.labels
            )
        }
        return tuple(
            ClaimCandidate(
                claim_id=claim_id,
                asserted=bool(claim_id in parsed and parsed[claim_id].r1_value is True),
                source="legacy_stage4_scanner",
                source_text=(parsed[claim_id].r1_text if claim_id in parsed else None),
                priority=claims[claim_id].priority,
            )
            for claim_id in self.candidate_claims
        )


class LegacyStage4Expert:
    def __init__(self, instruments: Mapping[str, Mapping[str, Any]]) -> None:
        from ecg_agent.experts import ToolSimulator

        self.tools = ToolSimulator(instruments)

    def inspect(
        self, context: RunContext, candidate: ClaimCandidate, spec: ClaimSpec
    ) -> Evidence:
        return _evidence_from_legacy(
            self.tools.call(spec.expert, context.case.record_id, candidate.claim_id)
        )


class LegacyStage4Verifier:
    def __init__(self) -> None:
        from ecg_agent.stage3_direct.verifier import ClinicalEvidenceVerifier

        self.verifier = ClinicalEvidenceVerifier()

    def verify(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        spec: ClaimSpec,
        evidence: Evidence,
    ) -> Evidence:
        del context
        from ecg_agent.schema import Claim as LegacyClaim

        legacy_evidence = _evidence_to_legacy(evidence)
        relation = self.verifier.verify(
            LegacyClaim(
                claim_id=candidate.claim_id,
                protocol_step=spec.clinical_group,
                source=candidate.source,
                r1_present=candidate.asserted,
                r1_value=candidate.asserted,
                r1_text=candidate.source_text,
                present_in_behavior_report=candidate.asserted,
            ),
            legacy_evidence,
        )
        return replace(evidence, relation=EvidenceRelation(relation.value))


class LegacyStage4OutcomePolicy:
    """Adapt a mapping of fold -> Stage-4 post-inspection policy."""

    ACTIONS: ClassVar[Mapping[ReportAction, str]] = {
        ReportAction.ADD: "add",
        ReportAction.REVISE: "revise_evidence",
        ReportAction.WITHHOLD: "withhold",
    }

    def __init__(self, policies: Mapping[int, Any]) -> None:
        self.policies = dict(policies)

    def estimate(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        action: ReportAction,
        evidence: Evidence,
    ) -> ValueEstimate | None:
        from ecg_agent.stage3_direct.schema import DirectClaimAction

        fold = int(context.case.metadata.get("fold", -1))
        policy = self.policies.get(fold)
        if policy is None or action not in self.ACTIONS:
            return None
        value = policy.estimate(
            context.case.record_id,
            context.report.text,
            context.report.labels,
            frozenset(context.committed_claims),
            candidate.claim_id,
            DirectClaimAction(self.ACTIONS[action]),
            _evidence_to_legacy(evidence),
            query_fold=fold,
        )
        if value is None:
            return None
        return ValueEstimate(
            mean_return=value.mean_return,
            uncertainty=value.uncertainty,
            safe_return=value.safe_return,
            policy_id=value.model,
            memory_folds=tuple(value.memory_folds),
            query_fold=value.query_fold,
        )


class LegacyStage4Composer:
    ACTIONS: ClassVar[Mapping[ReportAction, str]] = {
        ReportAction.ADD: "add",
        ReportAction.REVISE: "revise_evidence",
        ReportAction.WITHHOLD: "withhold",
    }

    def __init__(self) -> None:
        from ecg_agent.stage3_direct.composer import EvidenceLockedComposer

        self.composer = EvidenceLockedComposer()

    def apply(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        action: ReportAction,
        evidence: Evidence,
    ) -> Report:
        from ecg_agent.stage3_direct.schema import DirectClaimAction

        if action not in self.ACTIONS:
            return context.report
        result = self.composer.compose(
            context.report.text,
            context.report.labels,
            candidate.claim_id,
            DirectClaimAction(self.ACTIONS[action]),
            _evidence_to_legacy(evidence),
        )
        return Report(
            text=result.report,
            labels=tuple(result.labels),
            model_id=context.report.model_id,
            metadata={
                **context.report.metadata,
                "composer": "legacy-stage4-evidence-locked",
            },
        )


def build_stage4_framework(
    *,
    base_rows: Mapping[str, Mapping[str, Any]],
    instruments: Mapping[str, Mapping[str, Any]],
    policies: Mapping[int, Any],
    candidate_claims: Sequence[str],
    config: AgentConfig | None = None,
    presenter: EvidencePresenter | None = None,
) -> ECGFramework:
    """Build a public-SDK runtime backed by the current Stage-4 components."""

    from ecg_agent.registry import CERTIFIED_CLAIMS, CLAIM_REGISTRY

    registry = FrameworkRegistry()
    expert = LegacyStage4Expert(instruments)
    owners = {CLAIM_REGISTRY[claim_id].owner_expert for claim_id in candidate_claims}
    for owner in sorted(owners):
        registry.register_expert(owner, expert)
    for claim_id in candidate_claims:
        legacy = CLAIM_REGISTRY[claim_id]
        actions = [ReportAction.ADD, ReportAction.REVISE, ReportAction.ABSTAIN]
        if claim_id in CERTIFIED_CLAIMS:
            actions.append(ReportAction.WITHHOLD)
        registry.register_claim(
            ClaimSpec(
                claim_id=claim_id,
                clinical_group=legacy.protocol_step,
                expert=legacy.owner_expert,
                aliases=legacy.aliases,
                legal_actions=tuple(actions),
                priority=legacy.importance,
                description=legacy.positive_template,
            )
        )
    return ECGFramework(
        base_model=FrozenMappingBaseModel(base_rows, model_id="frozen-ECG-R1"),
        scanner=LegacyStage4Scanner(candidate_claims),
        planner=PriorityInspectionPlanner(),
        verifier=LegacyStage4Verifier(),
        outcome_policy=LegacyStage4OutcomePolicy(policies),
        composer=LegacyStage4Composer(),
        registry=registry,
        config=config,
        presenter=presenter,
    )
