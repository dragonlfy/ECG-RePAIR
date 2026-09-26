"""Network-free minimal ECG-Agent Framework example."""

from __future__ import annotations

import json

from ecg_framework import (
    AgentConfig,
    ClaimSpec,
    ECGCase,
    ECGFramework,
    Evidence,
    EvidenceAppendComposer,
    FixedOutcomePolicy,
    FrameworkRegistry,
    PredicateVerifier,
    PriorityInspectionPlanner,
    RegistryClaimScanner,
    ReportAction,
    SeedReportModel,
)


class HeartRateExpert:
    def inspect(self, context, candidate, spec):
        del spec
        heart_rate = float(context.case.metadata["heart_rate_bpm"])
        return Evidence(
            evidence_id=f"{context.case.record_id}:heart-rate",
            claim_id=candidate.claim_id,
            expert="heart_rate",
            measurement="heart_rate_bpm",
            value=heart_rate,
            unit="bpm",
            reliability=0.95,
        )


def build_agent() -> ECGFramework:
    registry = FrameworkRegistry()
    registry.register_expert("heart_rate", HeartRateExpert())
    registry.register_claim(
        ClaimSpec(
            claim_id="bradycardia",
            clinical_group="rate_rhythm",
            expert="heart_rate",
            aliases=("bradycardia", "slow rate"),
            legal_actions=(ReportAction.ADD, ReportAction.ABSTAIN),
            priority=0.8,
        )
    )
    return ECGFramework(
        base_model=SeedReportModel(model_id="example-frozen-model"),
        scanner=RegistryClaimScanner(),
        planner=PriorityInspectionPlanner(),
        verifier=PredicateVerifier({"bradycardia": lambda value: float(value) < 60.0}),
        outcome_policy=FixedOutcomePolicy(safe_return=1.0),
        composer=EvidenceAppendComposer(),
        registry=registry,
        config=AgentConfig(max_inspections=1, max_committed_actions=1),
    )


if __name__ == "__main__":
    case = ECGCase(
        record_id="demo-001",
        metadata={
            "base_report": "Sinus rhythm.",
            "base_labels": ["sinus rhythm"],
            "heart_rate_bpm": 52.0,
        },
    )
    print(json.dumps(build_agent().run(case).as_dict(), indent=2))
