from __future__ import annotations

from pathlib import Path

import pytest

from ecg_framework import (
    AgentConfig,
    ClaimCandidate,
    ClaimSpec,
    ComponentRegistry,
    ECGCase,
    ECGFramework,
    Evidence,
    EvidenceAppendComposer,
    FixedOutcomePolicy,
    FrameworkRegistry,
    OutcomeDataset,
    OutcomeRecord,
    PredicateVerifier,
    PriorityInspectionPlanner,
    RegistryClaimScanner,
    Report,
    ReportAction,
    RunContext,
    SeedReportModel,
    TracePhase,
    audit_result,
    crossfit_grouped_policies,
    paired_summary,
)
from ecg_framework.adapters import build_stage4_framework, build_stage5_framework


class HeartRateExpert:
    def inspect(self, context, candidate, spec):
        del spec
        return Evidence(
            evidence_id=f"{context.case.record_id}:hr",
            claim_id=candidate.claim_id,
            expert="heart_rate",
            measurement="heart_rate_bpm",
            value=context.case.metadata["heart_rate_bpm"],
            unit="bpm",
            reliability=0.95,
        )


def _agent(safe_return: float = 1.0, presenter=None) -> ECGFramework:
    registry = FrameworkRegistry()
    registry.register_expert("heart_rate", HeartRateExpert())
    registry.register_claim(
        ClaimSpec(
            claim_id="bradycardia",
            clinical_group="rate_rhythm",
            expert="heart_rate",
            aliases=("bradycardia",),
            legal_actions=(ReportAction.ADD, ReportAction.ABSTAIN),
            priority=0.8,
        )
    )
    return ECGFramework(
        base_model=SeedReportModel(),
        scanner=RegistryClaimScanner(),
        planner=PriorityInspectionPlanner(),
        verifier=PredicateVerifier({"bradycardia": lambda value: float(value) < 60.0}),
        outcome_policy=FixedOutcomePolicy(safe_return),
        composer=EvidenceAppendComposer(),
        registry=registry,
        config=AgentConfig(max_inspections=1, max_committed_actions=1),
        presenter=presenter,
    )


def _case() -> ECGCase:
    return ECGCase(
        "case-1",
        metadata={
            "base_report": "Sinus rhythm.",
            "base_labels": ["sinus rhythm"],
            "heart_rate_bpm": 52.0,
        },
    )


def test_framework_runs_complete_evidence_first_lifecycle() -> None:
    result = _agent().run(_case())

    assert result.changed
    assert result.inspections == 1
    assert result.committed_actions == 1
    assert result.decisions[0].action is ReportAction.ADD
    assert "bradycardia" in result.final_report.labels
    assert result.graph["evidence_count"] == 1
    assert result.graph["claims"]["bradycardia"]["status"] == "supported"
    assert audit_result(result, config=_agent().config).passed
    phases = [event.phase for event in result.trace]
    assert phases == [
        TracePhase.BASE_MODEL,
        TracePhase.SCAN,
        TracePhase.PLAN,
        TracePhase.INSPECT,
        TracePhase.VERIFY,
        TracePhase.VALUE,
        TracePhase.COMMIT,
        TracePhase.STOP,
    ]


def test_framework_abstains_when_safe_value_is_not_positive() -> None:
    result = _agent(safe_return=0.0).run(_case())

    assert not result.changed
    assert result.committed_actions == 0
    assert result.decisions[0].action is ReportAction.ABSTAIN
    assert result.stopped_reason == "no_positive_safe_action"


def test_framework_fails_closed_if_presenter_changes_diagnosis() -> None:
    class UnsafePresenter:
        def present(self, context, evidence, graph):
            del evidence, graph
            return Report(
                context.report.text,
                labels=(*context.report.labels, "invented diagnosis"),
                model_id=context.report.model_id,
            )

    result = _agent(presenter=UnsafePresenter()).run(_case())

    assert result.stopped_reason == "component_error_fail_closed"
    assert "invented diagnosis" not in result.final_report.labels
    assert [event.phase for event in result.trace][-2:] == [
        TracePhase.ERROR,
        TracePhase.STOP,
    ]


def test_registry_rejects_claim_without_expert() -> None:
    registry = FrameworkRegistry()
    registry.register_claim(ClaimSpec("axis_left", "axis", "missing", priority=0.7))
    with pytest.raises(ValueError, match="missing experts"):
        registry.validate()


def test_outcome_memory_excludes_query_fold() -> None:
    dataset = OutcomeDataset(
        (
            OutcomeRecord("a", 0, "bradycardia", ReportAction.ADD, 10.0),
            OutcomeRecord("b", 0, "bradycardia", ReportAction.ADD, 0.0),
            OutcomeRecord("c", 1, "bradycardia", ReportAction.ADD, 20.0),
            OutcomeRecord("d", 1, "bradycardia", ReportAction.ADD, 10.0),
        )
    )
    policy = crossfit_grouped_policies(dataset, risk_z=0.0, shrinkage=0.0)[0]
    context = RunContext(_case(), _agent().base_model.infer(_case()), 0)
    value = policy.estimate(
        context,
        ClaimCandidate("bradycardia", False, "test"),
        ReportAction.ADD,
        Evidence("e", "bradycardia", "heart_rate", "hr", 52.0),
    )

    assert value is not None
    assert value.mean_return == 15.0
    assert value.query_fold == 0
    assert value.memory_folds == (1,)


def test_config_and_component_plugin_contract(tmp_path: Path) -> None:
    path = tmp_path / "framework.yaml"
    path.write_text(
        "framework:\n  framework_id: test\n  max_inspections: 2\n",
        encoding="utf-8",
    )
    config = AgentConfig.from_yaml(path)
    assert config.framework_id == "test"
    assert config.max_inspections == 2

    plugins = ComponentRegistry()
    plugins.register("expert", "heart_rate", HeartRateExpert)
    assert isinstance(plugins.create("expert", "heart_rate"), HeartRateExpert)
    assert paired_summary({"a": 50.0}, {"a": 100.0})["delta"] == 50.0
    with pytest.raises(ValueError, match="keys differ"):
        paired_summary({"a": 50.0}, {"b": 100.0})


def test_current_stage4_components_are_available_through_public_sdk() -> None:
    from ecg_agent.stage3_direct.schema import ActionValue

    class PositivePolicy:
        def estimate(
            self,
            record_id,
            report,
            labels,
            selected_claims,
            claim_id,
            action,
            evidence,
            *,
            query_fold=-1,
        ):
            del report, labels, action, evidence
            return ActionValue(
                record_id,
                selected_claims,
                claim_id,
                10.0,
                0.0,
                10.0,
                query_fold,
                (1, 2, 3, 4),
                "test-policy",
            )

    base = {
        "case-1": {
            "prediction_raw_base": "Sinus rhythm.",
            "prediction_labels_base": ["sinus rhythm"],
        }
    }
    instruments = {
        "case-1": {
            "record_id": "case-1",
            "instrument_valid": True,
            "oracle_agreement_rate": 1.0,
            "scalar": {
                "heart_rate_bpm": {
                    "value": 52.0,
                    "agree": True,
                    "spread": 1.0,
                }
            },
            "p_wave": {"value": 1.0, "agree": True},
            "morphology": {"p_wave_measurable": True},
        }
    }
    agent = build_stage4_framework(
        base_rows=base,
        instruments=instruments,
        policies={0: PositivePolicy()},
        candidate_claims=("bradycardia",),
        config=AgentConfig(max_inspections=1, max_committed_actions=1),
    )
    result = agent.run(ECGCase("case-1", metadata={"fold": 0}))

    assert result.changed
    assert result.decisions[0].claim_id == "bradycardia"
    assert result.decisions[0].value is not None
    assert result.decisions[0].value.memory_folds == (1, 2, 3, 4)
    assert audit_result(result, config=agent.config, query_fold=0).passed

    stage5 = build_stage5_framework(
        base_rows=base,
        instruments=instruments,
        policies={0: PositivePolicy()},
        candidate_claims=("bradycardia",),
        config=AgentConfig(max_inspections=1, max_committed_actions=1),
    )
    presented = stage5.run(ECGCase("case-1", metadata={"fold": 0}))
    assert presented.final_report.labels == result.final_report.labels
    assert "Diagnosis-linked ECG evidence" in presented.final_report.text
    assert [event.phase for event in presented.trace][-2:] == [
        TracePhase.PRESENT,
        TracePhase.STOP,
    ]
    assert audit_result(presented, config=stage5.config, query_fold=0).passed
