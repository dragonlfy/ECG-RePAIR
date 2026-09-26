"""Fabricated inputs and outcomes for exercising the repair API, not evaluation."""

from ecg_agent.schema import Evidence, VerificationStatus
from ecg_agent.stage3_direct.evidence_advantage import EvidenceActionRow
from ecg_agent.stage3_direct.schema import DirectClaimAction

from .pipeline import repair_reports


def demo_inputs():
    reports = [{
        "record_id": "synthetic-query", "fold": 0,
        "report": "<think>Sinus rhythm. The PR interval is normal.</think>"
                  "<answer>Sinus rhythm</answer>",
        "labels": ["sinus rhythm"],
    }]
    # Hand-authored measurements test interfaces only; these are not measurements
    # from a patient, an attached image, or a diagnostic benchmark.
    instruments = [{
        "record_id": "synthetic-query", "instrument_valid": True,
        "oracle_agreement_rate": 1.0,
        "scalar": {"pr_ms": {"value": 224.0, "agree": True, "spread": 6.0}},
        "morphology": {"p_wave_measurable": True},
    }]
    memory = []
    for i in range(12):
        evidence = Evidence(
            evidence_id=f"synthetic-history-{i}:pr", claim_id="pr_prolonged",
            expert="intervals", measurement="pr_ms", value=220.0 + i,
            unit="ms", lead_set=("II",), relation=VerificationStatus.SUPPORTED,
            reliability=1.0, backend_agreement=1.0,
            provenance={"p_wave_measurable": True, "spread": 6.0},
        )
        memory.append(EvidenceActionRow(
            record_id=f"synthetic-history-{i}", fold=1 + i % 4,
            report="", labels=("sinus rhythm",), selected_claims=frozenset(),
            claim_id="pr_prolonged", action=DirectClaimAction.ADD,
            evidence=evidence, reward=4.0,
        ))
    return reports, instruments, memory


def run_demo():
    reports, instruments, memory = demo_inputs()
    return {
        "notice": "Synthetic API demo: fabricated observations and historical rewards; not clinical results.",
        "results": repair_reports(reports, instruments, memory, candidate_claims=("pr_prolonged",)),
    }
