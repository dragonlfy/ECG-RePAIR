"""Public API for ECG-Agent Framework."""

from .config import AgentConfig
from .defaults import (
    EvidenceAppendComposer,
    FixedOutcomePolicy,
    PredicateVerifier,
    PriorityInspectionPlanner,
    RegistryClaimScanner,
    SeedReportModel,
)
from .evaluation import AuditReport, audit_result, paired_summary
from .graph import EvidenceGraph
from .memory import (
    GroupedOutcomePolicy,
    OutcomeDataset,
    OutcomeRecord,
    crossfit_grouped_policies,
)
from .protocols import (
    BaseECGModel,
    ClaimScanner,
    ClinicalVerifier,
    ECGExpert,
    EvidencePresenter,
    InspectionPlanner,
    OutcomePolicy,
    ReportComposer,
)
from .registry import ComponentRegistry, FrameworkRegistry, components
from .runtime import FRAMEWORK_VERSION, ECGFramework
from .schema import (
    AgentResult,
    ClaimCandidate,
    ClaimSpec,
    Decision,
    ECGCase,
    Evidence,
    EvidenceRelation,
    Report,
    ReportAction,
    RunContext,
    TraceEvent,
    TracePhase,
    ValueEstimate,
    jsonable,
)

__version__ = FRAMEWORK_VERSION


def _register_builtin_components() -> None:
    components.register("base_model", "seed_report", SeedReportModel, replace=True)
    components.register("scanner", "registry", RegistryClaimScanner, replace=True)
    components.register("planner", "priority", PriorityInspectionPlanner, replace=True)
    components.register("verifier", "predicate", PredicateVerifier, replace=True)
    components.register("outcome_policy", "fixed", FixedOutcomePolicy, replace=True)
    components.register(
        "outcome_policy", "grouped_memory", GroupedOutcomePolicy, replace=True
    )
    components.register(
        "composer", "evidence_append", EvidenceAppendComposer, replace=True
    )


_register_builtin_components()

__all__ = [
    "AgentConfig",
    "AgentResult",
    "AuditReport",
    "BaseECGModel",
    "ClaimCandidate",
    "ClaimScanner",
    "ClaimSpec",
    "ClinicalVerifier",
    "ComponentRegistry",
    "Decision",
    "ECGCase",
    "ECGExpert",
    "ECGFramework",
    "Evidence",
    "EvidenceAppendComposer",
    "EvidenceGraph",
    "EvidencePresenter",
    "EvidenceRelation",
    "FixedOutcomePolicy",
    "FrameworkRegistry",
    "GroupedOutcomePolicy",
    "InspectionPlanner",
    "OutcomeDataset",
    "OutcomePolicy",
    "OutcomeRecord",
    "PredicateVerifier",
    "PriorityInspectionPlanner",
    "RegistryClaimScanner",
    "Report",
    "ReportAction",
    "ReportComposer",
    "RunContext",
    "SeedReportModel",
    "TraceEvent",
    "TracePhase",
    "ValueEstimate",
    "audit_result",
    "components",
    "crossfit_grouped_policies",
    "jsonable",
    "paired_summary",
]
