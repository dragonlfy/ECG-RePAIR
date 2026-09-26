"""Evidence-first, outcome-calibrated ECG Agent runtime."""

from __future__ import annotations

from typing import Any

from .config import AgentConfig
from .graph import EvidenceGraph
from .protocols import (
    BaseECGModel,
    ClaimScanner,
    ClinicalVerifier,
    EvidencePresenter,
    InspectionPlanner,
    OutcomePolicy,
    ReportComposer,
)
from .registry import FrameworkRegistry
from .schema import (
    AgentResult,
    ClaimCandidate,
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

FRAMEWORK_VERSION = "0.1.0"


class ECGFramework:
    """Compose replaceable ECG components under one auditable lifecycle."""

    def __init__(
        self,
        *,
        base_model: BaseECGModel,
        scanner: ClaimScanner,
        planner: InspectionPlanner,
        verifier: ClinicalVerifier,
        outcome_policy: OutcomePolicy,
        composer: ReportComposer,
        registry: FrameworkRegistry,
        config: AgentConfig | None = None,
        presenter: EvidencePresenter | None = None,
    ) -> None:
        self.base_model = base_model
        self.scanner = scanner
        self.planner = planner
        self.verifier = verifier
        self.outcome_policy = outcome_policy
        self.composer = composer
        self.registry = registry
        self.presenter = presenter
        self.config = config or AgentConfig()
        self.config.validate()
        self.registry.validate()

    def run(self, case: ECGCase, base_report: Report | None = None) -> AgentResult:
        trace: list[TraceEvent] = []
        decisions: list[Decision] = []
        evidence_cache: dict[str, Evidence] = {}
        graph = EvidenceGraph(self.registry.claims)
        finalized: set[str] = set()
        committed: list[str] = []
        inspections = 0
        event_step = 0

        def emit(phase: TracePhase, component: Any, **details: Any) -> None:
            nonlocal event_step
            trace.append(
                TraceEvent(
                    step=event_step,
                    phase=phase,
                    component=component.__class__.__name__,
                    details=jsonable(details),
                )
            )
            event_step += 1

        def apply_presentation() -> None:
            nonlocal report
            if self.presenter is None:
                return
            context = RunContext(
                case=case,
                report=report,
                step=len(committed),
                committed_claims=tuple(committed),
                inspected_claims=tuple(sorted(evidence_cache)),
            )
            before_labels = report.labels
            presented = self.presenter.present(
                context,
                tuple(evidence_cache[key] for key in sorted(evidence_cache)),
                graph.snapshot(),
            )
            if not isinstance(presented, Report):
                raise TypeError("presenter must return ecg_framework.Report")
            if presented.labels != before_labels:
                raise ValueError("presenter violated the diagnosis lock")
            report = presented
            emit(
                TracePhase.PRESENT,
                self.presenter,
                diagnosis_locked=True,
                labels=list(report.labels),
            )

        try:
            report = base_report or self.base_model.infer(case)
            base = report
            emit(
                TracePhase.BASE_MODEL,
                self.base_model,
                model_id=report.model_id,
                labels=list(report.labels),
            )
            stopped_reason = "no_positive_safe_action"
            while len(committed) < self.config.max_committed_actions:
                context = RunContext(
                    case=case,
                    report=report,
                    step=len(committed),
                    committed_claims=tuple(committed),
                    inspected_claims=tuple(sorted(evidence_cache)),
                )
                candidates = [
                    item
                    for item in self.scanner.scan(context, self.registry.claims)
                    if item.claim_id not in finalized
                ]
                unknown = sorted(
                    {item.claim_id for item in candidates} - set(self.registry.claims)
                )
                if unknown:
                    raise KeyError(
                        f"scanner emitted unregistered claims: {', '.join(unknown)}"
                    )
                emit(
                    TracePhase.SCAN,
                    self.scanner,
                    candidate_claims=[item.claim_id for item in candidates],
                )
                ranked = tuple(self.planner.rank(context, candidates))
                emit(
                    TracePhase.PLAN,
                    self.planner,
                    ranked_claims=[item.claim_id for item in ranked],
                )
                for candidate in ranked:
                    if candidate.claim_id in evidence_cache:
                        continue
                    if inspections >= self.config.max_inspections:
                        break
                    spec = self.registry.claims[candidate.claim_id]
                    expert = self.registry.experts[spec.expert]
                    raw_evidence = expert.inspect(context, candidate, spec)
                    if raw_evidence.claim_id != candidate.claim_id:
                        raise ValueError(
                            f"expert {spec.expert} returned evidence for "
                            f"{raw_evidence.claim_id}, expected {candidate.claim_id}"
                        )
                    inspections += 1
                    emit(
                        TracePhase.INSPECT,
                        expert,
                        claim_id=candidate.claim_id,
                        evidence=raw_evidence,
                    )
                    verified = self.verifier.verify(
                        context, candidate, spec, raw_evidence
                    )
                    evidence_cache[candidate.claim_id] = verified
                    graph_change = graph.add_evidence(verified)
                    emit(
                        TracePhase.VERIFY,
                        self.verifier,
                        claim_id=candidate.claim_id,
                        relation=verified.relation,
                        reliability=verified.reliability,
                        graph_change=graph_change,
                    )

                valued: list[
                    tuple[
                        float,
                        float,
                        float,
                        ClaimCandidate,
                        ReportAction,
                        Evidence,
                        ValueEstimate,
                    ]
                ] = []
                rejected: list[Decision] = []
                for candidate in ranked:
                    evidence = evidence_cache.get(candidate.claim_id)
                    if evidence is None:
                        continue
                    spec = self.registry.claims[candidate.claim_id]
                    action = self._action(candidate, evidence)
                    if action not in spec.legal_actions:
                        rejected.append(
                            Decision(
                                candidate.claim_id,
                                ReportAction.ABSTAIN,
                                f"{action.value} is not legal for this claim",
                                evidence,
                            )
                        )
                        continue
                    estimate = self.outcome_policy.estimate(
                        context, candidate, action, evidence
                    )
                    emit(
                        TracePhase.VALUE,
                        self.outcome_policy,
                        claim_id=candidate.claim_id,
                        action=action,
                        estimate=estimate,
                    )
                    if estimate is None:
                        rejected.append(
                            Decision(
                                candidate.claim_id,
                                ReportAction.ABSTAIN,
                                "no outcome estimate",
                                evidence,
                            )
                        )
                        continue
                    valued.append(
                        (
                            estimate.safe_return,
                            estimate.mean_return,
                            candidate.priority,
                            candidate,
                            action,
                            evidence,
                            estimate,
                        )
                    )

                positive = [
                    item
                    for item in valued
                    if item[0] > self.config.min_safe_return
                    and item[4] is not ReportAction.ABSTAIN
                ]
                if not positive:
                    abstentions = [
                        *rejected,
                        *(
                            Decision(
                                item[3].claim_id,
                                ReportAction.ABSTAIN,
                                "safe outcome value is not positive",
                                item[5],
                                item[6],
                            )
                            for item in valued
                        ),
                    ]
                    decisions.extend(abstentions)
                    for decision in abstentions:
                        graph.set_decision(decision)
                    apply_presentation()
                    emit(
                        TracePhase.STOP,
                        self,
                        reason="no_positive_safe_action",
                    )
                    break

                selected = max(
                    positive,
                    key=lambda item: (item[0], item[1], item[2], item[3].claim_id),
                )
                _, _, _, candidate, action, evidence, estimate = selected
                next_report = self.composer.apply(context, candidate, action, evidence)
                if not isinstance(next_report, Report):
                    raise TypeError("composer must return ecg_framework.Report")
                report = next_report
                finalized.add(candidate.claim_id)
                committed.append(candidate.claim_id)
                decision = Decision(
                    candidate.claim_id,
                    action,
                    "highest positive conservative outcome value",
                    evidence,
                    estimate,
                )
                decisions.append(decision)
                graph.set_decision(decision)
                emit(
                    TracePhase.COMMIT,
                    self.composer,
                    claim_id=candidate.claim_id,
                    action=action,
                    safe_return=estimate.safe_return,
                )
                if action is ReportAction.WITHHOLD and self.config.stop_after_withhold:
                    stopped_reason = "withhold_committed"
                    apply_presentation()
                    emit(TracePhase.STOP, self, reason=stopped_reason)
                    break
            else:
                stopped_reason = "action_budget_exhausted"
                apply_presentation()
                emit(TracePhase.STOP, self, reason=stopped_reason)
        except Exception as error:
            if not self.config.fail_closed:
                raise
            if "report" not in locals():
                report = base_report or Report("", model_id="failed-base-model")
                base = report
            stopped_reason = "component_error_fail_closed"
            emit(
                TracePhase.ERROR,
                self,
                error_type=type(error).__name__,
                error=str(error),
            )
            emit(TracePhase.STOP, self, reason=stopped_reason)

        return AgentResult(
            record_id=case.record_id,
            base_report=base,
            final_report=report,
            decisions=tuple(decisions),
            evidence=tuple(evidence_cache[key] for key in sorted(evidence_cache)),
            trace=tuple(trace),
            graph=graph.snapshot(),
            inspections=inspections,
            committed_actions=len(committed),
            stopped_reason=stopped_reason,
            framework_version=FRAMEWORK_VERSION,
        )

    @staticmethod
    def _action(candidate: ClaimCandidate, evidence: Evidence) -> ReportAction:
        if evidence.relation is EvidenceRelation.SUPPORTED:
            return ReportAction.REVISE if candidate.asserted else ReportAction.ADD
        if evidence.relation is EvidenceRelation.CONTRADICTED and candidate.asserted:
            return ReportAction.WITHHOLD
        return ReportAction.ABSTAIN
