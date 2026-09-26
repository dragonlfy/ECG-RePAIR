"""Replaceable component interfaces for ECG-Agent Framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from .schema import (
    ClaimCandidate,
    ClaimSpec,
    ECGCase,
    Evidence,
    Report,
    ReportAction,
    RunContext,
    ValueEstimate,
)


@runtime_checkable
class BaseECGModel(Protocol):
    def infer(self, case: ECGCase) -> Report: ...


@runtime_checkable
class ClaimScanner(Protocol):
    def scan(
        self, context: RunContext, claims: Mapping[str, ClaimSpec]
    ) -> Sequence[ClaimCandidate]: ...


@runtime_checkable
class InspectionPlanner(Protocol):
    def rank(
        self, context: RunContext, candidates: Sequence[ClaimCandidate]
    ) -> Sequence[ClaimCandidate]: ...


@runtime_checkable
class ECGExpert(Protocol):
    def inspect(
        self, context: RunContext, candidate: ClaimCandidate, spec: ClaimSpec
    ) -> Evidence: ...


@runtime_checkable
class ClinicalVerifier(Protocol):
    def verify(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        spec: ClaimSpec,
        evidence: Evidence,
    ) -> Evidence: ...


@runtime_checkable
class OutcomePolicy(Protocol):
    def estimate(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        action: ReportAction,
        evidence: Evidence,
    ) -> ValueEstimate | None: ...


@runtime_checkable
class ReportComposer(Protocol):
    def apply(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        action: ReportAction,
        evidence: Evidence,
    ) -> Report: ...


@runtime_checkable
class EvidencePresenter(Protocol):
    """Render diagnosis-linked evidence without changing diagnostic labels."""

    def present(
        self,
        context: RunContext,
        evidence: Sequence[Evidence],
        graph: Mapping[str, object],
    ) -> Report: ...
