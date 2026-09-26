"""Typed clinical evidence graph used by the Stage-5 report renderer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MeasurementRef:
    """One measured quantity and its provenance inside the ECG tool record."""

    path: str
    value: float | bool | str
    unit: str | None = None
    threshold: str | None = None


@dataclass(frozen=True)
class EvidenceCard:
    """A clinically coherent group of measurements rendered as one evidence item."""

    card_id: str
    domain: str
    statement: str
    measurements: tuple[MeasurementRef, ...]
    lead_set: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()
    claim_id: str | None = None
    reliability: float = 1.0
    criterion: str | None = None


@dataclass(frozen=True)
class ClinicalEvidenceGraph:
    """Evidence admitted for one report plus explicit local abstentions."""

    record_id: str
    diagnoses: tuple[str, ...]
    selected_claims: tuple[str, ...]
    cards: tuple[EvidenceCard, ...]
    abstentions: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable audit record."""

        return asdict(self)
