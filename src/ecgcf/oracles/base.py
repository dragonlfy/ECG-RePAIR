"""Measurement contracts shared by oracle adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ecgcf.record import ECGRecord


@dataclass(frozen=True)
class Measurements:
    """ECG measurements; failure of an individual measurement is `None`."""

    heart_rate_bpm: float | None
    rr_intervals_ms: list[float] | None
    pr_ms: float | None
    qrs_ms: float | None
    qt_ms: float | None
    qtc_ms: float | None
    st_mv: dict[str, float] | None
    t_polarity: dict[str, int] | None
    t_amplitude_mv: dict[str, float] | None
    p_wave_presence: list[bool] | None
    qtc_formula: str
    oracle: str
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""

        return asdict(self)


class Oracle(Protocol):
    """Protocol for an ECG delineation and measurement implementation."""

    name: str

    def measure(self, rec: ECGRecord) -> Measurements | None:
        """Measure a record or return `None` when delineation fails."""
