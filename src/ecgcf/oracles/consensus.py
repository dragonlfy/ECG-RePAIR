"""Tolerance-aware consensus across two independently delineated oracles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ecgcf.oracles.base import Measurements, Oracle
from ecgcf.record import ECGRecord


@dataclass(frozen=True)
class ConsensusMeasurement:
    """Consensus value with agreement and denominator provenance."""

    value: Any
    agree: bool
    spread: float | dict[str, float] | None
    n_oracles: int
    oracle_values: dict[str, Any]


@dataclass(frozen=True)
class ConsensusMeasurements:
    """All consensus fields and overall agreement rate."""

    measurements: dict[str, ConsensusMeasurement]
    agreement_rate: float
    n_oracles_succeeded: int
    oracle_backends: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""

        return asdict(self)

    def get(self, name: str) -> ConsensusMeasurement | None:
        """Return a named consensus measurement."""

        return self.measurements.get(name)


def _numeric_consensus(
    values: dict[str, float], tolerance: float
) -> ConsensusMeasurement:
    array = np.asarray(list(values.values()), dtype=np.float64)
    spread = float(np.max(array) - np.min(array)) if array.size > 1 else 0.0
    return ConsensusMeasurement(
        value=float(np.mean(array)),
        agree=array.size >= 2 and spread <= tolerance,
        spread=spread,
        n_oracles=int(array.size),
        oracle_values=values,
    )


def _dictionary_consensus(
    values: dict[str, dict[str, float | int]], tolerance: float
) -> ConsensusMeasurement:
    common_keys = set.intersection(*(set(value) for value in values.values()))
    consensus: dict[str, float] = {}
    spread: dict[str, float] = {}
    per_key_agreement: list[bool] = []
    for key in sorted(common_keys):
        array = np.asarray([float(value[key]) for value in values.values()])
        consensus[key] = float(np.mean(array))
        spread[key] = float(np.max(array) - np.min(array))
        per_key_agreement.append(spread[key] <= tolerance)
    return ConsensusMeasurement(
        value=consensus or None,
        agree=len(values) >= 2 and bool(per_key_agreement) and all(per_key_agreement),
        spread=spread or None,
        n_oracles=len(values),
        oracle_values=values,
    )


def _list_consensus(
    values: dict[str, list[float] | list[bool]], tolerance: float
) -> ConsensusMeasurement:
    medians = {
        name: float(np.median(np.asarray(value, dtype=np.float64)))
        for name, value in values.items()
        if value
    }
    result = _numeric_consensus(medians, tolerance)
    return ConsensusMeasurement(
        value=result.value,
        agree=result.agree,
        spread=result.spread,
        n_oracles=result.n_oracles,
        oracle_values=values,
    )


class ConsensusOracle:
    """Run two or more oracles and compare all supported measurements."""

    name = "oracle_consensus"

    def __init__(self, oracles: list[Oracle], tolerances: dict[str, float]) -> None:
        if len(oracles) < 2:
            raise ValueError("consensus requires at least two oracles")
        self.oracles = oracles
        self.tolerances = tolerances

    def measure(self, rec: ECGRecord) -> ConsensusMeasurements | None:
        """Return consensus or `None` when every oracle fails."""

        outputs: dict[str, Measurements] = {}
        for oracle in self.oracles:
            measured = oracle.measure(rec)
            if measured is not None:
                outputs[oracle.name] = measured
        if not outputs:
            return None
        fields = (
            "heart_rate_bpm",
            "rr_intervals_ms",
            "pr_ms",
            "qrs_ms",
            "qt_ms",
            "qtc_ms",
            "st_mv",
            "t_polarity",
            "t_amplitude_mv",
            "p_wave_presence",
        )
        tolerance_names = {
            "rr_intervals_ms": "rr_ms",
            "st_mv": "st_mv",
            "t_polarity": "t_amplitude_mv",
            "t_amplitude_mv": "t_amplitude_mv",
            "p_wave_presence": "p_wave_presence",
        }
        consensus: dict[str, ConsensusMeasurement] = {}
        for field in fields:
            present = {
                name: getattr(output, field)
                for name, output in outputs.items()
                if getattr(output, field) is not None
            }
            if not present:
                consensus[field] = ConsensusMeasurement(None, False, None, 0, {})
                continue
            tolerance_key = tolerance_names.get(field, field)
            tolerance = float(
                self.tolerances.get(
                    tolerance_key, 0.0 if field == "t_polarity" else 1e-6
                )
            )
            first = next(iter(present.values()))
            if isinstance(first, dict):
                consensus[field] = _dictionary_consensus(present, tolerance)
            elif isinstance(first, list):
                consensus[field] = _list_consensus(present, tolerance)
            else:
                numeric = {name: float(value) for name, value in present.items()}
                consensus[field] = _numeric_consensus(numeric, tolerance)
        evaluable = [item for item in consensus.values() if item.n_oracles >= 2]
        agreement_rate = (
            float(np.mean([item.agree for item in evaluable])) if evaluable else 0.0
        )
        return ConsensusMeasurements(
            measurements=consensus,
            agreement_rate=agreement_rate,
            n_oracles_succeeded=len(outputs),
            oracle_backends={name: value.backend for name, value in outputs.items()},
        )
