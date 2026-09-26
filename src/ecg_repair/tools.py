"""Compute the instrument schema used by the clinical verifier from an ECG."""

from __future__ import annotations

import numpy as np

from ecgcf.oracles import ConsensusOracle, NeuroKitOracle, WFDBOracle
from ecgcf.oracles.frontal_axis import frontal_qrs_axis_evidence
from ecgcf.oracles.morphology import morphology_evidence
from ecgcf.oracles.voltage import qrs_voltage_evidence
from ecgcf.record import ECGRecord

TOLERANCES = {
    "heart_rate_bpm": 8.0, "rr_ms": 40.0, "pr_ms": 30.0,
    "qrs_ms": 20.0, "qt_ms": 35.0, "qtc_ms": 40.0,
    "st_mv": 0.08, "t_amplitude_mv": 0.12,
}


def _series_summary(measurement, *, rhythm: bool):
    values = {}
    for backend, series in (measurement.oracle_values if measurement else {}).items():
        array = np.asarray(series, dtype=float)
        if not array.size or not np.isfinite(array).all():
            continue
        if rhythm:
            if array.size < 3 or array.mean() <= 0:
                continue
            values[backend] = float(array.std() / array.mean())
        else:
            values[backend] = float(array.mean())
    categories = {value > 0.12 if rhythm else value >= 0.5 for value in values.values()}
    return {
        "value": float(np.mean(list(values.values()))) if values else None,
        "agree": len(values) >= 2 and len(categories) == 1,
        "backend_values": values,
    }


def measure_ecg(record: ECGRecord) -> dict:
    """Measure a waveform; never use synthetic landmark metadata as evidence."""
    # Import explicitly so a missing dependency is not silently reported as a
    # patient with missing measurements by optional legacy measurement paths.
    import neurokit2  # noqa: F401
    import wfdb  # noqa: F401

    record = record.replace(meta={})
    result = {"record_id": record.record_id, "instrument_valid": False}
    measured = ConsensusOracle([NeuroKitOracle(), WFDBOracle()], TOLERANCES).measure(record)
    if measured is None:
        result["failure_reason"] = "all ECG measurement backends failed"
        return result

    def scalar(name):
        value = measured.get(name)
        return {
            "value": value.value if value else None,
            "agree": bool(value.agree) if value else False,
            "spread": value.spread if value else None,
        }

    result.update({
        "instrument_valid": True,
        "oracle_agreement_rate": measured.agreement_rate,
        "oracle_backends": measured.oracle_backends,
        "scalar": {key: scalar(key) for key in (
            "heart_rate_bpm", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms",
        )},
        "st_mv": scalar("st_mv"), "t_amplitude_mv": scalar("t_amplitude_mv"),
        "rr_cv": _series_summary(measured.get("rr_intervals_ms"), rhythm=True),
        "p_wave": _series_summary(measured.get("p_wave_presence"), rhythm=False),
        "frontal_axis": frontal_qrs_axis_evidence(record),
        "voltage": qrs_voltage_evidence(record), "morphology": morphology_evidence(record),
    })
    return result
