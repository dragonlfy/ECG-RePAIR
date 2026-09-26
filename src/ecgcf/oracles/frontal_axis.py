"""Reference-free frontal QRS axis estimation from independent beat detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ecgcf.record import ECGRecord


def _axis_category(angle: float) -> str:
    if -30.0 <= angle < 90.0:
        return "normal"
    if -90.0 <= angle < -30.0:
        return "left_axis_deviation"
    if 90.0 <= angle <= 180.0:
        return "right_axis_deviation"
    return "extreme_axis_deviation"


def _estimate_from_peaks(
    filtered: np.ndarray,
    peaks: np.ndarray,
    fs: int,
    lead_i: int,
    lead_avf: int,
) -> tuple[float, int] | None:
    left = round(0.060 * fs)
    right = round(0.080 * fs)
    valid = np.asarray(peaks, dtype=int)
    valid = valid[(valid >= left) & (valid + right < filtered.shape[1])]
    if valid.size < 3:
        return None
    vectors: list[tuple[float, float]] = []
    for peak in valid:
        segment = filtered[:, peak - left : peak + right]
        net_i = float(np.max(segment[lead_i]) + np.min(segment[lead_i]))
        net_avf = float(np.max(segment[lead_avf]) + np.min(segment[lead_avf]))
        vectors.append((net_i, net_avf))
    median_i, median_avf = np.median(np.asarray(vectors), axis=0)
    if np.hypot(median_i, median_avf) < 0.02:
        return None
    angle = float(np.degrees(np.arctan2(median_avf, median_i)))
    return angle, int(valid.size)


def _circular_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _circular_mean(values: list[float]) -> float:
    radians = np.radians(values)
    return float(
        np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    )


def frontal_qrs_axis_evidence(record: ECGRecord) -> dict[str, Any] | None:
    """Estimate frontal QRS axis with NeuroKit and WFDB peak paths.

    The net QRS deflection is the sum of the largest positive and negative
    deflections in a fixed beat window. The result is marked agreed only when
    both independently detected beat sets yield the same clinical category and
    their circular angular spread is no more than 15 degrees.
    """

    lead_index = {lead: index for index, lead in enumerate(record.leads)}
    required = {"I", "II", "aVF"}
    if not required.issubset(lead_index) or record.signal.shape[1] < record.fs:
        return None
    try:
        filtered = sosfiltfilt(
            butter(3, [0.5, 40.0], btype="bandpass", fs=record.fs, output="sos"),
            record.signal.astype(np.float64),
            axis=1,
        )
    except ValueError:
        return None

    estimates: dict[str, float] = {}
    beat_counts: dict[str, int] = {}
    lead_ii = record.signal[lead_index["II"]].astype(np.float64)
    try:
        import neurokit2 as nk

        cleaned = nk.ecg_clean(lead_ii, record.fs, method="neurokit")
        _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=record.fs)
        estimate = _estimate_from_peaks(
            filtered,
            np.asarray(peak_info["ECG_R_Peaks"]),
            record.fs,
            lead_index["I"],
            lead_index["aVF"],
        )
        if estimate is not None:
            estimates["neurokit"] = estimate[0]
            beat_counts["neurokit"] = estimate[1]
    except (
        ImportError,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        RuntimeError,
        ZeroDivisionError,
    ):
        pass
    try:
        from wfdb.processing import xqrs_detect

        estimate = _estimate_from_peaks(
            filtered,
            np.asarray(xqrs_detect(sig=lead_ii, fs=record.fs, verbose=False)),
            record.fs,
            lead_index["I"],
            lead_index["aVF"],
        )
        if estimate is not None:
            estimates["wfdb_xqrs"] = estimate[0]
            beat_counts["wfdb_xqrs"] = estimate[1]
    except (ImportError, ValueError, TypeError, IndexError, RuntimeError):
        pass
    if not estimates:
        return None

    values = list(estimates.values())
    category_by_backend = {
        backend: _axis_category(angle) for backend, angle in estimates.items()
    }
    spread = _circular_difference(values[0], values[1]) if len(values) >= 2 else None
    agreed = (
        len(values) >= 2
        and len(set(category_by_backend.values())) == 1
        and spread is not None
        and spread <= 15.0
    )
    angle = _circular_mean(values)
    return {
        "value_degrees": round(angle, 1),
        "category": _axis_category(angle),
        "agree": agreed,
        "spread_degrees": round(spread, 1) if spread is not None else None,
        "n_backends": len(values),
        "backend_values_degrees": {
            name: round(value, 1) for name, value in estimates.items()
        },
        "beats_by_backend": beat_counts,
        "method": "median_net_qrs_deflection_i_avf",
    }
