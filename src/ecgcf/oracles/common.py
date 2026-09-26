"""Formula-only helpers operating on oracle-specific landmarks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ecgcf.oracles.base import Measurements
from ecgcf.oracles.st_measure import measure_st_j80
from ecgcf.record import ECGRecord


def _valid(values: list[int], last: int) -> list[int]:
    return [value for value in values if 0 <= value <= last]


def measurements_from_landmarks(
    rec: ECGRecord,
    landmarks: dict[str, list[int]],
    *,
    oracle: str,
    backend: str,
    qtc_formula: str = "Bazett",
    diagnostics: dict[str, Any] | None = None,
) -> Measurements | None:
    """Compute formula-level measurements from one oracle's landmark output."""

    needed = ("r_peak", "p_on", "qrs_on", "qrs_off", "t_peak", "t_off")
    if any(not landmarks.get(key) for key in needed):
        return None
    last = rec.signal.shape[1] - 1
    r_peaks = np.asarray(_valid(landmarks["r_peak"], last), dtype=np.float64)
    if r_peaks.size < 3:
        return None
    rr_ms_array = np.diff(r_peaks) * 1000.0 / rec.fs
    median_rr_ms = float(np.median(rr_ms_array))
    heart_rate = 60000.0 / median_rr_ms if median_rr_ms > 0 else None

    def median_interval(start_key: str, stop_key: str) -> float | None:
        pairs = [
            (start, stop)
            for start, stop in zip(
                landmarks[start_key], landmarks[stop_key], strict=False
            )
            if 0 <= start < stop <= last
        ]
        if not pairs:
            return None
        samples = [stop - start for start, stop in pairs]
        return float(np.median(samples) * 1000.0 / rec.fs)

    pr_ms = median_interval("p_on", "qrs_on")
    qrs_ms = median_interval("qrs_on", "qrs_off")
    qt_ms = median_interval("qrs_on", "t_off")
    qtc_ms: float | None = None
    if qt_ms is not None and median_rr_ms > 0:
        rr_seconds = median_rr_ms / 1000.0
        if qtc_formula.lower() == "bazett":
            qtc_ms = float(qt_ms / np.sqrt(rr_seconds))
        elif qtc_formula.lower() == "fridericia":
            qtc_ms = float(qt_ms / np.cbrt(rr_seconds))
        else:
            raise ValueError(f"unknown QTc formula: {qtc_formula}")

    st_mv = measure_st_j80(
        rec,
        qrs_offsets=landmarks["qrs_off"],
        qrs_onsets=landmarks["qrs_on"],
        p_onsets=landmarks["p_on"],
    )
    t_polarity: dict[str, int] = {}
    t_amplitude: dict[str, float] = {}
    p_presence: list[bool] = []
    for lead_index, lead in enumerate(rec.leads):
        t_values: list[float] = []
        for p_on, qrs_on, t_peak in zip(
            landmarks["p_on"], landmarks["qrs_on"], landmarks["t_peak"], strict=False
        ):
            if not (0 <= p_on < qrs_on <= last and 0 <= t_peak <= last):
                continue
            pr_start = max(p_on, qrs_on - round(0.055 * rec.fs))
            pr_stop = max(pr_start + 1, qrs_on - round(0.012 * rec.fs))
            baseline = float(np.median(rec.signal[lead_index, pr_start:pr_stop]))
            t_values.append(float(rec.signal[lead_index, t_peak] - baseline))
        if t_values:
            amplitude = float(np.median(t_values))
            t_amplitude[lead] = amplitude
            t_polarity[lead] = int(np.sign(amplitude))
    for p_on, qrs_on in zip(landmarks["p_on"], landmarks["qrs_on"], strict=False):
        if not 0 <= p_on < qrs_on <= last:
            continue
        segment = rec.signal[1, p_on:qrs_on]
        edge = max(min(round(0.02 * rec.fs), segment.size // 3), 1)
        baseline = float(np.median(np.concatenate((segment[:edge], segment[-edge:]))))
        p_presence.append(float(np.max(segment) - baseline) >= 0.025)
    return Measurements(
        heart_rate_bpm=heart_rate,
        rr_intervals_ms=rr_ms_array.astype(float).tolist(),
        pr_ms=pr_ms,
        qrs_ms=qrs_ms,
        qt_ms=qt_ms,
        qtc_ms=qtc_ms,
        st_mv=st_mv,
        t_polarity=t_polarity or None,
        t_amplitude_mv=t_amplitude or None,
        p_wave_presence=p_presence or None,
        qtc_formula=qtc_formula,
        oracle=oracle,
        backend=backend,
        diagnostics=dict(diagnostics or {}),
    )
