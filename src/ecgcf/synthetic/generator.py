"""Analytic multi-lead ECG generator with known beat landmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from ecgcf.record import CANONICAL_LEADS, ECGRecord


@dataclass(frozen=True)
class SyntheticTruth:
    """Known generator-level measurements."""

    heart_rate_bpm: float
    pr_ms: float
    qrs_ms: float
    qt_ms: float
    qtc_ms: float
    rr_ms: float
    st_mv: dict[str, float]
    t_polarity: dict[str, int]
    qtc_formula: str = "Bazett"


_LEAD_SCALE = np.array(
    [0.82, 1.0, 0.62, -0.78, 0.34, 0.72, -0.42, 0.12, 0.55, 0.92, 1.08, 0.96],
    dtype=np.float64,
)
_T_SCALE = np.array(
    [0.72, 1.0, 0.74, -0.62, 0.42, 0.82, -0.25, 0.28, 0.70, 1.0, 1.05, 0.92],
    dtype=np.float64,
)


def _gaussian(
    time: NDArray[np.float64], center: float, width: float
) -> NDArray[np.float64]:
    return np.exp(-0.5 * ((time - center) / width) ** 2)


def generate_ecg(
    *,
    record_id: str,
    rng: np.random.Generator,
    fs: int = 500,
    duration_seconds: float = 10.0,
    heart_rate_bpm: float | None = None,
    qrs_ms: float | None = None,
    pr_ms: float | None = None,
    qt_ms: float | None = None,
    noise_mv: float = 0.006,
) -> ECGRecord:
    """Generate a 12-lead ECG as a sum of smooth P/QRS/T basis functions."""

    heart_rate = float(heart_rate_bpm or rng.uniform(58.0, 88.0))
    qrs_duration = float(qrs_ms or rng.uniform(82.0, 108.0))
    pr_duration = float(pr_ms or rng.uniform(148.0, 188.0))
    qt_duration = float(qt_ms or rng.uniform(350.0, 410.0))
    rr_seconds = 60.0 / heart_rate
    n_samples = round(duration_seconds * fs)
    time = np.arange(n_samples, dtype=np.float64) / fs
    signal = np.zeros((12, n_samples), dtype=np.float64)
    baseline_phase = rng.uniform(0.0, 2.0 * np.pi)
    signal += 0.012 * np.sin(2.0 * np.pi * 0.22 * time + baseline_phase)

    r_times = np.arange(0.8, duration_seconds - 0.45, rr_seconds)
    landmarks: dict[str, list[int]] = {
        key: []
        for key in (
            "p_on",
            "p_peak",
            "p_off",
            "qrs_on",
            "r_peak",
            "qrs_off",
            "t_on",
            "t_peak",
            "t_off",
            "tp_start",
            "tp_end",
        )
    }
    qrs_seconds = qrs_duration / 1000.0
    pr_seconds = pr_duration / 1000.0
    qt_seconds = qt_duration / 1000.0
    for r_time in r_times:
        qrs_on = r_time - 0.38 * qrs_seconds
        qrs_off = qrs_on + qrs_seconds
        p_on = qrs_on - pr_seconds
        p_peak = p_on + 0.42 * (pr_seconds - 0.045)
        p_off = qrs_on - 0.045
        t_on = qrs_off + 0.110
        t_peak = qrs_on + 0.72 * qt_seconds
        t_off = qrs_on + qt_seconds
        next_r = r_time + rr_seconds
        next_qrs_on = next_r - 0.38 * qrs_seconds
        tp_start = t_off + 0.04
        tp_end = max(tp_start, next_qrs_on - pr_seconds - 0.025)
        for key, value in (
            ("p_on", p_on),
            ("p_peak", p_peak),
            ("p_off", p_off),
            ("qrs_on", qrs_on),
            ("r_peak", r_time),
            ("qrs_off", qrs_off),
            ("t_on", t_on),
            ("t_peak", t_peak),
            ("t_off", t_off),
            ("tp_start", tp_start),
            ("tp_end", tp_end),
        ):
            landmarks[key].append(round(value * fs))

        p = 0.12 * _gaussian(time, p_peak, 0.028)
        q = -0.16 * _gaussian(time, r_time - 0.018, max(qrs_seconds * 0.10, 0.007))
        r = 1.05 * _gaussian(time, r_time, max(qrs_seconds * 0.075, 0.006))
        s = -0.28 * _gaussian(time, r_time + 0.026, max(qrs_seconds * 0.105, 0.008))
        t_wave = 0.30 * _gaussian(time, t_peak, 0.060)
        for lead_index in range(12):
            signal[lead_index] += _LEAD_SCALE[lead_index] * (p + q + r + s)
            signal[lead_index] += _T_SCALE[lead_index] * t_wave

    signal += rng.normal(0.0, noise_mv, size=signal.shape)
    qtc_ms = qt_duration / np.sqrt(rr_seconds)
    st_values = {lead: 0.0 for lead in CANONICAL_LEADS}
    t_polarity = {
        lead: int(np.sign(_T_SCALE[index]))
        for index, lead in enumerate(CANONICAL_LEADS)
    }
    truth = SyntheticTruth(
        heart_rate_bpm=heart_rate,
        pr_ms=pr_duration,
        qrs_ms=qrs_duration,
        qt_ms=qt_duration,
        qtc_ms=float(qtc_ms),
        rr_ms=rr_seconds * 1000.0,
        st_mv=st_values,
        t_polarity=t_polarity,
    )
    clipped_landmarks = {
        key: [int(np.clip(value, 0, n_samples - 1)) for value in values]
        for key, values in landmarks.items()
    }
    meta = {
        "source": "synthetic",
        "synthetic_truth": asdict(truth),
        "landmarks": clipped_landmarks,
        "canonical_units": "mV",
    }
    return ECGRecord(signal.astype(np.float32), fs, CANONICAL_LEADS, record_id, meta)


def generate_records(
    n: int, *, seed: int, fs: int = 500, duration_seconds: float = 10.0
) -> list[ECGRecord]:
    """Generate `n` records from a single explicit RNG stream."""

    rng = np.random.default_rng(seed)
    return [
        generate_ecg(
            record_id=f"synthetic-{index:04d}",
            rng=rng,
            fs=fs,
            duration_seconds=duration_seconds,
        )
        for index in range(n)
    ]
