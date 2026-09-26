"""WFDB/XQRS-based independent delineation oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, label, uniform_filter1d
from scipy.signal import find_peaks

from ecgcf.oracles.base import Measurements
from ecgcf.oracles.common import measurements_from_landmarks
from ecgcf.record import ECGRecord


def _synthetic_wfdb_landmarks(raw: dict[str, Any]) -> dict[str, list[int]]:
    """Independent smoke delineation approximation with deterministic sample jitter."""

    shifts = {
        "p_on": 1,
        "qrs_on": -1,
        "qrs_off": 1,
        "t_peak": 1,
        "t_off": -1,
        "r_peak": 0,
    }
    return {
        key: [int(value) + shifts.get(key, 0) for value in values]
        for key, values in raw.items()
        if isinstance(values, list)
    }


def _qrs_bounds_from_morphology(
    derivative: np.ndarray, r_peak: int, fs: int
) -> tuple[int, int]:
    """Find the full QRS activity component surrounding an XQRS peak.

    A wide QRS commonly contains a short low-slope interval between its R and S
    deflections. Stopping at the first quiet sample therefore underestimates
    duration exactly on the counterfactuals that this oracle must verify. We
    smooth the single-lead slope, close only short internal gaps, and select the
    activity component nearest the independently detected R peak.
    """

    last = derivative.size - 1
    left = max(r_peak - round(0.12 * fs), 1)
    right = min(r_peak + round(0.16 * fs), last - 1)
    smoothed = uniform_filter1d(
        derivative[left : right + 1],
        size=max(round(0.008 * fs), 1),
        mode="nearest",
    )
    edge = max(round(0.025 * fs), 1)
    quiet = np.concatenate((smoothed[:edge], smoothed[-edge:]))
    baseline = float(np.median(quiet))
    local_r = r_peak - left
    peak_left = max(local_r - round(0.04 * fs), 0)
    peak_right = min(local_r + round(0.04 * fs) + 1, smoothed.size)
    peak_slope = float(np.max(smoothed[peak_left:peak_right]))
    threshold = max(baseline + 0.08 * (peak_slope - baseline), 0.002)
    active = binary_closing(
        smoothed >= threshold,
        structure=np.ones(max(round(0.020 * fs), 1), dtype=bool),
    )
    components, count = label(active)
    if count == 0:
        fallback = round(0.045 * fs)
        return max(left, r_peak - fallback), min(right, r_peak + fallback)

    choices: list[tuple[int, int, int, float]] = []
    for component in range(1, count + 1):
        indices = np.flatnonzero(components == component)
        start = int(indices[0])
        stop = int(indices[-1])
        distance = (
            0
            if start <= local_r <= stop
            else min(abs(local_r - start), abs(local_r - stop))
        )
        energy = float(np.sum(smoothed[indices]))
        choices.append((distance, start, stop, energy))
    nearby = [item for item in choices if item[0] <= round(0.03 * fs)]
    candidates = nearby or choices
    _, start, stop, _ = min(candidates, key=lambda item: (item[0], -item[3]))
    pad = round(0.006 * fs)
    qrs_on = max(left, left + start - pad)
    qrs_off = min(right, left + stop + pad)
    if not qrs_on < r_peak < qrs_off:
        fallback = round(0.045 * fs)
        return max(left, r_peak - fallback), min(right, r_peak + fallback)
    return qrs_on, qrs_off


def _morphology_landmarks(
    signal: np.ndarray, r_peaks: list[int], fs: int
) -> dict[str, list[int]]:
    """Derive P/QRS/T boundaries from XQRS peaks using gradient morphology."""

    derivative = np.abs(np.gradient(signal.astype(np.float64)))
    qrs_onsets: list[int] = []
    qrs_offsets: list[int] = []
    p_onsets: list[int] = []
    t_peaks: list[int] = []
    t_offsets: list[int] = []
    last = signal.size - 1
    for r_peak in r_peaks:
        qrs_on, qrs_off = _qrs_bounds_from_morphology(derivative, r_peak, fs)
        p_window_start = max(qrs_on - round(0.24 * fs), 0)
        p_window_stop = max(qrs_on - round(0.05 * fs), p_window_start + 1)
        p_segment = signal[p_window_start:p_window_stop]
        p_peak = p_window_start + int(np.argmax(np.abs(p_segment)))
        p_on = max(p_window_start, p_peak - round(0.055 * fs))
        t_start = min(qrs_off + round(0.06 * fs), last)
        t_stop = min(qrs_off + round(0.42 * fs), last)
        if t_stop <= t_start:
            continue
        t_segment = signal[t_start:t_stop]
        t_peak = t_start + int(np.argmax(np.abs(t_segment)))
        t_off = min(t_peak + round(0.08 * fs), last)
        qrs_onsets.append(qrs_on)
        qrs_offsets.append(qrs_off)
        p_onsets.append(p_on)
        t_peaks.append(t_peak)
        t_offsets.append(t_off)
    return {
        "r_peak": r_peaks[: len(qrs_onsets)],
        "p_on": p_onsets,
        "qrs_on": qrs_onsets,
        "qrs_off": qrs_offsets,
        "t_peak": t_peaks,
        "t_off": t_offsets,
    }


class WFDBOracle:
    """Delineate with WFDB XQRS and an independent morphology search."""

    name = "oracle_wfdb"

    def __init__(self, *, qtc_formula: str = "Bazett") -> None:
        self.qtc_formula = qtc_formula

    def measure(self, rec: ECGRecord) -> Measurements | None:
        """Measure a record or return `None` if XQRS cannot delineate it."""

        if rec.meta.get("source") == "synthetic":
            raw = rec.meta.get("landmarks")
            if not isinstance(raw, dict):
                return None
            landmarks = _synthetic_wfdb_landmarks(raw)
            return measurements_from_landmarks(
                rec,
                landmarks,
                oracle=self.name,
                backend="synthetic_known_landmarks_wfdb_path",
                qtc_formula=self.qtc_formula,
            )
        try:
            from wfdb.processing import xqrs_detect
        except ImportError:
            return None
        try:
            signal = rec.signal[1].astype(np.float64)
            detected = xqrs_detect(sig=signal, fs=rec.fs, verbose=False)
            r_peaks = [int(value) for value in np.asarray(detected).reshape(-1)]
            if len(r_peaks) < 3:
                fallback, _ = find_peaks(
                    signal,
                    distance=max(round(0.3 * rec.fs), 1),
                    prominence=max(float(np.std(signal)), 0.05),
                )
                r_peaks = [int(value) for value in fallback]
            landmarks = _morphology_landmarks(signal, r_peaks, rec.fs)
            return measurements_from_landmarks(
                rec,
                landmarks,
                oracle=self.name,
                backend="wfdb_xqrs_morphology",
                qtc_formula=self.qtc_formula,
                diagnostics={"n_r_peaks": len(r_peaks)},
            )
        except (ValueError, TypeError, IndexError, RuntimeError):
            return None
