"""Reference-free QRS voltage criteria from independent beat detectors.

ECG-R1's third protocol step asserts chamber hypertrophy by name and cites the
Sokolow-Lyon and aVL criteria to justify the assertion. Checking whether that
assertion is conditioned on the waveform needs the criteria themselves, which
the delineation oracles do not report, so they are computed here in the same
dual-backend shape as :mod:`ecgcf.oracles.frontal_axis`: two independently
detected beat sets, and a result marked agreed only when both agree on the
clinical category and their amplitudes are close.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ecgcf.record import ECGRecord

#: Sokolow-Lyon threshold for left ventricular hypertrophy, in millivolts.
SOKOLOW_LYON_MV = 3.5
#: The protocol's third voltage criterion, which we had omitted.
R_LEAD_I_MV = 1.5
#: R wave in aVL threshold for left ventricular hypertrophy, in millivolts.
R_AVL_MV = 1.1
#: Amplitude spread above which the two beat sets are not treated as agreeing.
AMPLITUDE_TOLERANCE_MV = 0.5


def _beat_amplitudes(
    filtered: np.ndarray, peaks: np.ndarray, fs: int, lead_index: dict[str, int]
) -> dict[str, float] | None:
    """Return median per-lead R height and S depth over detected beats."""

    left = round(0.060 * fs)
    right = round(0.080 * fs)
    valid = np.asarray(peaks, dtype=int)
    valid = valid[(valid >= left) & (valid + right < filtered.shape[1])]
    if valid.size < 3:
        return None
    windows = np.stack(
        [filtered[:, peak - left : peak + right] for peak in valid], axis=0
    )
    highs = np.median(np.max(windows, axis=2), axis=0)
    lows = np.median(np.min(windows, axis=2), axis=0)
    return {
        "s_v1_mv": float(-lows[lead_index["V1"]]),
        "r_v5_mv": float(highs[lead_index["V5"]]),
        "r_v6_mv": float(highs[lead_index["V6"]]),
        "r_avl_mv": float(highs[lead_index["aVL"]]),
        "r_lead_i_mv": float(highs[lead_index["I"]]) if "I" in lead_index else None,
        "n_beats": float(valid.size),
    }


def _criteria(amplitudes: dict[str, float]) -> dict[str, Any]:
    sokolow = amplitudes["s_v1_mv"] + max(amplitudes["r_v5_mv"], amplitudes["r_v6_mv"])
    return {
        "sokolow_lyon_mv": sokolow,
        "r_avl_mv": amplitudes["r_avl_mv"],
        "r_lead_i_mv": amplitudes.get("r_lead_i_mv"),
        "lvh_by_voltage": bool(
            sokolow > SOKOLOW_LYON_MV
            or amplitudes["r_avl_mv"] > R_AVL_MV
            or (amplitudes.get("r_lead_i_mv") or 0.0) > R_LEAD_I_MV
        ),
    }


def qrs_voltage_evidence(record: ECGRecord) -> dict[str, Any] | None:
    """Estimate the voltage hypertrophy criteria, or `None` when unmeasurable."""

    lead_index = {lead: index for index, lead in enumerate(record.leads)}
    required = {"II", "V1", "V5", "V6", "aVL"}
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

    lead_ii = record.signal[lead_index["II"]].astype(np.float64)
    per_backend: dict[str, dict[str, Any]] = {}
    try:
        import neurokit2 as nk

        cleaned = nk.ecg_clean(lead_ii, record.fs, method="neurokit")
        _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=record.fs)
        amplitudes = _beat_amplitudes(
            filtered, np.asarray(peak_info["ECG_R_Peaks"]), record.fs, lead_index
        )
        if amplitudes is not None:
            per_backend["neurokit"] = _criteria(amplitudes)
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

        amplitudes = _beat_amplitudes(
            filtered,
            np.asarray(xqrs_detect(sig=lead_ii, fs=record.fs, verbose=False)),
            record.fs,
            lead_index,
        )
        if amplitudes is not None:
            per_backend["wfdb_xqrs"] = _criteria(amplitudes)
    except (ImportError, ValueError, TypeError, IndexError, RuntimeError):
        pass
    if not per_backend:
        return None

    sokolow = [value["sokolow_lyon_mv"] for value in per_backend.values()]
    avl = [value["r_avl_mv"] for value in per_backend.values()]
    lead_i = [
        value["r_lead_i_mv"]
        for value in per_backend.values()
        if value.get("r_lead_i_mv") is not None
    ]
    categories = {value["lvh_by_voltage"] for value in per_backend.values()}
    spreads = [
        float(np.max(sokolow) - np.min(sokolow)),
        float(np.max(avl) - np.min(avl)),
    ]
    if len(lead_i) >= 2:
        spreads.append(float(np.max(lead_i) - np.min(lead_i)))
    spread = max(spreads)
    mean_lead_i = float(np.mean(lead_i)) if lead_i else None
    return {
        "sokolow_lyon_mv": float(np.mean(sokolow)),
        "r_avl_mv": float(np.mean(avl)),
        "r_lead_i_mv": mean_lead_i,
        "lvh_by_voltage": bool(np.mean(sokolow) > SOKOLOW_LYON_MV)
        or bool(np.mean(avl) > R_AVL_MV)
        or bool(mean_lead_i is not None and mean_lead_i > R_LEAD_I_MV),
        "agree": len(per_backend) >= 2
        and len(categories) == 1
        and spread <= AMPLITUDE_TOLERANCE_MV,
        "spread_mv": spread,
        "n_backends": len(per_backend),
        "backend_values": per_backend,
    }
