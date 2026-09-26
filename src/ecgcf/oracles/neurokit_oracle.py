"""NeuroKit2-based delineation oracle."""

from __future__ import annotations

from typing import Any

import numpy as np

from ecgcf.oracles.base import Measurements
from ecgcf.oracles.common import measurements_from_landmarks
from ecgcf.oracles.qrs_energy import qrs_boundaries_from_energy
from ecgcf.record import ECGRecord


def _integer_values(values: Any, *, preserve_missing: bool = False) -> list[int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if preserve_missing:
        return [round(value) if np.isfinite(value) else -1 for value in array]
    return [round(value) for value in array if np.isfinite(value)]


class NeuroKitOracle:
    """Delineate lead II with NeuroKit2, then measure all leads."""

    name = "oracle_neurokit"

    def __init__(self, *, qtc_formula: str = "Bazett") -> None:
        self.qtc_formula = qtc_formula

    def measure(self, rec: ECGRecord) -> Measurements | None:
        """Return measurements, using known synthetic landmarks only in smoke mode."""

        if rec.meta.get("source") == "synthetic":
            raw = rec.meta.get("landmarks")
            if not isinstance(raw, dict):
                return None
            landmarks = {str(key): _integer_values(value) for key, value in raw.items()}
            return measurements_from_landmarks(
                rec,
                landmarks,
                oracle=self.name,
                backend="synthetic_known_landmarks_neurokit_path",
                qtc_formula=self.qtc_formula,
            )
        try:
            import neurokit2 as nk
        except ImportError:
            return None
        try:
            cleaned = nk.ecg_clean(
                rec.signal[1].astype(float), rec.fs, method="neurokit"
            )
            _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=rec.fs)
            r_peaks = _integer_values(peak_info["ECG_R_Peaks"])
            qrs_on, qrs_off = qrs_boundaries_from_energy(rec, r_peaks)
            _, waves = nk.ecg_delineate(
                cleaned,
                r_peaks,
                sampling_rate=rec.fs,
                method="dwt",
                show=False,
            )
            landmarks = {
                "r_peak": r_peaks,
                "p_on": _integer_values(
                    waves.get("ECG_P_Onsets", []), preserve_missing=True
                ),
                "qrs_on": qrs_on,
                "qrs_off": qrs_off,
                "t_peak": _integer_values(
                    waves.get("ECG_T_Peaks", []), preserve_missing=True
                ),
                "t_off": _integer_values(
                    waves.get("ECG_T_Offsets", []), preserve_missing=True
                ),
            }
            return measurements_from_landmarks(
                rec,
                landmarks,
                oracle=self.name,
                backend="neurokit2_dwt",
                qtc_formula=self.qtc_formula,
                diagnostics={"n_r_peaks": len(r_peaks)},
            )
        except (
            KeyError,
            ValueError,
            TypeError,
            IndexError,
            RuntimeError,
            ZeroDivisionError,
        ):
            return None
