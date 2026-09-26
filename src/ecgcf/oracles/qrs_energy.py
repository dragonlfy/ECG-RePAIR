"""Multi-lead energy refinement for QRS boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d

from ecgcf.record import ECGRecord


def qrs_boundaries_from_energy(
    rec: ECGRecord,
    r_peaks: Sequence[int],
    *,
    threshold_fraction: float = 0.05,
    smooth_ms: float = 12.0,
    hold_ms: float = 10.0,
    pad_ms: float = 4.0,
) -> tuple[list[int], list[int]]:
    """Refine QRS boundaries around R peaks using all-lead slope energy."""

    gradient = np.gradient(rec.signal.astype(np.float64), axis=1)
    energy = np.sqrt(np.mean(gradient * gradient, axis=0))
    smooth_samples = max(round(smooth_ms * rec.fs / 1000.0), 1)
    energy = uniform_filter1d(energy, size=smooth_samples, mode="nearest")
    radius = max(round(0.13 * rec.fs), 1)
    quiet_gap = max(round(0.09 * rec.fs), 1)
    hold = max(round(hold_ms * rec.fs / 1000.0), 1)
    pad = max(round(pad_ms * rec.fs / 1000.0), 0)
    last = energy.size - 1
    onsets: list[int] = []
    offsets: list[int] = []

    for raw_r_peak in r_peaks:
        r_peak = int(raw_r_peak)
        left = max(0, r_peak - radius)
        right = min(last, r_peak + radius)
        quiet = np.concatenate(
            (
                energy[left : max(left, r_peak - quiet_gap)],
                energy[min(right, r_peak + quiet_gap) : right + 1],
            )
        )
        baseline = (
            float(np.median(quiet))
            if quiet.size
            else float(np.median(energy[left : right + 1]))
        )
        peak_left = max(left, r_peak - round(0.04 * rec.fs))
        peak_right = min(right + 1, r_peak + round(0.04 * rec.fs))
        peak_energy = float(np.max(energy[peak_left:peak_right]))
        threshold = baseline + threshold_fraction * (peak_energy - baseline)

        onset = left
        for index in range(r_peak - hold, left - 1, -1):
            if np.all(energy[index : index + hold] <= threshold):
                onset = index + hold
                break
        offset = right
        for index in range(r_peak, max(r_peak, right - hold + 1)):
            if np.all(energy[index : index + hold] <= threshold):
                offset = index
                break
        onset = max(left, onset - pad)
        offset = min(right, offset + pad)
        if not onset < r_peak < offset:
            onset = max(left, r_peak - round(0.045 * rec.fs))
            offset = min(right, r_peak + round(0.045 * rec.fs))
        onsets.append(onset)
        offsets.append(offset)
    return onsets, offsets
