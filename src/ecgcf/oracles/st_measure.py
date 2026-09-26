"""Shared J+80 ms ST amplitude measurement disclosed as a dependence."""

from __future__ import annotations

import numpy as np

from ecgcf.record import ECGRecord


def measure_st_j80(
    rec: ECGRecord,
    *,
    qrs_offsets: list[int],
    qrs_onsets: list[int],
    p_onsets: list[int],
) -> dict[str, float] | None:
    """Measure median per-lead J+80 amplitude relative to PR baseline."""

    delay = round(0.080 * rec.fs)
    margin = max(round(0.012 * rec.fs), 1)
    per_lead: dict[str, list[float]] = {lead: [] for lead in rec.leads}
    last = rec.signal.shape[1] - 1
    for p_on, qrs_on, qrs_off in zip(p_onsets, qrs_onsets, qrs_offsets, strict=False):
        sample = qrs_off + delay
        if not 0 <= p_on < qrs_on <= last or sample > last:
            continue
        baseline_start = max(p_on + margin, qrs_on - round(0.055 * rec.fs))
        baseline_stop = qrs_on - margin
        if baseline_stop <= baseline_start:
            continue
        baseline = np.median(
            rec.signal[:, baseline_start:baseline_stop].astype(np.float64), axis=1
        )
        amplitude = rec.signal[:, sample].astype(np.float64) - baseline
        for lead, value in zip(rec.leads, amplitude, strict=False):
            per_lead[lead].append(float(value))
    if not any(per_lead.values()):
        return None
    return {
        lead: float(np.median(values)) for lead, values in per_lead.items() if values
    }
