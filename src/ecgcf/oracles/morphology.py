"""Reference-free QRS and T wave morphology criteria from independent detectors.

The conditioning measurement can only ask about findings the instrument can
measure, and the delineation oracles report intervals and ST levels but not
morphology. The findings ECG-R1 misses most often are morphological --
pathologic Q waves, T wave inversion, low voltage, poor R wave progression --
so they are computed here, in the same dual-backend shape as
:mod:`ecgcf.oracles.frontal_axis` and :mod:`ecgcf.oracles.voltage`: two
independently detected beat sets, and a finding marked agreed only when both
sets put it on the same side of its clinical threshold.

Nothing here decides whether a finding is worth acting on. A criterion that
measures badly simply fails to win adjudication against the reference report
during calibration and is never allowed to name a diagnosis, so adding a weak
instrument costs coverage rather than correctness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ecgcf.record import ECGRecord

#: Contiguous lead territories. A morphological finding in a single lead is
#: noise as often as it is disease, so every criterion here requires two.
TERRITORIES: dict[str, tuple[str, ...]] = {
    "inferior": ("II", "III", "aVF"),
    "lateral": ("I", "aVL", "V5", "V6"),
    "anteroseptal": ("V1", "V2", "V3", "V4"),
}
#: aVR faces away from the ventricles, so its Q wave and T inversion are normal.
EXCLUDED_LEADS = frozenset({"aVR"})

#: The base model's corpus was generated against a monograph protocol, released
#: with its code, which calls a Q wave pathologic at "> 0.03 s wide OR > 0.1 mV
#: deep". We adopt those numbers rather than our own so that a disagreement
#: between the model and the instrument is a disagreement about the tracing and
#: not about the definition.
Q_DURATION_MS = 30.0
#: An absolute depth, as the protocol states it, rather than a fraction of R.
Q_DEPTH_MV = 0.1
#: Peak-to-peak QRS below this in every limb lead is low voltage.
LOW_VOLTAGE_LIMB_MV = 0.5
#: Peak-to-peak QRS below this in every precordial lead is low voltage.
LOW_VOLTAGE_PRECORDIAL_MV = 1.0
#: An R wave in V3 at or below this is the usual poor-progression criterion.
POOR_R_V3_MV = 0.3
#: A P wave at least this wide in lead II is a left atrial abnormality.
P_DURATION_MS = 120.0
#: Terminal force in V1 at or beyond this, in millivolt-seconds, is the other
#: half of the criterion: the classic 1 mm by 40 ms box, in the units measured.
P_TERMINAL_FORCE_MVS = 0.004
#: Below this the P wave is not distinguishable from the baseline.
P_MIN_AMPLITUDE_MV = 0.03
#: Outside this span the detected deflection is not a P wave. Fibrillatory
#: baseline has no discrete atrial deflection, and measuring one anyway reports
#: a 200 ms "P wave" that means nothing; such records get no verdict instead.
P_PLAUSIBLE_MS = (60.0, 200.0)
#: A T wave at or below this is counted as inverted.
T_INVERSION_MV = -0.1
#: Leads where an inverted T wave is a finding rather than a normal variant.
T_WAVE_LEADS = ("I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6")

# High-specificity morphology profile selected on 21,790 PTB-XL records with
# cardiologist statement codes, then frozen before application to MIMIC-IV.
# The legacy textbook flags above remain in the cache for backwards-compatible
# analyses; these stricter flags are the Agent's action-facing measurements.
STRICT_Q_DURATION_MS = 40.0
STRICT_Q_DEPTH_MV = 0.15
STRICT_Q_DEPTH_FRACTION = 0.25
STRICT_Q_MIN_R_MV = 0.5
STRICT_Q_TERRITORY_LEADS = 3
STRICT_T_DEPTH_MV = 0.5
STRICT_T_MIN_R_MV = 0.5
STRICT_T_TERRITORY_LEADS = 3
STRICT_POOR_R_V3_MV = 0.1

_LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
_PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")


def _filtered(record: ECGRecord) -> np.ndarray | None:
    try:
        return sosfiltfilt(
            butter(3, [0.5, 40.0], btype="bandpass", fs=record.fs, output="sos"),
            record.signal.astype(np.float64),
            axis=1,
        )
    except ValueError:
        return None


def _peaks(record: ECGRecord) -> dict[str, np.ndarray]:
    """Return R peak positions from each detector that succeeds."""

    lead_index = {lead: index for index, lead in enumerate(record.leads)}
    lead_ii = record.signal[lead_index["II"]].astype(np.float64)
    found: dict[str, np.ndarray] = {}
    try:
        import neurokit2 as nk

        cleaned = nk.ecg_clean(lead_ii, record.fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=record.fs)
        found["neurokit"] = np.asarray(info["ECG_R_Peaks"], dtype=int)
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

        found["wfdb_xqrs"] = np.asarray(
            xqrs_detect(sig=lead_ii, fs=record.fs, verbose=False), dtype=int
        )
    except (ImportError, ValueError, TypeError, IndexError, RuntimeError):
        pass
    return {name: value for name, value in found.items() if value.size >= 3}


def _per_lead(
    filtered: np.ndarray, peaks: np.ndarray, fs: int, leads: tuple[str, ...]
) -> dict[str, dict[str, float]] | None:
    """Measure Q, R and T geometry per lead, taking the median across beats."""

    before = round(0.080 * fs)
    after = round(0.100 * fs)
    baseline_from = round(0.200 * fs)
    baseline_to = round(0.120 * fs)
    t_from = round(0.100 * fs)
    t_to = round(0.400 * fs)
    valid = peaks[(peaks >= baseline_from) & (peaks + t_to < filtered.shape[1])]
    if valid.size < 3:
        return None

    measured: dict[str, dict[str, list[float]]] = {
        lead: {"q_ms": [], "q_mv": [], "r_mv": [], "pp_mv": [], "t_mv": []}
        for lead in leads
    }
    atrial = {"p_ms": [], "p_terminal_force_mvs": []}
    p_from = round(0.300 * fs)
    p_to = round(0.050 * fs)
    lead_of = {lead: index for index, lead in enumerate(leads)}
    if "II" in lead_of and valid[0] >= p_from:
        for peak in valid:
            if peak < p_from:
                continue
            window = filtered[lead_of["II"], peak - p_from : peak - p_to]
            base = float(np.median(window[: max(1, window.size // 8)]))
            centred = window - base
            extreme = float(np.max(np.abs(centred)))
            if extreme < P_MIN_AMPLITUDE_MV:
                continue
            inside = np.flatnonzero(np.abs(centred) >= 0.2 * extreme)
            if inside.size < 2:
                continue
            onset, offset = int(inside[0]), int(inside[-1])
            atrial["p_ms"].append(1000.0 * (offset - onset) / fs)
            if "V1" not in lead_of:
                continue
            v1 = filtered[lead_of["V1"], peak - p_from : peak - p_to]
            v1 = v1 - float(np.median(v1[: max(1, v1.size // 8)]))
            terminal = v1[onset + (offset - onset) // 2 : offset + 1]
            negative = terminal[terminal < 0.0]
            if negative.size:
                atrial["p_terminal_force_mvs"].append(
                    abs(float(np.min(negative))) * negative.size / fs
                )
    for index, lead in enumerate(leads):
        trace = filtered[index]
        for peak in valid:
            baseline = float(
                np.median(trace[peak - baseline_from : peak - baseline_to])
            )
            window = trace[peak - before : peak + after] - baseline
            r_amplitude = float(np.max(window))
            measured[lead]["r_mv"].append(r_amplitude)
            measured[lead]["pp_mv"].append(r_amplitude - float(np.min(window)))
            # The Q wave is the negative excursion immediately before the R peak.
            rise = window[:before]
            negative = rise < 0.0
            q_samples = 0
            q_depth = 0.0
            for position in range(negative.size - 1, -1, -1):
                if not negative[position]:
                    break
                q_samples += 1
                q_depth = min(q_depth, float(rise[position]))
            measured[lead]["q_ms"].append(1000.0 * q_samples / fs)
            measured[lead]["q_mv"].append(q_depth)
            t_window = trace[peak + t_from : peak + t_to] - baseline
            extreme = t_window[int(np.argmax(np.abs(t_window)))]
            measured[lead]["t_mv"].append(float(extreme))
    result = {
        lead: {key: float(np.median(values)) for key, values in fields.items()}
        for lead, fields in measured.items()
    }
    result["_atrial"] = {
        key: (float(np.median(values)) if values else float("nan"))
        for key, values in atrial.items()
    }
    return result


def _flags(per_lead: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Turn per-lead geometry into the categorical findings a report would state."""

    atrial = per_lead.get("_atrial", {})
    pathologic: list[str] = []
    inverted: list[str] = []
    for lead, values in per_lead.items():
        if lead == "_atrial" or lead in EXCLUDED_LEADS:
            continue
        deep_enough = values["q_mv"] <= -Q_DEPTH_MV
        if values["q_ms"] >= Q_DURATION_MS or deep_enough:
            pathologic.append(lead)
        if lead in T_WAVE_LEADS and values["t_mv"] <= T_INVERSION_MV:
            inverted.append(lead)

    def in_a_territory(leads: list[str]) -> tuple[str | None, int]:
        best: tuple[str | None, int] = (None, 0)
        for name, members in TERRITORIES.items():
            count = sum(lead in leads for lead in members)
            if count > best[1]:
                best = (name, count)
        return best if best[1] >= 2 else (None, best[1])

    q_territory, q_leads = in_a_territory(pathologic)
    t_territory, _ = in_a_territory(inverted)

    strict_q = [
        lead
        for lead, values in per_lead.items()
        if lead != "_atrial"
        and lead not in EXCLUDED_LEADS
        and values["r_mv"] >= STRICT_Q_MIN_R_MV
        and values["q_mv"] <= -STRICT_Q_DEPTH_MV
        and (
            values["q_ms"] >= STRICT_Q_DURATION_MS
            or values["q_mv"] <= -STRICT_Q_DEPTH_FRACTION * values["r_mv"]
        )
    ]
    strict_t = [
        lead
        for lead, values in per_lead.items()
        if lead != "_atrial"
        and lead not in EXCLUDED_LEADS
        and lead != "V1"
        and values["r_mv"] >= STRICT_T_MIN_R_MV
        and values["t_mv"] <= -STRICT_T_DEPTH_MV
    ]

    def strict_territory(leads: list[str], needed: int) -> str | None:
        for name, members in TERRITORIES.items():
            if sum(lead in leads for lead in members) >= needed:
                return name
        return None

    strict_q_territory = strict_territory(strict_q, STRICT_Q_TERRITORY_LEADS)
    strict_t_territory = strict_territory(strict_t, STRICT_T_TERRITORY_LEADS)
    v2_r = per_lead.get("V2", {}).get("r_mv")
    v4_r = per_lead.get("V4", {}).get("r_mv")
    # The depth at which two leads of a territory still qualify. Thresholding it
    # is what a margin does: it asks how far past the clinical line the finding
    # sits, on the same scale for every record.
    t_depth = 0.0
    if t_territory is not None:
        members = [
            per_lead[lead]["t_mv"]
            for lead in TERRITORIES[t_territory]
            if lead in inverted
        ]
        t_depth = sorted(members)[1] if len(members) >= 2 else 0.0

    p_ms = atrial.get("p_ms", float("nan"))
    terminal = atrial.get("p_terminal_force_mvs", float("nan"))
    limb = [per_lead[lead]["pp_mv"] for lead in _LIMB if lead in per_lead]
    precordial = [per_lead[lead]["pp_mv"] for lead in _PRECORDIAL if lead in per_lead]
    r_v3 = per_lead.get("V3", {}).get("r_mv")
    return {
        "pathologic_q_leads": sorted(pathologic),
        "pathologic_q_territory": q_territory,
        "pathologic_q_n_leads": q_leads,
        "pathologic_q": q_territory is not None,
        "pathologic_q_strict_leads": sorted(strict_q),
        "pathologic_q_strict_territory": strict_q_territory,
        "pathologic_q_strict_n_leads": max(
            (
                sum(lead in strict_q for lead in members)
                for members in TERRITORIES.values()
            ),
            default=0,
        ),
        "pathologic_q_strict": strict_q_territory is not None,
        "t_inversion_leads": sorted(inverted),
        "t_inversion_territory": t_territory,
        "t_inversion_depth_mv": t_depth,
        "t_wave_inversion": t_territory is not None,
        "t_inversion_strict_leads": sorted(strict_t),
        "t_inversion_strict_territory": strict_t_territory,
        "t_wave_inversion_strict": strict_t_territory is not None,
        "low_voltage": bool(limb and all(v < LOW_VOLTAGE_LIMB_MV for v in limb))
        or bool(precordial and all(v < LOW_VOLTAGE_PRECORDIAL_MV for v in precordial)),
        "min_limb_pp_mv": min(limb) if limb else None,
        "max_limb_pp_mv": max(limb) if limb else None,
        "max_precordial_pp_mv": max(precordial) if precordial else None,
        "r_v3_mv": r_v3,
        "poor_r_progression": r_v3 is not None and r_v3 <= POOR_R_V3_MV,
        "poor_r_progression_strict": bool(
            r_v3 is not None
            and v2_r is not None
            and v4_r is not None
            and r_v3 <= STRICT_POOR_R_V3_MV
            and not (v2_r < r_v3 < v4_r)
        ),
        "p_duration_ms": None if np.isnan(p_ms) else p_ms,
        "p_terminal_force_mvs": None if np.isnan(terminal) else terminal,
        "p_wave_measurable": bool(
            not np.isnan(p_ms) and P_PLAUSIBLE_MS[0] <= p_ms <= P_PLAUSIBLE_MS[1]
        ),
        "left_atrial_abnormality": bool(
            not np.isnan(p_ms)
            and P_PLAUSIBLE_MS[0] <= p_ms <= P_PLAUSIBLE_MS[1]
            and (
                p_ms >= P_DURATION_MS
                or (not np.isnan(terminal) and terminal >= P_TERMINAL_FORCE_MVS)
            )
        ),
    }


#: The categorical findings both detectors must agree on.
_CATEGORICAL = (
    "pathologic_q",
    "t_wave_inversion",
    "low_voltage",
    "poor_r_progression",
    "left_atrial_abnormality",
    "pathologic_q_strict",
    "t_wave_inversion_strict",
    "poor_r_progression_strict",
)


def morphology_evidence(record: ECGRecord) -> dict[str, Any] | None:
    """Measure QRS and T morphology, or `None` when it cannot be measured."""

    lead_index = {lead: index for index, lead in enumerate(record.leads)}
    if "II" not in lead_index or record.signal.shape[1] < record.fs:
        return None
    filtered = _filtered(record)
    if filtered is None:
        return None

    per_backend: dict[str, dict[str, Any]] = {}
    geometry: dict[str, dict[str, dict[str, float]]] = {}
    for name, peaks in _peaks(record).items():
        measured = _per_lead(filtered, peaks, record.fs, record.leads)
        if measured is None:
            continue
        geometry[name] = measured
        per_backend[name] = _flags(measured)
    if not per_backend:
        return None

    first = next(iter(per_backend.values()))
    result: dict[str, Any] = {
        "n_backends": len(per_backend),
        "backend_findings": per_backend,
    }
    for finding in _CATEGORICAL:
        values = {backend[finding] for backend in per_backend.values()}
        result[finding] = bool(first[finding])
        result[f"{finding}_agree"] = len(per_backend) >= 2 and len(values) == 1
    # The per-lead geometry is kept so a criterion can be re-derived against
    # external labels without re-measuring 21,799 recordings for every threshold
    # that is worth trying. It is the measurement; the thresholds are a reading
    # of it, and the two should not be welded together in a cache.
    first_backend = next(iter(geometry))
    result["per_lead"] = {
        lead: {key: round(value, 4) for key, value in fields.items()}
        for lead, fields in geometry[first_backend].items()
        if lead != "_atrial"
    }
    for detail in (
        "pathologic_q_leads",
        "pathologic_q_territory",
        "pathologic_q_n_leads",
        "t_inversion_leads",
        "t_inversion_territory",
        "t_inversion_depth_mv",
        "min_limb_pp_mv",
        "max_limb_pp_mv",
        "max_precordial_pp_mv",
        "r_v3_mv",
        "p_duration_ms",
        "p_terminal_force_mvs",
        "p_wave_measurable",
        "pathologic_q_strict_leads",
        "pathologic_q_strict_territory",
        "pathologic_q_strict_n_leads",
        "t_inversion_strict_leads",
        "t_inversion_strict_territory",
    ):
        result[detail] = first[detail]
    result["criterion_profile"] = "ptbxl_external_specificity_v1"
    return result
