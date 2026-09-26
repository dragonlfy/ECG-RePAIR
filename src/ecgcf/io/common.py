"""Unit, order, and sampling-rate normalization."""

from __future__ import annotations

from collections.abc import Sequence
from math import gcd
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from ecgcf.record import CANONICAL_LEADS, ECGRecord

_LEAD_ALIASES = {
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "MDC_ECG_LEAD_AVR": "aVR",
    "MDC_ECG_LEAD_AVL": "aVL",
    "MDC_ECG_LEAD_AVF": "aVF",
}


def convert_to_mv(
    signal: NDArray[np.floating[Any]], units: str | Sequence[str]
) -> NDArray[np.float32]:
    """Convert volts, millivolts, or microvolts to float32 millivolts."""

    array = np.asarray(signal, dtype=np.float64)
    unit_values = [units] * array.shape[0] if isinstance(units, str) else list(units)
    if len(unit_values) != array.shape[0]:
        raise ValueError("one unit is required per lead")
    output = np.empty_like(array, dtype=np.float64)
    for index, raw_unit in enumerate(unit_values):
        unit = raw_unit.strip().lower().replace("μ", "u").replace("µ", "u")
        if unit in {"mv", "millivolt", "millivolts"}:
            scale = 1.0
        elif unit in {"uv", "microvolt", "microvolts"}:
            scale = 1e-3
        elif unit in {"v", "volt", "volts"}:
            scale = 1e3
        else:
            raise ValueError(f"unknown ECG unit: {raw_unit!r}")
        output[index] = array[index] * scale
    return output.astype(np.float32)


def _canonical_name(name: str) -> str:
    stripped = name.strip()
    return _LEAD_ALIASES.get(stripped.upper(), stripped)


def reorder_leads(
    signal: NDArray[np.floating[Any]], leads: Sequence[str]
) -> NDArray[np.float32]:
    """Reorder exactly 12 named leads into the canonical order."""

    array = np.asarray(signal)
    if array.ndim != 2:
        raise ValueError("signal must be two-dimensional")
    names = [_canonical_name(name) for name in leads]
    missing = [lead for lead in CANONICAL_LEADS if lead not in names]
    if missing:
        raise ValueError(f"missing canonical leads: {missing}")
    indices = [names.index(lead) for lead in CANONICAL_LEADS]
    return np.asarray(array[indices], dtype=np.float32)


def resample_signal(
    signal: NDArray[np.floating[Any]], source_fs: int, target_fs: int
) -> NDArray[np.float32]:
    """Polyphase-resample all leads to an exact target rate."""

    if source_fs <= 0 or target_fs <= 0:
        raise ValueError("sampling rates must be positive")
    array = np.asarray(signal, dtype=np.float64)
    if source_fs == target_fs:
        return array.astype(np.float32, copy=True)
    divisor = gcd(source_fs, target_fs)
    result = resample_poly(array, target_fs // divisor, source_fs // divisor, axis=1)
    return np.asarray(result, dtype=np.float32)


def canonicalize(
    signal: NDArray[np.floating[Any]],
    *,
    fs: int,
    leads: Sequence[str],
    units: str | Sequence[str],
    record_id: str,
    meta: dict[str, Any] | None = None,
    canonical_fs: int = 500,
) -> ECGRecord:
    """Apply unit conversion, lead ordering, and sampling-rate normalization."""

    array = np.asarray(signal)
    if array.ndim != 2:
        raise ValueError("signal must be two-dimensional")
    if array.shape[0] != len(leads) and array.shape[1] == len(leads):
        array = array.T
    mv = convert_to_mv(array, units)
    ordered = reorder_leads(mv, leads)
    sampled = resample_signal(ordered, fs, canonical_fs)
    metadata = dict(meta or {})
    metadata.update(
        {
            "source_fs": fs,
            "canonical_fs": canonical_fs,
            "source_units": list(units) if not isinstance(units, str) else units,
            "canonical_units": "mV",
        }
    )
    return ECGRecord(sampled, canonical_fs, CANONICAL_LEADS, record_id, metadata)
