"""WFDB loader for PTB-XL, CPSC, CSN, G12EC, and MIMIC-IV-ECG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ecgcf.io.common import canonicalize
from ecgcf.record import ECGRecord


def load_wfdb_record(path: str | Path, *, canonical_fs: int = 500) -> ECGRecord:
    """Load and canonicalize a WFDB record; require physical units."""

    try:
        import wfdb
    except ImportError as error:
        raise RuntimeError("SKIPPED: wfdb is not installed") from error
    record = wfdb.rdrecord(str(path), physical=True)
    if record.p_signal is None:
        raise ValueError("WFDB record has no physical signal")
    signal = record.p_signal.T
    leads = list(record.sig_name)
    units = list(record.units or [])
    if len(units) != len(leads):
        raise ValueError("WFDB physical unit metadata is missing")
    metadata: dict[str, Any] = {
        "loader": "wfdb",
        "comments": list(record.comments or []),
    }
    return canonicalize(
        signal,
        fs=int(record.fs),
        leads=leads,
        units=units,
        record_id=Path(path).name,
        meta=metadata,
        canonical_fs=canonical_fs,
    )
