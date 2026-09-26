"""Portable signal storage for evaluation items."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ecgcf.record import ECGRecord


def save_npz(rec: ECGRecord, path: str | Path) -> Path:
    """Write one canonical record and JSON metadata to a compressed NPZ."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        signal=rec.signal,
        fs=np.int64(rec.fs),
        leads=np.asarray(rec.leads),
        record_id=np.asarray(rec.record_id),
        meta=np.asarray(json.dumps(rec.meta, sort_keys=True)),
    )
    return target


def load_npz(path: str | Path) -> ECGRecord:
    """Load a record written by :func:`save_npz`."""

    with np.load(path, allow_pickle=False) as payload:
        meta: dict[str, Any] = json.loads(str(payload["meta"]))
        return ECGRecord(
            signal=np.asarray(payload["signal"], dtype=np.float32),
            fs=int(payload["fs"]),
            leads=tuple(str(value) for value in payload["leads"]),
            record_id=str(payload["record_id"]),
            meta=meta,
        )
