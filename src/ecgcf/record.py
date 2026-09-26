"""Canonical immutable ECG record representation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

CANONICAL_LEADS: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


@dataclass(frozen=True)
class ECGRecord:
    """A 12-lead ECG in canonical order and millivolts."""

    signal: NDArray[np.float32]
    fs: int
    leads: tuple[str, ...]
    record_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal = np.asarray(self.signal, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError("signal must have shape (12, N)")
        if signal.shape[1] < 2:
            raise ValueError("signal must contain at least two samples")
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        if tuple(self.leads) != CANONICAL_LEADS:
            raise ValueError("leads must be in canonical 12-lead order")
        if not np.isfinite(signal).all():
            raise ValueError("signal contains NaN or infinity")
        signal.setflags(write=False)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "leads", tuple(self.leads))

    @property
    def duration_seconds(self) -> float:
        """Return record duration in seconds."""

        return self.signal.shape[1] / self.fs

    def replace(
        self,
        *,
        signal: NDArray[np.float32] | None = None,
        meta: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> ECGRecord:
        """Return a validated record with selected fields replaced."""

        return ECGRecord(
            signal=self.signal if signal is None else signal,
            fs=self.fs,
            leads=self.leads,
            record_id=self.record_id if record_id is None else record_id,
            meta=dict(self.meta) if meta is None else meta,
        )

    def content_hash(self) -> str:
        """Hash signal bytes plus calibration-critical metadata."""

        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.signal).tobytes())
        digest.update(str(self.signal.shape).encode())
        digest.update(str(self.fs).encode())
        digest.update("|".join(self.leads).encode())
        return digest.hexdigest()
