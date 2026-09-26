from __future__ import annotations

import numpy as np

from ecgcf.oracles.frontal_axis import (
    _axis_category,
    _circular_difference,
    _estimate_from_peaks,
)


def test_axis_category_uses_standard_frontal_ranges() -> None:
    assert _axis_category(20.0) == "normal"
    assert _axis_category(-45.0) == "left_axis_deviation"
    assert _axis_category(120.0) == "right_axis_deviation"
    assert _axis_category(-120.0) == "extreme_axis_deviation"


def test_net_qrs_deflection_recovers_left_axis() -> None:
    fs = 500
    signal = np.zeros((12, 2500), dtype=np.float64)
    peaks = np.asarray([300, 800, 1300, 1800, 2300])
    for peak in peaks:
        signal[0, peak] = 1.0
        signal[5, peak] = -1.0

    result = _estimate_from_peaks(signal, peaks, fs, 0, 5)

    assert result is not None
    angle, n_beats = result
    assert -50.0 < angle < -40.0
    assert n_beats == 5


def test_circular_difference_handles_wraparound() -> None:
    assert _circular_difference(179.0, -179.0) == 2.0
