from __future__ import annotations

import numpy as np
import pytest

from ecgcf.oracles.morphology import (
    LOW_VOLTAGE_LIMB_MV,
    POOR_R_V3_MV,
    Q_DURATION_MS,
    T_INVERSION_MV,
    TERRITORIES,
    _flags,
)


def _lead(**overrides) -> dict[str, float]:
    values = {"q_ms": 10.0, "q_mv": -0.02, "r_mv": 1.0, "pp_mv": 1.5, "t_mv": 0.3}
    values.update(overrides)
    return values


def _per_lead(**overrides) -> dict[str, dict[str, float]]:
    leads = {
        name: _lead()
        for name in (
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
    }
    leads.update(overrides)
    leads["_atrial"] = {"p_ms": 90.0, "p_terminal_force_mvs": 0.001}
    return leads


def test_a_finding_in_one_lead_is_not_a_territory() -> None:
    """A single lead is noise as often as disease, so two are required."""

    one = _flags(_per_lead(II=_lead(q_ms=Q_DURATION_MS + 10)))
    two = _flags(
        _per_lead(II=_lead(q_ms=Q_DURATION_MS + 10), aVF=_lead(q_ms=Q_DURATION_MS + 10))
    )

    assert one["pathologic_q"] is False
    assert two["pathologic_q"] is True
    assert two["pathologic_q_territory"] == "inferior"
    assert two["pathologic_q_n_leads"] == 2


def test_avr_is_excluded_from_every_morphological_finding() -> None:
    """aVR faces away from the ventricles, so its Q wave and T are normal."""

    flags = _flags(
        _per_lead(
            aVR=_lead(q_ms=Q_DURATION_MS + 20, t_mv=T_INVERSION_MV - 0.2),
        )
    )

    assert "aVR" not in flags["pathologic_q_leads"]
    assert "aVR" not in flags["t_inversion_leads"]


def test_a_deep_q_wave_is_pathologic_without_being_wide() -> None:
    deep = _lead(q_ms=5.0, q_mv=-0.4, r_mv=1.0)
    flags = _flags(_per_lead(II=deep, III=deep))

    assert flags["pathologic_q"] is True


def test_a_deep_q_wave_beside_a_tiny_r_wave_is_not_counted() -> None:
    """Below a usable R wave the depth ratio is dividing noise by noise."""

    noise = _lead(q_ms=5.0, q_mv=-0.05, r_mv=0.1)
    flags = _flags(_per_lead(II=noise, III=noise))

    assert flags["pathologic_q"] is False


def test_low_voltage_needs_every_lead_of_a_group() -> None:
    small = _lead(pp_mv=LOW_VOLTAGE_LIMB_MV - 0.1)
    limb = {name: small for name in ("I", "II", "III", "aVR", "aVL", "aVF")}

    assert _flags(_per_lead(**limb))["low_voltage"] is True
    assert _flags(_per_lead(**{**limb, "I": _lead(pp_mv=2.0)}))["low_voltage"] is False


def test_poor_r_progression_reads_v3() -> None:
    assert _flags(_per_lead(V3=_lead(r_mv=POOR_R_V3_MV - 0.05)))["poor_r_progression"]
    assert not _flags(_per_lead(V3=_lead(r_mv=POOR_R_V3_MV + 0.05)))[
        "poor_r_progression"
    ]


def test_the_inversion_depth_is_the_second_deepest_of_a_territory() -> None:
    """The margin thresholds the depth at which two leads still qualify."""

    flags = _flags(
        _per_lead(V3=_lead(t_mv=-0.50), V4=_lead(t_mv=-0.20), V5=_lead(t_mv=-0.11))
    )

    assert flags["t_wave_inversion"] is True
    assert flags["t_inversion_depth_mv"] == pytest.approx(-0.20)


def test_a_fibrillatory_baseline_gets_no_atrial_verdict() -> None:
    """There is no discrete P wave to measure, so none is reported."""

    leads = _per_lead()
    leads["_atrial"] = {"p_ms": 240.0, "p_terminal_force_mvs": 0.02}

    flags = _flags(leads)

    assert flags["p_wave_measurable"] is False
    assert flags["left_atrial_abnormality"] is False


def test_every_territory_has_at_least_two_leads() -> None:
    assert all(len(leads) >= 2 for leads in TERRITORIES.values())
    assert not np.isnan(0.0)
