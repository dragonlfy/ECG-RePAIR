from __future__ import annotations

import numpy as np
import pytest

from ecgcf.oracles import ConsensusOracle, NeuroKitOracle, WFDBOracle
from ecgcf.oracles.qrs_energy import qrs_boundaries_from_energy
from ecgcf.synthetic import generate_records


def test_synthetic_is_deterministic() -> None:
    first = generate_records(2, seed=9)
    second = generate_records(2, seed=9)
    assert np.array_equal(first[0].signal, second[0].signal)
    assert first[0].meta["synthetic_truth"] == second[0].meta["synthetic_truth"]


def test_dual_oracles_agree_on_known_record(
    synthetic_record, intervention_config
) -> None:
    neurokit = NeuroKitOracle().measure(synthetic_record)
    wfdb = WFDBOracle().measure(synthetic_record)
    assert neurokit is not None
    assert wfdb is not None
    assert neurokit.backend != wfdb.backend
    consensus = ConsensusOracle(
        [NeuroKitOracle(), WFDBOracle()], intervention_config["tolerances"]
    ).measure(synthetic_record)
    assert consensus is not None
    assert consensus.n_oracles_succeeded == 2
    assert consensus.agreement_rate >= 0.8
    assert consensus.get("qrs_ms").agree
    assert consensus.get("qrs_ms").value == pytest.approx(96.0, abs=4.0)
    assert consensus.get("qtc_ms").value == pytest.approx(
        synthetic_record.meta["synthetic_truth"]["qtc_ms"], abs=8.0
    )


def test_multilead_qrs_energy_tracks_synthetic_truth(synthetic_record) -> None:
    onsets, offsets = qrs_boundaries_from_energy(
        synthetic_record, synthetic_record.meta["landmarks"]["r_peak"]
    )
    measured_ms = float(
        np.median(np.asarray(offsets) - np.asarray(onsets))
        * 1000.0
        / synthetic_record.fs
    )
    assert measured_ms == pytest.approx(
        synthetic_record.meta["synthetic_truth"]["qrs_ms"], abs=12.0
    )
