import numpy as np
import pytest

from ecgcf.synthetic import generate_ecg
from ecg_repair.tools import TOLERANCES


@pytest.fixture
def intervention_config():
    return {"tolerances": dict(TOLERANCES)}


@pytest.fixture
def synthetic_record():
    return generate_ecg(
        record_id="test-ecg", rng=np.random.default_rng(123), fs=500,
        duration_seconds=10.0, heart_rate_bpm=72.0, qrs_ms=96.0,
        pr_ms=170.0, qt_ms=380.0, noise_mv=0.0,
    )
