# Data formats

Keep local inputs under `data/` and outputs under `outputs/`; both are ignored by
Git. All examples below are fabricated interface examples.

## ECG waveform

```python
import numpy as np
from ecgcf import ECGRecord, CANONICAL_LEADS
from ecgcf.io import save_npz

# signal_mv: your calibrated NumPy array with shape (12, samples).
record = ECGRecord(signal=signal_mv, fs=500, leads=CANONICAL_LEADS,
                   record_id="example-001")
save_npz(record, "data/example.npz")
```

The NPZ schema includes `signal`, `fs`, `leads`, `record_id`, and JSON `meta`.
`load_npz` disables pickle loading. For a WFDB source use
`ecgcf.io.load_wfdb_record`; inspect its optional lead/resampling arguments before
converting the record. Do not manually change units without recalibration.

`ecg-repair measure` removes synthetic landmark metadata before measurement.
The test generator has separate known-landmark paths for unit tests; these are
not used by the public real-waveform entry point.

## Query report JSONL

Required fields are `record_id`, `fold`, and report text (`report` or
`prediction_raw`). Provide the diagnostic list through `labels` or
`prediction_labels`. Report IDs must be unique and have matching instruments.

```json
{"record_id":"example-001","fold":0,"report":"<think>Initial interpretation.</think><answer>Sinus rhythm</answer>","labels":["sinus rhythm"]}
```

`fold` is a user-assigned evaluation partition, not a patient attribute. Partition
at patient level when repeated recordings from one patient are present. The
release checks record/fold isolation; it cannot infer patient identities from
anonymous record IDs. Supply the same fold assignment for every episode from a
given record. Reference reports and query rewards have no online role.

## Instrument JSONL

`ecg-repair measure` creates these objects. The minimal schematic PR observation
below illustrates the nested format; it must not replace actual measurement.

```json
{"record_id":"example-001","instrument_valid":true,"oracle_agreement_rate":1.0,"scalar":{"pr_ms":{"value":224.0,"agree":true,"spread":6.0}},"morphology":{"p_wave_measurable":true}}
```

Additional tool fields include `rr_cv`, `p_wave`, `frontal_axis`, `voltage`,
`morphology`, `st_mv`, and backend provenance. The verifier determines whether a
field is reliable enough for a particular claim. Missing evidence is not a
negative diagnosis.

## Historical memory

Each JSONL row stores a historical state/action observation and its outcome.
`reward` is the measured score change **relative to that historical record's
unedited report**, in the same units used for the decision margin. The values in
this example are illustrative, not released training data.

```json
{
  "record_id": "history-001",
  "fold": 1,
  "labels": ["sinus rhythm"],
  "selected_claims": [],
  "claim_id": "pr_prolonged",
  "action": "add",
  "reward": 4.0,
  "evidence": {
    "evidence_id": "history-001:pr",
    "claim_id": "pr_prolonged",
    "expert": "intervals",
    "measurement": "pr_ms",
    "value": 224.0,
    "unit": "ms",
    "lead_set": ["II"],
    "relation": "supported",
    "reliability": 1.0,
    "backend_agreement": 1.0,
    "provenance": {"p_wave_measurable": true, "spread": 6.0}
  }
}
```

Store each complete object on **one line**. The action spelling for a revision
in the retained research schema is `revise_evidence`.

Construct episodes offline using your data and evaluation protocol. The package
does not fabricate useful clinical rewards, ship fitted memories, or request a
judge while repairing a query. With no eligible historical data it keeps the
diagnostic set. The ridge implementation and nested cross-fold trainer are in
`ecg_agent/stage3_direct/evidence_advantage.py`; the CLI uses explicit fixed
`alpha` and `margin` instead of automatically choosing them on the query set.

## Repair output

An output row contains the `AgentResult` fields and `release_audit`. Inspect
`decisions` for claim/action/value records, `evidence` for observations, and
`trace` for lifecycle events. The final report metadata contains the clinical
evidence graph when presentation succeeds. A run may change evidence wording
without committing a diagnostic action; use `committed_actions` to distinguish
these outcomes. Check the audit and trace for fallbacks before evaluating results.
