# Architecture

## Two decisions

Verification asks whether the current waveform supports a claim. Action selection
asks whether an eligible change is likely to improve the report. The framework
keeps these decisions separate: positive historical outcomes cannot authorize an
edit whose evidence fails verification.

The public runtime scans claims, ranks inspections, measures or reads the required
observations, verifies support, evaluates candidate actions, and executes local
edits under a budget. A final presentation pass links retained diagnoses to
available evidence without changing the diagnosis set. The default portable
entry point inspects up to six claims and commits at most one edit.

## Implementation map

| Responsibility | Implementation |
| --- | --- |
| Input validation and fold isolation | `ecg_repair/pipeline.py` |
| Waveform measurement | `ecg_repair/tools.py`, `ecgcf/oracles/` |
| Clinical claim definitions | `ecg_agent/registry.py` |
| Instrument access | `ecg_agent/experts.py` |
| Claim parsing and verification | `ecg_agent/stage3_direct/claim_parser.py`, `verifier.py` |
| Evidence features, ridge advantage, nested training | `ecg_agent/stage3_direct/evidence_advantage.py` |
| Local composition and numeric checks | `ecg_agent/stage3_direct/composer.py`, `locality.py` |
| Diagnosis/evidence graph and rendering | `ecg_agent/stage5_evidence_graph/` |
| Passage retrieval and provenance | `ecg_agent/clinical_rag/` |
| Interchangeable component contracts | `ecg_framework/protocols.py` |
| Agent execution and trajectory auditing | `ecg_framework/runtime.py`, `evaluation.py` |

Some module names (`stage3_direct`, `stage5_evidence_graph`, and `LegacyStage4`)
are retained from the research implementation to keep component behavior
traceable. They do not prescribe stage names for a new application. Likewise,
`safe_return` is the historical API name for a conservative action value; it is
not a claim of clinical safety.

## Knowledge and waveform evidence

Executable clinical rules live in the registry, verifier, and evidence graph
builder. The TF-IDF index retrieves source passages and retains their page and
source metadata. `ClinicalCriterionRAG` can attach relevant source information to
an existing clinical evidence graph. Its optional retrieval does not independently
measure an ECG or determine the clinical verdict, and it is not automatically
wired into the default repair command.

## Memory

The ridge estimator consumes claim/action identity, diagnostic context, agreement,
reliability, and measurement features. It predicts an advantage learned from
historical score changes, then subtracts a decision margin. Historical report
prose and query references are not estimator features. The CLI excludes query-fold
episodes before fitting and rejects query IDs appearing in the remaining memory.
No new query reward is inserted during inference.

The SDK also contains fixed-value and grouped-outcome policies for testing and
extensions. The fixed-value policy in `examples/quickstart.py` is a component demo,
not the ridge policy used by `ecg-repair repair`.

## Extension example

Implement an `ECGExpert.inspect(context, candidate, spec)` method that returns
typed `Evidence`, then register it with a `FrameworkRegistry`. A new finding also
needs a `ClaimSpec`, a verifier, and an eligible action/composer implementation.
Adding a claim name alone does not implement its clinical verification.

Model adapters implement `BaseECGModel.infer(case) -> Report`. Stored ECG-R1, GEM,
or other model outputs can use the mapping adapter. The repository contains no
model checkpoints or inference service settings.
