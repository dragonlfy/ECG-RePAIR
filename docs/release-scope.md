# Release scope

## Included

- Typed Agent SDK, replaceable component protocols, trajectory and fold audits.
- Existing claim registry, waveform verification, report parser, local composer.
- Evidence-conditioned ridge advantage model and nested cross-fold training code.
- Diagnosis-linked evidence graph, renderer, and diagnosis-preservation checks.
- NeuroKit2/WFDB wrappers and axis, voltage, morphology, interval routines.
- Clinical passage index and optional evidence-graph attribution.
- Portable CLI, synthetic examples, tests, and CI configuration.

## Not bundled

Patient ECGs, images, reports, reference text, historical patient episodes,
judge responses, expert assessments, trained weights, private endpoints, manuscript
sources/figures, cached experiment outputs, and old routing experiment launchers.
The public commands read local files; they do not submit ECGs to a remote service.

## Relationship to the research workflow

The release extracts reusable components and adds a portable wrapper. It does not
bundle the cached study core/expansion orchestration or claim that its default
single-pass CLI reproduces the exact main-table experiment. Reproducing those
results requires the appropriately authorized dataset, base report set, historical
episodes, split assignments, evaluation configuration, and orchestration.

The demo uses fabricated measurements and rewards. Unit tests may use known
synthetic landmarks. The real-waveform entry point removes this metadata before
calling the measurement tools. Tests validate software behavior, not clinical
accuracy.

## Implementation changes made for the release

The clinical kernels are retained. Package initialization no longer imports
unreleased early prototypes or experiment caches. The numeric-locality helper
owns its regular expression instead of importing an obsolete policy module.
Evidence provenance uses the supplied instrument source and backend metadata
instead of a hard-coded study cache name.
The public wrapper adds file schemas, duplicate-ID and memory-isolation checks,
explicit hyperparameters, and a portable command-line interface.

Base-model report generation and automated judge inference remain external.
Report quality can depend on report formatting and the available claim/tool
coverage. Local correction does not certify all retained diagnoses or guarantee
that every contradiction elsewhere in the report has been removed.
