"""Template for a separately distributed ECG Framework plugin."""

from __future__ import annotations

from ecg_framework import Evidence


class MyECGExpert:
    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint

    def inspect(self, context, candidate, spec):
        del spec
        # Replace with a signal model, image model, device API, or deterministic
        # measurement algorithm. Never read reference reports here.
        return Evidence(
            evidence_id=f"{context.case.record_id}:{candidate.claim_id}:my-expert",
            claim_id=candidate.claim_id,
            expert="my_ecg_expert",
            measurement="plugin_measurement",
            value=None,
            reliability=0.0,
            provenance={"checkpoint": self.checkpoint},
        )


def register(registry) -> None:
    registry.register("expert", "my_ecg_expert", MyECGExpert)


# In the plugin package's pyproject.toml:
# [project.entry-points."ecg_framework.plugins"]
# my_plugin = "my_package.plugin:register"
