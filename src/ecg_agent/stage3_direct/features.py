"""Leakage-free state-action features for direct ECG Agent planning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .schema import DirectEpisode, ReplayTransition


def _flatten(value: Any, prefix: str, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            output[prefix] = number
    elif isinstance(value, str):
        output[f"{prefix}={value.lower()}"] = 1.0
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key == "record_id":
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten(item, child, output)
    elif isinstance(value, (list, tuple)):
        output[f"{prefix}.count"] = float(len(value))
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                output[f"{prefix}={str(item).lower()}"] = 1.0


def state_action_features(
    episode: DirectEpisode,
    transition: ReplayTransition,
    instrument: Mapping[str, Any],
) -> dict[str, float]:
    """Use only pre-inspection frozen R1 state; never expose expert results."""

    output: dict[str, float] = {}
    output["base.label_count"] = float(len(episode.base_labels))
    for label in episode.base_labels:
        output[f"base.label={label.lower()}"] = 1.0
    normal = any("normal ecg" in label.lower() for label in episode.base_labels)
    output["base.asserts_normal_ecg"] = float(normal)
    output["state.selected_count"] = float(len(transition.selected_claims))
    for claim_id in transition.selected_claims:
        output[f"state.selected={claim_id}"] = 1.0
    output[f"action.claim={transition.action_claim}"] = 1.0
    output["state.selected_set=" + "+".join(sorted(transition.selected_claims))] = 1.0
    return output
