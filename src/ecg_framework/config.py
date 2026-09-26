"""Configuration model and YAML validation for ECG-Agent Framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgentConfig:
    framework_id: str = "ecg-agent-framework"
    max_inspections: int = 8
    max_committed_actions: int = 1
    min_safe_return: float = 0.0
    stop_after_withhold: bool = True
    fail_closed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.framework_id.strip():
            raise ValueError("framework_id cannot be empty")
        if self.max_inspections < 0:
            raise ValueError("max_inspections must be non-negative")
        if self.max_committed_actions < 0:
            raise ValueError("max_committed_actions must be non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AgentConfig:
        block = value.get("framework", value)
        known = {
            "framework_id",
            "max_inspections",
            "max_committed_actions",
            "min_safe_return",
            "stop_after_withhold",
            "fail_closed",
            "metadata",
        }
        unknown = sorted(set(block) - known)
        if unknown:
            raise ValueError(f"unknown framework config keys: {', '.join(unknown)}")
        config = cls(**block)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: Path) -> AgentConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            raise ValueError("framework YAML root must be an object")
        return cls.from_dict(value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
