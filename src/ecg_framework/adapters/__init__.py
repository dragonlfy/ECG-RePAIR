"""Adapters for existing ECG models and ECG-Agent Stage-4 components."""

from .common import FrozenMappingBaseModel
from .stage4 import (
    LegacyStage4Composer,
    LegacyStage4Expert,
    LegacyStage4OutcomePolicy,
    LegacyStage4Scanner,
    LegacyStage4Verifier,
    build_stage4_framework,
)
from .stage5 import DiagnosisLockedEvidencePresenter, build_stage5_framework

__all__ = [
    "DiagnosisLockedEvidencePresenter",
    "FrozenMappingBaseModel",
    "LegacyStage4Composer",
    "LegacyStage4Expert",
    "LegacyStage4OutcomePolicy",
    "LegacyStage4Scanner",
    "LegacyStage4Verifier",
    "build_stage4_framework",
    "build_stage5_framework",
]
