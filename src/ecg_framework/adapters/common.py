"""Generic adapters shared by multiple base ECG models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import ECGCase, Report


class FrozenMappingBaseModel:
    """Read frozen model reports from any record-keyed mapping."""

    def __init__(
        self,
        rows: Mapping[str, Mapping[str, Any]],
        *,
        report_fields: tuple[str, ...] = (
            "prediction_raw_base",
            "prediction_raw",
            "report",
        ),
        label_fields: tuple[str, ...] = (
            "prediction_labels_base",
            "prediction_labels",
            "labels",
        ),
        model_id: str = "frozen-ecg-model",
    ) -> None:
        self.rows = rows
        self.report_fields = report_fields
        self.label_fields = label_fields
        self.model_id = model_id

    def infer(self, case: ECGCase) -> Report:
        row = self.rows[case.record_id]
        text = next((str(row[key]) for key in self.report_fields if row.get(key)), "")
        labels = next(
            (
                tuple(str(value) for value in row[key])
                for key in self.label_fields
                if row.get(key)
            ),
            (),
        )
        if not text:
            raise ValueError(f"no frozen report found for {case.record_id}")
        return Report(text=text, labels=labels, model_id=self.model_id)
