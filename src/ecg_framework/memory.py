"""A small JitRL-style outcome memory with fold-safe construction."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .schema import (
    ClaimCandidate,
    Evidence,
    ReportAction,
    RunContext,
    ValueEstimate,
)


@dataclass(frozen=True)
class OutcomeRecord:
    record_id: str
    fold: int
    claim_id: str
    action: ReportAction
    reward: float
    context: str = "global"


class OutcomeDataset:
    def __init__(self, records: Iterable[OutcomeRecord] = ()) -> None:
        self.records = tuple(records)

    @classmethod
    def from_jsonl(cls, path: Path) -> OutcomeDataset:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                OutcomeRecord(
                    record_id=str(row["record_id"]),
                    fold=int(row["fold"]),
                    claim_id=str(row["claim_id"]),
                    action=ReportAction(str(row["action"])),
                    reward=float(row["reward"]),
                    context=str(row.get("context", "global")),
                )
            )
        return cls(records)

    def to_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                row = asdict(record)
                row["action"] = record.action.value
                handle.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True)
class _GroupEstimate:
    mean: float
    standard_error: float
    count: int


class GroupedOutcomePolicy:
    """Conservative partial pooling over claim/action outcome groups."""

    def __init__(
        self,
        records: Sequence[OutcomeRecord],
        *,
        risk_z: float = 0.5,
        shrinkage: float = 8.0,
        margin: float = 0.0,
        query_fold: int | None = None,
    ) -> None:
        self.risk_z = float(risk_z)
        self.shrinkage = float(shrinkage)
        self.margin = float(margin)
        self.query_fold = query_fold
        usable = [
            row for row in records if query_fold is None or row.fold != query_fold
        ]
        if not usable:
            raise ValueError("outcome policy has no usable memory records")
        self.memory_folds = tuple(sorted({row.fold for row in usable}))
        self.global_mean = sum(row.reward for row in usable) / len(usable)
        grouped: dict[tuple[str, ReportAction, str], list[float]] = defaultdict(list)
        for row in usable:
            grouped[(row.claim_id, row.action, row.context)].append(row.reward)
        self.estimates = {
            key: self._estimate(values) for key, values in grouped.items()
        }

    def _estimate(self, values: Sequence[float]) -> _GroupEstimate:
        count = len(values)
        local_mean = sum(values) / count
        pooled = (count * local_mean + self.shrinkage * self.global_mean) / (
            count + self.shrinkage
        )
        if count < 2:
            standard_error = float("inf")
        else:
            variance = sum((value - local_mean) ** 2 for value in values) / (count - 1)
            standard_error = math.sqrt(variance / count)
        return _GroupEstimate(pooled, standard_error, count)

    def estimate(
        self,
        context: RunContext,
        candidate: ClaimCandidate,
        action: ReportAction,
        evidence: Evidence,
    ) -> ValueEstimate | None:
        del evidence
        context_key = str(candidate.metadata.get("outcome_context", "global"))
        estimate = self.estimates.get((candidate.claim_id, action, context_key))
        if estimate is None:
            return None
        uncertainty = estimate.standard_error
        safe = estimate.mean - self.margin
        if math.isfinite(uncertainty):
            safe -= self.risk_z * uncertainty
        else:
            safe = float("-inf")
        return ValueEstimate(
            mean_return=estimate.mean,
            uncertainty=uncertainty,
            safe_return=safe,
            policy_id="grouped-outcome-memory",
            memory_folds=self.memory_folds,
            query_fold=self.query_fold,
            metadata={"count": estimate.count, "context": context_key},
        )


def crossfit_grouped_policies(
    dataset: OutcomeDataset,
    folds: Sequence[int] | None = None,
    **kwargs: float,
) -> dict[int, GroupedOutcomePolicy]:
    selected_folds = tuple(sorted(set(folds or (row.fold for row in dataset.records))))
    return {
        fold: GroupedOutcomePolicy(dataset.records, query_fold=fold, **kwargs)
        for fold in selected_folds
    }
