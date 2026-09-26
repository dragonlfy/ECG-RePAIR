"""A small auditable TF-IDF index over licensed ECG knowledge passages."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.lower()))


@dataclass(frozen=True)
class KnowledgePassage:
    """One immutable passage plus the fields needed to cite it."""

    chunk_id: str
    text: str
    content_sha256: str
    source_id: str
    title: str
    page: int
    license: str
    catalog_url: str
    authority_tier: str

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "KnowledgePassage":
        metadata = row.get("metadata") or {}
        return cls(
            chunk_id=str(row["chunk_id"]),
            text=str(row["text"]),
            content_sha256=str(row["content_sha256"]),
            source_id=str(metadata["source_id"]),
            title=str(metadata["title"]),
            page=int(metadata["page"]),
            license=str(metadata["license"]),
            catalog_url=str(metadata["catalog_url"]),
            authority_tier=str(metadata["authority_tier"]),
        )


@dataclass(frozen=True)
class SearchHit:
    passage: KnowledgePassage
    score: float


class ClinicalKnowledgeIndex:
    """Retrieve source passages without using a generator or patient labels."""

    def __init__(self, passages: Sequence[KnowledgePassage]) -> None:
        self.passages = tuple(passages)
        documents = [Counter(_tokens(passage.text)) for passage in self.passages]
        n_documents = max(len(documents), 1)
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(document.keys())
        self._idf = {
            term: math.log((n_documents + 1) / (count + 1)) + 1.0
            for term, count in document_frequency.items()
        }
        self._vectors = [self._vector(document) for document in documents]
        self._norms = [
            math.sqrt(sum(value * value for value in vector.values())) or 1.0
            for vector in self._vectors
        ]

    @classmethod
    def from_jsonl(cls, paths: Iterable[Path]) -> "ClinicalKnowledgeIndex":
        passages: list[KnowledgePassage] = []
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    passages.append(KnowledgePassage.from_json(json.loads(line)))
        if not passages:
            raise ValueError("clinical knowledge index cannot be empty")
        return cls(passages)

    def _vector(self, terms: Counter[str]) -> dict[str, float]:
        total = sum(terms.values()) or 1
        return {
            term: (count / total) * self._idf.get(term, 1.0)
            for term, count in terms.items()
        }

    def retrieve(
        self,
        query: str,
        *,
        k: int = 3,
        source_ids: frozenset[str] | None = None,
    ) -> tuple[SearchHit, ...]:
        query_vector = self._vector(Counter(_tokens(query)))
        query_norm = (
            math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
        )
        scored: list[SearchHit] = []
        for passage, vector, norm in zip(
            self.passages, self._vectors, self._norms, strict=True
        ):
            if source_ids is not None and passage.source_id not in source_ids:
                continue
            shared = set(query_vector) & set(vector)
            score = sum(query_vector[t] * vector[t] for t in shared) / (
                query_norm * norm
            )
            if score > 0.0:
                scored.append(SearchHit(passage, score))
        scored.sort(key=lambda hit: (-hit.score, hit.passage.chunk_id))
        return tuple(scored[:k])
