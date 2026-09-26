"""External, provenance-preserving clinical knowledge retrieval for ECG-Agent."""

from .grounding import ClinicalCriterionRAG, RetrievedCriterion
from .index import ClinicalKnowledgeIndex, KnowledgePassage

__all__ = [
    "ClinicalCriterionRAG",
    "ClinicalKnowledgeIndex",
    "KnowledgePassage",
    "RetrievedCriterion",
]
