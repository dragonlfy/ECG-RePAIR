"""Conservative claim parsing for the direct agent's terminal diagnosis state."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ecg_agent.registry import CLAIM_REGISTRY
from ecg_agent.schema import Claim


class DiagnosticClaimParser:
    """Use final labels as truth; inspect prose only when labels are unavailable."""

    PRE_NEGATION = re.compile(
        r"\b(?:no|not|without|absent|neither|nor|lack(?:s|ing)?|"
        r"rule(?:s|d)?\s+out|ruling\s+out|exclude(?:s|d)?|excluding|"
        r"negative\s+for|free\s+of)\b",
        re.IGNORECASE,
    )
    POST_NEGATION = re.compile(
        r"^\s*(?:is|are|was|were|has\s+been)?\s*"
        r"(?:absent|excluded|ruled\s+out|not\s+present)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _polarity(cls, text: str, alias: str) -> bool | None:
        lower = text.lower()
        start = lower.find(alias)
        if start < 0:
            return None
        before = lower[max(0, start - 100) : start]
        after = lower[start + len(alias) : start + len(alias) + 50]
        return not (cls.PRE_NEGATION.search(before) or cls.POST_NEGATION.search(after))

    @classmethod
    def _match_claim(
        cls, texts: Sequence[str], aliases: Sequence[str]
    ) -> tuple[str, bool] | None:
        for text in texts:
            for alias in aliases:
                polarity = cls._polarity(text, alias)
                if polarity is not None:
                    return text.strip(), polarity
        return None

    def parse(self, report: str, labels: Sequence[str] = ()) -> list[Claim]:
        clean_labels = tuple(str(label) for label in labels if str(label).strip())
        if clean_labels:
            texts = clean_labels
            source = "ECG-R1-final-label"
        else:
            texts = tuple(
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+|\n+", report)
                if part.strip()
            )
            source = "ECG-R1-report-fallback"

        output: list[Claim] = []
        for claim_id, spec in CLAIM_REGISTRY.items():
            matched = self._match_claim(texts, spec.aliases)
            if matched is None:
                continue
            text, value = matched
            output.append(
                Claim(
                    claim_id=claim_id,
                    protocol_step=spec.protocol_step,
                    source=source,
                    r1_present=True,
                    r1_value=value,
                    r1_text=text,
                    present_in_behavior_report=value,
                )
            )
        return output
