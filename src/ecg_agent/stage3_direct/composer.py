"""Evidence-locked report actions for the direct ECG Agent."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from ecg_agent.registry import CLAIM_REGISTRY
from ecg_agent.schema import Evidence
from harness.metrics.concept_match import concept_asserted

from .locality import target_numeric_correction_allowed
from .schema import DirectClaimAction

CANONICAL_LABELS = {
    "axis_left": "left axis deviation",
    "axis_right": "right axis deviation",
    "axis_extreme": "extreme axis deviation",
    "pr_prolonged": "first-degree AV block",
    "pr_short": "short PR interval",
    "qrs_prolonged": "prolonged QRS duration",
    "qrs_intermediate": "intraventricular conduction delay",
    "low_voltage": "low QRS voltage",
    "poor_r_progression": "poor R-wave progression",
    "pathologic_q": "pathologic Q waves",
    "t_wave_inversion": "T-wave inversion",
    "rv_infarction": "right ventricular involvement",
    "rhythm_irregular": "irregular rhythm",
    "bradycardia": "bradycardia",
    "tachycardia": "tachycardia",
}
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")

NORMAL_META_LABELS = frozenset({"normal ecg", "normal electrocardiogram"})

# These patterns identify statements that a positive registered measurement
# directly supersedes. They are deliberately narrower than the claim aliases:
# mentioning an alternative diagnosis is not enough to delete a sentence.
DENIAL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "axis_left": (
        re.compile(r"\b(?:frontal(?: plane)?|qrs|electrical)?\s*axis\s+(?:is|appears|remains|was)\s+(?:within\s+)?normal(?:\s+limits)?\b", re.I),
        re.compile(r"\bnormal\s+(?:frontal(?: plane)?|qrs|electrical)?\s*axis\b", re.I),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?left(?:ward)?\s+axis\s+deviation\b", re.I),
    ),
    "axis_right": (
        re.compile(r"\b(?:frontal(?: plane)?|qrs|electrical)?\s*axis\s+(?:is|appears|remains|was)\s+(?:within\s+)?normal(?:\s+limits)?\b", re.I),
        re.compile(r"\bnormal\s+(?:frontal(?: plane)?|qrs|electrical)?\s*axis\b", re.I),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?right(?:ward)?\s+axis\s+deviation\b", re.I),
    ),
    "axis_extreme": (
        re.compile(r"\b(?:frontal(?: plane)?|qrs|electrical)?\s*axis\s+(?:is|appears|remains|was)\s+(?:within\s+)?normal(?:\s+limits)?\b", re.I),
        re.compile(r"\bnormal\s+(?:frontal(?: plane)?|qrs|electrical)?\s*axis\b", re.I),
    ),
    "pr_prolonged": (
        re.compile(
            r"\bpr\s+(?:interval|duration)\b\s*"
            r"(?:(?:is|appears?|remains?|was)\s+)?"
            r"(?:within\s+normal\s+limits|normal|not\s+prolonged)\b",
            re.I,
        ),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?(?:first[- ]degree\s+)?(?:av|atrioventricular)\s+block\b", re.I),
        re.compile(r"\brul(?:e[sd]?|ing)\s+out\s+(?:first[- ]degree\s+)?(?:av|atrioventricular)\s+block\b", re.I),
    ),
    "pr_short": (
        re.compile(
            r"\bpr\s+(?:interval|duration)\b\s*"
            r"(?:(?:is|appears?|remains?|was)\s+)?"
            r"(?:within\s+normal\s+limits|normal|not\s+short)\b",
            re.I,
        ),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?(?:pre[- ]?excitation|short\s+pr)\b", re.I),
    ),
    "qrs_prolonged": (
        re.compile(r"\bqrs(?:\s+(?:complex|duration))?\b[^.!?]{0,55}\b(?:normal|narrow|not\s+(?:wide|prolonged))\b", re.I),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?(?:bundle\s+branch\s+block|conduction\s+delay|wide\s+qrs)\b", re.I),
    ),
    "qrs_intermediate": (
        re.compile(r"\bqrs(?:\s+(?:complex|duration))?\b[^.!?]{0,55}\b(?:normal|narrow|not\s+prolonged)\b", re.I),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?(?:intraventricular\s+)?conduction\s+delay\b", re.I),
    ),
    "low_voltage": (
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?low\s+(?:qrs\s+)?voltage\b", re.I),
        re.compile(r"\b(?:qrs\s+)?(?:voltage|amplitude)s?\s+(?:is|are|appears?|remains?)\s+(?:within\s+)?normal(?:\s+limits)?\b", re.I),
    ),
    "poor_r_progression": (
        re.compile(r"\b(?:r[- ]?wave\s+)?progression\b[^.!?]{0,45}\b(?:normal|appropriate|preserved|orderly)\b", re.I),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?poor\s+r[- ]?wave\s+progression\b", re.I),
    ),
    "pathologic_q": (
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?patholog(?:ic|ical)\s+q\s+waves?\b", re.I),
        re.compile(r"\bpatholog(?:ic|ical)\s+q\s+waves?\b[^.!?]{0,35}\b(?:absent|not\s+present)\b", re.I),
    ),
    "t_wave_inversion": (
        re.compile(r"\bno\s+(?:primary\s+)?t[- ]?wave\s+inversions?\b", re.I),
        re.compile(r"\bt\s+waves?\b[^.!?]{0,45}\b(?:normal|upright|not\s+inverted)\b", re.I),
    ),
    "rv_infarction": (
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?right\s+ventricular\s+(?:infarction|involvement)\b", re.I),
    ),
    "rhythm_irregular": (
        re.compile(r"\b(?:the\s+)?rhythm\s+(?:is|appears|remains|was)\s+(?:regular|regularly\s+regular)\b", re.I),
        re.compile(r"\brr\s+intervals?\b[^.!?]{0,45}\b(?:regular|stable|uniform)\b", re.I),
    ),
    "bradycardia": (
        re.compile(r"\b(?:heart|ventricular)\s+rate\b[^.!?]{0,55}\b(?:normal|within\s+normal\s+limits|not\s+bradycardic)\b", re.I),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?bradycardia\b", re.I),
    ),
    "tachycardia": (
        re.compile(r"\b(?:heart|ventricular)\s+rate\b[^.!?]{0,55}\b(?:normal|within\s+normal\s+limits|not\s+tachycardic)\b", re.I),
        re.compile(r"\bno\s+(?:evidence\s+of\s+)?tachycardia\b", re.I),
    ),
}

CLAUSE_REWRITES: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "axis_left": (
        (re.compile(r"\bnormal\s+frontal(?:\s+plane)?\s+axis\b", re.I), "left axis deviation"),
        (re.compile(r"\bnormal\s+(?:qrs\s+|electrical\s+)?axis\b", re.I), "left axis deviation"),
        (re.compile(r"\b(?:frontal(?:\s+plane)?|qrs|electrical)?\s*axis\s+is\s+(?:within\s+)?normal(?:\s+limits)?\b", re.I), "axis shows left axis deviation"),
    ),
    "axis_right": (
        (re.compile(r"\bnormal\s+frontal(?:\s+plane)?\s+axis\b", re.I), "right axis deviation"),
        (re.compile(r"\bnormal\s+(?:qrs\s+|electrical\s+)?axis\b", re.I), "right axis deviation"),
        (re.compile(r"\b(?:frontal(?:\s+plane)?|qrs|electrical)?\s*axis\s+is\s+(?:within\s+)?normal(?:\s+limits)?\b", re.I), "axis shows right axis deviation"),
    ),
    "axis_extreme": (
        (re.compile(r"\bnormal\s+frontal(?:\s+plane)?\s+axis\b", re.I), "extreme axis deviation"),
        (re.compile(r"\bnormal\s+(?:qrs\s+|electrical\s+)?axis\b", re.I), "extreme axis deviation"),
        (re.compile(r"\b(?:frontal(?:\s+plane)?|qrs|electrical)?\s*axis\s+is\s+(?:within\s+)?normal(?:\s+limits)?\b", re.I), "axis shows extreme axis deviation"),
    ),
    "pr_prolonged": (
        (
            re.compile(
                r"\b(?:the\s+)?pr\s+interval\s+is\s+within\s+normal\s+limits"
                r"(?:\s+(?:at\s+)?(?:approximately\s+)?[-+]?\d+(?:\.\d+)?\s*"
                r"(?:ms|milliseconds?|s|seconds?))?",
                re.I,
            ),
            "The PR interval is prolonged",
        ),
        (re.compile(r"\bnormal\s+pr\s+(?:interval|duration)\b", re.I), "prolonged PR interval"),
        (re.compile(r"\bpr\s+(?:interval|duration)\s+is\s+normal\b", re.I), "PR interval is prolonged"),
    ),
    "pr_short": (
        (
            re.compile(
                r"\b(?:the\s+)?pr\s+interval\s+is\s+within\s+normal\s+limits"
                r"(?:\s+(?:at\s+)?(?:approximately\s+)?[-+]?\d+(?:\.\d+)?\s*"
                r"(?:ms|milliseconds?|s|seconds?))?",
                re.I,
            ),
            "The PR interval is short",
        ),
        (re.compile(r"\bnormal\s+pr\s+(?:interval|duration)\b", re.I), "short PR interval"),
    ),
    "low_voltage": (
        (re.compile(r"\bnormal\s+(?:qrs\s+)?voltages?\b", re.I), "low QRS voltages"),
        (re.compile(r"\bno\s+(?:evidence\s+of\s+)?low\s+(?:qrs\s+)?voltage\b", re.I), "low QRS voltage"),
    ),
}

# Used only when a legacy response has no answer tag and its stored label is the
# entire narrative. This fixed, inference-time vocabulary preserves common base
# diagnoses without consulting a reference report.
CORE_DIAGNOSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sinus rhythm", ("sinus rhythm", "normal sinus rhythm")),
    ("atrial fibrillation", ("atrial fibrillation", "afib", "a fib")),
    ("atrial flutter", ("atrial flutter",)),
    ("junctional rhythm", ("junctional rhythm",)),
    ("paced rhythm", ("paced rhythm", "ventricular paced")),
    ("right bundle branch block", ("right bundle branch block", "rbbb")),
    ("left bundle branch block", ("left bundle branch block", "lbbb")),
    ("left ventricular hypertrophy", ("left ventricular hypertrophy", "lvh")),
    ("right ventricular hypertrophy", ("right ventricular hypertrophy", "rvh")),
    ("myocardial infarction", ("myocardial infarction", "infarct")),
    ("myocardial ischemia", ("myocardial ischemia", "ischemia")),
    ("premature ventricular complexes", ("premature ventricular", "pvc")),
    ("premature atrial complexes", ("premature atrial", "pac")),
    ("first-degree AV block", ("first degree av block", "first-degree av block")),
    ("prolonged QT", ("prolonged qt", "long qt")),
)


@dataclass(frozen=True)
class ComposedReport:
    report: str
    labels: tuple[str, ...]
    action: DirectClaimAction
    claim_id: str
    evidence_id: str
    deterministic: bool = True
    contradictions_removed: int = 0
    unresolved_contradictions: int = 0
    normal_labels_removed: int = 0

    def json(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        return value


class EvidenceLockedComposer:
    """Apply local claim edits without free-form diagnostic generation."""

    @staticmethod
    def _normalized(text: str) -> str:
        return " ".join(text.lower().split())

    @classmethod
    def _is_normal_label(cls, text: str) -> bool:
        return cls._normalized(text).strip(" .") in NORMAL_META_LABELS

    @classmethod
    def _deduplicate(cls, labels: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for label in labels:
            key = cls._normalized(label).strip(" .")
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(label.strip())
        return output

    @classmethod
    def _fallback_answer_items(
        cls, report: str, labels: tuple[str, ...] | list[str]
    ) -> list[str]:
        concise = [
            str(label).strip()
            for label in labels
            if str(label).strip()
            and len(str(label)) <= 160
            and len(str(label).split()) <= 20
        ]
        if concise:
            return cls._deduplicate(concise)
        return [
            canonical
            for canonical, aliases in CORE_DIAGNOSES
            if concept_asserted(report, aliases)
        ]

    @staticmethod
    def _asserts_target(sentence: str, claim_id: str) -> bool:
        canonical = CANONICAL_LABELS[claim_id]
        aliases = (canonical, *CLAIM_REGISTRY[claim_id].aliases)
        return concept_asserted(sentence, aliases)

    @classmethod
    def _is_conflicting_sentence(
        cls,
        sentence: str,
        claim_id: str,
        action: DirectClaimAction,
    ) -> bool:
        if action is DirectClaimAction.WITHHOLD:
            return cls._asserts_target(sentence, claim_id)
        if concept_asserted(sentence, NORMAL_META_LABELS):
            return True
        return any(
            pattern.search(sentence)
            for pattern in DENIAL_PATTERNS.get(claim_id, ())
        )

    @classmethod
    def _remove_conflicting_sentences(
        cls,
        narrative: str,
        claim_id: str,
        action: DirectClaimAction,
    ) -> tuple[str, int, int]:
        output_lines: list[str] = []
        removed = 0
        unresolved = 0
        for line in narrative.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("<") or stripped.startswith("**"):
                output_lines.append(line)
                continue
            kept: list[str] = []
            for sentence in re.split(r"(?<=[.!?])\s+", stripped):
                if not cls._is_conflicting_sentence(sentence, claim_id, action):
                    kept.append(sentence)
                    continue
                numeric = NUMBER.findall(sentence)
                removable = all(
                    target_numeric_correction_allowed(
                        narrative, token, frozenset({claim_id})
                    )
                    for token in numeric
                )
                if removable:
                    removed += 1
                else:
                    revised = sentence
                    if action is not DirectClaimAction.WITHHOLD:
                        for pattern, replacement in CLAUSE_REWRITES.get(
                            claim_id, ()
                        ):
                            revised = pattern.sub(replacement, revised)
                    if revised != sentence and not cls._is_conflicting_sentence(
                        revised, claim_id, action
                    ):
                        kept.append(revised)
                        removed += 1
                    else:
                        kept.append(sentence)
                        unresolved += 1
            output_lines.append(" ".join(kept))
        cleaned = "\n".join(output_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, removed, unresolved

    @staticmethod
    def _measurement(evidence: Evidence) -> str:
        provenance = evidence.provenance
        if evidence.claim_id == "low_voltage" and evidence.value is True:
            basis = provenance.get("criterion_basis")
            limb = provenance.get("max_limb_pp_mv")
            chest = provenance.get("max_precordial_pp_mv")
            criteria: list[str] = []
            if basis in ("limb", "limb_and_precordial") and isinstance(
                limb, (int, float)
            ):
                criteria.append(f"maximum limb-lead QRS={float(limb):.2f} mV <0.50 mV")
            if basis in ("precordial", "limb_and_precordial") and isinstance(
                chest, (int, float)
            ):
                criteria.append(
                    f"maximum precordial QRS={float(chest):.2f} mV <1.00 mV"
                )
            if criteria:
                return "; ".join(criteria)
        if evidence.claim_id == "poor_r_progression" and evidence.value is True:
            r_v3 = provenance.get("r_v3_mv")
            measured = (
                f"R(V3)={float(r_v3):.2f} mV"
                if isinstance(r_v3, (int, float))
                else "delayed precordial R-wave growth"
            )
            return f"{measured}; V2-V4 progression criterion not satisfied"
        if evidence.claim_id == "pathologic_q" and evidence.value is True:
            territory = provenance.get("pathologic_q_territory") or provenance.get(
                "territory"
            )
            where = f" in the {territory} territory" if territory else ""
            leads = ", ".join(evidence.lead_set)
            lead_text = f" ({leads})" if leads else ""
            return f"strict pathologic-Q criterion{where}{lead_text}"
        if evidence.claim_id == "t_wave_inversion" and evidence.value is True:
            territory = provenance.get("territory")
            depth = provenance.get("depth_mv")
            leads = ", ".join(evidence.lead_set)
            parts = ["contiguous T-wave inversion"]
            if territory:
                parts.append(f"{territory} territory")
            if leads:
                parts.append(f"leads {leads}")
            if isinstance(depth, (int, float)):
                parts.append(f"second-deepest T={float(depth):.2f} mV")
            return "; ".join(parts)
        if isinstance(evidence.value, bool):
            rendered = "present" if evidence.value else "absent"
        elif isinstance(evidence.value, float):
            rendered = f"{evidence.value:g}"
        else:
            rendered = json.dumps(evidence.value, sort_keys=True)
        unit = f" {evidence.unit}" if evidence.unit else ""
        leads = f" in leads {', '.join(evidence.lead_set)}" if evidence.lead_set else ""
        return f"{evidence.measurement}={rendered}{unit}{leads}"

    @staticmethod
    def _answer_items(report: str) -> tuple[re.Match[str] | None, list[str]]:
        match = ANSWER.search(report)
        if match is None:
            return None, []
        items = [item.strip() for item in re.split(r"[;\n]+", match.group(1))]
        return match, [item for item in items if item]

    @staticmethod
    def _insert_addendum(report: str, addendum: str) -> str:
        match = ANSWER.search(report)
        if match is None:
            return report.rstrip() + "\n\n" + addendum
        return report[: match.start()] + addendum + "\n\n" + report[match.start() :]

    def compose(
        self,
        report: str,
        labels: tuple[str, ...] | list[str],
        claim_id: str,
        action: DirectClaimAction,
        evidence: Evidence,
    ) -> ComposedReport:
        if claim_id not in CLAIM_REGISTRY:
            raise KeyError(f"unknown claim: {claim_id}")
        if action not in (
            DirectClaimAction.ADD,
            DirectClaimAction.REVISE_EVIDENCE,
            DirectClaimAction.WITHHOLD,
        ):
            return ComposedReport(
                report,
                tuple(labels),
                action,
                claim_id,
                evidence.evidence_id,
            )
        canonical = CANONICAL_LABELS[claim_id]
        measurement = self._measurement(evidence)
        verb = "contradicts" if action is DirectClaimAction.WITHHOLD else "supports"
        addendum = (
            "Verified correction (superseding any conflicting template statement): "
            f"the registered ECG measurement ({measurement}) {verb} {canonical}."
        )
        if action is DirectClaimAction.WITHHOLD:
            addendum += " This claim is withheld from the final diagnosis."

        match, answer_items = self._answer_items(report)
        if match is None:
            answer_items = self._fallback_answer_items(report, labels)
        narrative = (
            report[: match.start()] + report[match.end() :]
            if match is not None
            else report
        )
        narrative, removed, unresolved = self._remove_conflicting_sentences(
            narrative, claim_id, action
        )
        aliases = tuple({canonical.lower(), *CLAIM_REGISTRY[claim_id].aliases})
        concise_labels = [
            str(label).strip()
            for label in labels
            if str(label).strip()
            and len(str(label)) <= 160
            and len(str(label).split()) <= 20
        ]
        output_labels = self._deduplicate(concise_labels) or list(answer_items)
        normal_removed = 0
        if action is DirectClaimAction.ADD:
            normal_removed = sum(self._is_normal_label(label) for label in output_labels)
            output_labels = [
                label for label in output_labels if not self._is_normal_label(label)
            ]
            answer_items = [
                item for item in answer_items if not self._is_normal_label(item)
            ]
            if not any(canonical.lower() == label.lower() for label in output_labels):
                output_labels.append(canonical)
            if not any(
                canonical.lower() == item.lower() for item in answer_items
            ):
                answer_items.append(canonical)
        elif action is DirectClaimAction.WITHHOLD:
            output_labels = [
                label
                for label in output_labels
                if not any(alias in label.lower() for alias in aliases)
            ]
            answer_items = [
                item
                for item in answer_items
                if not any(alias in item.lower() for alias in aliases)
            ]

        output_labels = self._deduplicate(output_labels)
        answer_items = self._deduplicate(answer_items)
        answer = "; ".join(answer_items) or "No retained diagnostic label"
        edited = narrative.rstrip() + f"\n\n{addendum}\n\n<answer>{answer}</answer>"
        return ComposedReport(
            edited,
            tuple(output_labels),
            action,
            claim_id,
            evidence.evidence_id,
            contradictions_removed=removed,
            unresolved_contradictions=unresolved,
            normal_labels_removed=normal_removed,
        )
