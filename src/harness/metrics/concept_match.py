"""Uniform negation-aware detection of a target diagnosis in free-text output.

Different ECG models emit different response shapes. ECG-R1 is reinforcement-trained
to wrap its conclusion in ``<answer>`` tags, while GEM and PULSE answer the native
ECG-Grounding prompt with a single clinical narrative. Restricting concept detection
to a parsed answer block therefore measures response formatting rather than diagnosis,
and would structurally suppress the flip rate of any model that does not emit tags.

The primary benchmark surface is consequently the complete response, scanned with one
predeclared rule for every model. Because a narrative frequently mentions a concept in
order to rule it out, a bare substring test would count "no evidence of atrial
fibrillation" as an acquired atrial-fibrillation diagnosis. Occurrences preceded by a
negation cue inside the same clause are therefore not treated as assertions.

The rule is deliberately shallow and lexical. It does not resolve hedging
("cannot exclude"), coreference, or clause-crossing scope, and it is applied
identically to every model so that no model is advantaged by output shape.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: Cues that negate a concept mentioned after them within the same clause.
NEGATION_CUES: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "non",
        "without",
        "absent",
        "absence",
        "denies",
        "denied",
        "negative",
        "exclude",
        "excludes",
        "excluded",
        "excluding",
        "rule",
        "rules",
        "ruled",
        "ruling",
        "lacks",
        "lacking",
        "devoid",
        "unlikely",
        "rather",
        "instead",
        "neither",
        "nor",
        "free",
        "resolved",
        "unremarkable",
    }
)

#: Tokens that end a negation's forward scope by pivoting back to an assertion.
#:
#: Clinical narratives routinely negate a list and then affirm something else in
#: the same clause: "the absence of distinct P waves and the presence of irregular
#: RR intervals", or "rules out bundle branch block suggesting a nonspecific
#: intraventricular conduction delay". Without terminators, a negation cue would
#: swallow the affirmative half.
SCOPE_TERMINATORS: frozenset[str] = frozenset(
    {
        "presence",
        "present",
        "consistent",
        "show",
        "shows",
        "showed",
        "showing",
        "exhibit",
        "exhibits",
        "exhibiting",
        "seen",
        "noted",
        "revealed",
        "suggesting",
        "suggests",
        "indicating",
        "indicates",
        "revealing",
        "reveals",
        "demonstrating",
        "demonstrates",
        "compatible",
        "confirming",
        "confirms",
        "confirmed",
    }
)

#: Nouns after which "negative" describes a waveform's polarity, not a finding.
#:
#: "a predominantly negative complex in lead II, confirming left axis deviation"
#: is an assertion, not a denial. Treating "negative" as a cue everywhere makes
#: the scorer miss the most common way an ECG report states its own evidence, so
#: the cue is suppressed when the next token names a waveform or its amplitude.
POLARITY_NOUNS: frozenset[str] = frozenset(
    {
        "complex",
        "complexes",
        "deflection",
        "deflections",
        "qrs",
        "wave",
        "waves",
        "amplitude",
        "amplitudes",
        "voltage",
        "voltages",
        "polarity",
        "concordance",
        "component",
        "components",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
    }
)

_CLAUSE_SPLIT = re.compile(r"[.;:\n\r]+|(?<=\s)\bbut\b|(?<=\s)\bhowever\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and collapse punctuation without dropping meaningful words."""

    return " ".join(_NON_ALNUM.sub(" ", text.lower()).split())


def _clauses(text: str) -> list[str]:
    return [
        normalized
        for part in _CLAUSE_SPLIT.split(text or "")
        if (normalized := normalize(part))
    ]


def _negated_positions(tokens: list[str]) -> list[bool]:
    """Mark each token position that lies inside an active negation scope.

    A cue opens a scope that runs forward to the end of the clause or until a
    scope terminator, whichever comes first. A six-token window is not enough:
    "no evidence of chamber hypertrophy, bundle branch block, or acute ischemic
    changes such as ST segment elevation" negates a concept sixteen tokens away.
    """

    negated = [False] * len(tokens)
    active = False
    for index, token in enumerate(tokens):
        if token in SCOPE_TERMINATORS:
            active = False
        if token in NEGATION_CUES and not (
            token == "negative"
            and index + 1 < len(tokens)
            and tokens[index + 1] in POLARITY_NOUNS
        ):
            active = True
            continue
        negated[index] = active
    return negated


def _asserted_in_clause(clause: str, alias: str) -> bool:
    """Return whether ``alias`` appears in ``clause`` outside any negation scope."""

    tokens = clause.split()
    alias_tokens = alias.split()
    if not alias_tokens or len(alias_tokens) > len(tokens):
        return False
    negated = _negated_positions(tokens)
    for start in range(len(tokens) - len(alias_tokens) + 1):
        if tokens[start : start + len(alias_tokens)] != alias_tokens:
            continue
        if not negated[start]:
            return True
    return False


def concept_asserted(text: str, aliases: Iterable[str]) -> bool:
    """Return whether the response asserts any configured alias of a concept.

    Matching is on whole normalized tokens, so ``"af"`` does not match ``"afib"``
    and ``"mi"`` does not match ``"minimal"``.
    """

    normalized_aliases = [
        normalized for alias in aliases if (normalized := normalize(alias))
    ]
    if not normalized_aliases:
        return False
    return any(
        _asserted_in_clause(clause, alias)
        for clause in _clauses(text)
        for alias in normalized_aliases
    )


def concept_mentions(text: str, aliases: Iterable[str]) -> dict[str, int]:
    """Count asserted and negated clause-level mentions, for auditing."""

    normalized_aliases = [
        normalized for alias in aliases if (normalized := normalize(alias))
    ]
    asserted = 0
    negated = 0
    for clause in _clauses(text):
        for alias in normalized_aliases:
            if not alias:
                continue
            tokens = clause.split()
            alias_tokens = alias.split()
            scope = _negated_positions(tokens)
            for start in range(len(tokens) - len(alias_tokens) + 1):
                if tokens[start : start + len(alias_tokens)] != alias_tokens:
                    continue
                if scope[start]:
                    negated += 1
                else:
                    asserted += 1
    return {"asserted": asserted, "negated": negated}
