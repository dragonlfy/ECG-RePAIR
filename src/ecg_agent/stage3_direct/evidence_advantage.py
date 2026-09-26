"""Evidence-conditioned terminal advantages learned from judged action outcomes."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from ecg_agent.experts import ToolSimulator
from ecg_agent.registry import CLAIM_REGISTRY
from ecg_agent.schema import Claim, Evidence, VerificationStatus

from .claim_parser import DiagnosticClaimParser
from .q_memory import CACHE_EVALUABLE_CLAIMS, build_inspection_transitions
from .schema import ActionValue, DirectClaimAction, DirectEpisode
from .verifier import ClinicalEvidenceVerifier

TOKEN = re.compile(r"[a-z0-9]+")
DIAGNOSTIC_GROUPS = {
    "atrial_fibrillation": ("atrial fibrillation", "afib"),
    "abnormal": ("abnormal ecg",),
    "borderline": ("borderline ecg",),
    "bradycardia": ("bradycardia", "bradycardic"),
    "infarction": ("infarct", "myocardial infarction"),
    "ischemia": ("ischemia", "ischemic"),
    "lbbb": ("left bundle branch block", "lbbb"),
    "lvh": ("left ventricular hypertrophy", "lvh"),
    "normal": ("normal ecg",),
    "pac": ("premature atrial", "pac"),
    "pvc": ("premature ventricular", "pvc"),
    "qtc": ("prolonged qt", "prolonged qtc"),
    "rbbb": ("right bundle branch block", "rbbb"),
    "tachycardia": ("tachycardia", "tachycardic"),
}


@dataclass(frozen=True)
class EvidenceAdvantageConfig:
    """Regularization and conservative action margin."""

    alpha: float
    margin: float
    feature_mode: str = "simple"

    @property
    def key(self) -> str:
        return (
            f"evidence_advantage:features={self.feature_mode}:"
            f"ridge={self.alpha:g}:margin={self.margin:g}"
        )


@dataclass(frozen=True)
class EvidenceActionRow:
    """A judged terminal action paired with only inference-time information."""

    record_id: str
    fold: int
    report: str
    labels: tuple[str, ...]
    selected_claims: frozenset[str]
    claim_id: str
    action: DirectClaimAction
    evidence: Evidence
    reward: float


def _positive_claims(report: str, labels: Sequence[str]) -> frozenset[str]:
    return frozenset(
        claim.claim_id
        for claim in DiagnosticClaimParser().parse(report, labels)
        if claim.r1_value is True
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _clinical_evidence_fingerprint(
    claim_id: str, evidence: Evidence
) -> dict[str, float]:
    """Compact ECG-criterion features; no free-text or outcome information."""

    features: dict[str, float] = {
        f"relation={evidence.relation.value}": 1.0,
        "lead_count": float(len(evidence.lead_set)) / 12.0,
    }
    for lead in evidence.lead_set:
        features[f"lead={lead}"] = 1.0

    provenance = evidence.provenance
    categorical = (
        "category",
        "criterion_basis",
        "territory",
        "pathologic_q_territory",
    )
    for key in categorical:
        value = provenance.get(key)
        if isinstance(value, str) and value:
            features[f"clinical.{key}={value.lower()}"] = 1.0
    for key in (
        "p_wave_measurable",
        "p_wave_agree",
        "limb_criterion_met",
        "precordial_criterion_met",
    ):
        value = provenance.get(key)
        if isinstance(value, bool):
            features[f"clinical.{key}"] = float(value)

    numeric_scales = {
        "spread": 100.0,
        "spread_degrees": 180.0,
        "spread_mv": 1.0,
        "n_backends": 2.0,
        "p_duration_ms": 200.0,
        "p_wave_score": 1.0,
        "heart_rate_bpm": 200.0,
        "min_limb_pp_mv": 1.0,
        "max_limb_pp_mv": 1.0,
        "max_precordial_pp_mv": 2.0,
        "r_v3_mv": 1.0,
        "pathologic_q_n_leads": 6.0,
        "depth_mv": 1.0,
        "sokolow_lyon_mv": 5.0,
        "r_avl_mv": 2.0,
        "r_lead_i_mv": 2.0,
    }
    for key, scale in numeric_scales.items():
        number = _finite_number(provenance.get(key))
        if number is not None:
            features[f"clinical.{key}"] = number / scale

    # Explicit signed distance to the criterion is more useful than raw values
    # for ECG's discontinuous, threshold-shaped decision surface.
    value = _finite_number(evidence.value)
    if value is not None:
        if claim_id == "axis_left":
            margin, boundary = (-30.0 - value) / 30.0, abs(value + 30.0) / 30.0
        elif claim_id == "axis_right":
            margin, boundary = (value - 90.0) / 30.0, abs(value - 90.0) / 30.0
        elif claim_id == "axis_extreme":
            margin = max((-90.0 - value) / 30.0, (value - 180.0) / 30.0)
            boundary = min(abs(value + 90.0), abs(value - 180.0)) / 30.0
        elif claim_id == "pr_prolonged":
            margin, boundary = (value - 200.0) / 40.0, abs(value - 200.0) / 40.0
        elif claim_id == "pr_short":
            margin, boundary = (120.0 - value) / 40.0, abs(value - 120.0) / 40.0
        else:
            margin = boundary = None
        if margin is not None:
            features["clinical.criterion_margin"] = float(margin)
            features["clinical.boundary_distance"] = float(boundary)
            features["clinical.near_boundary"] = float(boundary <= 0.5)

    if claim_id == "low_voltage":
        limb = _finite_number(provenance.get("max_limb_pp_mv"))
        chest = _finite_number(provenance.get("max_precordial_pp_mv"))
        margins = []
        if limb is not None:
            margins.append((0.5 - limb) / 0.5)
        if chest is not None:
            margins.append((1.0 - chest) / 1.0)
        if margins:
            criterion_margin = max(margins)
            features["clinical.criterion_margin"] = criterion_margin
            features["clinical.boundary_distance"] = abs(criterion_margin)
            features["clinical.near_boundary"] = float(abs(criterion_margin) <= 0.2)
    elif claim_id == "poor_r_progression":
        r_v3 = _finite_number(provenance.get("r_v3_mv"))
        if r_v3 is not None:
            features["clinical.criterion_margin"] = (0.3 - r_v3) / 0.3
            features["clinical.near_boundary"] = float(abs(r_v3 - 0.3) <= 0.05)
    return features


def evidence_action_features(
    report: str,
    labels: Sequence[str],
    selected_claims: frozenset[str],
    claim_id: str,
    action: DirectClaimAction,
    evidence: Evidence,
    feature_mode: str = "simple",
) -> dict[str, float]:
    """Represent the post-inspection state without reference or judge features."""

    del report  # Template prose is deliberately excluded from the reward model.
    normal = any("normal ecg" in label.lower() for label in labels)
    label_count = min(8, len(labels))
    features: dict[str, float] = {
        f"claim={claim_id}": 1.0,
        f"action={action.value}": 1.0,
        f"claim_action={claim_id}:{action.value}": 1.0,
        "label_count": float(label_count) / 8.0,
        "selected_count": float(len(selected_claims)) / 4.0,
        "asserts_normal": float(normal),
        "evidence_reliability": float(evidence.reliability),
        "backend_agreement": float(evidence.backend_agreement or 0.0),
    }
    if feature_mode not in ("simple", "interactions", "clinical"):
        raise ValueError(f"unknown feature mode: {feature_mode}")
    if feature_mode == "interactions":
        features[f"claim_normal={claim_id}:{normal}"] = 1.0
        features[f"claim_label_count={claim_id}:{label_count}"] = 1.0
    for selected in selected_claims:
        features[f"selected={selected}"] = 1.0
    for present in _positive_claims("", labels):
        features[f"present_claim={present}"] = 1.0

    label_text = " ".join(labels).lower()
    tokens = TOKEN.findall(label_text)
    if feature_mode == "interactions":
        features["label_word_count"] = float(min(100, len(tokens))) / 100.0
        features["narrative_label"] = float(len(tokens) > 40)
    if feature_mode != "clinical":
        for token in set(tokens):
            features[f"label_token={token}"] = 1.0
            if feature_mode == "interactions":
                features[f"claim_label_token={claim_id}:{token}"] = 1.0
        for left, right in set(pairwise(tokens)):
            features[f"label_bigram={left}_{right}"] = 1.0
            if feature_mode == "interactions":
                features[f"claim_label_bigram={claim_id}:{left}_{right}"] = 1.0
    if feature_mode in ("interactions", "clinical"):
        for group, aliases in DIAGNOSTIC_GROUPS.items():
            if any(alias in label_text for alias in aliases):
                features[f"diagnostic_group={group}"] = 1.0
                if feature_mode == "interactions":
                    features[f"claim_diagnostic_group={claim_id}:{group}"] = 1.0

    value = evidence.value
    if isinstance(value, bool):
        features["evidence_boolean"] = float(value)
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if np.isfinite(numeric):
            if claim_id.startswith("axis_"):
                features["measurement_scaled"] = numeric / 180.0
                if claim_id == "axis_left":
                    features["threshold_margin"] = (-30.0 - numeric) / 60.0
                elif claim_id == "axis_right":
                    features["threshold_margin"] = (numeric - 90.0) / 90.0
            elif claim_id.startswith("pr_"):
                features["measurement_scaled"] = numeric / 300.0
                features["threshold_margin"] = (numeric - 200.0) / 100.0
            else:
                features["measurement_scaled"] = numeric
            if feature_mode == "interactions":
                features[f"claim_measurement={claim_id}"] = features[
                    "measurement_scaled"
                ]
    spread = evidence.provenance.get("spread")
    if isinstance(spread, (int, float)) and np.isfinite(float(spread)):
        features["measurement_spread"] = float(spread) / 100.0
    if feature_mode == "clinical":
        features.update(_clinical_evidence_fingerprint(claim_id, evidence))
    return features


def build_evidence_action_rows(
    episodes: Sequence[DirectEpisode],
    instruments: Mapping[str, Mapping[str, Any]],
    candidate_claims: Sequence[str] = CACHE_EVALUABLE_CLAIMS,
) -> tuple[EvidenceActionRow, ...]:
    """Pair executable supportive edits with their exact judged reward."""

    tools = ToolSimulator(instruments)
    verifier = ClinicalEvidenceVerifier()
    by_record = {episode.record_id: episode for episode in episodes}
    rows: list[EvidenceActionRow] = []
    for transition in build_inspection_transitions(
        episodes, instruments, candidate_claims
    ):
        if not transition.edit_available:
            continue
        episode = by_record[transition.record_id]
        variant = episode.variants[transition.selected_claims]
        present = transition.action_claim in _positive_claims(
            variant.report, variant.labels
        )
        action = (
            DirectClaimAction.REVISE_EVIDENCE if present else DirectClaimAction.ADD
        )
        spec = CLAIM_REGISTRY[transition.action_claim]
        evidence = tools.call(
            spec.owner_expert, transition.record_id, transition.action_claim
        )
        claim = Claim(
            claim_id=transition.action_claim,
            protocol_step=spec.protocol_step,
            source="evidence_advantage_training",
            r1_present=present,
            r1_value=present,
            present_in_behavior_report=present,
        )
        if verifier.verify(claim, evidence) is not VerificationStatus.SUPPORTED:
            raise RuntimeError("an executable edit lacks supportive evidence")
        rows.append(
            EvidenceActionRow(
                transition.record_id,
                transition.fold,
                variant.report,
                variant.labels,
                transition.selected_claims,
                transition.action_claim,
                action,
                evidence,
                transition.reward,
            )
        )
    return tuple(rows)


class EvidenceConditionedAdvantagePolicy:
    """Estimate terminal action value after the ECG expert has returned evidence."""

    def __init__(
        self,
        vectorizer: DictVectorizer,
        model: Ridge,
        config: EvidenceAdvantageConfig,
        memory_folds: Sequence[int],
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.config = config
        self.memory_folds = tuple(sorted(memory_folds))
        self.audit = {
            "record_keyed_lookup": False,
            "post_inspection": True,
            "features_exclude_template_prose": True,
            "features_exclude_reference": True,
            "features_exclude_judge": True,
            "config": asdict(config),
            "memory_folds": list(self.memory_folds),
        }

    @classmethod
    def fit(
        cls,
        rows: Sequence[EvidenceActionRow],
        config: EvidenceAdvantageConfig,
    ) -> EvidenceConditionedAdvantagePolicy:
        if not rows:
            raise ValueError("cannot fit an evidence advantage policy without rows")
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(
            [
                evidence_action_features(
                    row.report,
                    row.labels,
                    row.selected_claims,
                    row.claim_id,
                    row.action,
                    row.evidence,
                    config.feature_mode,
                )
                for row in rows
            ]
        )
        model = Ridge(alpha=config.alpha, solver="lsqr")
        model.fit(matrix, np.asarray([row.reward for row in rows], dtype=float))
        return cls(vectorizer, model, config, {row.fold for row in rows})

    def estimate(
        self,
        record_id: str,
        report: str,
        labels: Sequence[str],
        selected_claims: frozenset[str],
        claim_id: str,
        action: DirectClaimAction,
        evidence: Evidence,
        *,
        query_fold: int = -1,
    ) -> ActionValue:
        matrix = self.vectorizer.transform(
            [
                evidence_action_features(
                    report,
                    labels,
                    selected_claims,
                    claim_id,
                    action,
                    evidence,
                    self.config.feature_mode,
                )
            ]
        )
        mean = float(self.model.predict(matrix)[0])
        return ActionValue(
            record_id=record_id,
            selected_claims=selected_claims,
            action_claim=claim_id,
            mean_return=mean,
            uncertainty=0.0,
            safe_return=mean - self.config.margin,
            query_fold=query_fold,
            memory_folds=self.memory_folds,
            model=self.config.key,
        )


class NestedEvidenceAdvantageTrainer:
    """Tune the post-inspection reward model without reading an outer fold."""

    @staticmethod
    def default_grid() -> tuple[EvidenceAdvantageConfig, ...]:
        return tuple(
            EvidenceAdvantageConfig(alpha, margin, feature_mode)
            for feature_mode in ("simple", "clinical")
            for alpha in (0.1, 1.0, 10.0, 100.0)
            for margin in (0.0, 1.0, 2.0, 5.0, 10.0)
        )

    def fit_nested(
        self,
        rows: Sequence[EvidenceActionRow],
        configs: Sequence[EvidenceAdvantageConfig] | None = None,
    ) -> tuple[dict[int, EvidenceConditionedAdvantagePolicy], dict[str, Any]]:
        configs = tuple(configs or self.default_grid())
        folds = sorted({row.fold for row in rows})
        policies: dict[int, EvidenceConditionedAdvantagePolicy] = {}
        tuning: dict[str, Any] = {}
        for outer_fold in folds:
            training_rows = [row for row in rows if row.fold != outer_fold]
            predictions: dict[
                tuple[float, str],
                dict[tuple[str, frozenset[str], str], float],
            ] = {}
            model_configs = sorted(
                {(config.alpha, config.feature_mode) for config in configs}
            )
            for alpha, feature_mode in model_configs:
                oof: dict[tuple[str, frozenset[str], str], float] = {}
                base_config = EvidenceAdvantageConfig(alpha, 0.0, feature_mode)
                for inner_fold in folds:
                    if inner_fold == outer_fold:
                        continue
                    model = EvidenceConditionedAdvantagePolicy.fit(
                        [
                            row
                            for row in rows
                            if row.fold not in (outer_fold, inner_fold)
                        ],
                        base_config,
                    )
                    for row in rows:
                        if row.fold != inner_fold:
                            continue
                        estimate = model.estimate(
                            row.record_id,
                            row.report,
                            row.labels,
                            row.selected_claims,
                            row.claim_id,
                            row.action,
                            row.evidence,
                            query_fold=inner_fold,
                        )
                        oof[(row.record_id, row.selected_claims, row.claim_id)] = (
                            estimate.mean_return
                        )
                predictions[(alpha, feature_mode)] = oof

            scores: dict[EvidenceAdvantageConfig, tuple[float, int]] = {}
            for config in configs:
                selected_rewards = [
                    row.reward
                    if predictions[(config.alpha, config.feature_mode)][
                        (row.record_id, row.selected_claims, row.claim_id)
                    ]
                    > config.margin
                    else 0.0
                    for row in training_rows
                ]
                actions = sum(
                    predictions[(config.alpha, config.feature_mode)][
                        (row.record_id, row.selected_claims, row.claim_id)
                    ]
                    > config.margin
                    for row in training_rows
                )
                scores[config] = (float(np.sum(selected_rewards)), actions)
            selected = max(
                configs,
                key=lambda config: (
                    scores[config][0],
                    -scores[config][1],
                    config.margin,
                    config.alpha,
                    config.feature_mode == "simple",
                ),
            )
            policies[outer_fold] = EvidenceConditionedAdvantagePolicy.fit(
                training_rows, selected
            )
            tuning[str(outer_fold)] = {
                "selected": selected.key,
                "inner_total_gain": scores[selected][0],
                "inner_actions": scores[selected][1],
                "grid": {
                    config.key: {
                        "total_gain": score[0],
                        "actions": score[1],
                    }
                    for config, score in scores.items()
                },
            }
        return policies, {
            "selection": "nested_cross_validation",
            "estimator": "evidence_conditioned_terminal_advantage",
            "record_keyed_lookup": False,
            "outer_folds": folds,
            "tuning": tuning,
            "features_exclude_reference": True,
            "features_exclude_judge": True,
            "judge_used_only_as_offline_action_reward": True,
        }


class ClaimAdaptiveEvidenceAdvantagePolicy:
    """Delegate each clinical action to its nested-selected reward model."""

    def __init__(
        self,
        policies: Mapping[str, EvidenceConditionedAdvantagePolicy],
    ) -> None:
        self.policies = dict(policies)
        self.audit = {
            "record_keyed_lookup": False,
            "claim_adaptive": True,
            "action_specific_values": True,
            "policies": {
                claim_id: policy.audit for claim_id, policy in policies.items()
            },
        }

    def estimate(
        self,
        record_id: str,
        report: str,
        labels: Sequence[str],
        selected_claims: frozenset[str],
        claim_id: str,
        action: DirectClaimAction,
        evidence: Evidence,
        *,
        query_fold: int = -1,
    ) -> ActionValue | None:
        policy = self.policies.get(claim_id)
        if policy is None:
            return None
        return policy.estimate(
            record_id,
            report,
            labels,
            selected_claims,
            claim_id,
            action,
            evidence,
            query_fold=query_fold,
        )


class ClaimAdaptiveEvidenceAdvantageTrainer:
    """Select a separate post-inspection value model for every claim."""

    @staticmethod
    def default_grid() -> tuple[EvidenceAdvantageConfig, ...]:
        return tuple(
            EvidenceAdvantageConfig(alpha, margin, feature_mode)
            for feature_mode in ("simple", "interactions")
            for alpha in (1.0, 10.0, 100.0)
            for margin in (0.0, 1.0, 2.0, 5.0, 10.0)
        )

    def fit_nested(
        self,
        rows: Sequence[EvidenceActionRow],
        configs: Sequence[EvidenceAdvantageConfig] | None = None,
    ) -> tuple[dict[int, ClaimAdaptiveEvidenceAdvantagePolicy], dict[str, Any]]:
        configs = tuple(configs or self.default_grid())
        folds = sorted({row.fold for row in rows})
        claims = sorted({row.claim_id for row in rows})
        policies: dict[int, ClaimAdaptiveEvidenceAdvantagePolicy] = {}
        tuning: dict[str, Any] = {}
        model_configs = sorted(
            {(config.alpha, config.feature_mode) for config in configs}
        )
        for outer_fold in folds:
            training_rows = [row for row in rows if row.fold != outer_fold]
            predictions: dict[
                tuple[float, str],
                dict[tuple[str, frozenset[str], str], float],
            ] = {}
            for alpha, feature_mode in model_configs:
                oof: dict[tuple[str, frozenset[str], str], float] = {}
                base_config = EvidenceAdvantageConfig(alpha, 0.0, feature_mode)
                for inner_fold in folds:
                    if inner_fold == outer_fold:
                        continue
                    model = EvidenceConditionedAdvantagePolicy.fit(
                        [
                            row
                            for row in rows
                            if row.fold not in (outer_fold, inner_fold)
                        ],
                        base_config,
                    )
                    for row in rows:
                        if row.fold != inner_fold:
                            continue
                        value = model.estimate(
                            row.record_id,
                            row.report,
                            row.labels,
                            row.selected_claims,
                            row.claim_id,
                            row.action,
                            row.evidence,
                            query_fold=inner_fold,
                        )
                        oof[(row.record_id, row.selected_claims, row.claim_id)] = (
                            value.mean_return
                        )
                predictions[(alpha, feature_mode)] = oof

            chosen: dict[str, EvidenceAdvantageConfig] = {}
            claim_tuning: dict[str, Any] = {}
            for claim_id in claims:
                claim_rows = [
                    row for row in training_rows if row.claim_id == claim_id
                ]
                scores: dict[EvidenceAdvantageConfig, tuple[float, int]] = {}
                for config in configs:
                    values = predictions[(config.alpha, config.feature_mode)]
                    selected = [
                        row
                        for row in claim_rows
                        if values[(row.record_id, row.selected_claims, row.claim_id)]
                        > config.margin
                    ]
                    scores[config] = (
                        float(sum(row.reward for row in selected)),
                        len(selected),
                    )
                selected_config = max(
                    configs,
                    key=lambda config: (
                        scores[config][0],
                        -scores[config][1],
                        config.margin,
                        config.alpha,
                        config.feature_mode == "simple",
                    ),
                )
                chosen[claim_id] = selected_config
                claim_tuning[claim_id] = {
                    "selected": selected_config.key,
                    "inner_total_gain": scores[selected_config][0],
                    "inner_actions": scores[selected_config][1],
                }
            policies[outer_fold] = ClaimAdaptiveEvidenceAdvantagePolicy(
                {
                    claim_id: EvidenceConditionedAdvantagePolicy.fit(
                        training_rows, config
                    )
                    for claim_id, config in chosen.items()
                }
            )
            tuning[str(outer_fold)] = claim_tuning
        return policies, {
            "selection": "claim_adaptive_nested_cross_validation",
            "estimator": "claim_adaptive_evidence_conditioned_advantage",
            "record_keyed_lookup": False,
            "action_specific_values": True,
            "outer_folds": folds,
            "tuning": tuning,
            "features_exclude_reference": True,
            "features_exclude_judge": True,
            "judge_used_only_as_offline_action_reward": True,
        }
