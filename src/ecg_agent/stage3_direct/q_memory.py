"""Fitted-Q replay memory and nested cross-fold planner training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from ecg_agent.experts import ToolSimulator
from ecg_agent.registry import CLAIM_REGISTRY
from ecg_agent.schema import Claim, VerificationStatus

from .claim_parser import DiagnosticClaimParser
from .features import state_action_features
from .schema import ActionValue, DirectEpisode, ReplayTransition
from .verifier import ClinicalEvidenceVerifier

CACHE_EVALUABLE_CLAIMS = (
    "axis_left",
    "axis_right",
    "pr_prolonged",
    "low_voltage",
)


@dataclass(frozen=True)
class PlannerConfig:
    model: str
    min_leaf: int
    risk_z: float

    @property
    def key(self) -> str:
        return f"{self.model}:leaf={self.min_leaf}:risk={self.risk_z:g}"


@dataclass(frozen=True)
class HierarchicalConfig:
    shrinkage: float
    risk_z: float
    context_level: str = "label_count"

    @property
    def key(self) -> str:
        return (
            f"hierarchical_q:context={self.context_level}:"
            f"shrinkage={self.shrinkage:g}:risk={self.risk_z:g}"
        )


class QMemory:
    def __init__(
        self,
        estimates: Mapping[tuple[str, frozenset[str], str], ActionValue],
        *,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        self.estimates = dict(estimates)
        self.audit = dict(audit or {})

    def retrieve(
        self, record_id: str, selected_claims: frozenset[str], action_claim: str
    ) -> ActionValue | None:
        return self.estimates.get((record_id, selected_claims, action_claim))

    def claims_for_record(self, record_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    claim_id
                    for row_id, _, claim_id in self.estimates
                    if row_id == record_id
                }
            )
        )


def build_replay_transitions(
    episodes: Sequence[DirectEpisode],
) -> tuple[ReplayTransition, ...]:
    transitions: list[ReplayTransition] = []
    for episode in episodes:
        for selected, variant in episode.variants.items():
            for action_claim in episode.proposal_ids:
                if action_claim in selected:
                    continue
                target = selected | {action_claim}
                if target not in episode.variants:
                    continue
                transitions.append(
                    ReplayTransition(
                        record_id=episode.record_id,
                        fold=episode.fold,
                        selected_claims=selected,
                        action_claim=action_claim,
                        next_claims=target,
                        reward=(
                            episode.variants[target].diagnosis_score
                            - variant.diagnosis_score
                        ),
                    )
                )
    return tuple(transitions)


def build_inspection_transitions(
    episodes: Sequence[DirectEpisode],
    instruments: Mapping[str, Mapping[str, Any]],
    candidate_claims: Sequence[str] = CACHE_EVALUABLE_CLAIMS,
) -> tuple[ReplayTransition, ...]:
    """Build pre-inspection actions, retaining censored edits as query states."""

    tools = ToolSimulator(instruments)
    verifier = ClinicalEvidenceVerifier()
    transitions: list[ReplayTransition] = []
    for episode in episodes:
        for selected, variant in episode.variants.items():
            for action_claim in candidate_claims:
                if action_claim in selected:
                    continue
                spec = CLAIM_REGISTRY[action_claim]
                evidence = tools.call(
                    spec.owner_expert, episode.record_id, action_claim
                )
                claim = Claim(
                    claim_id=action_claim,
                    protocol_step=spec.protocol_step,
                    source="planner_query",
                    r1_present=False,
                    r1_value=None,
                )
                supported = (
                    verifier.verify(claim, evidence) is VerificationStatus.SUPPORTED
                )
                target = selected | {action_claim}
                edit_available = supported and target in episode.variants
                reward_observed = edit_available or not supported
                transitions.append(
                    ReplayTransition(
                        record_id=episode.record_id,
                        fold=episode.fold,
                        selected_claims=selected,
                        action_claim=action_claim,
                        next_claims=target if edit_available else selected,
                        reward=(
                            episode.variants[target].diagnosis_score
                            - variant.diagnosis_score
                            if edit_available
                            else 0.0
                        ),
                        reward_observed=reward_observed,
                        edit_available=edit_available,
                    )
                )
    return tuple(transitions)


class CrossFittedQTrainer:
    """Fit action values only from other folds, optionally tuning in nested CV."""

    def __init__(
        self,
        *,
        outer_trees: int = 500,
        inner_trees: int = 150,
        seed: int = 9,
    ) -> None:
        self.outer_trees = int(outer_trees)
        self.inner_trees = int(inner_trees)
        self.seed = int(seed)

    @staticmethod
    def default_grid() -> tuple[PlannerConfig, ...]:
        return tuple(
            PlannerConfig(model, leaf, risk)
            for model in ("random_forest", "extra_trees")
            for leaf in (12, 20, 30)
            for risk in (0.0, 0.1, 0.25)
        )

    def _model(self, config: PlannerConfig, trees: int):
        model_class = (
            RandomForestRegressor
            if config.model == "random_forest"
            else ExtraTreesRegressor
        )
        return model_class(
            n_estimators=trees,
            min_samples_leaf=config.min_leaf,
            max_features=0.5,
            random_state=self.seed,
            n_jobs=-1,
        )

    def _predict(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        train: np.ndarray,
        query: np.ndarray,
        config: PlannerConfig,
        trees: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = self._model(config, trees)
        model.fit(matrix[train], target[train])
        tree_predictions = np.asarray(
            [tree.predict(matrix[query]) for tree in model.estimators_]
        )
        return tree_predictions.mean(axis=0), tree_predictions.std(axis=0, ddof=0)

    @staticmethod
    def _rollout(
        episode: DirectEpisode,
        transition_index: Mapping[tuple[str, frozenset[str], str], int],
        means: np.ndarray,
        uncertainty: np.ndarray,
        risk_z: float,
    ) -> tuple[frozenset[str], int]:
        selected: frozenset[str] = frozenset()
        attempted: set[str] = set()
        calls = 0
        for _ in CACHE_EVALUABLE_CLAIMS:
            candidates: list[tuple[float, float, str]] = []
            for action_claim in CACHE_EVALUABLE_CLAIMS:
                key = (episode.record_id, selected, action_claim)
                if action_claim in attempted or key not in transition_index:
                    continue
                index = transition_index[key]
                safe = means[index] - risk_z * uncertainty[index]
                candidates.append((float(safe), float(means[index]), action_claim))
            if not candidates:
                break
            safe, _, action_claim = max(candidates)
            if safe <= 0.0:
                break
            attempted.add(action_claim)
            calls += 1
            target = selected | {action_claim}
            if target in episode.variants:
                selected = target
        return selected, calls

    def fit_fixed_crossfit(
        self,
        episodes: Sequence[DirectEpisode],
        instruments: Mapping[str, Mapping[str, Any]],
        config: PlannerConfig,
    ) -> QMemory:
        transitions = build_inspection_transitions(episodes, instruments)
        matrix, target, folds, _ = self._matrix(episodes, transitions, instruments)
        observed = np.asarray(
            [transition.reward_observed for transition in transitions], dtype=bool
        )
        means = np.zeros(len(transitions), dtype=float)
        uncertainty = np.zeros(len(transitions), dtype=float)
        for query_fold in sorted(set(folds)):
            train = np.where((folds != query_fold) & observed)[0]
            query = np.where(folds == query_fold)[0]
            means[query], uncertainty[query] = self._predict(
                matrix,
                target,
                train,
                query,
                config,
                self.outer_trees,
            )
        return self._memory(
            transitions,
            means,
            uncertainty,
            {fold: config for fold in sorted(set(folds))},
            {"selection": "fixed_development_configuration"},
        )

    def fit_nested_crossfit(
        self,
        episodes: Sequence[DirectEpisode],
        instruments: Mapping[str, Mapping[str, Any]],
        configs: Sequence[PlannerConfig] | None = None,
    ) -> QMemory:
        configs = tuple(configs or self.default_grid())
        transitions = build_inspection_transitions(episodes, instruments)
        matrix, target, folds, transition_index = self._matrix(
            episodes, transitions, instruments
        )
        means = np.zeros(len(transitions), dtype=float)
        uncertainty = np.zeros(len(transitions), dtype=float)
        observed = np.asarray(
            [transition.reward_observed for transition in transitions], dtype=bool
        )
        selected_configs: dict[int, PlannerConfig] = {}
        tuning: dict[str, Any] = {}
        episode_folds = sorted({episode.fold for episode in episodes})
        for outer_fold in episode_folds:
            training_folds = [fold for fold in episode_folds if fold != outer_fold]
            model_predictions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
            for model_name, leaf in sorted(
                {(config.model, config.min_leaf) for config in configs}
            ):
                inner_means = np.zeros(len(transitions), dtype=float)
                inner_uncertainty = np.zeros(len(transitions), dtype=float)
                model_config = PlannerConfig(model_name, leaf, 0.0)
                for inner_fold in training_folds:
                    train = np.where(
                        (folds != outer_fold) & (folds != inner_fold) & observed
                    )[0]
                    query = np.where(folds == inner_fold)[0]
                    prediction, spread = self._predict(
                        matrix,
                        target,
                        train,
                        query,
                        model_config,
                        self.inner_trees,
                    )
                    inner_means[query] = prediction
                    inner_uncertainty[query] = spread
                model_predictions[(model_name, leaf)] = (
                    inner_means,
                    inner_uncertainty,
                )

            scores: dict[PlannerConfig, tuple[float, int]] = {}
            training_episodes = [
                episode for episode in episodes if episode.fold != outer_fold
            ]
            for config in configs:
                prediction, spread = model_predictions[(config.model, config.min_leaf)]
                gains: list[float] = []
                actions = 0
                for episode in training_episodes:
                    selected, calls = self._rollout(
                        episode,
                        transition_index,
                        prediction,
                        spread,
                        config.risk_z,
                    )
                    gains.append(
                        episode.variants[selected].diagnosis_score - episode.base_score
                    )
                    actions += calls
                scores[config] = (float(np.mean(gains)), actions)
            selected_config = max(
                configs,
                key=lambda config: (
                    scores[config][0],
                    -scores[config][1],
                    config.risk_z,
                    config.min_leaf,
                    config.model,
                ),
            )
            selected_configs[outer_fold] = selected_config
            train = np.where((folds != outer_fold) & observed)[0]
            query = np.where(folds == outer_fold)[0]
            means[query], uncertainty[query] = self._predict(
                matrix,
                target,
                train,
                query,
                selected_config,
                self.outer_trees,
            )
            tuning[str(outer_fold)] = {
                "selected": selected_config.key,
                "inner_mean_gain": scores[selected_config][0],
                "inner_actions": scores[selected_config][1],
                "grid": {
                    config.key: {
                        "mean_gain": values[0],
                        "actions": values[1],
                    }
                    for config, values in scores.items()
                },
            }
        return self._memory(
            transitions,
            means,
            uncertainty,
            selected_configs,
            {
                "selection": "nested_cross_validation",
                "outer_trees": self.outer_trees,
                "inner_trees": self.inner_trees,
                "tuning": tuning,
            },
        )

    @staticmethod
    def _matrix(
        episodes: Sequence[DirectEpisode],
        transitions: Sequence[ReplayTransition],
        instruments: Mapping[str, Mapping[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple, int]]:
        by_record = {episode.record_id: episode for episode in episodes}
        features = [
            state_action_features(
                by_record[transition.record_id],
                transition,
                instruments[transition.record_id],
            )
            for transition in transitions
        ]
        matrix = DictVectorizer(sparse=False).fit_transform(features)
        target = np.asarray(
            [transition.reward for transition in transitions], dtype=float
        )
        folds = np.asarray([transition.fold for transition in transitions], dtype=int)
        index = {
            transition.key: position for position, transition in enumerate(transitions)
        }
        return matrix, target, folds, index

    @staticmethod
    def _memory(
        transitions: Sequence[ReplayTransition],
        means: np.ndarray,
        uncertainty: np.ndarray,
        configs: Mapping[int, PlannerConfig],
        audit: Mapping[str, Any],
    ) -> QMemory:
        all_folds = sorted({transition.fold for transition in transitions})
        estimates: dict[tuple[str, frozenset[str], str], ActionValue] = {}
        for index, transition in enumerate(transitions):
            config = configs[transition.fold]
            estimates[transition.key] = ActionValue(
                record_id=transition.record_id,
                selected_claims=transition.selected_claims,
                action_claim=transition.action_claim,
                mean_return=float(means[index]),
                uncertainty=float(uncertainty[index]),
                safe_return=float(means[index] - config.risk_z * uncertainty[index]),
                query_fold=transition.fold,
                memory_folds=tuple(
                    fold for fold in all_folds if fold != transition.fold
                ),
                model=config.key,
            )
        return QMemory(
            estimates,
            audit={
                **dict(audit),
                "transitions": len(transitions),
                "observed_rewards": sum(row.reward_observed for row in transitions),
                "censored_rewards": sum(not row.reward_observed for row in transitions),
                "planner_observes_expert_result_before_action": False,
                "features_exclude_reference": True,
                "features_exclude_judge": True,
                "features_exclude_RAG": True,
                "features_exclude_Memory_reports": True,
                "selected_configs": {
                    str(fold): asdict(config) for fold, config in configs.items()
                },
            },
        )


class HierarchicalQTrainer:
    """Empirical-Bayes Q memory over claim, report context, and graph depth."""

    def __init__(self) -> None:
        self._context_cache: dict[tuple[str, str], tuple[bool, int, bool]] = {}

    @staticmethod
    def default_grid() -> tuple[HierarchicalConfig, ...]:
        return tuple(
            HierarchicalConfig(shrinkage, risk_z, "normal")
            for shrinkage in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
            for risk_z in (0.0, 0.25, 0.5, 1.0)
        )

    @staticmethod
    def _normal_context(episode: DirectEpisode) -> bool:
        return any("normal ecg" in label.lower() for label in episode.base_labels)

    def _contexts(
        self, episodes: Sequence[DirectEpisode]
    ) -> Mapping[tuple[str, str], tuple[bool, int, bool]]:
        for episode in episodes:
            if (episode.record_id, CACHE_EVALUABLE_CLAIMS[0]) in self._context_cache:
                continue
            positive_claims = {
                claim.claim_id
                for claim in DiagnosticClaimParser().parse(
                    episode.base_report, list(episode.base_labels)
                )
                if claim.r1_value is True
            }
            base = (self._normal_context(episode), min(3, len(episode.base_labels)))
            for claim_id in CACHE_EVALUABLE_CLAIMS:
                self._context_cache[(episode.record_id, claim_id)] = (
                    *base,
                    claim_id in positive_claims,
                )
        return self._context_cache

    def _fit_query(
        self,
        episodes: Sequence[DirectEpisode],
        transitions: Sequence[ReplayTransition],
        query_fold: int,
        excluded_folds: set[int],
        config: HierarchicalConfig,
    ) -> dict[tuple[str, frozenset[str], str], ActionValue]:
        memory = [
            transition
            for transition in transitions
            if transition.fold not in excluded_folds and transition.reward_observed
        ]
        memory_folds = tuple(sorted({transition.fold for transition in memory}))
        global_buckets: dict[str, list[float]] = defaultdict(list)
        context = self._contexts(episodes)

        def local_key(row: ReplayTransition) -> tuple[Any, ...]:
            normal, label_count, present = context[(row.record_id, row.action_claim)]
            prefix: tuple[Any, ...] = (
                row.action_claim,
                len(row.selected_claims),
            )
            if config.context_level == "global":
                return prefix
            if config.context_level == "normal":
                return (*prefix, normal)
            if config.context_level == "label_count":
                return (*prefix, label_count)
            if config.context_level == "full":
                return (*prefix, normal, label_count, present)
            raise ValueError(f"unknown context level: {config.context_level}")

        local_buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in memory:
            global_buckets[row.action_claim].append(row.reward)
            local_buckets[local_key(row)].append(row.reward)
        estimates: dict[tuple[str, frozenset[str], str], ActionValue] = {}
        for transition in transitions:
            if transition.fold != query_fold:
                continue
            global_values = np.asarray(
                global_buckets[transition.action_claim], dtype=float
            )
            local_values = np.asarray(local_buckets[local_key(transition)], dtype=float)
            if not len(global_values):
                raise RuntimeError(
                    f"no historical action values for {transition.action_claim}"
                )
            global_mean = float(global_values.mean())
            denominator = len(local_values) + config.shrinkage
            mean = (
                float(
                    (local_values.sum() + config.shrinkage * global_mean) / denominator
                )
                if denominator
                else global_mean
            )
            if len(local_values) >= 2:
                uncertainty = float(
                    local_values.std(ddof=1) / np.sqrt(len(local_values))
                )
            elif len(global_values) >= 2:
                uncertainty = float(
                    global_values.std(ddof=1) / np.sqrt(len(global_values))
                )
            else:
                uncertainty = float("inf")
            estimates[transition.key] = ActionValue(
                record_id=transition.record_id,
                selected_claims=transition.selected_claims,
                action_claim=transition.action_claim,
                mean_return=mean,
                uncertainty=uncertainty,
                safe_return=float(mean - config.risk_z * uncertainty),
                query_fold=query_fold,
                memory_folds=memory_folds,
                model=config.key,
            )
        return estimates

    @staticmethod
    def _rollout(episode: DirectEpisode, memory: QMemory) -> tuple[frozenset[str], int]:
        selected: frozenset[str] = frozenset()
        attempted: set[str] = set()
        calls = 0
        for _ in CACHE_EVALUABLE_CLAIMS:
            candidates: list[tuple[float, float, str]] = []
            for action_claim in CACHE_EVALUABLE_CLAIMS:
                if action_claim in attempted:
                    continue
                estimate = memory.retrieve(episode.record_id, selected, action_claim)
                if estimate is not None:
                    candidates.append(
                        (
                            estimate.safe_return,
                            estimate.mean_return,
                            action_claim,
                        )
                    )
            if not candidates:
                break
            safe, _, action_claim = max(candidates)
            if safe <= 0.0:
                break
            attempted.add(action_claim)
            calls += 1
            target = selected | {action_claim}
            if target in episode.variants:
                selected = target
        return selected, calls

    def fit_fixed_crossfit(
        self,
        episodes: Sequence[DirectEpisode],
        instruments: Mapping[str, Mapping[str, Any]],
        config: HierarchicalConfig,
    ) -> QMemory:
        transitions = build_inspection_transitions(episodes, instruments)
        folds = sorted({episode.fold for episode in episodes})
        estimates = {}
        for query_fold in folds:
            estimates.update(
                self._fit_query(
                    episodes,
                    transitions,
                    query_fold,
                    {query_fold},
                    config,
                )
            )
        return QMemory(
            estimates,
            audit={
                "selection": "fixed_development_configuration",
                "estimator": "empirical_Bayes_hierarchical_Q",
                "transitions": len(transitions),
                "observed_rewards": sum(row.reward_observed for row in transitions),
                "censored_rewards": sum(not row.reward_observed for row in transitions),
                "planner_observes_expert_result_before_action": False,
                "selected_configs": {str(fold): asdict(config) for fold in folds},
                "features_exclude_reference": True,
                "features_exclude_judge": True,
                "features_exclude_RAG": True,
                "features_exclude_Memory_reports": True,
            },
        )

    def fit_nested_crossfit(
        self,
        episodes: Sequence[DirectEpisode],
        instruments: Mapping[str, Mapping[str, Any]],
        configs: Sequence[HierarchicalConfig] | None = None,
    ) -> QMemory:
        configs = tuple(configs or self.default_grid())
        transitions = build_inspection_transitions(episodes, instruments)
        folds = sorted({episode.fold for episode in episodes})
        final_estimates = {}
        tuning: dict[str, Any] = {}
        selected_configs: dict[str, dict[str, Any]] = {}
        for outer_fold in folds:
            training_episodes = [
                episode for episode in episodes if episode.fold != outer_fold
            ]
            scores: dict[HierarchicalConfig, tuple[float, int]] = {}
            for config in configs:
                inner_estimates = {}
                for inner_fold in folds:
                    if inner_fold == outer_fold:
                        continue
                    inner_estimates.update(
                        self._fit_query(
                            episodes,
                            transitions,
                            inner_fold,
                            {outer_fold, inner_fold},
                            config,
                        )
                    )
                memory = QMemory(inner_estimates)
                gains = []
                actions = 0
                for episode in training_episodes:
                    selected, calls = self._rollout(episode, memory)
                    gains.append(
                        episode.variants[selected].diagnosis_score - episode.base_score
                    )
                    actions += calls
                scores[config] = (float(np.mean(gains)), actions)
            selected = max(
                configs,
                key=lambda config: (
                    scores[config][0],
                    -scores[config][1],
                    config.risk_z,
                    -config.shrinkage,
                ),
            )
            final_estimates.update(
                self._fit_query(
                    episodes,
                    transitions,
                    outer_fold,
                    {outer_fold},
                    selected,
                )
            )
            selected_configs[str(outer_fold)] = asdict(selected)
            tuning[str(outer_fold)] = {
                "selected": selected.key,
                "inner_mean_gain": scores[selected][0],
                "inner_actions": scores[selected][1],
                "grid": {
                    config.key: {
                        "mean_gain": values[0],
                        "actions": values[1],
                    }
                    for config, values in scores.items()
                },
            }
        return QMemory(
            final_estimates,
            audit={
                "selection": "nested_cross_validation",
                "estimator": "empirical_Bayes_hierarchical_Q",
                "transitions": len(transitions),
                "observed_rewards": sum(row.reward_observed for row in transitions),
                "censored_rewards": sum(not row.reward_observed for row in transitions),
                "planner_observes_expert_result_before_action": False,
                "selected_configs": selected_configs,
                "tuning": tuning,
                "features_exclude_reference": True,
                "features_exclude_judge": True,
                "features_exclude_RAG": True,
                "features_exclude_Memory_reports": True,
            },
        )
