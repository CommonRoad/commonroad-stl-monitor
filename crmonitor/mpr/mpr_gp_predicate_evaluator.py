import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crmonitor.common import ScenarioType
from crmonitor.common.world import World
from crmonitor.mpr.learning import FeatureExtractor, read_model
from crmonitor.predicates.base import AbstractPredicate, PredicateName

_LOGGER = logging.getLogger(__name__)


@dataclass
class MprGpPredicateEvaluationResult:
    """
    Results from Gaussian Process-based Model Predictive Predicate Robustness evaluation.
    """

    robustness: float
    """Predicted robustness value from the GP model."""

    satisfied: bool
    """Boolean satisfaction of the predicate in the current state."""

    std: float
    """Standard deviation/uncertainty of the GP prediction."""

    gradient: float | None = None
    """Optional gradient information for sensitivity analysis."""

    @property
    def characteristic_value(self) -> float:
        characteristic_value = -1.0 if self.satisfied else 1.0
        return characteristic_value

    def prediction_matches_reality(self) -> bool:
        """
        Check if the GP model's robustness prediction aligns with actual predicate satisfaction.

        Returns:
            True if the prediction sign matches the expected sign based on satisfaction.
        """
        return np.sign(self.robustness) == np.sign(self.characteristic_value)


@dataclass(kw_only=True, frozen=True)
class MprGpPredicateEvaluatorConfig:
    model_path: Path | None = None
    """Optional path to GP models. If no path is provided, the models are loaded from the repo."""

    rectification: bool = False
    """Whether to apply rectification to ensure prediction consistency."""

    extract_gradient: bool = False
    """Whether to compute gradients for sensitivity analysis. Gradients show how robustness changes with input features."""

    eps: float = 1e-3


class MprGpPredicateEvaluator:
    """Gaussian Process-based Model Predictive Predicate Robustness evaluator."""

    def __init__(
        self,
        predicates: Iterable[AbstractPredicate],
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
        config: MprGpPredicateEvaluatorConfig | None = None,
    ) -> None:
        self._predicates = predicates
        if config is None:
            config = MprGpPredicateEvaluatorConfig()
        self._config = config

        self._gp_models = {}
        all_predicates_desired_features = defaultdict(set)
        for predicate in predicates:
            model_container = read_model(
                str(predicate.predicate_name), self._config.model_path, scenario_type
            )
            self._gp_models[predicate.predicate_name] = model_container

            for agent_combination, feature_variables in model_container.features.items():
                all_predicates_desired_features[agent_combination].update(feature_variables)

        self._feature_extractor = FeatureExtractor(
            dict(all_predicates_desired_features),
        )

    @property
    def config(self) -> MprGpPredicateEvaluatorConfig:
        return self._config

    def evaluate(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> dict[PredicateName, MprGpPredicateEvaluationResult]:
        """
        Evaluate all predicates using GPs at a given time step.

        Args:
            world: Complete scenario world state containing all vehicles and environment.
            time_step: Current time step for evaluation.
            vehicle_ids: Tuple of vehicle IDs (ego vehicle first, then other vehicles).

        Returns:
            Mapping from predicate names to their GP-based evaluation results.
        """
        feature_values = self._feature_extractor.extract_feature_values(
            world=world, time_step=time_step, vehicle_ids=vehicle_ids
        )

        results = {}
        for predicate in self._predicates:
            model_container = self._gp_models[predicate.predicate_name]

            satisfied = predicate.evaluate_boolean(world, time_step, vehicle_ids)
            characteristic_value = 1.0 if satisfied else -1.0

            # The feature extractor is configured to extract the union of all desired features variables.
            # Therefore, we need to narrow the feature values to only the desired values for the current predicate.
            feature_values_list = feature_values.as_list(model_container.features)

            # Each model always requires the characteristic values as feature.
            # TODO: This is not very elegant.
            feature_values_list.append(characteristic_value)

            predicted_robustness, std = model_container.model.predict([feature_values_list])

            gradient = None
            if self._config.extract_gradient:
                gradient = model_container.model.get_gradient([feature_values_list])

            result = MprGpPredicateEvaluationResult(
                robustness=predicted_robustness, satisfied=satisfied, std=std, gradient=gradient
            )
            results[predicate.predicate_name] = result

        return results
