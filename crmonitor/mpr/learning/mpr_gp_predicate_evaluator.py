from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crmonitor.common import ScenarioType
from crmonitor.common.world import World
from crmonitor.predicates.base import BasePredicateEvaluator, PredicateName

from .exact_gp_model import read_model
from .feature_extractor import FeatureExtractor, FeatureVariableAgentCombination


@dataclass
class MprGpPredicateEvaluationResult:
    robustness: float
    satisfied: bool
    std: float
    gradient: float | None = None

    def prediction_matches_reality(self) -> bool:
        characteristic_value = -1.0 if self.satisfied else 1.0
        return np.sign(self.robustness) == np.sign(characteristic_value)


@dataclass(kw_only=True, frozen=True)
class MprGpPredicateEvaluatorConfig:
    model_path: Path | None = None
    rectification: bool = False
    extract_gradient: bool = False
    eps: float = 1e-3


class MprGpPredicateEvaluator:
    def __init__(
        self,
        predicates: Iterable[BasePredicateEvaluator],
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
        config: MprGpPredicateEvaluatorConfig | None = None,
    ) -> None:
        """
        Creates an evaluator for `predicate_names` by loading the models from `model_path`. If one of the models cannot be found, an error is raised.
        """
        self._predicates = predicates
        if config is None:
            config = MprGpPredicateEvaluatorConfig()
        self._config = config

        self._feature_extractors = {
            predicate.predicate_name: FeatureExtractor.for_predicate_evaluator(
                predicate, scenario_type
            )
            for predicate in self._predicates
        }
        self._gp_models = {
            predicate.predicate_name: read_model(
                str(predicate.predicate_name), self._config.model_path, scenario_type
            )
            for predicate in self._predicates
        }

    def evaluate(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> dict[PredicateName, float]:
        """evaluate predicate robustness using machine learning models

        Args:
            world_state (WroldState):
            vehicles (List[Vehicle]): [ego_vehicle, other_vehicle]. If world_state is missing, generate from it
            time_step (int):
            scenario (Scenario): If world_state is missing, generate from it
            vehicle_ids (List[int]): [ego_id, other_id]. If vehicles is missing, get vehicles from it.
            states (List[VehicleState]): [ego_state, other_state]. If world_state, vehicles, or time_step
                are missing, get them from it.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (robustness, std)

        """
        feature_values = self._feature_extractor.extract_feature_values(
            world=world, time_step=time_step, vehicle_ids=vehicle_ids
        )

        results = {}
        for predicate in self._predicates:
            model_container = self._gp_models[predicate.predicate_name]

            satisfied = predicate.evaluate_boolean(world, time_step, vehicle_ids)
            characteristic_value = 1.0 if satisfied else -1.0
            feature_values.add_feature_variable_value(
                FeatureVariableAgentCombination.EGO, {"characteristic_value": characteristic_value}
            )
            predicted_robustness, std = model_container.model.predict(feature_values.as_list())

            gradient = None
            if self._config.extract_gradient:
                gradient = model_container.model.get_gradient(feature_values.as_list())

            result = MprGpPredicateEvaluationResult(
                robustness=predicted_robustness, satisfied=satisfied, std=std, gradient=gradient
            )
            results[predicate.predicate_name] = result

        return results
