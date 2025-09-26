from collections.abc import Iterable
from enum import Enum
from typing import Any

from crmonitor.common import ScenarioType, World
from crmonitor.mpr.learning.feature_variables import (
    FeatureVariableAgentCombination,
    default_feature_variable_classes_for_scenario_type,
)
from crmonitor.mpr.state_context import StateContext
from crmonitor.predicates import AbstractPredicate

from .feature_variables import AbstractFeatureVariable, DesiredFeatureVariables


class DesiredPredicateEvaluation(Enum):
    BOOL = "bool"
    # TODO: More desired predicate evaluation options?


class FeatureVariablesValueCollection:
    def __init__(self) -> None:
        self._feature_values = {
            agent_combination: {} for agent_combination in FeatureVariableAgentCombination
        }

    def add_feature_variable_value(
        self,
        agent_combination: FeatureVariableAgentCombination,
        feature_values: dict[str, Any],
    ) -> None:
        self._feature_values[agent_combination].update(feature_values)

    def as_list(self, include: DesiredFeatureVariables) -> list[float]:
        feature_values_list = []
        label_list = []

        for agent_combination, desired_features in include.items():
            for desired_feature in desired_features:
                for label, value in self._feature_values[agent_combination].items():
                    if not desired_feature.provides(label):
                        continue

                    feature_values_list.append(value)
                    label_list.append(label)

        return feature_values_list

    def as_dict(self) -> dict[FeatureVariableAgentCombination, dict[str, float]]:
        return self._feature_values


class FeatureExtractor:
    def __init__(
        self,
        feature_variables: dict[
            FeatureVariableAgentCombination, Iterable[type[AbstractFeatureVariable]]
        ],
        desired_predicate_evaluation: DesiredPredicateEvaluation = DesiredPredicateEvaluation.BOOL,
    ) -> None:
        self._feature_variables = feature_variables
        self._desired_predicate_evaluation = desired_predicate_evaluation

    @property
    def desired_predicate_evaluation(self) -> DesiredPredicateEvaluation:
        return self._desired_predicate_evaluation

    @property
    def desired_feature_variables(self) -> DesiredFeatureVariables:
        return self._feature_variables

    @classmethod
    def for_scenario_type(
        cls, scenario_type: ScenarioType = ScenarioType.INTERSTATE
    ) -> "FeatureExtractor":
        return cls(default_feature_variable_classes_for_scenario_type(scenario_type))

    def extract_feature_values(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> FeatureVariablesValueCollection:
        ctx = StateContext(time_step=time_step, world=world, vehicle_ids=vehicle_ids)
        reversed_ctx = ctx.reversed()
        feature_variables_value_collection = FeatureVariablesValueCollection()
        for agent_combination, feature_variables in self._feature_variables.items():
            if agent_combination.is_reversed():
                current_ctx = reversed_ctx
            else:
                current_ctx = ctx

            for feature_variable in feature_variables:
                values = feature_variable.extract(current_ctx)
                if values is None:
                    raise RuntimeError(
                        f"Feature variable {feature_variable.name} extracted invalid value 'None'"
                    )
                feature_variables_value_collection.add_feature_variable_value(
                    agent_combination, values
                )

        return feature_variables_value_collection

    def feature_variable_labels(
        self, predicate: type[AbstractPredicate] | AbstractPredicate | str | None = None
    ) -> list[tuple[str, str, str]]:
        labels = []
        for agent_combination, feature_variables in self._feature_variables.items():
            for feature_variable in feature_variables:
                labels.append(("inputs", agent_combination.value, feature_variable.name))

        if predicate is not None:
            if isinstance(predicate, str):
                predicate_name = predicate
            else:
                predicate_name = str(predicate.predicate_name)
            labels.append(
                (
                    "predicates",
                    predicate_name,
                    self.desired_predicate_evaluation.value,
                )
            )
        return labels
