import logging

from crmonitor.common.scenario_type import ScenarioType
from crmonitor.mpr.learning.feature_variables import VelocityFeatureVariable
from crmonitor.predicates.base import BasePredicateEvaluator, PredicateName
from crmonitor.predicates.predicate_registry import PredicateRegistry, PredicateRegistryExtension
from crmonitor.predicates.velocity import VelocityPredicates

from .feature_variables import (
    DesiredFeatureVariables,
    FeatureVariableAgentCombination,
    default_feature_variable_classes_for_scenario_type,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_PREDICATES_DESIRED_FEATURES = {
    VelocityPredicates.VelocityBelow2: {
        FeatureVariableAgentCombination.EGO: VelocityFeatureVariable
    },
    VelocityPredicates.VelocityBelow5: {
        FeatureVariableAgentCombination.EGO: VelocityFeatureVariable
    },
}


class FeatureRegistryExtension(PredicateRegistryExtension):
    def __init__(self) -> None:
        self._desired_features: dict[PredicateName, DesiredFeatureVariables] = {}

    def set_desired_features(
        self, predicate_name: PredicateName, desired_features: DesiredFeatureVariables
    ) -> None:
        self._desired_features[predicate_name] = desired_features

    def get_desired_features(self, predicate_name: PredicateName) -> DesiredFeatureVariables | None:
        if predicate_name in self._desired_features:
            return self._desired_features[predicate_name]
        return None


def initialize_feature_predicate_registry_extension():
    registry = PredicateRegistry.get_registry()
    extension = registry.get_extension(FeatureRegistryExtension)
    if extension is None:
        extension = FeatureRegistryExtension()
        registry.register_extension(extension)
        for predicate_name, desired_features in DEFAULT_PREDICATES_DESIRED_FEATURES.items():
            extension.set_desired_features(predicate_name, desired_features)


def get_desired_features_for_predicate(
    predicate: BasePredicateEvaluator, scenario_type: ScenarioType = ScenarioType.INTERSTATE
) -> DesiredFeatureVariables | None:
    registry = PredicateRegistry.get_registry()
    extension = registry.get_extension(FeatureRegistryExtension)
    if extension is None:
        raise RuntimeError("Feature registry extension was not registered")

    desired_features = extension.get_desired_features(predicate.predicate_name)
    if desired_features is not None:
        return desired_features

    default_desired_features = default_feature_variable_classes_for_scenario_type(scenario_type)

    arity_adjusted_desired_features = {}
    for agent_combination, feature_variables in default_desired_features.items():
        if agent_combination.arity > predicate.arity:
            continue

        arity_adjusted_desired_features[agent_combination] = feature_variables

    return arity_adjusted_desired_features
