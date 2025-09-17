import copy
import csv
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from crmonitor.common import ScenarioType, Vehicle, World
from crmonitor.mpr.prediction import (
    FutureStateSampler,
    FutureStateSamplerConfig,
    StateBasedSamplingResult,
)
from crmonitor.predicates.base import AbstractPredicate, PredicateName

_LOGGER = logging.getLogger(__name__)


@dataclass
class MprPredicateEvaluationResult:
    """
    Results from Model Predictive Robustness (MPR) evaluation of a single predicate.

    This class encapsulates the statistical results of evaluating a predicate across
    multiple sampled future states, providing both satisfaction counts and robustness metrics.

    Attributes:
        count_valid: Number of successfully evaluated samples (no errors).
        count_true: Number of samples where the predicate was satisfied.
        count_error: Number of samples that resulted in evaluation errors.
        satisfied: Boolean satisfaction of the predicate in the original (non-sampled) state.
        robustness: MPR robustness value derived from sample satisfaction probability.
        mfr_robustness: Model Free Robustness value (optional).
    """

    count_valid: int
    count_true: int
    count_error: int
    satisfied: bool
    robustness: float
    mfr_robustness: float | None = None

    def as_dict(self) -> dict[str, int | bool | float | None]:
        """
        Convert the evaluation result to a dictionary format.

        Returns:
            Dictionary representation with all result fields.
        """
        return {
            "count_valid": self.count_valid,
            "count_true": self.count_true,
            "count_error": self.count_error,
            "bool": self.satisfied,
            "robustness": self.robustness,
            "mfr_robustness": self.mfr_robustness,
        }


class AbstractRobustnessNormalizationProvider(ABC):
    """
    Abstract base class for normalizing robustness values.

    Robustness normalization ensures that robustness values from different predicates
    can be compared on a consistent scale, typically [-1, 1].
    """

    @abstractmethod
    def normalize(self, probability: float, satisfied: bool) -> float: ...


@dataclass
class PredicateNormalizationValues:
    """
    Normalization parameters for a specific predicate's robustness calculation.

    These values define the expected ranges for positive and negative robustness
    to enable proper scaling to the [-1, 1] range.

    Attributes:
        pos_min: Minimum probability when predicate is satisfied (positive case).
        pos_max: Maximum probability when predicate is satisfied (positive case).
        neg_min: Minimum probability when predicate is not satisfied (negative case).
        neg_max: Maximum probability when predicate is not satisfied (negative case).
    """

    pos_min: float
    pos_max: float
    neg_min: float
    neg_max: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PredicateNormalizationValues":
        """
        Create normalization values from a dictionary (typically from CSV row).

        Args:
            values: Dictionary with keys 'p+min', 'p+max', 'p-min', 'p-max'.

        Returns:
            Parsed normalization parameters.
        """
        return cls(
            pos_min=float(values["p+min"]),
            pos_max=float(values["p+max"]),
            neg_min=float(values["p-min"]),
            neg_max=float(values["p-max"]),
        )


class RobustnessNormalizationProvider:
    """
    Concrete implementation of robustness normalization using predicate-specific ranges.

    This provider loads normalization parameters from a CSV file and applies
    linear scaling to map probability values to normalized robustness values.
    """

    def __init__(self, normalization_values: dict[str, PredicateNormalizationValues]):
        self._normalization_values = normalization_values

    @classmethod
    def from_csv(cls, csv_file_path: Path) -> "RobustnessNormalizationProvider":
        """
        Load normalization parameters from a CSV file.

        Expected CSV format:
        predicate,p+min,p+max,p-min,p-max
        safe_distance,0.1,0.9,0.05,0.95
        ...

        Args:
            csv_file_path: Path to CSV file containing normalization parameters.

        Returns:
            Initialized provider with loaded parameters.
        """
        normalization_values: dict[str, PredicateNormalizationValues] = {}
        with csv_file_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                predicate_name = row["predicate"]
                normalization_values[predicate_name] = PredicateNormalizationValues.from_dict(row)

        return cls(normalization_values)

    def normalize(self, predicate_name: str, probability: float, satisfied: bool) -> float:
        """
        Normalize probability to robustness value using predicate-specific parameters.

        The normalization applies linear scaling:
        - For satisfied predicates: map [pos_min, pos_max] to [0, 1]
        - For unsatisfied predicates: map [neg_min, neg_max] to [0, -1]

        Args:
            predicate_name: Name of the predicate being normalized.
            probability: Raw satisfaction probability from sampling.
            satisfied: Whether predicate is satisfied in original state.

        Returns:
            Normalized robustness value in [-1, 1] range.
        """
        normalization_values = self._normalization_values[predicate_name]
        if satisfied:
            robustness = (probability - normalization_values.pos_min) / (
                normalization_values.pos_max - normalization_values.pos_min
            )
        else:
            robustness = -(probability - normalization_values.neg_min) / (
                normalization_values.neg_max - normalization_values.neg_min
            )

        robustness = np.clip(robustness, -1, 1)
        return robustness


class MprSampledStatesCache(Protocol):
    def set_sampling_result(
        self,
        time_step: int,
        vehicle_id: int,
        result: StateBasedSamplingResult,
    ) -> None: ...

    def get_sampling_result(
        self, time_step: int, vehicle_id: int
    ) -> StateBasedSamplingResult | None: ...


@dataclass(frozen=True, kw_only=True)
class MprPredicateEvaluatorConfig:
    eps: float = 1e-17
    normalization_provider: AbstractRobustnessNormalizationProvider | None = None
    sampler_config: FutureStateSamplerConfig | None = None


class MprPredicateEvaluator:
    """
    Evaluate multiple predicates with model predictive robustness.
    """

    def __init__(
        self,
        predicates: Iterable[AbstractPredicate],
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
        config: MprPredicateEvaluatorConfig | None = None,
        state_sampling_cache: MprSampledStatesCache | None = None,
    ) -> None:
        self._predicates = predicates
        if config is None:
            config = MprPredicateEvaluatorConfig()

        self._config = config
        self._state_sampler = FutureStateSampler(scenario_type, config.sampler_config)

        self._state_sampling_cache = state_sampling_cache

    def evaluate(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> dict[PredicateName, MprPredicateEvaluationResult]:
        """
        Perform MPR evaluation of all predicates at a given time step.

        This method samples multiple future states for the ego vehicle and evaluates
        each predicate on these samples to compute the robustness values.

        Args:
            world: World in which the vehicles can be found.
            time_step: Current time step for evaluation.
            vehicle_ids: Tuple of vehicle IDs (ego vehicle first, then other vehicles).

        Returns:
            Mapping from predicate names to their MPR evaluation results.

        Raises:
            ValueError: If binary predicates are used but only one vehicle ID is provided.
            RuntimeError: If required vehicles are not found in the world.
        """
        only_unary_predicates = all(predicate.arity for predicate in self._predicates)
        if not only_unary_predicates and len(vehicle_ids) == 1:
            raise ValueError("Other vehicle is required if predicates with arity >1 are evaluated")

        ego_vehicle_id = vehicle_ids[0]
        orig_ego_vehicle = world.vehicle_by_id(ego_vehicle_id)
        if orig_ego_vehicle is None:
            raise RuntimeError(f"Ego vehicle with ID {ego_vehicle_id} not found in world")

        ego_vehicle = copy.deepcopy(orig_ego_vehicle)
        # Clear signals at current time step to avoid accidentanl interference, since they are currently not considered in sampling.
        ego_vehicle.signal_series[time_step] = None

        # Determine time bounds in which all required vehicles are active in the scenario.
        if not only_unary_predicates:
            # For binary predicates the bounds are given by the start and end time of the other vehicle.
            other_vehicle_id = vehicle_ids[1]
            other_vehicle = world.vehicle_by_id(other_vehicle_id)
            if other_vehicle is None:
                raise RuntimeError(f"Other vehicle with ID {other_vehicle_id} not found in world")

            other_vehicle_start_time = other_vehicle.start_time
            other_vehicle_end_time = other_vehicle.end_time
        else:
            # For unary predicates, use ego vehicle's time bounds.
            other_vehicle_start_time = ego_vehicle.start_time
            other_vehicle_end_time = ego_vehicle.end_time

        # The `ego_vehicle` will be modified for each sampled stated.
        # To make sure this does not interfere with other predicates, the vehicle is swapped with a copy before and after the evaluation.
        world.remove_vehicle(orig_ego_vehicle)
        world.add_vehicle(ego_vehicle)

        # Initialize counters for statistical analysis. Each dict is indexed with the predicate names.
        predicates_valid_count = defaultdict(lambda: 0)
        predicates_satisfied_count = defaultdict(lambda: 0)
        predicates_error_count = defaultdict(lambda: 0)
        general_error_count = 0

        for ego_future_state in self._get_state_samples(world, time_step, ego_vehicle_id):
            injection_successfull = _inject_sampled_state_into_vehicle_trajectory(
                world, ego_vehicle, ego_future_state
            )
            if not injection_successfull:
                general_error_count += 1

            sampled_time_step = ego_future_state.time_step

            # Evaluate each predicate on the current sampled state.
            for predicate in self._predicates:
                # Skip binary predicates if other vehicle is not active at sampled time.
                if predicate.arity == 2 and (
                    other_vehicle_start_time > sampled_time_step
                    or other_vehicle_end_time < sampled_time_step
                ):
                    predicates_error_count[predicate.predicate_name] += 1
                    continue

                try:
                    # Evaluate the predicate at the *sampled* time step, so that the correct
                    # state of the other vehicle is considered.
                    satisfied = predicate.evaluate_boolean(world, sampled_time_step, vehicle_ids)
                    if satisfied:
                        predicates_satisfied_count[predicate.predicate_name] += 1
                    predicates_valid_count[predicate.predicate_name] += 1
                except Exception as e:
                    _LOGGER.debug(
                        "Encountered exception while evaluating predicate %s at time step %s in %s: %s",
                        predicate.predicate_name,
                        time_step,
                        world.scenario.scenario_id,
                        e,
                    )
                    predicates_error_count[predicate.predicate_name] += 1

        # Restore original ego vehicle in the world.
        world.remove_vehicle(ego_vehicle)
        world.add_vehicle(orig_ego_vehicle)

        results: dict[PredicateName, MprPredicateEvaluationResult] = {}
        for predicate in self._predicates:
            count_true = predicates_satisfied_count[predicate.predicate_name]
            count_valid = predicates_valid_count[predicate.predicate_name]
            count_error = predicates_error_count[predicate.predicate_name] + general_error_count

            probability = count_true / (count_valid + self._config.eps)

            # Evaluate predicate satisfaction in the original (non-sampled) state.
            satisfied = predicate.evaluate_boolean(world, time_step, vehicle_ids)

            # Convert probability to robustness metric and optionally apply normalization.
            predicate_robustness = self._get_robustness_from_probability(probability, satisfied)

            results[predicate.predicate_name] = MprPredicateEvaluationResult(
                count_valid=count_valid,
                count_true=count_true,
                count_error=count_error,
                satisfied=satisfied,
                robustness=predicate_robustness,
                mfr_robustness=predicate.evaluate_robustness(world, time_step, vehicle_ids),
            )

        return results

    def _get_robustness_from_probability(self, probability: float, satisfied: bool) -> float:
        """
        Convert satisfaction probability to robustness value.

        If a normalization provider is configured, it will be used to map the probability
        to a normalized robustness value. Otherwise, a simple sign-based mapping is used.

        Args:
            probability: Satisfaction probability.
            satisfied: Whether predicate is satisfied in the original state.

        Returns:
            Robustness value.
        """
        if self._config.normalization_provider:
            return self._config.normalization_provider.normalize(probability, satisfied)
        else:  # no normalization, just setting the sign
            return probability if satisfied else -(1 - probability)

    def _get_state_samples(
        self, world: World, time_step: int, ego_vehicle_id: int
    ) -> StateBasedSamplingResult:
        if self._state_sampling_cache is not None:
            result = self._state_sampling_cache.get_sampling_result(time_step, ego_vehicle_id)
            if result is not None:
                return result

        result = self._state_sampler.sample(world, time_step, ego_vehicle_id)

        if self._state_sampling_cache is not None:
            self._state_sampling_cache.set_sampling_result(time_step, ego_vehicle_id, result)

        return result


def _inject_sampled_state_into_vehicle_trajectory(
    world: World, vehicle: Vehicle, sampled_state
) -> bool:
    """
    Inject a sampled future state into a vehicle's trajectory for predicate evaluation.

    This function updates both the vehicle's state and its lanelet assignment to ensure
    consistency with the sampled state. Many predicates rely on correct lanelet
    assignments to access the curvilinear state.

    Args:
        world: World containing the road network and vehicles.
        vehicle: Vehicle whose trajectory should be updated.
        sampled_state: The sampled future state to inject.

    Returns:
        True if injection was successful, False if state is invalid (e.g., off-road)
    """
    sampled_time_step = sampled_state.time_step

    vehicle.set_state_at_time_step(sampled_time_step, sampled_state)

    # Verify that the lanelet assignment is valid.
    lanelet_assignment = vehicle.lanelet_ids_at_time_step(sampled_time_step)
    if len(lanelet_assignment) == 0:
        # The state sampler created a state outside the lanelet network.
        return False

    return True
