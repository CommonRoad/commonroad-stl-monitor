import abc
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from commonroad.visualization.renderer import IRenderer
from commonroad_mpr.common.observation import World as WorldMPR
from commonroad_mpr.learning import FeatureExtrator, read_model
from commonroad_mpr.learning.gp_regression import ModelLoadError
from commonroad_mpr.prediction.ego_sampling import StateBasedSampling
from commonroad_mpr.utils.configuration_builder import ScenarioType

from crmonitor.common.world import World
from crmonitor.predicates.scaling import RobustnessScaler

_LOGGER = logging.getLogger(__name__)


@dataclass
class PredicateMprConfig:
    enabled: bool = False
    """Enable model-predictive robustness evaluation."""

    ml: bool = True
    """Enable gaussian processes for model-predictive robustness evaluation. This can be used e.g. for the training of new models."""

    extract_gradient: bool = False
    """Extract gradient from the gaussian process models during evaluation."""

    model_path: Optional[Path] = None
    """Path to the pre-trained models. If None, the models from the commonroad-mpr package are used."""

    rectification: bool = True
    """Control the behaviour if the sign of the MPR value does not match the sign of the characteristic value during MPR evaluation with GPs. If rectification is disabled, the evaluation will fallback to MPR evaluation without GPs."""

    sampler_time_horizon: float = 1.5
    """Set the time horizon for the MPR `StateBasedSampler`."""

    sample_number: int = 1000
    """Set the number of samples for the MPR `StateBasedSampler`."""


@dataclass
class PredicateEvaluatorConfig:
    scale_rob: bool = True
    mpr: PredicateMprConfig = field(default_factory=PredicateMprConfig)
    eps: float = 1e-17

    min_interstate_width: float = 7.0

    max_congestion_velocity: float = 2.78
    """Determines the velocity of vehicles, when they are considered in congestion. Used for the predicates `PredInCongestion` and `PredHasCongestionVelocity`."""

    num_veh_congestion: float = 3.0
    """Determines the number of vehicles, when it is considered as congestion. Used for the predicate `PredInCongestion`."""

    max_slow_moving_traffic_velocity: float = 8.33
    """Determines the velocity of vehicles when they are considered in slow moving traffic. Used for the predicates `PredInSlowMovingTraffic` and `PredHasSlowMovingVelocity`."""

    num_veh_slow_moving_traffic: float = 3.0
    """Determines the number of vehicles, when it is considered as in slow moving traffic. Used for the predicate `PredInSlowMovingTraffic`."""

    max_queue_of_vehicles_velocity: float = 16.67
    """Determines the velocity of vehicles when they are considered in a queue of vehicles. Used for the predicates `PredInQueueOfVehicles` and `PredHasQueueVelocity`."""

    num_veh_queue_of_vehicles: float = 3.0
    """Determines the number of vehicles, when it is considered as in a queue of vehicles. Used for the predicate `PredInQueueOfVehicles`."""

    max_interstate_speed_truck: float = 22.22
    desired_interstate_velocity: float = 36.11

    u_turn: float = 1.57

    standstill_error: float = 0.01

    min_velocity_diff: float = 15

    slightly_higher_speed_difference: float = 5.55

    close_to_other_vehicle: float = 0.75
    close_to_lane_border: float = 0.2

    d_sl: float = 1.0
    d_br: float = 15.0
    a_br: float = -1.0

    a_abrupt: float = -2.0

    country: str = "DEU"


class BasePredicateEvaluator(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name = "interface"
    arity: int

    def __init__(self, config: PredicateEvaluatorConfig, scaler=None):
        self.config = config
        self._scaler = scaler or RobustnessScaler(self.config.scale_rob)

        if self.config.mpr.enabled and self.config.mpr.ml:
            try:
                self._mpr_model = read_model(
                    self.predicate_name,
                    self.config.mpr.model_path,
                    ScenarioType.INTERSTATE,
                )
            except ModelLoadError as e:
                _LOGGER.warning(
                    "Failed to load pre-trained model for predicate '%s' from path '%s'. Falling back to model-predictive evaluation without a pre-trained model.",
                    e.predicate_name,
                    e.model_path,
                )
                self._mpr_model = None
        else:
            self._mpr_model = None

        self._mpr_gradients = []

    @property
    def gradients(self):
        return self._mpr_gradients

    @property
    def last_gradient(self):
        if len(self._mpr_gradients) == 0:
            return None
        return self._mpr_gradients[-1]

    def _scale_speed(self, x):
        return self._scaler.scale_speed(x)

    def _scale_acc(self, x):
        return self._scaler.scale_acc(x)

    def _scale_lon_dist(self, x):
        return self._scaler.scale_lon_dist(x)

    def _scale_lat_dist(self, x):
        return self._scaler.scale_lat_dist(x)

    def _scale_angle(self, x):
        return self._scaler.scale_angle(x)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        pass

    def evaluate_mpr_ml(
        self,
        world: World,
        world_mpr: WorldMPR,
        time_step: int,
        vehicle_ids: List[int],
    ) -> float:
        """
        Evaluate this predicate with model-predicitive robustness using pre-trained models.
        This method should usually not be called directly. Instead use `evaluate_robustness_with_cache`.

        :param world: The model-free world, required for boolean evaluation of this predicate.
        :param world_mpr: The model-predictive world, required for feature extraction.
        :param time_step: The time step for which this predicate should be evaluated.
        :param vehicle_ids: The vehicles that should be considered for the evaluation.
        :param rectification: Whether to rectificate the robustness.

        :returns: The predicted robustness value.

        :raises RuntimeError: If the pre-trained model was not already loaded.
        """
        # The feature extraction is usually performed by `PredicateEvaluatorML`. It must be adapted here, because
        # it originally does not support predicates which are not part of commonroad-mpr.
        desired_features = FeatureExtrator.get_desired_features(self.predicate_name)

        vehicles = []
        for vehicle_id in vehicle_ids:
            vehicles.append(world_mpr.vehicle_by_id(vehicle_id))
        all_feature_variables = FeatureExtrator.all_feature_variables(
            world_state=world_mpr, vehicles=vehicles, time_step=time_step
        )

        features = {
            vehicle: {name: all_feature_variables[vehicle][name] for name in desired_names}
            for vehicle, desired_names in desired_features.items()
            if desired_names is not None
        }

        # For consistency, the limits of the scaler are used here.
        # However, it is unclear whether the pre-trained models can handled values outside of
        # the interval [-1.0, 1.0].
        characteristic_value = (
            self._scaler.max
            if self.evaluate_boolean(world, time_step, vehicle_ids)
            else self._scaler.min
        )
        features.update({"predicate_evaluation": {"bool": characteristic_value}})

        list_features = FeatureExtrator.featrue_dict_to_list(features)

        if self._mpr_model is None:
            raise RuntimeError(
                f"Failed to evaluate predicate {self.predicate_name} with model-predictive robustness and pre-trained model: No pre-trained model was loaded!"
            )
        robustness, _ = self._mpr_model.predict([list_features])
        if self.config.mpr.extract_gradient:
            gradient = self._mpr_model.get_gradient([list_features])
            self._mpr_gradients.append(gradient)

        if robustness * characteristic_value < 0:
            if self.config.mpr.rectification:
                # rectification
                _LOGGER.debug(
                    f"Apply rectification. MPR: {robustness:.3f}, Characteristic value: {characteristic_value}"
                )
                robustness = np.float64(1e-3) * np.sign(characteristic_value)
            else:
                _LOGGER.debug(
                    f"Sign of characterstic value {characteristic_value} and MPR {robustness:.3f} do not match. Falling back to evaluation without GPs."
                )
                mpr_dict = self.evaluate_mpr(world, world_mpr, time_step, vehicle_ids)
                robustness = mpr_dict["robustness"]

        clipped_robustness = np.clip(robustness, self._scaler.min, self._scaler.max)

        return clipped_robustness

    def evaluate_mpr(
        self, world: World, world_mpr: WorldMPR, time_step: int, vehicle_ids: List[int]
    ) -> Dict[str, Union[bool, float]]:
        """
        Evaluate this predicate with model-predictive robustness. Its return value follows the standard from commonroad_mpr, so this method can be used for data generation for the gaussian processes.
        This method should usually not be called directly. Instead use `evaluate_robustness_with_cache`.

        :param world: The model-free world, required for boolean evaluation of this predicate.
        :param world_mpr: The model-predictive world, required for state sampling.
        :param time_step: The time step for which this predicate should be evaluated.
        :param vehicle_ids: The vehicles that should be considered for the evaluation. The first one is considered the ego vehicle.

        :returns: A dict with:
            - robustness: The model-predictive robustness.
            - count_valid: The number of successfull evaulations of the model-free predicate.
            - count_true: The number of successfull evaluations of the model-free predicate, where its return value is True.
            - count_error: The number of failed evaluations of the model-free predicate.
            - bool: The single boolean evaluation of the model-free predicate.

        :raises RuntimeError: If the pre-trained model was not already loaded.
        """
        ego_vehicle_id = vehicle_ids[0]
        ego_vehicle_mpr = world_mpr.vehicle_by_id(ego_vehicle_id)

        if self.arity == 2:
            # If arity is 2, the other vehicle must also be moved forward in time.
            # We extract it from the world, so that we can later check whether it has a state at the sampled time step.
            other_vehicle_id = vehicle_ids[1]
            other_vehicle = world.vehicle_by_id(other_vehicle_id)

        ego_sampler = StateBasedSampling(
            ego_vehicle_mpr,
            time_step,
            # TODO: support for intersection scenarios.
            ScenarioType.INTERSTATE.to_str(),
            time_horizon=self.config.mpr.sampler_time_horizon,
            # TODO: hardcoded values and weird way to pass the arguments.
            samplingxd={"distribution": "uniform", "simulation": "monte-carlo"},
            end_state_options={
                "sample_long_lat": [[1], [0, 1]],
                "zero_long_lat": [[2], [2]],
                "size": [12, 12, 12],
                "number": self.config.mpr.sample_number,
                "d_radius": 5,
                "d_dot_radius": 3,
            },
        )

        orig_ego_vehicle = world.vehicle_by_id(ego_vehicle_id)
        ego_vehicle = copy.deepcopy(orig_ego_vehicle)
        ego_vehicle.signal_series[time_step] = None

        # The `ego_vehicle` will be modified for each sampled stated.
        # To make sure this does not interfer with other predicates, the vehicle is swapped with a copy before and after the evaluation.
        world.remove_vehicle(orig_ego_vehicle)
        world.add_vehicle(ego_vehicle)

        count_valid = 0
        count_true = 0
        count_error = 0
        for (
            ego_future_state_mpr
        ) in ego_sampler.sample():  # iterates over all states of all predictions.
            try:
                # To be able to use the states, they need to be converted to CommonRoad states, which are relative to the world.
                ego_future_state = (
                    ego_future_state_mpr.get_state_in_world_frame().convert_to_commonroad_state()
                )
                # From now on we use the time step of the sampled state as the current time step.
                sampled_time_step = ego_future_state.time_step
                if self.arity == 2 and (
                    other_vehicle.start_time > sampled_time_step
                    or other_vehicle.end_time < sampled_time_step
                ):
                    # If the sampled state lies outside the time frame of the other vehicle we
                    # cannot evaluate the predicate.
                    # Usually, the evaluation of the predicate should just fail in this case,
                    # because they cannot access the state at the requested time step.
                    # However, we take a shortcut here and speed things up a bit.
                    count_error += 1
                    continue

                # Inject the sampled state at the specific time step into the model-free world.
                ego_vehicle.states_cr[sampled_time_step] = ego_future_state

                # Many predicates rely on the pre-computed lanelet assignments.
                # To make sure they match the sampled state, the lanelet assignment must also be updated.
                ego_loc_shape = ego_vehicle.shape.rotate_translate_local(
                    ego_future_state.position, ego_future_state.orientation
                )
                lanelet_assignment = world.road_network.lanelet_network.find_lanelet_by_shape(
                    ego_loc_shape
                )
                if len(lanelet_assignment) == 0:
                    # The state sampler created a state outside the lanelet network.
                    # TODO mitigate by activating phantom lanes?
                    count_error += 1
                    continue
                ego_vehicle.lanelet_assignment[sampled_time_step] = lanelet_assignment

                # Evaluate the predicate at the *sampled* time step, so that the correct
                # state of the other vehicle is considered.
                satisfied = self.evaluate_boolean(world, sampled_time_step, vehicle_ids)
                if satisfied:
                    count_true += 1
                count_valid += 1
            except Exception as e:
                _LOGGER.debug(
                    "Encountered exception while evaluating predicate %s at time step %s in %s: %s",
                    self.predicate_name,
                    time_step,
                    world.scenario.scenario_id,
                    e,
                )
                count_error += 1

        world.remove_vehicle(ego_vehicle)
        world.add_vehicle(orig_ego_vehicle)

        satisfied = self.evaluate_boolean(world, time_step, vehicle_ids)
        probability = count_true / (count_valid + self.config.eps)
        robustness = probability if satisfied else -(1 - probability)

        # This is the format used by the original MPR evaluator.
        # It is used here too, to keep backwards compatibility with GP regression for the time being.
        ret = {
            "robustness": robustness,
            "count_valid": count_valid,
            "count_error": count_error,
            "count_true": count_true,
            "bool": satisfied,
        }

        return ret

    def evaluate_robustness_with_cache(
        self, world: World, mpr_world: WorldMPR, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        vehicle_ids_tuple = tuple(vehicle_ids)
        value = vehicle.predicate_cache.get_robustness(
            time_step, self.predicate_name, vehicle_ids_tuple[1:]
        )
        if value is None:
            if self._mpr_model is not None and mpr_world is not None:
                _LOGGER.debug(
                    "Evaluating predicate %s at time step %d with vehicles %s using model-predictive robustness, with pre-trained models.",
                    self.predicate_name,
                    time_step,
                    ",".join(str(vehicle_id) for vehicle_id in vehicle_ids_tuple),
                )
                value = self.evaluate_mpr_ml(world, mpr_world, time_step, vehicle_ids)
            elif self.config.mpr.enabled and mpr_world is not None:
                _LOGGER.debug(
                    "Evaluating predicate %s at time step %d with vehicles %s using model-predictive robustness, without pre-trained models.",
                    self.predicate_name,
                    time_step,
                    ",".join(str(vehicle_id) for vehicle_id in vehicle_ids_tuple),
                )
                mpr_ret = self.evaluate_mpr(world, mpr_world, time_step, vehicle_ids)
                value = mpr_ret["robustness"]
            else:
                _LOGGER.debug(
                    "Evaluating predicate %s at time step %d with vehicles %s using model-free robustness.",
                    self.predicate_name,
                    time_step,
                    ",".join(str(vehicle_id) for vehicle_id in vehicle_ids_tuple),
                )
                value = self.evaluate_robustness(world, time_step, vehicle_ids)
            vehicle.predicate_cache.set_robustness(
                time_step, self.predicate_name, vehicle_ids_tuple[1:], value
            )
        return value

    def visualize(
        self,
        vehicle_ids: List[int],
        add_vehicle_draw_params: Callable[[int, any], None],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ) -> Tuple[Callable[[IRenderer], None], ...]:
        """
        Overwrite this function for visualizing a predicate in a certain way within the scenario plot.
        """
        self._gather_predicate_values_to_plot(
            vehicle_ids, world, time_step, predicate_names2vehicle_ids2values
        )
        return ()

    def _gather_predicate_values_to_plot(
        self,
        vehicle_ids: List[int],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ):
        predicate_names2vehicle_ids2values[self.predicate_name][tuple(vehicle_ids)] = (
            self.evaluate_robustness_with_cache(world, time_step, vehicle_ids)
        )

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.axis("off")
        ax.text(0.1, 0.5, "[not visualized]", fontsize=12)

    def reset(self) -> None:
        self._mpr_gradients = []
