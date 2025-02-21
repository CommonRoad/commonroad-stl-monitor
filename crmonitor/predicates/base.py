import abc
import copy
import logging
import warnings
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from commonroad.visualization.renderer import IRenderer
from ruamel.yaml.comments import CommentedMap

from commonroad_mpr.common.observation import World as WorldMPR
from commonroad_mpr.learning import FeatureExtrator, PredicateEvaluatorML, read_model
from commonroad_mpr.learning.gp_regression import ModelLoadError
from commonroad_mpr.prediction.ego_sampling import StateBasedSampling
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from commonroad_mpr.utils.configuration_builder import ScenarioType
from crmonitor.common.world import World
from crmonitor.predicates.scaling import RobustnessScaler

_LOGGER = logging.getLogger(__name__)


class BasePredicateEvaluator(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name = "interface"

    def __init__(self, config: CommentedMap, scaler=None):
        self.config = config
        self.eps = 1e-5
        scale_rob = config.get("scale_rob", True)
        self._scaler = scaler or RobustnessScaler(scale_rob)

        if self.config["use_mpr"]:
            try:
                self._mpr_model = read_model(
                    self.predicate_name,
                    self.config.get("model_path"),
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

        self._use_mpr_for_evaluation = self.config["use_mpr"]

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
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        pass

    def evaluate_mpr_ml(
        self, world: World, world_mpr: WorldMPR, time_step: int, vehicle_ids: List[int]
    ) -> float:
        """
        Evaluate this predicate with model-predicitive robustness using pre-trained models.
        This method should usually not be called directly. Instead use `evaluate_robustness_with_cache`.

        :param world: The model-free world, required for boolean evaluation of this predicate.
        :param world_mpr: The model-predictive world, required for feature extraction.
        :param time_step: The time step for which this predicate should be evaluated.
        :param vehicle_ids: The vehicles that should be considered for the evaluation.

        :returns: The predictated robustness value.

        :raises RuntimeError: If the pre-trained model was not already loaded.
        """
        # The feature extraction is usually performed by `PredicateEvaluatorML`. It must be adapted here, because
        # it originally does not support predicates which are not part of commonroad-mpr.
        desired_features = dict(
            MprCfg["feature_variable"][MprCfg["common"]["scenario"]]["desired_features"]
        )
        # TODO: arity should be a standard attribute of all predicates.
        # Currently it is not clear whether all predicates have this attribute.
        if self.arity == 1:
            desired_features.pop("other", None)
            desired_features.pop("ego_other", None)
            desired_features.pop("other_ego", None)

        vehicles = []
        for vehicle_id in vehicle_ids:
            vehicles.append(world_mpr.vehicle_by_id(vehicle_id))
        all_feature_variables = FeatureExtrator.all_feature_variables(
            world_state=world_mpr, vehicles=vehicles, time_step=time_step
        )

        features = {
            vehicle: {
                name: all_feature_variables[vehicle][name] for name in desired_names
            }
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

        if robustness * characteristic_value < 0:
            robustness = np.float64(1e-3)

        return np.copysign(
            np.clip(robustness, self._scaler.min, self._scaler.max),
            characteristic_value,
        )

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

        ego_sampler = StateBasedSampling(
            ego_vehicle_mpr,
            time_step,
            MprCfg["common"]["scenario"],
            **MprCfg["sampling_approach"]["state_based_sampling"][
                MprCfg["common"]["scenario"]
            ],
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
        for ego_future_state_mpr in ego_sampler.sample():
            try:
                # MPR states are sometimes (always?) not relative to the world.
                # To be able to use the states, they need to be converted to CommonRoad states, which are relative to the world.
                ego_future_state = (
                    ego_future_state_mpr.get_state_in_world_frame().convert_to_commonroad_state()
                )
                # Inject the sampled state at the specific time step into the model-free world.
                ego_vehicle.states_cr[time_step] = ego_future_state

                # Many predicates rely on the pre-computed lanelet assignments.
                # To make sure they match the sampled state, the lanelet assignment must also be updated.
                ego_loc_shape = ego_vehicle.shape.rotate_translate_local(
                    ego_future_state.position, ego_future_state.orientation
                )
                lanelet_assignment = (
                    world.road_network.lanelet_network.find_lanelet_by_shape(
                        ego_loc_shape
                    )
                )
                if len(lanelet_assignment) == 0:
                    # The state sampler created a state that is outside of the lanelet network.
                    count_error += 1
                    continue
                ego_vehicle.lanelet_assignment[time_step] = lanelet_assignment

                satisfied = self.evaluate_boolean(world, time_step, vehicle_ids)
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
        probability = count_true / (count_valid + MprCfg["robustness"]["eps"])
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

    def gradient_mpr(self):
        """
        Computes the gradient of the MPR w.r.t. the input values.
        """
        default = [0.0] * 35
        # TODO: The gradient is always requested, even if this predicate is not evaluated with pre-trained models.
        return default
        if not self._use_mpr_for_evaluation:
            # TODO: If the user tries to extract the gradient for a comosed/exempted predicate, should it just be skipped?
            warnings.warn(
                f"Tried to extract the gradient of the model predictive evaluation, but model predictive evaluation is not enabled for '{self.predicate_name}'. This is either because mpr is disabled or this predicate is exempted from MPR."
            )
            return default

        if self._mpr_model is None:
            warnings.warn(
                f"Tried to extract gradient of the model predictive robustness, but no gaussian processes were used for the evaluation of '{self.predicate_name}' and therefore no gradient is available."
            )
            return default

        return self._mpr_model.derivative()[0]

    def evaluate_robustness_with_cache(
        self, world: World, mpr_world: WorldMPR, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        vehicle_ids_tuple = tuple(vehicle_ids)
        value = vehicle.predicate_cache.get_robustness(
            time_step, self.predicate_name, vehicle_ids_tuple[1:]
        )
        if value is None:
            if self._mpr_model is not None:
                _LOGGER.debug(
                    "Evaluating predicate %s at time step %d with vehicles %s using model-predictive robustness, with pre-trained models.",
                    self.predicate_name,
                    time_step,
                    ",".join(str(vehicle_id) for vehicle_id in vehicle_ids_tuple),
                )
                value = self.evaluate_mpr_ml(world, mpr_world, time_step, vehicle_ids)
            elif self._use_mpr_for_evaluation:
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
        predicate_names2vehicle_ids2values[self.predicate_name][
            tuple(vehicle_ids)
        ] = self.evaluate_robustness_with_cache(world, time_step, vehicle_ids)

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.axis("off")
        ax.text(0.1, 0.5, "[not visualized]", fontsize=12)
