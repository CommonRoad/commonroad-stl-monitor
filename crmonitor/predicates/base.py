import abc
import logging
import math
from typing import Callable, Dict, List, Tuple

import numpy as np
from commonroad.visualization.renderer import IRenderer
from ruamel.yaml.comments import CommentedMap

from crmonitor.common.world import World, Vehicle
from crmonitor.predicates.utils import (
    distance_veh_center_to_lane_boundaries, bool_to_num
)
from commonroad_mpr.learning import PredicateEvaluatorML as PEML

logger = logging.getLogger(__name__)

MAX_LONG_DIST = 200.0
MAX_LAT_DIST = 20.0


class BasePredicateEvaluator(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name = "interface"

    def __init__(self, config: CommentedMap):
        self.config = config
        self.scale = config.setdefault("scale_rob", True)
        self.eps = 1e-5

        # usage of model predictive robustness
        if self.config["use_mpr"]:
            self.peml = PEML([self.predicate_name])

    # todo: decouple the scaler
    def _scale(self, x, max_value):
        return np.clip(x / max_value, -1.0, 1.0) if self.scale else x

    def _scale_speed(self, x):
        return self._scale(x, 250.0 / 3.6)

    def _scale_acc(self, x):
        return self._scale(x, 10.5)

    def _scale_lon_dist(self, x):
        return self._scale(x, MAX_LONG_DIST)

    def _scale_lat_dist(self, x):
        return self._scale(x, MAX_LAT_DIST)

    def _scale_angle(self, x):
        return self._scale(x, math.pi)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        pass

    def evaluate_mpr(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        """
        Evaluation of model predictive robustness
        """

        def get_veh_state_long_features(veh: Vehicle):
            return [
                veh.get_lon_state(time_step).s,  # position
                veh.get_lon_state(time_step).v,  # velocity
                veh.get_lon_state(time_step).a,  # acceleration
                veh.get_lon_state(time_step).j,  # jerk
            ]

        def get_veh_input_long_features(veh: Vehicle):
            return [veh.get_lon_state(time_step).j_dot]  # jerk_dot

        def get_veh_state_lat_features(veh: Vehicle):
            return [
                veh.get_lat_state(time_step).d,  # lateral_position
                veh.get_lat_state(time_step).theta,  # orientation
                veh.get_lat_state(time_step).kappa,  # curvature
                veh.get_lat_state(time_step).kappa_dot,  # curvature_dot
            ]

        def get_veh_input_lat_features(veh: Vehicle):
            return [veh.get_lat_state(time_step).kappa_dot_dot]  # curvature_ddot

        def get_veh_env_features(veh: Vehicle):
            d_left, d_right = distance_veh_center_to_lane_boundaries(
                veh,
                world.road_network.find_lane_by_lanelet(
                    list(veh.lanelet_assignment[time_step])[0]
                ),
                time_step,
            )
            return [
                veh.shape.length,  # length
                veh.shape.width,  # width
                # road from right to the left is: 0, 1, 2, ... for distance_to_road_left/right
                distance_veh_center_to_lane_boundaries(
                    veh, world.road_network.lanes[-1], time_step
                )[0],
                distance_veh_center_to_lane_boundaries(
                    veh, world.road_network.lanes[0], time_step
                )[-1],
                # distance_to_ref_lane_left, distance_to_ref_lane_right
                d_left,
                d_right,
            ]

        def get_v2v_features(veh_1: Vehicle, veh_2: Vehicle):
            ref_lane = veh_1.get_lane(time_step)
            # ego_other_distance, ego_other_lateral_distance,
            # ego_other_relative_longitudinal_velocity, ego_other_relative_lateral_velocity
            return [
                veh_2.rear_s(time_step, ref_lane) - veh_1.front_s(time_step, ref_lane),
                veh_2.get_lat_state(time_step, ref_lane).d
                - veh_1.get_lat_state(time_step, ref_lane).d,  # todo
                veh_2.get_lon_state(time_step, ref_lane).v
                * np.cos(veh_2.get_lat_state(time_step, ref_lane).theta)
                - veh_1.get_lon_state(time_step, ref_lane).v
                * np.cos(veh_1.get_lat_state(time_step, ref_lane).theta),
                veh_2.get_lon_state(time_step, ref_lane).v
                * np.sin(veh_2.get_lat_state(time_step, ref_lane).theta)
                - veh_1.get_lon_state(time_step, ref_lane).v
                * np.sin(veh_1.get_lat_state(time_step, ref_lane).theta),
            ]

        feature_list = []
        # extract feature variables
        # - single veh features
        ego_veh = world.vehicle_by_id(vehicle_ids[0])
        feature_list += (
            get_veh_state_long_features(ego_veh)
            + get_veh_input_long_features(ego_veh)
            + get_veh_state_lat_features(ego_veh)
            + get_veh_input_lat_features(ego_veh)
            + get_veh_env_features(ego_veh)
        )
        # - veh to veh features
        if len(vehicle_ids) == 2:
            other_veh = world.vehicle_by_id(vehicle_ids[1])
            feature_list += (
                get_veh_state_long_features(other_veh)
                + get_veh_state_lat_features(other_veh)
                + get_veh_env_features(other_veh)
            )
            feature_list += get_v2v_features(ego_veh, other_veh)
        # - characteristic function (Boolean evaluation)
        char_func = self.evaluate_boolean(world, time_step, vehicle_ids)
        feature_list += [bool_to_num(char_func)]
        # computation for single predicate
        robustness, _ = self.peml.robustness_models[0].predict([feature_list])
        if robustness * char_func < 0:
            robustness = self.config["eps"]
        return robustness

    def evaluate_robustness_with_cache(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        vehicle_ids_tuple = tuple(vehicle_ids)
        value = vehicle.predicate_cache.get_robustness(
            time_step, self.predicate_name, vehicle_ids_tuple[1:]
        )
        if value is None:
            logger.debug(
                "Evaluating predicate %s , t=%d, ids=%s",
                self.predicate_name,
                time_step,
                vehicle_ids_tuple,
            )
            if self.config["use_mpr"]:
                value = self.evaluate_mpr(world, time_step, vehicle_ids)
            else:
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
