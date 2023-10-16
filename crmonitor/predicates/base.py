import abc
import logging
import numpy as np
from typing import Callable, Dict, List, Tuple
import math

from commonroad.visualization.renderer import IRenderer
from commonroad_mpr.learning import PredicateEvaluatorML as PEML
from ruamel.yaml.comments import CommentedMap

from crmonitor.common.world import Vehicle, World
from crmonitor.predicates.utils import (
    bool_to_num,
    distance_veh_center_to_lane_boundaries,
    get_long_distance_stop_lines_from_lane,
)
from crmonitor.predicates.scaling import RobustnessScaler

logger = logging.getLogger(__name__)


class BasePredicateEvaluator(abc.ABC):
    """
    Base class for the predicate evaluator
    """

    predicate_name = "interface"

    def __init__(self, config: CommentedMap, scaler=None):
        self.config = config
        self.eps = 1e-5
        self._scaler = scaler or RobustnessScaler(
            scale=config.setdefault("scale_rob", True)
        )

        if self.config["use_mpr"]:
            try:
                self.peml = PEML([self.predicate_name])
            except:
                print("do not have model:", self.predicate_name)
                self.peml = None

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

    def extract_feature(self, world: World, time_step, vehicle_ids: List[int]) -> List:
        """
        Extract features for MPR computation.
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
        char_func = bool_to_num(self.evaluate_boolean(world, time_step, vehicle_ids))
        feature_list += [char_func]
        return feature_list

    def extract_feature_intersection(self, world: World, time_step, vehicle_ids: List[int]) -> List:
        """
        Extract features for MPR computation.
        """

        def get_veh_state_long_features(veh: Vehicle):
            return [
                veh.get_lon_state(time_step, veh.ref_path_lane).s,  # position
                veh.get_lon_state(time_step, veh.ref_path_lane).v,  # velocity
                veh.get_lon_state(time_step, veh.ref_path_lane).a,  # acceleration
                # veh.get_lon_state(time_step, veh.ref_path_lane).j,  # jerk
            ]

        def get_veh_input_long_features(veh: Vehicle):
            return [veh.get_lon_state(time_step, veh.ref_path_lane).j_dot]  # jerk_dot

        def get_veh_state_lat_features(veh: Vehicle):
            return [
                veh.get_lat_state(time_step, veh.ref_path_lane).d,  # lateral_position
                # veh.get_lat_state(time_step, veh.ref_path_lane).theta,  # orientation
                # veh.get_lat_state(time_step, veh.ref_path_lane).kappa,  # curvature
                # veh.get_lat_state(time_step, veh.ref_path_lane).kappa_dot,  # curvature_dot
            ]

        def get_veh_input_lat_features(veh: Vehicle):
            return [veh.get_lat_state(time_step, veh.ref_path_lane).kappa_dot_dot]  # curvature_ddot

        def get_veh_shape(veh: Vehicle):
            return [veh.shape.length,  # length
                    veh.shape.width,  # width
                    ]

        def get_veh_longitudinal_env_features(veh: Vehicle):
            intersection_lanelets_id = veh.incoming_intersection.successors_left.union(veh.incoming_intersection.successors_right).union(veh.incoming_intersection.successors_straight)
            ref_intersection_lanelet_id = list(intersection_lanelets_id.intersection(veh.ref_path_lane.contained_lanelets))
            enter_s, exit_s = world.road_network.get_lanelets_start_end_s(ref_intersection_lanelet_id, veh.ref_path_lane)
            return [
                veh.get_lon_state(time_step, veh.ref_path_lane).s - enter_s,
                exit_s - veh.get_lon_state(time_step, veh.ref_path_lane).s,
                min(veh.get_lon_state(time_step, veh.ref_path_lane).s - enter_s, exit_s - veh.get_lon_state(time_step, veh.ref_path_lane).s),
            ]

        def get_veh_stop_line_features(veh: Vehicle):
            intersection_lanelets_id = veh.incoming_intersection.successors_left.union(
                veh.incoming_intersection.successors_right).union(veh.incoming_intersection.successors_straight)
            ref_intersection_lanelet_id = list(
                intersection_lanelets_id.intersection(veh.ref_path_lane.contained_lanelets))
            s_stop_lines = get_long_distance_stop_lines_from_lane(world.road_network, veh.ref_path_lane)
            if len(s_stop_lines) == 0:
                return [50.0]
            closest_distance_index = np.argmin(abs(np.array(s_stop_lines) - veh.get_lon_state(time_step, veh.ref_path_lane).s))
            center_distance_stop_line = s_stop_lines[closest_distance_index] - veh.get_lon_state(time_step, veh.ref_path_lane).s
            if center_distance_stop_line >= 0:
                state_distance_stop_line = min(center_distance_stop_line, 50.0)
            else:
                state_distance_stop_line = max(center_distance_stop_line, -50.0)
            return [state_distance_stop_line]

        def get_veh_lateral_env_feature(veh: Vehicle):
            d_left, d_right = distance_veh_center_to_lane_boundaries(veh, veh.ref_path_lane, time_step)
            return [min(d_left, d_right)]

        def get_other_to_ego_features(veh_1: Vehicle, veh_2: Vehicle):
            ref_lane = veh_1.ref_path_lane
            return [
                veh_2.rear_s(time_step, ref_lane) - veh_1.front_s(time_step, ref_lane),
                veh_1.get_lat_state(time_step, ref_lane).d - veh_2.get_lat_state(time_step, ref_lane).d,
                veh_2.get_lon_state(time_step, ref_lane).v * np.cos(veh_2.get_lat_state(time_step, ref_lane).theta)
                - veh_1.get_lon_state(time_step, ref_lane).v * np.cos(veh_1.get_lat_state(time_step, ref_lane).theta),
                veh_2.get_lon_state(time_step, ref_lane).v * np.sin(veh_2.get_lat_state(time_step, ref_lane).theta)
                - veh_1.get_lon_state(time_step, ref_lane).v * np.sin(veh_1.get_lat_state(time_step, ref_lane).theta),
            ]

        def get_ego_to_other_features(veh_1: Vehicle, veh_2: Vehicle):
            ref_lane = veh_1.ref_path_lane
            return [
                veh_2.rear_s(time_step, ref_lane) - veh_1.front_s(time_step, ref_lane),
                veh_1.get_lat_state(time_step, ref_lane).d - veh_2.get_lat_state(time_step, ref_lane).d,
            ]

        feature_list = []
        # extract feature variables
        # - single veh features
        ego_veh = world.vehicle_by_id(vehicle_ids[0])
        feature_list += (
            get_veh_state_long_features(ego_veh)
            # + get_veh_input_long_features(ego_veh)
            + get_veh_state_lat_features(ego_veh)
            # + get_veh_input_lat_features(ego_veh)
            + get_veh_shape(ego_veh)
            + get_veh_longitudinal_env_features(ego_veh)
            + get_veh_stop_line_features(ego_veh)
            + get_veh_lateral_env_feature(ego_veh)
        )
        # - veh to veh features
        if len(vehicle_ids) == 2:
            other_veh = world.vehicle_by_id(vehicle_ids[1])
            feature_list += (
                get_veh_state_long_features(other_veh)
                + get_veh_state_lat_features(other_veh)
                + get_veh_shape(other_veh)
                + get_veh_longitudinal_env_features(other_veh)
                + get_veh_lateral_env_feature(other_veh)
            )
            feature_list += get_other_to_ego_features(ego_veh, other_veh)
            feature_list += get_ego_to_other_features(other_veh, ego_veh)
        # - characteristic function (Boolean evaluation)
        feature_list = [round(f, 6) for f in feature_list]
        char_func = bool_to_num(self.evaluate_boolean(world, time_step, vehicle_ids))
        feature_list += [char_func]
        return feature_list

    def evaluate_mpr(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        """
        Evaluation of model predictive robustness
        """
        if self.peml is not None:
            if self.config["mpr_scenario"] == "interstate":
                feature_list = self.extract_feature(world, time_step, vehicle_ids)
            else:
                feature_list = self.extract_feature_intersection(world, time_step, vehicle_ids)
            self.peml.list_feature_variables = [feature_list]
            # computation for single predicate
            robustness, _ = self.peml.robustness_models[0].predict([feature_list])
            rob_test = self.evaluate_robustness(world, time_step, vehicle_ids)
            if robustness * feature_list[-1] < 0:
                robustness = feature_list[-1] * self.eps
        else:
            robustness = self.evaluate_robustness(world, time_step, vehicle_ids)
        return robustness

    def gradient_mpr(self):
        """
        Computes the gradient of the MPR w.r.t. the input values
        """
        if self.config["use_mpr"]:
            return self.peml.derivative()[0]
        else:
            warnings.warn("The MPR is deactivated")
            return [0.0] * 35

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
