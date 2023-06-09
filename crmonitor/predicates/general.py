from enum import Enum
import logging
from typing import List, Tuple, Dict, Callable
import matplotlib.colors
import numpy as np

from matplotlib import pyplot as plt

from crmonitor.common.world import World
from crmonitor.predicates import utils
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.predicates.position import PredInSameLane, PredSingleLane, PredInFrontOf
from crmonitor.predicates.utils import (cal_road_width, )

logger = logging.getLogger(__name__)


class GeneralPredicates(str, Enum):
    CutIn = "cut_in"
    InterstateBroadEnough = "interstate_broad_enough"
    InCongestion = "in_congestion"
    InSlowMovingTraffic = "in_slow_moving_traffic"
    InQueueOfVehicles = "in_queue_of_vehicles"
    MakesUTurn = "makes_u_turn"
    NotEndangerIntersection = "not_endanger_intersection"
    TurningLeft = "turning_left"
    TurningRight = "turning_right"
    GoingStraight = "going_straight"


class PredCutIn(BasePredicateEvaluator):
    predicate_name = GeneralPredicates.CutIn
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._single_lane_evaluator = PredSingleLane(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        cutting_vehicle = world.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_boolean(
            world, time_step, [vehicle_ids[0]]
        )
        if single_lane:
            return False
        same_lane = self._same_lane_evaluator.evaluate_boolean(
            world, time_step, vehicle_ids
        )
        if not same_lane:
            return False
        cutting_lane = cutting_vehicle.get_lane(time_step)
        cutted_lat = cutted_vehicle.get_lat_state(time_step, cutting_lane)
        cutting_lat = cutting_vehicle.get_lat_state(time_step)
        d_p = cutted_lat.d
        d_k = cutting_lat.d
        orient_k = cutting_lat.theta

        result = (d_k < d_p and orient_k > self.eps) or (
            d_k > d_p and orient_k < -self.eps
        )
        return result

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        cutting_vehicle = world.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_robustness_with_cache(
            world,
            time_step,
            [
                vehicle_ids[0],
            ],
        )
        same_lane = self._same_lane_evaluator.evaluate_robustness_with_cache(
            world, time_step, vehicle_ids
        )

        cutting_lane = cutting_vehicle.get_lane(time_step)
        cutted_lat = cutted_vehicle.get_lat_state(time_step, cutting_lane)
        cutting_lat = cutting_vehicle.get_lat_state(time_step)
        r_l_dist = cutted_lat.d - cutting_lat.d
        r_l_orient = cutting_lat.theta - self.eps
        l_r_dist = cutting_lat.d - cutted_lat.d
        l_r_orient = -self.eps - cutting_lat.theta

        r_l_dist = self._scale_lat_dist(r_l_dist)
        l_r_dist = self._scale_lat_dist(l_r_dist)
        r_l_orient = self._scale_angle(r_l_orient)
        l_r_orient = self._scale_angle(l_r_orient)

        rob = min(
            -single_lane,
            same_lane,
            max(min(r_l_dist, r_l_orient), min(l_r_dist, l_r_orient)),
        )
        return rob

    @staticmethod
    def _get_color_map():
        return plt.get_cmap("bwr")

    def visualize(
        self,
        vehicle_ids: List[int],
        add_vehicle_draw_params: Callable[[int, any], None],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ):
        self._gather_predicate_values_to_plot(
            vehicle_ids, world, time_step, predicate_names2vehicle_ids2values
        )

        latest_value = self.evaluate_robustness_with_cache(
            world, time_step, vehicle_ids
        )
        latest_value_normalized = (latest_value + 1) / 2
        violation_color = self._get_color_map()(latest_value_normalized)
        violation_color_hex = matplotlib.colors.rgb2hex(violation_color)

        vehicle = vehicle_ids[0]
        draw_params = {
            "dynamic_obstacle": {
                "vehicle_shape": {
                    "occupancy": {
                        "shape": {"rectangle": {"facecolor": violation_color_hex}}
                    }
                }
            }
        }
        add_vehicle_draw_params(vehicle, draw_params)

        draw_functions1 = self._same_lane_evaluator.visualize(
            vehicle_ids,
            add_vehicle_draw_params,
            world,
            time_step,
            predicate_names2vehicle_ids2values,
        )
        draw_functions2 = self._single_lane_evaluator.visualize(
            [vehicle],
            add_vehicle_draw_params,
            world,
            time_step,
            predicate_names2vehicle_ids2values,
        )

        return () + draw_functions1 + draw_functions2

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        points = np.linspace(0, 1, 256)
        points = np.vstack((points, points))
        ax.imshow(points, cmap=PredCutIn._get_color_map(), extent=[-1, 1, 0, 1])
        ax.get_yaxis().set_ticks([])
        ax.set_ylabel("vehicle color")


class PredInterstateBroadEnough(BasePredicateEvaluator):
    """
    Evaluates if an interstate is broad enough to build a standard emergency lane.
    """

    predicate_name = GeneralPredicates.InterstateBroadEnough
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        s = vehicle.get_lon_state(time_step).s
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                cal_road_width(lanelet, world.road_network, s)
                <= self.config["min_interstate_width"]
            ):
                return False
        return True

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        s = vehicle.get_lon_state(time_step).s
        comparison_list = []
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            comparison_list.append(
                self._scale_lat_dist(
                    cal_road_width(lanelet, world.road_network, s)
                    - self.config["min_interstate_width"]
                    - 1.0e-17
                )
            )
        return min(comparison_list)


class PredInCongestion(BasePredicateEvaluator):
    """
    Evaluates if a vehicle is in a congestion.
    """

    predicate_name = GeneralPredicates.InCongestion
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._in_front_of_evaluator = PredInFrontOf(config)
        self._same_lane_evaluator = PredInSameLane(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        num_vehicles = 0
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if (
                self._in_front_of_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                )
                and self._same_lane_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                )
                and veh_o.get_lon_state(time_step).v
                <= self.config["max_congestion_velocity"]
            ):
                num_vehicles += 1
        if num_vehicles >= self.config["num_veh_congestion"]:
            return True
        else:
            return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]

        rob_cong_veh_list = [self._scale_speed(-np.inf)]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                rob_cong_veh_list.append(self._scale_speed(-np.inf))
            rob_cong_veh_list.append(
                min(
                    self._in_front_of_evaluator.evaluate_robustness(
                        world, time_step, [vehicle_ids[0], veh_o.id]
                    ),
                    self._same_lane_evaluator.evaluate_robustness(
                        world, time_step, [vehicle_ids[0], veh_o.id]
                    ),
                    self._scale_speed(
                        self.config["max_congestion_velocity"]
                        - veh_o.get_lon_state(time_step).v
                        - 1.0e-17
                    ),
                )
            )
        # values are already normalized
        if (
            sum(rob > 0 for rob in rob_cong_veh_list)
            >= self.config["num_veh_congestion"]
        ):
            return min(rob for rob in rob_cong_veh_list if rob > 0)
        else:
            return max(rob for rob in rob_cong_veh_list if rob < 0)


class PredInSlowMovingTraffic(BasePredicateEvaluator):
    """
    Evaluates if a vehicle is part of slow moving traffic.
    """

    predicate_name = GeneralPredicates.InSlowMovingTraffic
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._in_front_of_evaluator = PredInFrontOf(config)
        self._same_lane_evaluator = PredInSameLane(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        num_vehicles = 0
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if (
                self._in_front_of_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                )
                and self._same_lane_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                )
                and veh_o.get_lon_state(time_step).v
                <= self.config["max_slow_moving_traffic_velocity"]
            ):
                num_vehicles += 1
        if num_vehicles >= self.config["num_veh_slow_moving_traffic"]:
            return True
        else:
            return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]

        rob_cong_veh_list = [self._scale_speed(-np.inf)]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                rob_cong_veh_list.append(self._scale_speed(-np.inf))
            rob_cong_veh_list.append(
                min(
                    self._in_front_of_evaluator.evaluate_robustness(
                        world, time_step, [vehicle_ids[0], veh_o.id]
                    ),
                    self._same_lane_evaluator.evaluate_robustness(
                        world, time_step, [vehicle_ids[0], veh_o.id]
                    ),
                    self._scale_speed(
                        self.config["max_slow_moving_traffic_velocity"]
                        - veh_o.get_lon_state(time_step).v
                        - 1.0e-17
                    ),
                )
            )
        # values are already normalized
        if (
            sum(rob > 0 for rob in rob_cong_veh_list)
            >= self.config["num_veh_slow_moving_traffic"]
        ):
            return min(rob for rob in rob_cong_veh_list if rob > 0)
        else:
            return max(rob for rob in rob_cong_veh_list if rob < 0)


class PredInQueueOfVehicles(BasePredicateEvaluator):
    """
    Evaluates if a vehicle is part of a queue of vehicles
    """

    predicate_name = GeneralPredicates.InQueueOfVehicles
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._in_front_of_evaluator = PredInFrontOf(config)
        self._same_lane_evaluator = PredInSameLane(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        num_vehicles = 0
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if (
                self._in_front_of_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                )
                and self._same_lane_evaluator.evaluate_boolean(
                    world, time_step, [vehicle_ids[0], veh_o.id]
                )
                and veh_o.get_lon_state(time_step).v
                <= self.config["max_queue_of_vehicles_velocity"]
            ):
                num_vehicles += 1
        if num_vehicles >= self.config["num_veh_queue_of_vehicles"]:
            return True
        else:
            return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]

        rob_cong_veh_list = [self._scale_speed(-np.inf)]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                rob_cong_veh_list.append(self._scale_speed(-np.inf))
            rob_cong_veh_list.append(
                min(
                    self._in_front_of_evaluator.evaluate_robustness(
                        world, time_step, [vehicle_ids[0], veh_o.id]
                    ),
                    self._same_lane_evaluator.evaluate_robustness(
                        world, time_step, [vehicle_ids[0], veh_o.id]
                    ),
                    self._scale_speed(
                        self.config["max_queue_of_vehicles_velocity"]
                        - veh_o.get_lon_state(time_step).v
                        - 1.0e-17
                    ),
                )
            )
        # values are already normalized
        if (
            sum(rob > 0 for rob in rob_cong_veh_list)
            >= self.config["num_veh_queue_of_vehicles"]
        ):
            return min(rob for rob in rob_cong_veh_list if rob > 0)
        else:
            return max(rob for rob in rob_cong_veh_list if rob < 0)


class PredMakesUTurn(BasePredicateEvaluator):
    """
    Predicate which evaluates if vehicle makes U-turn
    """

    predicate_name = GeneralPredicates.MakesUTurn
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanes = world.road_network.find_lanes_by_lanelets(
            vehicle.lanelet_assignment[time_step]
        )
        for la in lanes:
            if self.config["u_turn"] <= abs(
                vehicle.get_lat_state(time_step, la).theta
                - la.orientation(vehicle.get_lon_state(time_step, la).s)
            ):
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        robustness_values = []
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanes = world.road_network.find_lanes_by_lanelets(
            vehicle.lanelet_assignment[time_step]
        )
        for la in lanes:
            robustness_values.append(
                self._scale_angle(
                    abs(
                        vehicle.get_lat_state(time_step, la).theta
                        - la.orientation(vehicle.get_lon_state(time_step, la).s)
                    )
                    - self.config["u_turn"]
                    - 1.0e-17
                )
            )
        return max(robustness_values)


##############
# intersection
##############

# ---------------------------------------------------------------------- #
class PredTurningRight(BasePredicateEvaluator):
    """
    evaluates if a vehicle is turning right
    """

    predicate_name = GeneralPredicates.TurningRight
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        lanelets_assignment_initial = vehicle.lanelet_assignment[0]
        lanelets_initial_occupied = lanelets_assignment_initial.intersection(set(vehicle.lanelets_dir))
        incoming = utils.get_incoming(list(lanelets_initial_occupied), road_network)
        if incoming is None:
            return self._scale_lon_dist(-np.inf)
        # find corresponding right turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_assignment[time_step]
        right_turn_lane = utils.get_right_turn_lane(road_network, incoming)
        # important bounds of vehicle and lanelet in Curvilinear Coordinate System
        front_s = vehicle.front_s(time_step, right_turn_lane)
        rear_s = vehicle.rear_s(time_step, right_turn_lane)
        right_turn_start_s, right_turn_end_s = utils.get_lanelets_start_end_s(right_turn_lane, incoming.successors_right, road_network)
        # case 1: vehicle only in incoming
        if (len(lanelets_assignment_current.intersection(incoming.incoming_lanelets)) > 0
                and len(lanelets_assignment_current.intersection(incoming.successors_right)) == 0):
            rob = self._scale_lon_dist(front_s - right_turn_start_s)
        # case 2: vehicle occupies right turning at intersection
        elif len(lanelets_assignment_current.intersection(incoming.successors_right)) > 0:
            state = vehicle.states_cr[time_step]
            d_center_to_left = right_turn_lane.clcs_left.convert_to_curvilinear_coords(*state.position)[1]
            if d_center_to_left > 0:
                # out of right turning lanelet
                d_left = utils.distance_to_left_bounds_clcs(vehicle, right_turn_lane, time_step)
                if len(d_left) == 0:
                    rob = -np.inf
                else:
                    rob = -np.max(d_left, initial=-np.inf)
                rob = self._scale_lat_dist(rob)
            else:
                # in right turning lanelet
                # TODO: should also consider lateral information (distance to left bound)?
                rob = np.min([front_s - right_turn_start_s, right_turn_end_s - rear_s])
                rob = self._scale_lon_dist(rob)
        # case 3: vehicle occupies left turning or straight lanelet at intersection instead of right turning
        elif (len(lanelets_assignment_current.intersection(incoming.successors_left)) > 0
              or len(lanelets_assignment_current.intersection(incoming.successors_straight)) > 0):
            d_left = utils.distance_to_left_bounds_clcs(vehicle, right_turn_lane, time_step)
            if len(d_left) == 0:
                rob = -np.inf
            else:
                rob = -np.max(d_left, initial=-np.inf)
            rob = self._scale_lat_dist(rob)
        # case 4: vehicle exits intersection
        else:
            incoming_right_turn = utils.get_right_turn_incoming(lanelets_assignment_current, road_network)
            right_turn_lane = utils.get_right_turn_lane(road_network, incoming_right_turn)
            rear_s = vehicle.rear_s(time_step, right_turn_lane)
            right_turn_end_s = utils.get_lanelets_end_s(right_turn_lane, incoming_right_turn.successors_right, road_network)
            rob = self._scale_lon_dist(right_turn_end_s - rear_s)
        return rob


class PredTurningLeft(BasePredicateEvaluator):
    """
    evaluates if a vehicle is turning left
    """

    predicate_name = GeneralPredicates.TurningLeft
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        lanelets_dir = utils.lanelets_dir(vehicle, time_step, road_network)
        lanelets_assignment_initial = vehicle.lanelet_assignment[0]
        lanelets_initial_occupied = list(lanelets_assignment_initial.intersection(set(lanelets_dir)))
        incoming = utils.get_incoming(lanelets_initial_occupied, road_network)
        if incoming is None:
            return self._scale_lon_dist(-np.inf)
        # find corresponding left turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_assignment[time_step]
        left_turn_lane = utils.get_left_turn_lane(road_network, incoming)
        # important bounds of vehicle and lanelet in Curvilinear Coordinate System
        front_s = vehicle.front_s(time_step, left_turn_lane)
        rear_s = vehicle.rear_s(time_step, left_turn_lane)
        left_turn_start_s, left_turn_end_s = utils.get_lanelets_start_end_s(left_turn_lane, incoming.successors_left, road_network)
        # case 1: vehicle only in incoming
        if (len(lanelets_assignment_current.intersection(incoming.incoming_lanelets)) > 0
                and len(lanelets_assignment_current.intersection(incoming.successors_left)) == 0):
            rob = self._scale_lon_dist(front_s - left_turn_start_s)
        # case 2: vehicle occupies left turning at intersection
        elif len(lanelets_assignment_current.intersection(incoming.successors_left)) > 0:
            state = vehicle.states_cr[time_step]
            d_center_to_right = left_turn_lane.clcs_right.convert_to_curvilinear_coords(*state.position)[1]
            if d_center_to_right < 0:
                # out of left turning lanelet
                d_right = utils.distance_to_right_bounds_clcs(vehicle, left_turn_lane, time_step)
                if len(d_right) == 0:
                    rob = -np.inf
                else:
                    rob = np.min(d_right, initial=np.inf)
                rob = self._scale_lat_dist(rob)
            else:
                # in left turning lanelet
                rob = np.min([front_s - left_turn_start_s, left_turn_end_s - rear_s])
                rob = self._scale_lon_dist(rob)
        # case 3: vehicle occupies right turning or straight lanelet at intersection instead of right turning
        elif (len(lanelets_assignment_current.intersection(incoming.successors_right)) > 0
              or len(lanelets_assignment_current.intersection(incoming.successors_straight)) > 0):
            d_right = utils.distance_to_right_bounds_clcs(vehicle, left_turn_lane, time_step)
            if len(d_right) == 0:
                rob = -np.inf
            else:
                rob = np.min(d_right, initial=np.inf)
            rob = self._scale_lat_dist(rob)
        # case 4: vehicle exits intersection
        else:
            incoming_left_turn = utils.get_left_turn_incoming(lanelets_assignment_current, road_network)
            left_turn_lane = utils.get_left_turn_lane(road_network, incoming_left_turn)
            rear_s = vehicle.rear_s(time_step, left_turn_lane)
            left_turn_end_s = utils.get_lanelets_end_s(left_turn_lane, incoming_left_turn.successors_left, road_network)
            rob = self._scale_lon_dist(left_turn_end_s - rear_s)
        return rob


class PredGoingStraight(BasePredicateEvaluator):
    """
    evaluates if a vehicle is going straight
    """

    # same idea as turning left, also same limitations and same idea for future work

    predicate_name = GeneralPredicates.GoingStraight
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        lanelets_dir = utils.lanelets_dir(vehicle, time_step, road_network)
        lanelets_assignment_initial = vehicle.lanelet_assignment[0]
        lanelets_initial_occupied = list(lanelets_assignment_initial.intersection(set(lanelets_dir)))
        incoming = utils.get_incoming(lanelets_initial_occupied, road_network)
        if incoming is None:
            return self._scale_lon_dist(-np.inf)
        # find corresponding left turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_assignment[time_step]
        straight_lane = utils.get_straight_going_lane(road_network, incoming)
        # important bounds of vehicle and lanelet in Curvilinear Coordinate System
        front_s = vehicle.front_s(time_step, straight_lane)
        rear_s = vehicle.rear_s(time_step, straight_lane)
        straight_start_s, straight_end_s = utils.get_lanelets_start_end_s(straight_lane, incoming.successors_straight, road_network)
        # case 1: vehicle only in incoming
        if (len(lanelets_assignment_current.intersection(incoming.incoming_lanelets)) > 0
                and len(lanelets_assignment_current.intersection(incoming.successors_straight)) == 0):
            rob = self._scale_lon_dist(front_s - straight_start_s)
        # case 2: vehicle occupies left turning at intersection
        elif len(lanelets_assignment_current.intersection(incoming.successors_straight)) > 0:
            state = vehicle.states_cr[time_step]
            d_center_to_left = straight_lane.clcs_left.convert_to_curvilinear_coords(*state.position)[1]
            d_center_to_right = straight_lane.clcs_right.convert_to_curvilinear_coords(*state.position)[1]
            if d_center_to_left > 0:
                # out of left turning lanelet
                d_left = utils.distance_to_left_bounds_clcs(vehicle, straight_lane, time_step)
                if len(d_left) == 0:
                    rob = -np.inf
                else:
                    rob = -np.max(d_left, initial=-np.inf)
                rob = self._scale_lat_dist(rob)
            elif d_center_to_right < 0:
                # out of left turning lanelet
                d_right = utils.distance_to_right_bounds_clcs(vehicle, straight_lane, time_step)
                if len(d_right) == 0:
                    rob = -np.inf
                else:
                    rob = np.min(d_right, initial=np.inf)
                rob = self._scale_lat_dist(rob)
            else:
                # in left turning lanelet
                rob = np.min([front_s - straight_start_s, straight_end_s - rear_s])
                rob = self._scale_lon_dist(rob)
        # case 3: vehicle occupies right turning or straight lanelet at intersection instead of right turning
        elif (len(lanelets_assignment_current.intersection(incoming.successors_right)) > 0
              or len(lanelets_assignment_current.intersection(incoming.successors_left)) > 0):
            d_left = utils.distance_to_left_bounds_clcs(vehicle, straight_lane, time_step)
            if len(d_left) == 0:
                rob_left = -np.inf
            else:
                rob_left = -np.max(d_left, initial=-np.inf)
            d_right = utils.distance_to_right_bounds_clcs(vehicle, straight_lane, time_step)
            if len(d_right) == 0:
                rob_right = -np.inf
            else:
                rob_right = np.min(d_right, initial=np.inf)
            rob = self._scale_lat_dist(np.min([rob_left, rob_right]))
        # case 4: vehicle exits intersection
        else:
            incoming_straight = utils.get_straight_going_incoming(lanelets_assignment_current, road_network)
            straight_lane = utils.get_straight_going_lane(road_network, incoming_straight)
            rear_s = vehicle.rear_s(time_step, straight_lane)
            straight_end_s = utils.get_lanelets_end_s(straight_lane, incoming_straight.successors_straight, road_network)
            rob = self._scale_lon_dist(straight_end_s - rear_s)
        return rob

