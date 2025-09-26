import logging
from typing import Callable, Dict, List, Tuple

import matplotlib.colors
import numpy as np
from commonroad.common.util import subtract_orientations
from matplotlib import pyplot as plt
from typing_extensions import override

from crmonitor.common.world import World
from crmonitor.predicates import utils
from crmonitor.predicates.base import (
    AbstractPredicate,
    PredicateConfig,
    PredicateName,
)
from crmonitor.predicates.position import (
    PredInFrontOf,
    PredInSameLane,
    PredOnOncomOf,
    PredSingleLane,
)
from crmonitor.predicates.priority import (
    PredHasPriorityLeftLeft,
    PredHasPriorityLeftRight,
    PredHasPriorityLeftStraight,
    PredHasPriorityRightLeft,
    PredHasPriorityRightRight,
    PredHasPriorityRightStraight,
    PredHasPriorityStraightLeft,
    PredHasPriorityStraightRight,
    PredHasPriorityStraightStraight,
    PredSamePriorityLeftLeft,
    PredSamePriorityLeftRight,
    PredSamePriorityLeftStraight,
    PredSamePriorityRightLeft,
    PredSamePriorityRightRight,
    PredSamePriorityRightStraight,
    PredSamePriorityStraightLeft,
    PredSamePriorityStraightRight,
    PredSamePriorityStraightStraight,
)
from crmonitor.predicates.utils import cal_road_width

_LOGGER = logging.getLogger(__name__)


class GeneralPredicates(PredicateName):
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
    SlInFront = "sl_in_front"
    RightTurn = "on_right_turn"
    TlRed = "tl_red"
    InIntersection = "on_intersection"

    TurningSamePriorityBase = "turning_same_priority_base"
    RightEgoRightTargetSamePriority = "turning_right_ego_turning_right_target_same_priority"
    RightEgoLeftTargetSamePriority = "turning_right_ego_turning_left_target_same_priority"  # the ego vehicle is turning right, the target vehicle is turning left, and they have the same priority.
    RightEgoStraightTargetSamePriority = "turning_right_ego_going_straight_target_same_priority"
    LeftEgoRightTargetSamePriority = "turning_left_ego_turning_right_target_same_priority"
    LeftEgoLeftTargetSamePriority = "turning_left_ego_turning_left_target_same_priority"
    LeftEgoStraightTargetSamePriority = "turning_left_ego_going_straight_target_same_priority"
    StraightEgoRightTargetSamePriority = "going_straight_ego_turning_right_target_same_priority"
    StraightEgoLeftTargetSamePriority = "going_straight_ego_turning_left_target_same_priority"
    StraightEgoStraightTargetSamePriority = "going_straight_ego_going_straight_target_same_priority"

    TurningHasPriorityBase = "turning_has_priority_base"
    RightTargetRightEgoTargetHasPriority = (
        "turning_right_target_turning_right_ego_target_has_priority"
    )
    RightTargetLeftEgoTargetHasPriorityNotOncoming = (
        "turning_right_target_turning_left_ego_target_has_priority_not_oncoming"
    )
    RightTargetStraightEgoTargetHasPriority = "turning_right_target_going_straight_ego_target_has_priority"  # the target vehicle is turning right, the ego vehicle is going straight, and the target vehicle has priority
    LeftTargetRightEgoTargetHasPriority = (
        "turning_left_target_turning_right_ego_target_has_priority"
    )
    LeftTargetLeftEgoTargetHasPriority = "turning_left_target_turning_left_ego_target_has_priority"
    LeftTargetStraightEgoTargetHasPriority = (
        "turning_left_target_going_straight_ego_target_has_priority"
    )
    StraightTargetRightEgoTargetHasPriority = (
        "going_straight_target_turning_right_ego_target_has_priority"
    )
    StraightTargetLeftEgoTargetHasPriorityNotOncoming = (
        "going_straight_target_turning_left_ego_target_has_priority_not_oncoming"
    )
    StraightTargetStraightEgoTargetHasPriority = (
        "going_straight_target_going_straight_ego_target_has_priority"
    )

    RightTargetLeftEgoTargetHasPriorityOncoming = (
        "turning_right_target_turning_left_ego_target_has_priority_oncoming"
    )

    StraightTargetLeftEgoTargetHasPriorityOncoming = (
        "going_straight_target_turning_left_ego_target_has_priority_oncoming"
    )


class PredCutIn(AbstractPredicate):
    predicate_name = GeneralPredicates.CutIn
    arity = 2

    def __init__(self, config: PredicateConfig | None = None):
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
        same_lane = self._same_lane_evaluator.evaluate_boolean(world, time_step, vehicle_ids)
        if not same_lane:
            return False
        cutting_lane = cutting_vehicle.lane_at_time_step(time_step)
        cutted_lat = cutted_vehicle.get_lat_state(time_step, cutting_lane)
        cutting_lat = cutting_vehicle.get_lat_state(time_step)
        d_p = cutted_lat.d
        d_k = cutting_lat.d
        orient_k = cutting_lat.theta

        result = (d_k < d_p and orient_k > self.config.eps) or (
            d_k > d_p and orient_k < -self.config.eps
        )
        return result

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: tuple[int]) -> float:
        cutting_vehicle = world.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world.vehicle_by_id(vehicle_ids[1])
        # For model-free evaluation, there is no mpr_world.
        single_lane = self._single_lane_evaluator.evaluate_robustness(
            world,
            time_step,
            # TODO: Why is this rewrapped?
            (vehicle_ids[0],),
        )
        same_lane = self._same_lane_evaluator.evaluate_robustness(world, time_step, vehicle_ids)

        cutting_lane = cutting_vehicle.lane_at_time_step(time_step)
        cutted_lat = cutted_vehicle.get_lat_state(time_step, cutting_lane)
        cutting_lat = cutting_vehicle.get_lat_state(time_step)
        r_l_dist = cutted_lat.d - cutting_lat.d
        r_l_orient = subtract_orientations(cutting_lat.theta, self.config.eps)
        l_r_dist = cutting_lat.d - cutted_lat.d
        l_r_orient = subtract_orientations(-self.config.eps, cutting_lat.theta)

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
        # For model-free evaluation, there is no mpr_world
        latest_value = self.evaluate_robustness(world, time_step, vehicle_ids)
        latest_value_normalized = (latest_value + 1) / 2
        violation_color = self._get_color_map()(latest_value_normalized)
        violation_color_hex = matplotlib.colors.rgb2hex(violation_color)

        vehicle = vehicle_ids[0]
        draw_params = {
            "dynamic_obstacle": {
                "vehicle_shape": {
                    "occupancy": {"shape": {"rectangle": {"facecolor": violation_color_hex}}}
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


class PredInterstateBroadEnough(AbstractPredicate):
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
            if cal_road_width(lanelet, world.road_network, s) <= self.config.min_interstate_width:
                return False
        return True

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        s = vehicle.get_lon_state(time_step).s
        comparison_list = []
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            comparison_list.append(
                self._scale_lat_dist(
                    cal_road_width(lanelet, world.road_network, s)
                    - self.config.min_interstate_width
                    - self.config.eps
                )
            )
        return min(comparison_list)


class PredInCongestion(AbstractPredicate):
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
                and veh_o.get_lon_state(time_step).v <= self.config.max_congestion_velocity
            ):
                num_vehicles += 1
        if num_vehicles >= self.config.num_veh_congestion:
            return True
        else:
            return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
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
                        self.config.max_congestion_velocity
                        - veh_o.get_lon_state(time_step).v
                        - self.config.eps,
                    ),
                )
            )
        # values are already normalized
        if sum(rob > 0 for rob in rob_cong_veh_list) >= self.config.num_veh_congestion:
            return min(rob for rob in rob_cong_veh_list if rob > 0)
        else:
            return max(rob for rob in rob_cong_veh_list if rob < 0)


class PredInSlowMovingTraffic(AbstractPredicate):
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
                and veh_o.get_lon_state(time_step).v <= self.config.max_slow_moving_traffic_velocity
            ):
                num_vehicles += 1
        if num_vehicles >= self.config.num_veh_slow_moving_traffic:
            return True
        else:
            return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
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
                        self.config.max_slow_moving_traffic_velocity
                        - veh_o.get_lon_state(time_step).v
                        - self.config.eps,
                    ),
                )
            )
        # values are already normalized
        if sum(rob > 0 for rob in rob_cong_veh_list) >= self.config.num_veh_slow_moving_traffic:
            return min(rob for rob in rob_cong_veh_list if rob > 0)
        else:
            return max(rob for rob in rob_cong_veh_list if rob < 0)


class PredInQueueOfVehicles(AbstractPredicate):
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
                and veh_o.get_lon_state(time_step).v <= self.config.max_queue_of_vehicles_velocity
            ):
                num_vehicles += 1
        if num_vehicles >= self.config.num_veh_queue_of_vehicles:
            return True
        else:
            return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
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
                        self.config.max_queue_of_vehicles_velocity
                        - veh_o.get_lon_state(time_step).v
                        - self.config.eps
                    ),
                )
            )
        # values are already normalized
        if sum(rob > 0 for rob in rob_cong_veh_list) >= self.config.num_veh_queue_of_vehicles:
            return min(rob for rob in rob_cong_veh_list if rob > 0)
        else:
            return max(rob for rob in rob_cong_veh_list if rob < 0)


class PredMakesUTurn(AbstractPredicate):
    """
    Predicate which evaluates if vehicle makes U-turn
    """

    predicate_name = GeneralPredicates.MakesUTurn
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanes = vehicle.lanes_at_time_step(time_step)
        for lane in lanes:
            if self.config.u_turn <= abs(vehicle.get_lat_state(time_step, lane).theta):
                return True
        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        robustness_values = []
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanes = vehicle.lanes_at_time_step(time_step)
        for la in lanes:
            robustness_values.append(
                self._scale_angle(
                    abs(vehicle.get_lat_state(time_step, la).theta)
                    - self.config.u_turn
                    - self.config.eps,
                )
            )
        return max(robustness_values)  # TODO why not min?


##############
# intersection
##############


# ---------------------------------------------------------------------- #
class PredTurningRight(AbstractPredicate):
    """
    evaluates if a vehicle is turning right
    """

    predicate_name = GeneralPredicates.TurningRight
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        incoming = vehicle.incoming_intersection
        if incoming is None:
            return False
        # find corresponding right turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_ids_at_time_step(time_step)
        right_turn_lane = road_network.find_lanes_incoming_by_id(incoming.incoming_id)[
            0
        ]  # lanes: [right, straight, left]
        # vehicle odes not occupy right turning lanelet
        if len(lanelets_assignment_current.intersection(incoming.successors_right)) == 0:
            return False
        # vehicle occupies right turning lanelet
        else:
            state = vehicle.get_cr_state(time_step)
            d_center_to_left = right_turn_lane.distance_to_left(*state.position)
            # center of vehicle in right turning lanelet
            if d_center_to_left <= 0:
                return False
            # center of vehicle outside right turning lanelet
            else:
                return True

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        incoming = vehicle.incoming_intersection
        if incoming is None:
            return self._scale_lon_dist(-np.inf)
        # find corresponding right turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_ids_at_time_step(time_step)
        right_turn_lane = road_network.find_lanes_incoming_by_id(incoming.incoming_id)[
            0
        ]  # lanes: [right, straight, left]
        # important bounds of vehicle and lanelet in Curvilinear Coordinate System
        front_s = vehicle.front_s(time_step, right_turn_lane)
        rear_s = vehicle.rear_s(time_step, right_turn_lane)
        right_turn_start_s, right_turn_end_s = road_network.get_lanelets_start_end_s(
            incoming.successors_right, right_turn_lane
        )
        intersection_lanelets = incoming.successors_straight.union(incoming.successors_left).union(
            incoming.successors_right
        )
        # case 1: vehicle only in incoming
        if (
            len(lanelets_assignment_current.intersection(incoming.incoming_lanelets)) > 0
            and len(lanelets_assignment_current.intersection(intersection_lanelets)) == 0
        ):
            rob = self._scale_lon_dist(front_s - right_turn_start_s)
        # case 2: vehicle occupies right turning at intersection
        elif len(lanelets_assignment_current.intersection(incoming.successors_right)) > 0:
            state = vehicle.get_cr_state(time_step)
            d_center_to_left = right_turn_lane.distance_to_left(*state.position)
            if d_center_to_left > 0:
                # out of right turning lanelet
                d_left = utils.distance_to_left_bounds_clcs(vehicle, right_turn_lane, time_step)
                if len(d_left) == 0:
                    rob = -np.inf
                else:
                    rob = np.max(d_left, initial=-np.inf)
                rob = self._scale_lat_dist(rob)
            else:
                # in right turning lanelet
                # TODO: should also consider lateral information (distance to left bound)?
                rob = np.min([front_s - right_turn_start_s, right_turn_end_s - rear_s])
                rob = self._scale_lon_dist(rob)
        # case 3: vehicle occupies left turning or straight lanelet at intersection instead of right turning
        elif utils.check_in_intersection(road_network, lanelets_assignment_current):
            d_left = utils.distance_to_left_bounds_clcs(vehicle, right_turn_lane, time_step)
            if len(d_left) == 0:
                rob = -np.inf
            else:
                rob = -np.max(d_left, initial=-np.inf)
            rob = self._scale_lat_dist(rob)
        # case 4: vehicle exits intersection
        else:
            (
                incoming_right_turn,
                right_turn_lane,
            ) = utils.get_right_turning_lane_by_lanelets(lanelets_assignment_current, road_network)
            rear_s = vehicle.rear_s(time_step, right_turn_lane)
            right_turn_end_s = utils.get_lanelets_end_s(
                right_turn_lane, incoming_right_turn.successors_right, road_network
            )
            front_s = vehicle.front_s(time_step, right_turn_lane)
            right_turn_start_s = utils.get_lanelets_start_s(
                right_turn_lane, incoming_right_turn.successors_straight, road_network
            )
            rob = self._scale_lon_dist(min(right_turn_end_s - rear_s, front_s - right_turn_start_s))
        return rob


class PredTurningLeft(AbstractPredicate):
    """
    evaluates if a vehicle is turning left
    """

    predicate_name = GeneralPredicates.TurningLeft
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        incoming = vehicle.incoming_intersection
        if incoming is None:
            return False
        # find corresponding left turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_assignment[time_step]
        left_turn_lane = road_network.find_lanes_incoming_by_id(incoming.incoming_id)[
            2
        ]  # lanes: [right, straight, left]
        # vehicle does not occupy left turning lanelet
        if len(lanelets_assignment_current.intersection(incoming.successors_left)) == 0:
            return False
        else:
            state = vehicle.states_cr[time_step]
            d_center_to_right = left_turn_lane.distance_to_right(*state.position)
            if d_center_to_right < 0:
                # out of left turning lanelet
                return False
            else:
                # in left turning lanelet
                return True

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        incoming = vehicle.incoming_intersection
        if incoming is None:
            return self._scale_lon_dist(-np.inf)
        # find corresponding left turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_assignment[time_step]
        left_turn_lane = road_network.find_lanes_incoming_by_id(incoming.incoming_id)[
            2
        ]  # lanes: [right, straight, left]
        # important bounds of vehicle and lanelet in Curvilinear Coordinate System
        front_s = vehicle.front_s(time_step, left_turn_lane)
        rear_s = vehicle.rear_s(time_step, left_turn_lane)
        left_turn_start_s, left_turn_end_s = road_network.get_lanelets_start_end_s(
            incoming.successors_left, left_turn_lane
        )
        intersection_lanelets = incoming.successors_straight.union(incoming.successors_left).union(
            incoming.successors_right
        )
        # case 1: vehicle only in incoming
        if (
            len(lanelets_assignment_current.intersection(incoming.incoming_lanelets)) > 0
            and len(lanelets_assignment_current.intersection(intersection_lanelets)) == 0
        ):
            rob = self._scale_lon_dist(front_s - left_turn_start_s)
        # case 2: vehicle occupies left turning at intersection
        elif len(lanelets_assignment_current.intersection(incoming.successors_left)) > 0:
            state = vehicle.get_cr_state(time_step)
            d_center_to_right = left_turn_lane.distance_to_right(*state.position)
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
        elif utils.check_in_intersection(road_network, lanelets_assignment_current):
            d_right = utils.distance_to_right_bounds_clcs(vehicle, left_turn_lane, time_step)
            if len(d_right) == 0:
                rob = -np.inf
            else:
                rob = np.min(d_right, initial=np.inf)
            rob = self._scale_lat_dist(rob)
        # case 4: vehicle exits intersection
        else:
            (
                incoming_left_turn,
                left_turn_lane,
            ) = utils.get_left_turning_lane_by_lanelets(lanelets_assignment_current, road_network)
            rear_s = vehicle.rear_s(time_step, left_turn_lane)
            left_turn_end_s = utils.get_lanelets_end_s(
                left_turn_lane, incoming_left_turn.successors_left, road_network
            )
            front_s = vehicle.front_s(time_step, left_turn_lane)
            left_turn_start_s = utils.get_lanelets_start_s(
                left_turn_lane, incoming_left_turn.successors_straight, road_network
            )
            rob = self._scale_lon_dist(min(left_turn_end_s - rear_s, front_s - left_turn_start_s))
        return rob


class PredGoingStraight(AbstractPredicate):
    """
    evaluates if a vehicle is going straight
    """

    predicate_name = GeneralPredicates.GoingStraight
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        incoming = vehicle.incoming_intersection
        if incoming is None:
            return False
        # find corresponding going straight lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_assignment[time_step]
        straight_lane = road_network.find_lanes_incoming_by_id(incoming.incoming_id)[
            1
        ]  # lanes: [right, straight, left]
        if len(lanelets_assignment_current.intersection(incoming.successors_straight)) == 0:
            return False
        else:
            state = vehicle.states_cr[time_step]
            d_center_to_left = straight_lane.distance_to_left(*state.position)
            d_center_to_right = straight_lane.distance_to_right(*state.position)
            if (d_center_to_left < 0) or (d_center_to_right < 0):
                # out of going straight lanelet
                return False
            else:
                # in going straight lanelet
                return True

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find incoming lanelet
        incoming = vehicle.incoming_intersection
        if incoming is None:
            return self._scale_lon_dist(-np.inf)
        # find corresponding left turn lane and lanelets in intersection
        lanelets_assignment_current = vehicle.lanelet_ids_at_time_step(time_step)
        straight_lane = road_network.find_lanes_incoming_by_id(incoming.incoming_id)[
            1
        ]  # lanes: [right, straight, left]
        # important bounds of vehicle and lanelet in Curvilinear Coordinate System
        front_s = vehicle.front_s(time_step, straight_lane)
        rear_s = vehicle.rear_s(time_step, straight_lane)
        straight_start_s, straight_end_s = road_network.get_lanelets_start_end_s(
            incoming.successors_straight, straight_lane
        )
        intersection_lanelets = incoming.successors_straight.union(incoming.successors_left).union(
            incoming.successors_right
        )
        # case 1: vehicle only in incoming
        if (
            len(lanelets_assignment_current.intersection(incoming.incoming_lanelets)) > 0
            and len(lanelets_assignment_current.intersection(intersection_lanelets)) == 0
        ):
            rob = self._scale_lon_dist(front_s - straight_start_s)
        # case 2: vehicle occupies left turning at intersection
        elif len(lanelets_assignment_current.intersection(incoming.successors_straight)) > 0:
            state = vehicle.get_cr_state(time_step)
            d_center_to_left = straight_lane.distance_to_left(*state.position)
            d_center_to_right = straight_lane.distance_to_right(*state.position)
            if d_center_to_left > 0:
                # out of straight going lanelet
                d_left = utils.distance_to_left_bounds_clcs(vehicle, straight_lane, time_step)
                if len(d_left) == 0:
                    rob = -np.inf
                else:
                    rob = np.max(d_left, initial=-np.inf)
                rob = self._scale_lat_dist(rob)
            elif d_center_to_right < 0:
                # out of straight going lanelet
                d_right = utils.distance_to_right_bounds_clcs(vehicle, straight_lane, time_step)
                if len(d_right) == 0:
                    rob = -np.inf
                else:
                    rob = np.min(d_right, initial=np.inf)
                rob = self._scale_lat_dist(rob)
            else:
                # in straight going lanelet
                rob = np.min([front_s - straight_start_s, straight_end_s - rear_s])
                rob = self._scale_lon_dist(rob)
        # case 3: vehicle occupies right turning or straight lanelet at intersection instead of right turning
        elif utils.check_in_intersection(road_network, lanelets_assignment_current):
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
        # TODO: change to lateral distance to straight lane
        else:
            (
                incoming_straight,
                straight_lane,
            ) = utils.get_straight_going_lane_by_lanelets(lanelets_assignment_current, road_network)
            rear_s = vehicle.rear_s(time_step, straight_lane)
            straight_end_s = utils.get_lanelets_end_s(
                straight_lane, incoming_straight.successors_straight, road_network
            )
            front_s = vehicle.front_s(time_step, straight_lane)
            straight_start_s = utils.get_lanelets_start_s(
                straight_lane, incoming_straight.successors_straight, road_network
            )
            rob = self._scale_lon_dist(min(straight_end_s - rear_s, front_s - straight_start_s))
        return rob


class PredTurningSamePriorityBase(AbstractPredicate):
    predicate_name = GeneralPredicates.TurningSamePriorityBase
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityRightRight(config)
        self._turning_ego = PredTurningRight(config)
        self._turning_target = PredTurningRight(config)

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        bool_turning_ego = self._turning_ego.evaluate_boolean(world, time_step, (ego_vehicle_id,))
        bool_turning_target = self._turning_target.evaluate_boolean(
            world, time_step, (target_vehicle_id,)
        )
        bool_same_priority = self._same_priority.evaluate_boolean(world, time_step, vehicle_ids)
        return bool_turning_ego and bool_turning_target and bool_same_priority

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        rob_turning_ego = self._turning_ego.evaluate_robustness(world, time_step, (ego_vehicle_id,))
        rob_turning_target = self._turning_target.evaluate_robustness(
            world, time_step, (target_vehicle_id,)
        )
        rob_same_priority = self._same_priority.evaluate_robustness(world, time_step, vehicle_ids)
        rob = min(rob_turning_ego, rob_turning_target, rob_same_priority)
        return rob


class PredRightEgoRightTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.RightEgoRightTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityRightRight(config)
        self._turning_ego = PredTurningRight(config)
        self._turning_target = PredTurningRight(config)


class PredRightEgoLeftTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.RightEgoLeftTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityRightLeft(config)
        self._turning_ego = PredTurningRight(config)
        self._turning_target = PredTurningLeft(config)


class PredRightEgoStraightTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.RightEgoStraightTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityRightStraight(config)
        self._turning_ego = PredTurningRight(config)
        self._turning_target = PredGoingStraight(config)


class PredLeftEgoRightTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.LeftEgoRightTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityLeftRight(config)
        self._turning_ego = PredTurningLeft(config)
        self._turning_target = PredTurningRight(config)


class PredLeftEgoLeftTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.LeftEgoLeftTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityLeftLeft(config)
        self._turning_ego = PredTurningLeft(config)
        self._turning_target = PredTurningLeft(config)


class PredLeftEgoStraightTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.LeftEgoStraightTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityLeftStraight(config)
        self._turning_ego = PredTurningLeft(config)
        self._turning_target = PredGoingStraight(config)


class PredStraightEgoRightTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.StraightEgoRightTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityStraightRight(config)
        self._turning_ego = PredGoingStraight(config)
        self._turning_target = PredTurningRight(config)


class PredStraightEgoLeftTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.StraightEgoLeftTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityStraightLeft(config)
        self._turning_ego = PredGoingStraight(config)
        self._turning_target = PredTurningLeft(config)


class PredStraightEgoStraightTargetSamePriority(PredTurningSamePriorityBase):
    predicate_name = GeneralPredicates.StraightEgoStraightTargetSamePriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_priority = PredSamePriorityStraightStraight(config)
        self._turning_ego = PredGoingStraight(config)
        self._turning_target = PredGoingStraight(config)


class PredTurningHasPriorityBase(AbstractPredicate):
    predicate_name = GeneralPredicates.TurningHasPriorityBase
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityRightRight(config)
        self._turning_target = PredTurningRight(config)
        self._turning_ego = PredTurningRight(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        bool_turning_target = self._turning_target.evaluate_boolean(
            world, time_step, [target_vehicle_id]
        )
        bool_turning_ego = self._turning_ego.evaluate_boolean(world, time_step, [ego_vehicle_id])
        bool_target_has_priority = self._target_has_priority.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        return bool_turning_target and bool_turning_ego and bool_target_has_priority

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        rob_turning_target = self._turning_target.evaluate_robustness(
            world, time_step, (target_vehicle_id,)
        )
        rob_turning_ego = self._turning_ego.evaluate_robustness(world, time_step, (ego_vehicle_id,))
        rob_target_has_priority = self._target_has_priority.evaluate_robustness(
            world, time_step, (target_vehicle_id, ego_vehicle_id)
        )
        rob = min(rob_turning_target, rob_turning_ego, rob_target_has_priority)
        return rob


class PredRightTargetRightEgoTargetHasPriority(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.RightTargetRightEgoTargetHasPriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityRightRight(config)
        self._turning_target = PredTurningRight(config)
        self._turning_ego = PredTurningRight(config)


class PredRightTargetLeftEgoTargetHasPriorityNotOncoming(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.RightTargetLeftEgoTargetHasPriorityNotOncoming
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityRightLeft(config)
        self._turning_target = PredTurningRight(config)
        self._turning_ego = PredTurningLeft(config)
        self._on_oncoming_of = PredOnOncomOf(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        bool_turning_target = self._turning_target.evaluate_boolean(
            world, time_step, [target_vehicle_id]
        )
        bool_turning_ego = self._turning_ego.evaluate_boolean(world, time_step, [ego_vehicle_id])
        bool_target_has_priority = self._target_has_priority.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        bool_on_oncoming_of = self._on_oncoming_of.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        return (
            bool_turning_target
            and bool_turning_ego
            and bool_target_has_priority
            and not bool_on_oncoming_of
        )

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        rob_turning_target = self._turning_target.evaluate_robustness(
            world, time_step, (target_vehicle_id,)
        )
        rob_turning_ego = self._turning_ego.evaluate_robustness(world, time_step, (ego_vehicle_id,))
        rob_target_has_priority = self._target_has_priority.evaluate_robustness(
            world, time_step, (target_vehicle_id, ego_vehicle_id)
        )
        rob_on_oncoming_of = self._on_oncoming_of.evaluate_robustness(
            world, time_step, (target_vehicle_id, ego_vehicle_id)
        )
        rob = min(
            rob_turning_target,
            rob_turning_ego,
            rob_target_has_priority,
            -rob_on_oncoming_of,
        )
        return rob


class PredRightTargetLeftEgoTargetHasPriorityOncoming(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.RightTargetLeftEgoTargetHasPriorityOncoming
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityRightLeft(config)
        self._turning_target = PredTurningRight(config)
        self._turning_ego = PredTurningLeft(config)
        self._on_oncoming_of = PredOnOncomOf(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        bool_turning_target = self._turning_target.evaluate_boolean(
            world, time_step, [target_vehicle_id]
        )
        bool_turning_ego = self._turning_ego.evaluate_boolean(world, time_step, [ego_vehicle_id])
        bool_target_has_priority = self._target_has_priority.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        bool_on_oncoming_of = self._on_oncoming_of.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        return (
            bool_turning_target
            and bool_turning_ego
            and bool_target_has_priority
            and bool_on_oncoming_of
        )

    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        rob_turning_target = self._turning_target.evaluate_robustness(
            world, time_step, (target_vehicle_id,)
        )
        rob_turning_ego = self._turning_ego.evaluate_robustness(world, time_step, [ego_vehicle_id])
        rob_target_has_priority = self._target_has_priority.evaluate_robustness(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        rob_on_oncoming_of = self._on_oncoming_of.evaluate_robustness(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        rob = min(
            rob_turning_target,
            rob_turning_ego,
            rob_target_has_priority,
            rob_on_oncoming_of,
        )
        return rob


class PredRightTargetStraightEgoTargetHasPriority(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.RightTargetStraightEgoTargetHasPriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityRightStraight(config)
        self._turning_target = PredTurningRight(config)
        self._turning_ego = PredGoingStraight(config)


class PredLeftTargetRightEgoTargetHasPriority(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.LeftTargetRightEgoTargetHasPriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityLeftRight(config)
        self._turning_target = PredTurningLeft(config)
        self._turning_ego = PredTurningRight(config)


class PredLeftTargetLeftEgoTargetHasPriority(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.LeftTargetLeftEgoTargetHasPriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityLeftLeft(config)
        self._turning_target = PredTurningLeft(config)
        self._turning_ego = PredTurningLeft(config)


class PredLeftTargetStraightEgoTargetHasPriority(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.LeftTargetStraightEgoTargetHasPriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityLeftStraight(config)
        self._turning_target = PredTurningLeft(config)
        self._turning_ego = PredGoingStraight(config)


class PredStraightTargetRightEgoTargetHasPriority(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.StraightTargetRightEgoTargetHasPriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityStraightRight(config)
        self._turning_target = PredGoingStraight(config)
        self._turning_ego = PredTurningRight(config)


class PredStraightTargetLeftEgoTargetHasPriorityNotOncoming(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.StraightTargetLeftEgoTargetHasPriorityNotOncoming
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityStraightLeft(config)
        self._turning_target = PredGoingStraight(config)
        self._turning_ego = PredTurningLeft(config)
        self._on_oncoming_of = PredOnOncomOf(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        bool_turning_target = self._turning_target.evaluate_boolean(
            world, time_step, [target_vehicle_id]
        )
        bool_turning_ego = self._turning_ego.evaluate_boolean(world, time_step, [ego_vehicle_id])
        bool_target_has_priority = self._target_has_priority.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        bool_on_oncoming_of = self._on_oncoming_of.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        return (
            bool_turning_target
            and bool_turning_ego
            and bool_target_has_priority
            and not bool_on_oncoming_of
        )

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        rob_turning_target = self._turning_target.evaluate_robustness(
            world, time_step, [target_vehicle_id]
        )
        rob_turning_ego = self._turning_ego.evaluate_robustness(world, time_step, [ego_vehicle_id])
        rob_target_has_priority = self._target_has_priority.evaluate_robustness(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        rob_on_oncoming_of = self._on_oncoming_of.evaluate_robustness(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        rob = min(
            rob_turning_target,
            rob_turning_ego,
            rob_target_has_priority,
            -rob_on_oncoming_of,
        )
        return rob


class PredStraightTargetLeftEgoTargetHasPriorityOncoming(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.StraightTargetLeftEgoTargetHasPriorityOncoming
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityStraightLeft(config)
        self._turning_target = PredGoingStraight(config)
        self._turning_ego = PredTurningLeft(config)
        self._on_oncoming_of = PredOnOncomOf(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        bool_turning_target = self._turning_target.evaluate_boolean(
            world, time_step, [target_vehicle_id]
        )
        bool_turning_ego = self._turning_ego.evaluate_boolean(world, time_step, [ego_vehicle_id])
        bool_target_has_priority = self._target_has_priority.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        bool_on_oncoming_of = self._on_oncoming_of.evaluate_boolean(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        return (
            bool_turning_target
            and bool_turning_ego
            and bool_target_has_priority
            and bool_on_oncoming_of
        )

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        ego_vehicle_id = vehicle_ids[0]
        target_vehicle_id = vehicle_ids[1]
        rob_turning_target = self._turning_target.evaluate_robustness(
            world, time_step, [target_vehicle_id]
        )
        rob_turning_ego = self._turning_ego.evaluate_robustness(world, time_step, [ego_vehicle_id])
        rob_target_has_priority = self._target_has_priority.evaluate_robustness(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        rob_on_oncoming_of = self._on_oncoming_of.evaluate_robustness(
            world, time_step, [target_vehicle_id, ego_vehicle_id]
        )
        rob = min(
            rob_turning_target,
            rob_turning_ego,
            rob_target_has_priority,
            rob_on_oncoming_of,
        )
        return rob


class PredStraightTargetStraightEgoTargetHasPriority(PredTurningHasPriorityBase):
    predicate_name = GeneralPredicates.StraightTargetStraightEgoTargetHasPriority
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._target_has_priority = PredHasPriorityStraightStraight(config)
        self._turning_target = PredGoingStraight(config)
        self._turning_ego = PredGoingStraight(config)
