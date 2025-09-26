import logging
from collections import defaultdict
from typing import List

import numpy as np
from commonroad.scenario.traffic_sign import TrafficSignIDGermany

from crmonitor.common.world import World
from crmonitor.predicates import utils
from crmonitor.predicates.base import (
    AbstractPredicate,
    PredicateConfig,
    PredicateName,
)

_LOGGER = logging.getLogger(__name__)


class PriorityPredicates(PredicateName):
    SamePriorityBase = "same_priority_base"
    SamePriorityRightRight = "same_priority_right_right"
    SamePriorityRightLeft = "same_priority_right_left"
    SamePriorityLeftRight = "same_priority_left_right"
    SamePriorityRightStraight = "same_priority_right_straight"
    SamePriorityStraightRight = "same_priority_straight_right"
    SamePriorityLeftStraight = "same_priority_left_straight"
    SamePriorityStraightLeft = "same_priority_straight_left"
    SamePriorityLeftLeft = "same_priority_left_left"
    SamePriorityStraightStraight = "same_priority_straight_straight"

    HasPriorityBase = "has_priority_base"
    HasPriorityRightRight = "has_priority_right_right"
    HasPriorityRightLeft = "has_priority_right_left"
    HasPriorityRightStraight = "has_priority_right_straight"
    HasPriorityLeftRight = "has_priority_left_right"
    HasPriorityLeftLeft = "has_priority_left_left"
    HasPriorityLeftStraight = "has_priority_left_straight"
    HasPriorityStraightRight = "has_priority_straight_right"
    HasPriorityStraightLeft = "has_priority_straight_left"
    HasPriorityStraightStraight = "has_priority_straight_straight"

    AtTrafficSignStop = "at_traffic_sign_stop"
    RelevantTrafficLight = "relevant_traffic_light"


# --------------------------------------------------------------------------------------------------------------------#
class TrafficSignPriority:
    def __init__(self):
        self.priority = {}
        self.priority[
            TrafficSignIDGermany.ADDITION_LEFT_TURNING_PRIORITY_WITH_OPPOSITE_RIGHT_YIELD
        ] = self.PriorityIntersection(5, 4, 4, 1)
        self.priority[TrafficSignIDGermany.ADDITION_LEFT_TURNING_PRIORITY_WITH_OPPOSITE_YIELD] = (
            self.PriorityIntersection(5, 4, None, 2)
        )
        self.priority[TrafficSignIDGermany.ADDITION_LEFT_TURNING_PRIORITY_WITH_RIGHT_YIELD] = (
            self.PriorityIntersection(5, None, 4, 3)
        )
        self.priority[
            TrafficSignIDGermany.ADDITION_RIGHT_TURNING_PRIORITY_WITH_OPPOSITE_LEFT_YIELD
        ] = self.PriorityIntersection(4, 4, 5, 4)
        self.priority[TrafficSignIDGermany.ADDITION_RIGHT_TURNING_PRIORITY_WITH_OPPOSITE_YIELD] = (
            self.PriorityIntersection(None, 4, 5, 5)
        )
        self.priority[TrafficSignIDGermany.ADDITION_RIGHT_TURNING_PRIORITY_WITH_LEFT_YIELD] = (
            self.PriorityIntersection(4, None, 5, 6)
        )
        self.priority[
            TrafficSignIDGermany.ADDITION_LEFT_TRAFFIC_PRIORITY_WITH_STRAIGHT_RIGHT_YIELD
        ] = self.PriorityIntersection(2, 2, 2, 7)
        self.priority[TrafficSignIDGermany.ADDITION_LEFT_TRAFFIC_PRIORITY_WITH_STRAIGHT_YIELD] = (
            self.PriorityIntersection(2, 2, None, 8)
        )
        self.priority[
            TrafficSignIDGermany.ADDITION_RIGHT_TRAFFIC_PRIORITY_WITH_STRAIGHT_LEFT_YIELD
        ] = self.PriorityIntersection(2, 2, 2, 9)
        self.priority[TrafficSignIDGermany.ADDITION_RIGHT_TRAFFIC_PRIORITY_WITH_STRAIGHT_YIELD] = (
            self.PriorityIntersection(None, 2, 2, 10)
        )
        self.priority[TrafficSignIDGermany.PRIORITY] = self.PriorityIntersection(4, 5, 4, 11)
        self.priority[TrafficSignIDGermany.RIGHT_OF_WAY] = self.PriorityIntersection(4, 5, 4, 12)
        self.priority[TrafficSignIDGermany.YIELD] = self.PriorityIntersection(2, 2, 2, 13)
        self.priority[TrafficSignIDGermany.STOP] = self.PriorityIntersection(1, 1, 1, 14)
        self.priority[TrafficSignIDGermany.WARNING_RIGHT_BEFORE_LEFT] = self.PriorityIntersection(
            3, 3, 3, 15
        )
        self.priority[TrafficSignIDGermany.GREEN_ARROW] = self.PriorityIntersection(
            None, None, 0, 16
        )

    def get_priority(self):
        return self.priority

    class PriorityIntersection:
        def __init__(self, left_priority, straight_priority, right_priority, evaluation_index):
            self.left = left_priority
            self.straight = straight_priority
            self.right = right_priority
            self.evaluation_idx = evaluation_index


class PredAtTrafficSignStop(AbstractPredicate):
    predicate_name = PriorityPredicates.AtTrafficSignStop
    arity = 1
    stop_traffic_sign_deu = TrafficSignIDGermany.STOP

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        If the vehicle locates at the lanelet with a stop traffic sign (206), return True, otherwise, return False.
        """
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        # find all traffic sign elements with type stop (206) in lanelets_dir
        for lanelet_id in vehicle.lanelets_dir:
            traffic_sign_elements = utils.traffic_sign(
                lanelet_id, self.stop_traffic_sign_deu, world.road_network
            )
            if traffic_sign_elements is None:
                continue
            # check if vehicle in this lanelet in lateral horizon
            if not vehicle.lanelet_assignment[time_step].intersection([lanelet_id]):
                continue
            return True
        return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        # find all traffic sign elements with type stop (206) in lanelets_dir
        for lanelet_id in vehicle.lanelets_dir:
            traffic_sign_elements = utils.traffic_sign(
                lanelet_id, self.stop_traffic_sign_deu, road_network
            )
            if traffic_sign_elements is None:
                continue
            # check if vehicle in this lanelet in lateral horizon
            d_lane = utils.distance_to_lanes(vehicle, [lanelet_id], world, time_step)
            if d_lane < 0:
                continue
            return 1.0
        return -1.0


class PredRelevantTrafficLight(AbstractPredicate):
    """
    evaluates if an upcoming intersection is regulated by traffic lights
    """

    predicate_name = PriorityPredicates.RelevantTrafficLight
    arity = 1

    def __init__(self, config: PredicateConfig | None = None):
        super().__init__(config)
        self._dict_lanelets_traffic_light = defaultdict(lambda: None)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        check if there is an active traffic light in lanelet_dir or successors
        """
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        if self._dict_lanelets_traffic_light[vehicle.lanelets_dir[0]] is not None:
            traffic_light_lanelets = self._dict_lanelets_traffic_light[vehicle.lanelets_dir[0]]
        else:
            reach_suc_id = road_network.get_reach_suc_cache(vehicle.lanelets_dir[0])
            traffic_light_lanelets = list()
            for l_id in reach_suc_id:
                lanelet_suc = road_network.lanelet_network.find_lanelet_by_id(l_id)
                if len(lanelet_suc.traffic_lights) == 0:
                    continue
                assert len(lanelet_suc.traffic_lights) == 1, (
                    "TODO: Only works for one traffic light per lanelet!"
                )
                tl = road_network.lanelet_network.find_traffic_light_by_id(
                    list(lanelet_suc.traffic_lights)[0]
                )
                if tl.active:
                    traffic_light_lanelets.append(l_id)
            self._dict_lanelets_traffic_light[vehicle.lanelets_dir[0]] = traffic_light_lanelets
        # check if vehicle in this lanelet in lateral horizon
        if vehicle.lanelet_assignment[time_step].intersection(traffic_light_lanelets):
            return True
        else:
            return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        """
        returns the distance to the nearest active traffic light
        """
        lanelet_with_active_tl = list()
        robustness = -np.inf
        road_network = world.road_network
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        ref_path = vehicle.ref_path_lane
        reach_suc_id = road_network.get_reach_suc_cache(vehicle.lanelets_dir[0])
        # intersection between reference path and successors of lanelets_dir
        lanelets_ids = ref_path.contained_lanelets.intersection(reach_suc_id)
        for l_id in lanelets_ids:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(l_id)
            if len(lanelet.traffic_lights) == 0:
                continue
            # check if vehicle in this lanelet in lateral horizon
            d_lane = utils.distance_to_lanes(vehicle, [l_id], world, time_step)
            if d_lane < 0:
                continue
            assert len(lanelet.traffic_lights) == 1, (
                "TODO: Only works for one traffic light per lanelet!"
            )
            tl = road_network.lanelet_network.find_traffic_light_by_id(
                list(lanelet.traffic_lights)[0]
            )
            if tl.active:
                lanelet_with_active_tl.append(l_id)
        if len(lanelet_with_active_tl) == 0:
            return self._scale_lon_dist(float(robustness))
        # Get the front and rear longitudinal value of the vehicle and lanelets with traffic light
        front_s = vehicle.front_s(time_step, ref_path) or -np.inf
        rear_s = vehicle.rear_s(time_step, ref_path) or -np.inf
        lanelet_start_s = np.array(
            [
                ref_path.clcs.convert_to_curvilinear_coords(
                    *utils.get_lanelet_start_line(
                        world.road_network.lanelet_network.find_lanelet_by_id(lanelet)
                    )[0]
                )[0]
                for lanelet in lanelet_with_active_tl
            ]
        )
        lanelet_end_s = np.array(
            [
                ref_path.clcs.convert_to_curvilinear_coords(
                    *utils.get_lanelet_end_line(
                        world.road_network.lanelet_network.find_lanelet_by_id(lanelet)
                    )[0]
                )[0]
                for lanelet in lanelet_with_active_tl
            ]
        )
        for i in range(lanelet_start_s.shape[0]):
            # lanelet in front of vehicle
            if (front_s - lanelet_start_s[i]) < 0 < (lanelet_end_s[i] - front_s):
                robustness = max(robustness, front_s - lanelet_start_s[i])
            # vehicle in front of lanelet
            elif (lanelet_end_s[i] - rear_s) <= 0 <= (front_s - lanelet_start_s[i]):
                robustness = max(robustness, lanelet_end_s[i] - rear_s)
            # vehicle inside lanelet
            else:
                distance_robustness = min(front_s - lanelet_start_s[i], lanelet_end_s[i] - rear_s)
                robustness = max(robustness, distance_robustness)
        return self._scale_lon_dist(float(robustness))


class PredSamePriorityBase(AbstractPredicate):
    predicate_name = PriorityPredicates.SamePriorityBase
    arity = 2
    first_direction = None
    second_direction = None
    traffic_sign_priority = TrafficSignPriority()

    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        incoming_k = vehicle_k.incoming_intersection
        k_incoming_relevant_lanelets = incoming_k.incoming_lanelets.union(
            incoming_k.successors_left,
            incoming_k.successors_right,
            incoming_k.successors_straight,
        )
        incoming_p = vehicle_p.incoming_intersection
        p_incoming_relevant_lanelets = incoming_p.incoming_lanelets.union(
            incoming_p.successors_left,
            incoming_p.successors_right,
            incoming_p.successors_straight,
        )
        priority_k = utils.get_priority(
            k_incoming_relevant_lanelets,
            road_network,
            self.first_direction,
            self.traffic_sign_priority.get_priority(),
        )
        priority_p = utils.get_priority(
            p_incoming_relevant_lanelets,
            road_network,
            self.second_direction,
            self.traffic_sign_priority.get_priority(),
        )
        return priority_k == priority_p

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        incoming_k = vehicle_k.incoming_intersection
        k_incoming_relevant_lanelets = incoming_k.incoming_lanelets.union(
            incoming_k.successors_left,
            incoming_k.successors_right,
            incoming_k.successors_straight,
        )
        incoming_p = vehicle_p.incoming_intersection
        p_incoming_relevant_lanelets = incoming_p.incoming_lanelets.union(
            incoming_p.successors_left,
            incoming_p.successors_right,
            incoming_p.successors_straight,
        )
        priority_k = utils.get_priority(
            k_incoming_relevant_lanelets,
            road_network,
            self.first_direction,
            self.traffic_sign_priority.get_priority(),
        )
        priority_p = utils.get_priority(
            p_incoming_relevant_lanelets,
            road_network,
            self.second_direction,
            self.traffic_sign_priority.get_priority(),
        )
        if priority_k != priority_p:
            # rob = -abs(priority_k - priority_p) / 5
            rob = -1
        else:
            rob = 1
        return rob


class PredSamePriorityRightRight(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityRightRight
    arity = 2
    first_direction = "right"
    second_direction = "right"


class PredSamePriorityRightLeft(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityRightLeft
    arity = 2
    first_direction = "right"
    second_direction = "left"


class PredSamePriorityRightStraight(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityRightStraight
    arity = 2
    first_direction = "right"
    second_direction = "straight"


class PredSamePriorityLeftRight(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityLeftRight
    arity = 2
    first_direction = "left"
    second_direction = "right"


class PredSamePriorityLeftLeft(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityLeftLeft
    arity = 2
    first_direction = "left"
    second_direction = "left"


class PredSamePriorityLeftStraight(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityLeftStraight
    arity = 2
    first_direction = "left"
    second_direction = "straight"


class PredSamePriorityStraightRight(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityStraightRight
    arity = 2
    first_direction = "straight"
    second_direction = "right"


class PredSamePriorityStraightLeft(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityStraightLeft
    arity = 2
    first_direction = "straight"
    second_direction = "left"


class PredSamePriorityStraightStraight(PredSamePriorityBase):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """

    predicate_name = PriorityPredicates.SamePriorityStraightStraight
    arity = 2
    first_direction = "straight"
    second_direction = "straight"


class PredHasPriorityBase(AbstractPredicate):
    predicate_name = PriorityPredicates.HasPriorityBase
    arity = 2
    first_direction = None
    second_direction = None
    traffic_sign_priority = TrafficSignPriority()

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        incoming_k = vehicle_k.incoming_intersection
        k_incoming_relevant_lanelets = incoming_k.incoming_lanelets.union(
            incoming_k.successors_left,
            incoming_k.successors_right,
            incoming_k.successors_straight,
        )
        incoming_p = vehicle_p.incoming_intersection
        p_incoming_relevant_lanelets = incoming_p.incoming_lanelets.union(
            incoming_p.successors_left,
            incoming_p.successors_right,
            incoming_p.successors_straight,
        )
        priority_k = utils.get_priority(
            k_incoming_relevant_lanelets,
            road_network,
            self.first_direction,
            self.traffic_sign_priority.get_priority(),
        )
        priority_p = utils.get_priority(
            p_incoming_relevant_lanelets,
            road_network,
            self.second_direction,
            self.traffic_sign_priority.get_priority(),
        )
        rob = (priority_k - priority_p - 0.5) / 5
        if rob > 0:
            rob = 1
        else:
            rob = -1
        return rob


class PredHasPriorityRightRight(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityRightRight
    arity = 2
    first_direction = "right"
    second_direction = "right"


class PredHasPriorityRightLeft(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityRightLeft
    arity = 2
    first_direction = "right"
    second_direction = "left"


class PredHasPriorityRightStraight(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityRightStraight
    arity = 2
    first_direction = "right"
    second_direction = "straight"


class PredHasPriorityLeftRight(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityLeftRight
    arity = 2
    first_direction = "left"
    second_direction = "right"


class PredHasPriorityLeftLeft(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityLeftLeft
    arity = 2
    first_direction = "left"
    second_direction = "left"


class PredHasPriorityLeftStraight(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityLeftStraight
    arity = 2
    first_direction = "left"
    second_direction = "straight"


class PredHasPriorityStraightRight(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityStraightRight
    arity = 2
    first_direction = "straight"
    second_direction = "right"


class PredHasPriorityStraightLeft(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityStraightLeft
    arity = 2
    first_direction = "straight"
    second_direction = "left"


class PredHasPriorityStraightStraight(PredHasPriorityBase):
    predicate_name = PriorityPredicates.HasPriorityStraightStraight
    arity = 2
    first_direction = "straight"
    second_direction = "straight"
