import logging
from enum import Enum
from typing import List

import numpy as np
import math

from crmonitor.common.world import World
from crmonitor.predicates import utils
from crmonitor.predicates.base import BasePredicateEvaluator

logger = logging.getLogger(__name__)


class PriorityPredicates(str, Enum):
    SamePriority = "same_priority"
    RelevantTrafficLight = "relevant_traffic_light"
    HasPriority = "has_priority"
    SamePriorityRightRight = "same_priority_right_right"
    SamePriorityRightLeft = "same_priority_right_left"
    SamePriorityLeftRight = "same_priority_left_right"
    SamePriorityRightStraight = "same_priority_right_straight"
    SamePriorityStraightRight = "same_priority_straight_right"
    SamePriorityLeftStraight = "same_priority_left_straight"
    SamePriorityStraightLeft = "same_priority_straight_left"
    SamePriorityLeftLeft = "same_priority_left_left"
    SamePriorityStraightStraight = "same_priority_straight_straight"
    HasPriorityRightRight = "has_priority_right_right"
    HasPriorityRightLeft = "has_priority_right_left"
    HasPriorityLeftRight = "has_priority_left_right"
    HasPriorityRightStraight = "has_priority_right_straight"
    HasPriorityStraightRight = "has_priority_straight_right"
    HasPriorityLeftStraight = "has_priority_left_straight"
    HasPriorityStraightLeft = "has_priority_straight_left"
    HasPriorityLeftLeft = "has_priority_left_left"
    HasPriorityStraightStraight = "has_priority_straight_straight"
    AtTrafficSign = "at_traffic_sign"


class PredSamePriority(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    predicate_name = PriorityPredicates.SamePriority
    arity = 4

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, "right"
        )  #'102'
        priority_k = utils.get_priority(lanelets_dir_ids_of_k, road_network, "right")

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob

    def same_priority_dir_robustness(
        self,
        world: World,
        time_step,
        vehicle_ids: List[int],
        vehicle_dir_p: str,
        vehicle_dir_k: str,
    ) -> bool:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, vehicle_dir_k
        )

        if priority_p <= priority_k:
            rob = -1
        else:
            rob = 1

        return rob


class PredRelevantTrafficLight(BasePredicateEvaluator):
    """
    evaluates if an upcoming intersection is regulated by traffic lights
    """

    predicate_name = PriorityPredicates.RelevantTrafficLight
    arity = 1
    lanelet_type = utils.HelperLaneletTypes.RELEVANT_TRAFFIC_LIGHT

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        check if there is an active traffic light in lanelet_dir or successors
        """
        boolean_eval = False
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        lanelet_dir_ids = utils.lanelets_dir(vehicle, time_step, road_network)
        for lanelet_id in lanelet_dir_ids:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if utils.is_lanelet_of_type(lanelet, self.lanelet_type, world.road_network):
                boolean_eval = True
                return boolean_eval
            reach_suc_id = utils.reach_succ(lanelet, road_network.lanelet_network)
            for l_id in np.unique(reach_suc_id):
                lanelet_suc = road_network.lanelet_network.find_lanelet_by_id(l_id)
                if utils.is_lanelet_of_type(lanelet_suc, self.lanelet_type, world.road_network):
                    boolean_eval = True
                    return boolean_eval
        return boolean_eval

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        """
        returns the distance to the nearest active traffic light
        """

        # current implemented idea: return distance to tl position
        #   robustness = distance(ego_vehicle, tl )

        # TODO:
        # project distance from vehicle to stop line along vehicle path.

        # vehicle = world.vehicle_by_id(vehicle_ids[0])  # vehicle: x_ego
        # distance_from_nearest_tl = -1

        # lanelets_dir_ids = utils.lanelets_dir(vehicle, time_step, world.road_network)

        # # for l in lanelets_dir_ids:

        # lanelet_network = world.road_network.lanelet_network
        # for l_id in lanelets_dir_ids:
        #     lanelet = lanelet_network.find_lanelet_by_id(l_id)
        #     successors_paths = lanelet.find_lanelet_successors_in_range(
        #         world.road_network.lanelet_network, max_length=150
        #     )
        #     for successors_path in successors_paths:
        #         # find lanelet successors in range excludes the current lanelet, so we add it again
        #         successors_path.insert(0, l_id)
        #         for successor_id in successors_path:
        #             successor = lanelet_network.find_lanelet_by_id(successor_id)

        #             traffic_lights = successor.traffic_lights
        #             for tl_id in traffic_lights:

        #                 tl = lanelet_network.find_traffic_light_by_id(tl_id)

        #                 if tl.active:
        #                     stop_line = successor.stop_line
        #                     distance_to_ego = utils.distance_vehicle_to_stop_line(
        #                         vehicle, stop_line, time_step
        #                     )
        #                     if (
        #                         distance_to_ego < distance_from_nearest_tl
        #                         or distance_from_nearest_tl == -1
        #                     ):
        #                         distance_from_nearest_tl = distance_to_ego
        # return self._scale_lon_dist(distance_from_nearest_tl)

        return self._scale_lon_dist(
            utils.get_robustness_wrt_lanelet_type(
                world, time_step, vehicle_ids, self.lanelet_type, False, True
            )
        )


class PredHasPriority(BasePredicateEvaluator):
    """
    evaluates if the first vehicle has priority over the second vehicle.
    """

    predicate_name = PriorityPredicates.HasPriority
    arity = 4

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, "right"
        )  #'102'
        priority_k = utils.get_priority(lanelets_dir_ids_of_k, road_network, "right")

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityRightRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "right"
    vehicle_dir_p = "right"

    predicate_name = PriorityPredicates.SamePriorityRightRight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityRightLeft(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "right"
    vehicle_dir_p = "left"

    predicate_name = PriorityPredicates.SamePriorityRightLeft
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityLeftRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "left"
    vehicle_dir_p = "right"

    predicate_name = PriorityPredicates.SamePriorityLeftRight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityRightStraight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "right"
    vehicle_dir_p = "straight"

    predicate_name = PriorityPredicates.SamePriorityRightStraight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityStraightRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "straight"
    vehicle_dir_p = "right"

    predicate_name = PriorityPredicates.SamePriorityStraightRight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityLeftStraight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "left"
    vehicle_dir_p = "straight"

    predicate_name = PriorityPredicates.SamePriorityLeftStraight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityStraightLeft(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "straight"
    vehicle_dir_p = "left"

    predicate_name = PriorityPredicates.SamePriorityStraightLeft
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityLeftLeft(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "left"
    vehicle_dir_p = "left"

    predicate_name = PriorityPredicates.SamePriorityLeftLeft
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredSamePriorityStraightStraight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "straight"
    vehicle_dir_p = "straight"

    predicate_name = PriorityPredicates.SamePriorityStraightStraight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityRightRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "right"
    vehicle_dir_p = "right"

    predicate_name = PriorityPredicates.HasPriorityRightRight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityRightLeft(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "right"
    vehicle_dir_p = "left"

    predicate_name = PriorityPredicates.HasPriorityRightLeft
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityLeftRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "left"
    vehicle_dir_p = "right"

    predicate_name = PriorityPredicates.HasPriorityLeftRight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityRightStraight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "right"
    vehicle_dir_p = "straight"

    predicate_name = PriorityPredicates.HasPriorityRightStraight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityStraightRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "straight"
    vehicle_dir_p = "right"

    predicate_name = PriorityPredicates.HasPriorityStraightRight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityLeftStraight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "left"
    vehicle_dir_p = "straight"

    predicate_name = PriorityPredicates.HasPriorityLeftStraight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityStraightLeft(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "straight"
    vehicle_dir_p = "left"

    predicate_name = PriorityPredicates.HasPriorityStraightLeft
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityLeftLeft(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "left"
    vehicle_dir_p = "left"

    predicate_name = PriorityPredicates.HasPriorityLeftLeft
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob


class PredHasPriorityStraightStraight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "straight"
    vehicle_dir_p = "straight"

    predicate_name = PriorityPredicates.HasPriorityStraightStraight
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = utils.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = utils.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = utils.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  # '102'
        priority_k = utils.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_k > priority_p:
            rob = 1
        else:
            rob = -1

        return rob

class PredAtTrafficSign(BasePredicateEvaluator):
    predicate_name = PriorityPredicates.AtTrafficSign
    arity = 1
    stop_traffic_sign = '206'

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        If the vehicle locates at the lanelet with a stop traffic sign (206), return True, otherwise, return False.
        """
        traffic_sign_elements = list()
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        lanelets_dir_ids = utils.lanelets_dir(vehicle, time_step, road_network)
        for lanelet_id in lanelets_dir_ids:
            traffic_sign_elements = utils.traffic_sign(lanelet_id, self.stop_traffic_sign, road_network)
        if len(traffic_sign_elements) == 0:
            boolean_eval = False
        else:
            boolean_eval = True
        return boolean_eval


    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        """
        If the vehicle locates at the lanelet with a stop traffic sign (206), return distance to the start of lanelet,
        otherwise, return -1.
        idea to improve: use reference path of vehicle, so that the lanelet with a stop traffic sign in the reference
        path can be found, even though current occupied lanelets have no traffic sign.
        """
        d_start_lanelet = list()
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        lanelets_dir_ids = utils.lanelets_dir(vehicle, time_step, road_network)
        for lanelet_id in lanelets_dir_ids:
            traffic_sign_elements = (utils.traffic_sign(lanelet_id, self.stop_traffic_sign, road_network))
            if len(traffic_sign_elements) != 0:
                d_start_lanelet.append(utils.distance_start_lanelet(vehicle, lanelet_id, road_network, time_step))
            else:
                d_start_lanelet.append(math.inf * -1)
        return self._scale_lon_dist(np.max(d_start_lanelet))

