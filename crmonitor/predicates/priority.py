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
    AtTrafficSignStop = "at_traffic_sign_stop"
    RelevantTrafficLight = "relevant_traffic_light"


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


# --------------------------------------------------------------------------------------------------------------------#
class PredAtTrafficSignStop(BasePredicateEvaluator):
    predicate_name = PriorityPredicates.AtTrafficSignStop
    arity = 1
    stop_traffic_sign = '206'

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        If the vehicle locates at the lanelet with a stop traffic sign (206), return True, otherwise, return False.
        """
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        lanelets_dir_ids = utils.lanelets_dir(vehicle, time_step, world.road_network)
        # find all traffic sign elements with type stop (206) in lanelets_dir
        for lanelet_id in lanelets_dir_ids:
            traffic_sign_elements = utils.traffic_sign(lanelet_id, self.stop_traffic_sign, road_network)
            if traffic_sign_elements is None:
                continue
            # check if vehicle in this lanelet in lateral horizon
            d_lane = utils.distance_to_lanes(vehicle, [lanelet_id], world, time_step)
            if d_lane < 0:
                continue
            return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        """
        If the vehicle locates at the lanelet with a stop traffic sign (206), return distance to the start of lanelet,
        otherwise, return -1.
        idea to improve: use reference path of vehicle, so that the lanelet with a stop traffic sign in the reference
        path can be found, even though current occupied lanelets have no traffic sign.
        """
        robustness = -np.inf
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        lanelets_dir_ids = utils.lanelets_dir(vehicle, time_step, world.road_network)
        ref_path = utils.ref_path_lanelets(vehicle, world.road_network, time_step)
        reach_suc = np.array([], dtype=int)
        # find successors of lanelets_dir
        for lanelet_id in lanelets_dir_ids:
            test = utils.reach_suc(lanelet_id, road_network)
            reach_suc = np.append(reach_suc, utils.reach_suc(lanelet_id, road_network))
        reach_suc = np.unique(reach_suc)
        # intersection between reference path and successors of lanelets_dir
        lanelets_ids = ref_path.contained_lanelets.intersection(set(reach_suc))
        # find relevant lanelets with stop traffic sign
        lanelet_with_ts_stop = list()
        for lanelet_id in lanelets_ids:
            traffic_sign_elements = utils.traffic_sign(lanelet_id, self.stop_traffic_sign, road_network)
            if traffic_sign_elements is not None:
                lanelet_with_ts_stop.append(lanelet_id)
        if len(lanelet_with_ts_stop) == 0:
            return self._scale_lon_dist(float(robustness))
        # Get the front and rear longitudinal value of the vehicle and lanelets with stop sign
        front_s = vehicle.front_s(time_step, ref_path) or -np.inf
        rear_s = vehicle.rear_s(time_step, ref_path) or -np.inf
        lanelet_start_s = np.array([ref_path.clcs.convert_to_curvilinear_coords(
                *utils.get_lanelet_start_line(world.road_network.lanelet_network.find_lanelet_by_id(l))[0])[0] for l
                                    in lanelet_with_ts_stop])
        lanelet_end_s = np.array([ref_path.clcs.convert_to_curvilinear_coords(
                *utils.get_lanelet_end_line(world.road_network.lanelet_network.find_lanelet_by_id(l))[0])[0] for l
                                  in lanelet_with_ts_stop])
        for i in range(lanelet_start_s.shape[0]):
            # check if vehicle in this lanelet in lateral horizon
            d_lane = utils.distance_to_lanes(vehicle, [lanelet_with_ts_stop[i]], world, time_step)
            if d_lane < 0:
                continue
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


class PredRelevantTrafficLight(BasePredicateEvaluator):
    """
    evaluates if an upcoming intersection is regulated by traffic lights
    """

    predicate_name = PriorityPredicates.RelevantTrafficLight
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        check if there is an active traffic light in lanelet_dir or successors
        """
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        road_network = world.road_network
        reach_suc_id = np.array([], dtype=int)
        lanelet_dir_ids = utils.lanelets_dir(vehicle, time_step, road_network)
        for lanelet_id in lanelet_dir_ids:
            reach_suc_id = np.append(reach_suc_id, utils.reach_suc(lanelet_id, road_network))
        for l_id in np.unique(reach_suc_id):
            lanelet_suc = road_network.lanelet_network.find_lanelet_by_id(l_id)
            if len(lanelet_suc.traffic_lights) == 0:
                continue
            assert len(lanelet_suc.traffic_lights) == 1, "TODO: Only works for one " "traffic light per lanelet!"
            # check if vehicle in this lanelet in lateral horizon
            d_lane = utils.distance_to_lanes(vehicle, [l_id], world, time_step)
            if d_lane < 0:
                continue
            tl = road_network.lanelet_network.find_traffic_light_by_id(list(lanelet_suc.traffic_lights)[0])
            if tl.active:
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        """
        returns the distance to the nearest active traffic light
        """
        reach_suc_id = np.array([], dtype=int)
        lanelet_with_active_tl = list()
        robustness = -np.inf
        road_network = world.road_network
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_dir_ids = utils.lanelets_dir(vehicle, time_step, road_network)
        ref_path = utils.ref_path_lanelets(vehicle, world.road_network, time_step)
        for lanelet_id in lanelet_dir_ids:
            reach_suc_id = np.append(reach_suc_id, utils.reach_suc(lanelet_id, road_network))
        reach_suc_id = np.unique(reach_suc_id)
        # intersection between reference path and successors of lanelets_dir
        lanelets_ids = ref_path.contained_lanelets.intersection(set(reach_suc_id))
        for l_id in lanelets_ids:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(l_id)
            if len(lanelet.traffic_lights) == 0:
                continue
            # check if vehicle in this lanelet in lateral horizon
            d_lane = utils.distance_to_lanes(vehicle, [l_id], world, time_step)
            if d_lane < 0:
                continue
            assert len(lanelet.traffic_lights) == 1, "TODO: Only works for one " "traffic light per lanelet!"
            tl = road_network.lanelet_network.find_traffic_light_by_id(list(lanelet.traffic_lights)[0])
            if tl.active:
                lanelet_with_active_tl.append(l_id)
        if len(lanelet_with_active_tl) == 0:
            return self._scale_lon_dist(float(robustness))
        # Get the front and rear longitudinal value of the vehicle and lanelets with traffic light
        front_s = vehicle.front_s(time_step, ref_path) or -np.inf
        rear_s = vehicle.rear_s(time_step, ref_path) or -np.inf
        lanelet_start_s = np.array([ref_path.clcs.convert_to_curvilinear_coords(
                *utils.get_lanelet_start_line(world.road_network.lanelet_network.find_lanelet_by_id(l))[0])[0] for l
                                    in lanelet_with_active_tl])
        lanelet_end_s = np.array([ref_path.clcs.convert_to_curvilinear_coords(
                *utils.get_lanelet_end_line(world.road_network.lanelet_network.find_lanelet_by_id(l))[0])[0] for l
                                  in lanelet_with_active_tl])
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


class PredSamePriorityRightRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority in right and right turning
    """
    predicate_name = PriorityPredicates.SamePriorityRightRight
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        incoming_k = utils.get_incoming(vehicle_k.lanelets_dir, road_network)
        incoming_k_id = list(incoming_k.incoming_lanelets)[0]
        incoming_p = utils.get_incoming(vehicle_p.lanelets_dir, road_network)
        incoming_p_id = list(incoming_p.incoming_lanelets)[0]
        priority_k = utils.get_priority(incoming_k_id, road_network, 'right')
        priority_p = utils.get_priority(incoming_p_id, road_network, 'right')
        if priority_k != priority_p:
            rob = -abs(priority_k - priority_p)
        else:
            right_turning_lane_k, _ = utils.get_right_turn_lane(road_network, incoming_k)
            right_turning_lane_p, _ = utils.get_right_turn_lane(road_network, incoming_p)
            state_k = vehicle_k.states_cr[time_step]
            state_p = vehicle_p.states_cr[time_step]
            d_center_left_k = right_turning_lane_k.clcs_left.convert_to_curvilinear_coords(*state_k.position)[1]
            d_center_left_p = right_turning_lane_p.clcs_left.convert_to_curvilinear_coords(*state_p.position)[1]
            front_k_s = vehicle_k.front_s(time_step, right_turning_lane_k)
            front_p_s = vehicle_p.front_s(time_step, right_turning_lane_p)
            start_k_s = right_turning_lane_k.clcs_left.convert_to_curvilinear_coords(*utils.get_lanelet_start_line(road_network.lanelet_network.find_lanelet_by_id(incoming_k_id))[0])[0]
            start_p_s = right_turning_lane_p.clcs_left.convert_to_curvilinear_coords(
                *utils.get_lanelet_start_line(road_network.lanelet_network.find_lanelet_by_id(incoming_p_id))[0])[0]
            rob = self._scale_lat_dist(np.min(np.array([front_k_s - start_k_s, front_p_s - start_p_s])))
            rob = 1
        return rob
