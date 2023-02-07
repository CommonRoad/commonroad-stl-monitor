from enum import Enum
import logging
import numpy as np
from typing import List, Set
from crmonitor.common.world import World
from crmonitor.common import helper, world, vehicle
from crmonitor.predicates.base import BasePredicateEvaluator
from commonroad.scenario.traffic_sign import TrafficLightState
from commonroad.scenario import lanelet, traffic_sign
from commonroad.scenario.traffic_sign import TrafficLight, TrafficSignIDGermany
from crmonitor.common.road_network import Lane, RoadNetwork

logger = logging.getLogger(__name__)


class PriorityPredicates(str, Enum):
    SamePriority = "same_priority"
    RelevantTrafficLight = "relevant_traffic_light"
    HasPriority = "has_priority"
    SamePriorityRightRight = "same_priority_right_right"


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

        lanelets_dir_ids_of_p = helper.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = helper.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = helper.get_priority(
            lanelets_dir_ids_of_p, road_network, "right"
        )  #'102'
        priority_k = helper.get_priority(lanelets_dir_ids_of_k, road_network, "right")

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

        lanelets_dir_ids_of_p = helper.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = helper.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = helper.get_priority(
            lanelets_dir_ids_of_p, road_network, vehicle_dir_p
        )  #'102'
        priority_k = helper.get_priority(
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

        vehicle = world.vehicle_by_id(vehicle_ids[0])  # vehicle: x_ego
        distance_from_nearest_tl = -1

        lanelets_dir_ids = helper.lanelets_dir(vehicle, time_step, world.road_network)

        # for l in lanelets_dir_ids:

        lanelet_network = world.road_network.lanelet_network
        for l_id in lanelets_dir_ids:
            lanelet = lanelet_network.find_lanelet_by_id(l_id)
            successors_paths = lanelet.find_lanelet_successors_in_range(
                world.road_network.lanelet_network, max_length=150
            )
            for successors_path in successors_paths:
                # find lanelet successors in range excludes the current lanelet, so we add it again
                successors_path.insert(0, l_id)
                for successor_id in successors_path:
                    successor = lanelet_network.find_lanelet_by_id(successor_id)

                    traffic_lights = successor.traffic_lights
                    for tl_id in traffic_lights:

                        tl = lanelet_network.find_traffic_light_by_id(tl_id)

                        if tl.active:
                            stop_line = successor.stop_line
                            distance_to_ego = helper.distance_vehicle_to_stop_line(
                                vehicle, stop_line, time_step
                            )
                            if (
                                distance_to_ego < distance_from_nearest_tl
                                or distance_from_nearest_tl == -1
                            ):
                                distance_from_nearest_tl = distance_to_ego
        return distance_from_nearest_tl


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

        lanelets_dir_ids_of_p = helper.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = helper.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = helper.get_priority(
            lanelets_dir_ids_of_p, road_network, "right"
        )  #'102'
        priority_k = helper.get_priority(lanelets_dir_ids_of_k, road_network, "right")

        if priority_k <= priority_p:
            rob = -1
        else:
            rob = 1

        return rob


class PredSamePriorityRightRight(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    vehicle_dir_k = "right"
    vehicle_dir_p = "right"

    predicate_name = PriorityPredicates.SamePriorityRightRight
    arity = 4

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = helper.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = helper.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

        priority_p = helper.get_priority(
            lanelets_dir_ids_of_p, road_network, self.vehicle_dir_p
        )  #'102'
        priority_k = helper.get_priority(
            lanelets_dir_ids_of_k, road_network, self.vehicle_dir_k
        )

        if priority_p == priority_k:
            rob = 1
        else:
            rob = -1

        return rob
