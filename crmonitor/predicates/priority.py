from enum import Enum
import logging
import numpy as np
from typing import List, Set
from crmonitor.common.world import World
from crmonitor.predicates.base import BasePredicateEvaluator
from commonroad.scenario.traffic_sign import TrafficLightState
from commonroad.scenario import lanelet
from commonroad.scenario.traffic_sign import TrafficLight

logger = logging.getLogger(__name__)


class PriorityPredicates(str, Enum):
    SamePriority = "same_priority"
    RelevantTrafficLight = "relevant_traffic_light"
    HasPriority = "has_priority"


class PredSamePriority(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """
    predicate_name = PriorityPredicates.SamePriority
    arity = 4

    # TODO
    # def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

    # TODO
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        return self._scale_lat_dist(100)


class PredRelevantTrafficLight(BasePredicateEvaluator):
    """
    evaluates if an upcoming intersection is regulated by traffic lights
    """
    predicate_name = PriorityPredicates.RelevantTrafficLight
    arity = 1

    # TODO
    # def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

    # TODO

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        """
        returns the distance to the nearest active traffic light
        """

        vehicle = world.vehicle_by_id(vehicle_ids[0])  # vehicle: x_ego
        vehicle_position = vehicle.get_lon_state(time_step)[0];  
        distance_from_nearest_tl = -1
        # lanelet_ids_occ = lanelets(x_ego) #TODO lanelets_dir(x_ego)

        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        lanelet_network = world.road_network.lanelet_network
        for l_id in lanelet_ids_occ:
            lanelet = lanelet_network.find_lanelet_by_id(l_id)
            successors_paths = lanelet.find_lanelet_successors_in_range(
                world.road_network, max_length=150)
            for successors_path in successors_paths:
                for successor_id in successors_path:
                    # TODO: how to access traffic_lights of a lanelet
                    successor = lanelet_network.find_lanelet_by_id(successor_id)
                    traffic_lights = successor.traffic_lights
                    for tl_id in traffic_lights:
                        tl = lanelet_network.find_traffic_light_by_id(tl_id)
                        if tl.active:
                            distance_to_ego = np.sqrt((tl.position[0]-vehicle_position[0])**2 + (tl.position[1]-vehicle_position[1])**2)
                            if(distance_to_ego <distance_from_nearest_tl or distance_from_nearest_tl==-1):
                                distance_from_nearest_tl = distance_to_ego;                  
        return distance_from_nearest_tl


class PredHasPriority(BasePredicateEvaluator):
    """
    evaluates if the first vehicle has priority over the second vehicle.
    """
    predicate_name = PriorityPredicates.HasPriority
    arity = 4

    # TODO
    # def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

    # TODO
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        return self._scale_lat_dist(100)
