from enum import Enum
import logging
import numpy as np
from typing import List, Set
from crmonitor.common.world import World
from crmonitor.common import helper, world, vehicle
from crmonitor.predicates.base import BasePredicateEvaluator
from commonroad.scenario.traffic_sign import TrafficLightState
from commonroad.scenario import lanelet, traffic_sign
from commonroad.scenario.traffic_sign import TrafficLight
from crmonitor.common.road_network import Lane, RoadNetwork

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

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = vehicle_p.lanelets_dir(time_step, world.road_network)
        lanelets_dir_ids_of_k = vehicle_k.lanelets_dir(time_step, world.road_network)

        priority_p = PredHasPriority.get_priority(lanelets_dir_ids_of_p)
        priority_k = PredHasPriority.get_priority(lanelets_dir_ids_of_k)

        if priority_p == priority_k:
            rob = True
        else:
            rob = False

        return rob

    # TODO
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        return self._scale_lat_dist(100)

    def same_priority_dir(
        self,
        world: World,
        time_step,
        vehicle_ids: List[int],
        vehicle_dir_p: str,
        vehicle_dir_k: str,
    ) -> bool:

        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = vehicle_p.lanelets_dir(time_step)
        lanelets_dir_ids_of_k = vehicle_k.lanelets_dir(time_step)

        priority_p = self.get_priority_dir(lanelets_dir_ids_of_p, vehicle_dir_p)
        priority_k = self.get_priority_dir(lanelets_dir_ids_of_k, vehicle_dir_k)

        if priority_p == priority_k:
            rob = True
        else:
            rob = False

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

        lanelets_dir_ids = vehicle.lanelets_dir(time_step, world.road_network)

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

    sign_id_priority = {
        306: [4, 5, 4, 11],
        301: [4, 5, 4, 12],
        205: [2, 2, 2, 13],
        206: [1, 1, 1, 14],
        102: [3, 3, 3, 15],
    }

    def get_priority(self, lanelets_dir_ids, road_network):

        for l_id in lanelets_dir_ids:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(l_id)
            traffic_signs = lanelet.traffic_signs

            if traffic_signs != {None}:
                traffic_signs = traffic_signs
            else:
                traffic_signs = {102}

            for sign_id in traffic_signs:
                sign_element = lanelet.
                sign_priority = PredHasPriority.sign_id_priority[sign_element]
                eval_idx = sign_priority[3]
                eval_idx_arr = []
                eval_idx_arr = +[eval_idx]
                value = eval_idx_arr[(np.argmin(eval_idx_arr))]

                list_of_keys = [key for key, list_of_values in self.sign_id_priority.items()
                                if value in list_of_values][0]

                priority_all = self.sign_id_priority[list_of_keys]

                priority_left = priority_all[0]
                priority_straight = priority_all[1]
                priority_right = priority_all[2]

                priority_veh = priority_straight
            # orient = vehicle.Vehicle.compute_lanelet_relative_orientation(lanelets_dir_ids)
            #
            # if orient >= np.deg2rad(45):
            #     priority_veh = priority_left
            # elif orient <= np.deg2rad(-45):
            #     priority_veh = priority_right
            # else:
            #     priority_veh = priority_straight

                return priority_veh

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = vehicle_p.lanelets_dir(time_step, world.road_network)
        lanelets_dir_ids_of_k = vehicle_k.lanelets_dir(time_step, world.road_network)

        priority_p = self.get_priority(lanelets_dir_ids_of_p, road_network)
        priority_k = self.get_priority(lanelets_dir_ids_of_k, road_network)

        if priority_p <= priority_k:
            rob = False
        else:
            rob = True

        return rob

    def get_priority_dir(lanelets_dir_ids, vehicle_dir: str) -> int:

        for l_id in lanelets_dir_ids:
            l_sign_id = lanelet.traffic_sign_id(l_id)

            if len(l_sign_id):
                l_sign_id = l_sign_id
            else:
                l_sign_id = {102}

            for sign_id in l_sign_id:
                sign_priority = PredHasPriority.sign_id_priority[sign_id]
                eval_idx = sign_priority[3]
                eval_idx_arr = []
                eval_idx_arr = +[eval_idx]

        value = eval_idx_arr[(np.argmin(eval_idx_arr))]

        list_of_keys = [
            key
            for key, list_of_values in PredHasPriority.sign_id_priority.items()
            if value in list_of_values
        ][0]
        priority_all = PredHasPriority.sign_id_priority[list_of_keys]

        priority_left = priority_all[0]
        priority_straight = priority_all[1]
        priority_right = priority_all[2]

        if vehicle_dir == "left":
            priority_veh = priority_left
        elif vehicle_dir == "right":
            priority_veh = priority_right
        else:
            priority_veh = priority_straight

        return priority_veh

    def has_priority_dir(
        self,
        world: World,
        time_step,
        vehicle_ids: List[int],
        vehicle_dir_p: str,
        vehicle_dir_k: str,
    ) -> bool:

        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = vehicle_p.lanelets_dir(time_step)
        lanelets_dir_ids_of_k = vehicle_k.lanelets_dir(time_step)

        priority_p = self.get_priority_dir(lanelets_dir_ids_of_p, vehicle_dir_p)
        priority_k = self.get_priority_dir(lanelets_dir_ids_of_k, vehicle_dir_k)

        if priority_p <= priority_k:
            rob = False
        else:
            rob = True

        return rob

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        return self._scale_lat_dist(100)
