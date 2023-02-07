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


class PredSamePriority(BasePredicateEvaluator):
    """
    evaluates if two vehicles have the same priority
    """

    predicate_name = PriorityPredicates.SamePriority
    arity = 4

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelets_dir_ids_of_p = helper.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = helper.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

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

        lanelets_dir_ids_of_p = helper.lanelets_dir(
            vehicle_p, time_step, world.road_network
        )
        lanelets_dir_ids_of_k = helper.lanelets_dir(
            vehicle_k, time_step, world.road_network
        )

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

    sign_id_priority = {
        # sign_id :[prio_left, prio_straight, prio_right, evaluation_index]
        "306": [4, 5, 4, 11],  # TrafficSignIDGermany.PRIORITY
        "301": [4, 5, 4, 12],  # TrafficSignIDGermany.RIGHT_OF_WAY
        "205": [2, 2, 2, 13],  # TrafficSignIDGermany.YIELD
        "206": [1, 1, 1, 14],  # TrafficSignIDGermany.STOP
        "102": [3, 3, 3, 15],  # TrafficSignIDGermany.WARNING_RIGHT_BEFORE_LEFT
    }

    # def arg_min(self, traffic_signs):
    #
    # for sign_id in traffic_signs:
    #     sign_priority = PredHasPriority.sign_id_priority[sign_id]
    #     eval_idx = sign_priority[3]
    #     eval_idx_arr = eval_idx_arr.extend(eval_idx)
    # return eval_idx_arr

    def get_priority(
        self, lanelets_dir_ids: List[int], road_network: RoadNetwork, direction: str
    ):

        direction_index_dic = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}
        direction_index = direction_index_dic[direction.upper()]

        for l_id in lanelets_dir_ids:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(l_id)
            traffic_sign_ids = lanelet.traffic_signs
            # traffic_sign_object = road_network.lanelet_network.find_traffic_sign_by_id(traffic_sign_id)

            traffic_ids = list()
            for ts_id in traffic_sign_ids:
                traffic_sign_object = (
                    road_network.lanelet_network.find_traffic_sign_by_id(ts_id)
                )
                traffic_sign_elements = traffic_sign_object.traffic_sign_elements
                for ts_element in traffic_sign_elements:
                    ts_element_id = ts_element.traffic_sign_element_id
                    traffic_ids.append(ts_element_id.value)

            if len(traffic_ids) == 0:
                traffic_ids.append("102")

            print(f"traffic_ids: {traffic_ids}")

            min_priority = 3  # 3 by default is the priority for '102'
            min_evaluation_index = 15

            for id in traffic_ids:
                if self.sign_id_priority[id][3] < min_evaluation_index:
                    min_evaluation_index = self.sign_id_priority[id][3]
                    min_priority = self.sign_id_priority[id][direction_index]

            print(f"min_priority: {min_priority}")
            return min_priority

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

        priority_p = self.get_priority(
            lanelets_dir_ids_of_p, road_network, "right"
        )  #'102'
        priority_k = self.get_priority(lanelets_dir_ids_of_k, road_network, "right")

        if priority_p <= priority_k:
            rob = -1
        else:
            rob = 1

        return rob

    
    def has_priority_dir(
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

        priority_p = self.get_priority_dir(
            lanelets_dir_ids_of_p, road_network, vehicle_dir_p
        )
        priority_k = self.get_priority_dir(
            lanelets_dir_ids_of_k, road_network, vehicle_dir_k
        )

        if priority_p <= priority_k:
            rob = False
        else:
            rob = True

        return rob
