from typing import Dict, Set
from commonroad.scenario.lanelet import LaneletNetwork


class LanePredicateCollection():
    def __init__(self, lanelet_network: LaneletNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        self._ego_vehicle_param = ego_vehicle_param

    @staticmethod
    def in_fov(s_other: float, s_ego, fov: float) -> bool:
        if abs(s_other - s_ego) < fov:
            return True
        else:
            return False

    @staticmethod
    def same_lane_behind_other(s_ego: float, s_other: float, lanelet_ids_ego: Set[int],
                               lanelet_ids_other: Set[int]) -> bool:
        if s_ego < s_other:
            for lanelet_id in lanelet_ids_ego:
                if lanelet_id in lanelet_ids_other:
                    return True
            return False
        else:
            return False
