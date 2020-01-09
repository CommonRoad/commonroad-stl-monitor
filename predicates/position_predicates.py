from typing import List, Dict, Set
from predicates.predicate_collection import PredicateCollection
from commonroad.scenario.lanelet import LaneletNetwork, LaneletType
from common.vehicle import Vehicle
from common.road_network import RoadNetwork


class PositionPredicateCollection(PredicateCollection):
    def __init__(self, road_network: RoadNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, traffic_rules_param: Dict):
        """
        :param road_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        """
        super().__init__(road_network, simulation_param, ego_vehicle_param, other_vehicles_param,
                         traffic_rules_param)

    @staticmethod
    def in_fov(s_ego, fov: float, s_other: float) -> bool:
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

    @staticmethod
    def front_vehicle_same_lane(vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[List[Vehicle]]:
        front_vehicles_all_vehicle_lanes = []
        for lanelet in vehicle.lanelet_assignment[time_step]:
            front_vehicles_single_vehicle_lanes = []
            for veh in other_vehicles:
                if lanelet in veh.lanelet_assignment[time_step]:
                    if vehicle.states_lon[time_step].s < veh.states_lon[time_step].s:
                        front_vehicles_single_vehicle_lanes.append(vehicle)
            front_vehicles_all_vehicle_lanes.append(front_vehicles_single_vehicle_lanes)
        return front_vehicles_all_vehicle_lanes

    @staticmethod
    def same_lane(lanelet_ids_ego: Set[int], lanelet_ids_other: Set[int]) -> bool:
        for lanelet_id in lanelet_ids_ego:
            if lanelet_id in lanelet_ids_other:
                return True
        return False

    @staticmethod
    def behind(s_ego: float, s_other: float) -> bool:
        if s_ego < s_other:
            return True
        else:
            return False

    def _on_ramp(self, vehicle: Vehicle, time_step: int) -> bool:
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type or LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                return True
        return False

    def _urban(self, vehicle: Vehicle, time_step: int) -> bool:
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.URBAN in lanelet.lanelet_type:
                return True
        return False

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with trace of bool values for each predicate
        """
        pass