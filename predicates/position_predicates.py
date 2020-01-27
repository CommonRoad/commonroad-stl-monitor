from typing import List, Dict, Set
from predicates.predicate_collection import PredicateCollection
from common.vehicle import Vehicle
from common.road_network import RoadNetwork
from commonroad.scenario.lanelet import LaneletType


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
    def in_fov(s_ego, s_other: float, fov: float) -> bool:
        """
        Evaluates if another vehicle is inside the field of view of the ego vehicle

        :param s_ego: ego vehicle position
        :param s_other: other vehicle's position
        :param fov: field of view of the ego vehicle
        :returns boolean indicating satisfaction
        """
        if abs(s_other - s_ego) < fov:
            return True
        else:
            return False

    @staticmethod
    def same_lane_behind_other(s_ego: float, s_other: float, lanelet_ids_ego: Set[int],
                               lanelet_ids_other: Set[int]) -> bool:
        """
        Evaluates if another vehicle is inside the same lane and behind the ego vehicle

        :param s_ego: ego vehicle position
        :param s_other: other vehicle's position
        :param lanelet_ids_ego: lanelet IDs of the ego vehicle
        :param lanelet_ids_other: lanelet IDs of the other vehicle
        :returns boolean indicating satisfaction
        """
        if s_ego < s_other:
            for lanelet_id in lanelet_ids_ego:
                if lanelet_id in lanelet_ids_other:
                    return True
            return False
        else:
            return False

    @staticmethod
    def in_front_of(s_1: float, s_2: float) -> bool:
        """
        Evaluates if vehicle two is in front of vehicle one

        :param s_1: longitudinal position of vehicle one
        :param s_2: longitudinal position of vehicle two
        :returns boolean indicating satisfaction
        """
        if s_1 < s_2:
            return True
        else:
            return False

    @staticmethod
    def same_lane(lanelet_ids_ego: Set[int], lanelet_ids_other: Set[int]) -> bool:
        """
        Evaluates if another vehicle is within the same lane as the ego vehicle

        :param lanelet_ids_ego: lanelet IDs of lanelets the ego vehicle is on
        :param lanelet_ids_other: lanelet IDs of lanelets the other vehicle is on
        :returns boolean indicating satisfaction
        """
        for lanelet_id in lanelet_ids_ego:
            if lanelet_id in lanelet_ids_other:
                return True
        return False

    @staticmethod
    def behind(s_ego: float, s_other: float) -> bool:
        """
        Evaluates if another vehicle is behind the ego vehicle

        :param s_ego: longitudinal position of the ego vehicle
        :param s_other: longitudinal position of the other vehicle
        :returns boolean indicating satisfaction
        """
        if s_ego < s_other:
            return True
        else:
            return False

    def _on_access_ramp(self, vehicle: Vehicle, time_step: int) -> bool:
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
                return True
        return False

    def _on_exit_ramp(self, vehicle: Vehicle, time_step: int) -> bool:
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                return True
        return False

    def _on_shoulder(self, vehicle: Vehicle, time_step: int) -> bool:
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.SHOULDER in lanelet.lanelet_type:
                return True
        return False

    def _on_main_carriage_way(self, vehicle: Vehicle, time_step: int) -> bool:
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type:
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
        predicate_trace = {"same_lane_as_ego_vehicle": {},
                           "in_front_of_ego_vehicle": {}}

        for other_vehicle in other_vehicles:
            predicate_trace["same_lane_as_ego_vehicle"][other_vehicle.id] = {}
            predicate_trace["in_front_of_ego_vehicle"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    continue
                predicate_trace["same_lane_as_ego_vehicle"][other_vehicle.id][time_step] = \
                    self.same_lane(ego_vehicle.lanelet_assignment[time_step],
                                   other_vehicle.lanelet_assignment[time_step])
                predicate_trace["in_front_of_ego_vehicle"][other_vehicle.id][time_step] = \
                    self.in_front_of(ego_vehicle.states_lon[time_step].s,
                                     other_vehicle.states_lon[time_step].s)
        return predicate_trace
