from typing import List, Dict
from predicates.predicate_collection import PredicateCollection
from commonroad.scenario.lanelet import LaneletNetwork
from common.vehicle import Vehicle


class LanePredicateCollection(PredicateCollection):
    def __init__(self, lanelet_network: LaneletNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        super().__init__(lanelet_network, simulation_param, ego_vehicle_param, other_vehicles_param)

    @staticmethod
    def _behind_other_vehicle(s_ego: float, s_lead: float) -> bool:
        if s_ego < s_lead:
            return True
        else:
            return False

    def _same_lane_as_other_vehicle(self, vehicle_id_ego: int, vehicle_id_other: int, time_step: int) -> bool:
        if

    def evaluate_predicates(self, vehicle: Vehicle) -> Dict[str, List[bool]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param vehicle: vehicle object
        """
        predicate_trace = {"behind_other_vehicle": [],
                           "same_lane_as_other_vehicle": []}
        for idx in range(len(vehicle.state_list_cr)):
            predicate_trace["behind_other_vehicle"].append(
                self._behind_other_vehicle(vehicle.states_lon[idx].s))
            predicate_trace["same_lane_as_other_vehicle"].append(
                self._same_lane_as_other_vehicle(vehicle.states_lon[idx].v, vehicle.lanelet_assignment[idx]))
            #predicate_trace["keeps_safe_distance"][idx] = self.keeps_safe_distance()
            #predicate_trace["brakes_abruptly"][idx] = self.brakes_abruptly()

        return predicate_trace
