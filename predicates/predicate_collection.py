from abc import ABC, abstractmethod
from typing import List, Dict
from commonroad.scenario.lanelet import LaneletNetwork
from common.vehicle import Vehicle


class PredicateCollection(ABC):
    def __init__(self, lanelet_network: LaneletNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, traffic_rules_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        """
        self._lanelet_network = lanelet_network
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._traffic_rule_param = traffic_rules_param
        self._country = simulation_param.get("country")

    @abstractmethod
    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> Dict[str, List[bool]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with trace of bool values for each predicate
        """
        pass
