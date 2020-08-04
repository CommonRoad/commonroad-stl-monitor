from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from src.common.road_network import RoadNetwork
from src.common.vehicle import Vehicle
from src.common.helper import OperatingMode


class PredicateCollection(ABC):
    def __init__(self, road_network: RoadNetwork, simulation_param: Dict, traffic_rules_param: Dict,
                 necessary_predicates: Set[str], traffic_sign_interpreter: TrafficSigInterpreter):
        """
        Constructor

        :param road_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        :param traffic_sign_interpreter: CommonRoad traffic sign interpreter
        """
        self._road_network = road_network
        self._simulation_param = simulation_param
        self._traffic_rules_param = traffic_rules_param
        self._country = simulation_param.get("country")
        self._necessary_predicates = necessary_predicates
        self._traffic_sign_interpreter = traffic_sign_interpreter

    @abstractmethod
    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int], operating_mode: OperatingMode) -> Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with traces of bool values for each predicate
        """
        pass

    @abstractmethod
    def evaluate_constraints(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                             time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts constraints for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with traces of constraints for each predicate
        """
        pass

    @abstractmethod
    def evaluate_robustness(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts robustness values for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with traces of robustness values for each predicate
        """
        pass
