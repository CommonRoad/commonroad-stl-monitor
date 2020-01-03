from abc import ABC, abstractmethod
from commonroad.scenario.trajectory import State
from typing import List, Dict
from commonroad.scenario.lanelet import LaneletNetwork


class PredicateCollection(ABC):
    def __init__(self, lanelet_network: LaneletNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        self._lanelet_network = lanelet_network
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._country = simulation_param.get("country")

    @staticmethod
    def convert_to_curvilinear(state_cr: State, curvilinear_coord_system):
        s, d = curvilinear_coord_system.convert_to_curvilinear_coords(state_cr.position[0], state_cr.position[1])
        return [s, d]

    @abstractmethod
    def evaluate_predicates(self, trajectory: List[State]) -> Dict[str, List[bool]]:
        pass
