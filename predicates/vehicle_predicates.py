from typing import List, Dict
from predicates.predicate_collection import PredicateCollection
from commonroad.scenario.lanelet import LaneletNetwork
from common.vehicle import Vehicle
from commonroad.scenario.obstacle import SignalState


class VehiclePredicateCollection(PredicateCollection):
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
    def braking_lights(signal_state: SignalState):
        if signal_state.braking_lights:
            return True
        else:
            return False

    @staticmethod
    def brakes(a: float):
        if a < 0:
            return True
        else:
            return False

    def evaluate_predicates(self, vehicle: Vehicle) -> Dict[str, List[bool]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param vehicle: vehicle object
        """
        predicate_trace = {"braking_lights": []}

        for idx in range(len(vehicle.state_list_cr)):
            predicate_trace["braking_lights"].append(
                self._braking_lights(vehicle.signal_series[idx]))

        return predicate_trace
