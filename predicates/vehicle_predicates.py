from typing import List, Dict
from predicates.predicate_collection import PredicateCollection
from common.road_network import RoadNetwork
from common.vehicle import Vehicle
from commonroad.scenario.obstacle import SignalState


class VehiclePredicateCollection(PredicateCollection):
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
    def _braking_lights(signal_state: SignalState):
        """
        Evaluates if braking lights are active

        :param signal_state: CommonRoad signal state
        :returns boolean indicating satisfaction
        """
        if signal_state.braking_lights:
            return True
        else:
            return False

    @staticmethod
    def brakes(a: float):
        """
        Evaluates if vehicle brakes

        :param a: acceleration of vehicle
        :returns boolean indicating satisfaction
        """
        if a < 0:
            return True
        else:
            return False

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> Dict[str, List[bool]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        """
        predicate_trace = {"braking_lights": []}

        for idx in range(len(ego_vehicle.state_list_cr)):
            predicate_trace["braking_lights"].append(
                self._braking_lights(ego_vehicle.signal_series[idx]))

        return predicate_trace
