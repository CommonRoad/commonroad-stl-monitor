import mtl
from typing import Dict, List
from commonroad.scenario.scenario import Scenario
from vehicle import Vehicle
from commonroad.scenario.trajectory import State

class SimpleMonitor:
    def __init__(self, traffic_rules: Dict[str, str], predicates: Dict[str, List[str]],
                 simulation_param: Dict, ego_vehicle_param: Dict, other_vehicles_param: Dict):
        """
        :param traffic_rules: dictionary with MTl traffic rule formalizations
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        self._rules = self.create_monitors(traffic_rules)
        self._predicates = predicates

    @staticmethod
    def create_monitors(mtl_rules: Dict[str, str]) -> List[str]:
        monitors = []
        for key, value in mtl_rules.items():
            monitor = mtl.parse(value)
            monitors.append(monitor)
        return monitors

    def keeps_speed_limit(self, state: State, scenario: Scenario):
        lanelet_id = scenario.lanelet_network.find_lanelet_by_position([state.position])
        lanlet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0][0])
        if lanlet.speed_limit < state.velocity:
            return False
        else:
            return True

    def evaluate_trajectory(self, scenario: Scenario, trajectory: List[State]):
        """
        :param scenario: CommonRoad scenario
        :param trajectory: ego vehicle trajectory
        """

        data = {'keeps_speed_limit': []}
        for state in trajectory:
            data["keeps_speed_limit"].append((state.time_step, self.keeps_speed_limit(state, scenario)))

        print(self._rules[0](data, quantitative=False))
