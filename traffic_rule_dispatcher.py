from traffic_rule_monitor import TrafficRuleMonitor
from typing import List, Dict
from commonroad.scenario.scenario import Scenario
from safety_predicates import SafetyPredicateCollection
from common.vehicle import Vehicle


class TrafficRuleDispatcher:
    """Manages the different monitors for each traffic rule"""

    def __init__(self, traffic_rules: Dict[str, str], scenario: Scenario, simulation_param: Dict,
                 ego_vehicle_param: Dict, other_vehicles_param: Dict, traffic_rule_param: Dict):
        """
        :param traffic_rules: dictionary with MTL formulas of traffic rules
        :param scenario: CommonRoad scenario
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rule_param: dictionary with parameters of traffic rule parameters
        """
        self._monitors = self.create_monitors(traffic_rules)
        self._dt = simulation_param.get("dt")
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._traffic_rule_param = traffic_rule_param
        self._safety_predicates = SafetyPredicateCollection(scenario.lanelet_network, simulation_param,
                                                            ego_vehicle_param, other_vehicles_param)

    @staticmethod
    def create_monitors(traffic_rules: Dict[str, str]) -> List[TrafficRuleMonitor]:
        """
        Initialization of monitor for each MTL rule

        :param traffic_rules: dictionary with traffic rules in temporal logic
        :returns list of monitors
        """
        monitors = []
        for rule in traffic_rules.items():
            monitors.append(TrafficRuleMonitor(rule))

        return monitors

    def evaluate_predicates(self, vehicle: Vehicle) -> Dict[str, List[bool]]:
        """
        Calls different predicate classes for predicate evaluation

        :param vehicle: vehicle object containing trajectory and other relevant information
        """
        safety_predicates = self._safety_predicates.evaluate_predicates(vehicle)

        return safety_predicates

    def evaluate_trajectory(self, vehicle: Vehicle) -> Dict[str, bool]:
        """
        Evaluates trajectory for traffic rule compliance

        :param vehicle: vehicle object containing trajectory and other relevant information
        :returns each rule with boolean indicating satisfaction
        """
        evaluated_predicates = self.evaluate_predicates(vehicle)
        rule_evaluation = {}
        for rule in self._monitors:
            rule_predicates = {}
            for pred in rule.predicates:
                trace = []
                for idx, value in enumerate(evaluated_predicates[pred]):
                    trace.append((idx * self._dt, value))
                rule_predicates[pred] = trace
                rule_evaluation[rule.name] = rule.evaluate_monitor(rule_predicates)

        return rule_evaluation
