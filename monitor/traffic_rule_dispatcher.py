from monitor.traffic_rule_monitor import TrafficRuleMonitor
from typing import List, Dict
from commonroad.scenario.scenario import Scenario
from predicates.safety_predicates import SafetyPredicateCollection
from common.vehicle import Vehicle


class TrafficRuleDispatcher:
    """Manages the different monitors for each traffic rule"""

    def __init__(self, traffic_rules: Dict[str, str], traffic_rule_sets: Dict[str, str], scenario: Scenario,
                 simulation_param: Dict, ego_vehicle_param: Dict, other_vehicles_param: Dict, traffic_rule_param: Dict,
                 activated_traffic_rule_sets: List[int], vehicle_dependent_rules: List[str]):
        """
        :param traffic_rules: dictionary with MTL formulas of traffic rules
        :param traffic_rule_sets: dictionary with sets of related traffic rules
        :param scenario: CommonRoad scenario
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rule_param: dictionary with parameters of traffic rule parameters
        """
        self._dt = simulation_param.get("dt")
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._safety_predicates = SafetyPredicateCollection(scenario.lanelet_network, simulation_param,
                                                            ego_vehicle_param, other_vehicles_param, traffic_rule_param)
        self._monitors = self.create_monitors(traffic_rules, traffic_rule_sets, activated_traffic_rule_sets,
                                              vehicle_dependent_rules)

    @staticmethod
    def create_monitors(traffic_rules: Dict[str, str], traffic_rule_sets: Dict[str, str],
                        activated_traffic_rule_sets: List[int], vehicle_dependent_rules: List[str]) \
            -> List[TrafficRuleMonitor]:
        """
        Initialization of monitor for each MTL rule

        :param traffic_rules: dictionary with traffic rules in temporal logic
        :param traffic_rule_sets: dictionary with sets of related traffic rules
        :returns list of monitors
        """
        monitors = []
        for traffic_rule_set_id in activated_traffic_rule_sets:
            for rule in traffic_rule_sets.get(traffic_rule_set_id):
                vehicle_dependency = rule in vehicle_dependent_rules
                monitors.append(TrafficRuleMonitor((rule, traffic_rules.get(rule)), vehicle_dependency))

        return monitors

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Calls different predicate classes for predicate evaluation

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        """
        safety_predicates = self._safety_predicates.evaluate_predicates(ego_vehicle, other_vehicles)

        return safety_predicates

    def evaluate_trajectory(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> Dict[str, bool]:
        """
        Evaluates trajectory for traffic rule compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns each rule with boolean indicating satisfaction
        """
        evaluated_predicates = self.evaluate_predicates(ego_vehicle, other_vehicles)
        rule_evaluation = {}
        for rule in self._monitors:
            if rule.vehicle_dependency is False:
                rule_predicates = {}
                for pred in rule.predicates:
                    trace = []
                    for idx, value in enumerate(evaluated_predicates[pred][ego_vehicle.id].values()):
                        trace.append((idx * self._dt, value))
                    rule_predicates[pred] = trace
                    rule_evaluation[rule.name] = rule.evaluate_monitor(rule_predicates)
            else:
                rule_predicates = {}
                rule_evaluated = False
                for vehicle in other_vehicles:
                    for pred in rule.predicates:
                        trace = []
                        if len(evaluated_predicates[pred]) > 0 \
                                and evaluated_predicates[pred].get(vehicle.id) is not None:
                            for idx, value in enumerate(evaluated_predicates[pred][vehicle.id].values()):
                                trace.append((idx * self._dt, value))
                        elif len(evaluated_predicates[pred]) > 0 \
                                and evaluated_predicates[pred].get(ego_vehicle.id) is not None and \
                                rule_predicates.get(pred) is None:
                            for idx, value in enumerate(evaluated_predicates[pred][ego_vehicle.id].values()):
                                trace.append((idx * self._dt, value))
                        else:
                            break
                        rule_predicates[pred] = trace
                        rule_evaluated = True
                        rule_evaluation[rule.name + "_veh_" + str(vehicle.id)] = rule.evaluate_monitor(rule_predicates)
                if rule_evaluated is False:
                    rule_evaluation[rule.name] = True
        return rule_evaluation
