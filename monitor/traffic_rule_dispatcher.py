from monitor.traffic_rule_monitor import TrafficRuleMonitor
from typing import List, Dict
from predicates.velocity_predicates import VelocityPredicateCollection
from predicates.position_predicates import PositionPredicateCollection
from predicates.braking_predicates import BrakingPredicateCollection
from predicates.general_predicates import GeneralPredicateCollection
from common.vehicle import Vehicle
from common.road_network import RoadNetwork


class TrafficRuleDispatcher:
    """
    Manages the different monitors for each traffic rule
    """
    def __init__(self, traffic_rules: Dict[str, str], traffic_rule_sets: Dict[int, str], road_network: RoadNetwork,
                 simulation_param: Dict, ego_vehicle_param: Dict, other_vehicles_param: Dict, traffic_rule_param: Dict,
                 activated_traffic_rule_sets: List[str], vehicle_dependent_rules: List[str]):
        """
        :param traffic_rules: dictionary with MTL formulas of traffic rules
        :param traffic_rule_sets: dictionary with sets of related traffic rules
        :param road_network: road network with lanes based on CommonRoad scenario
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rule_param: dictionary with parameters of traffic rule parameters
        :param activated_traffic_rule_sets: set of rules which are activated
        :param vehicle_dependent_rules: set of rules which must be evaluated with respect to several vehicles
        """
        self._dt = simulation_param.get("dt")
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._road_network = road_network
        self._velocity_predicates = VelocityPredicateCollection(road_network, simulation_param,
                                                                ego_vehicle_param, other_vehicles_param,
                                                                traffic_rule_param)
        self._position_predicates = PositionPredicateCollection(road_network, simulation_param,
                                                                ego_vehicle_param, other_vehicles_param,
                                                                traffic_rule_param)
        self._braking_predicates = BrakingPredicateCollection(road_network, simulation_param,
                                                              ego_vehicle_param, other_vehicles_param,
                                                              traffic_rule_param)
        self._general_predicates = GeneralPredicateCollection(road_network, simulation_param,
                                                              ego_vehicle_param, other_vehicles_param,
                                                              traffic_rule_param)
        self._monitors = self.create_monitors(traffic_rules, traffic_rule_sets, activated_traffic_rule_sets,
                                              vehicle_dependent_rules)

    @staticmethod
    def create_monitors(traffic_rules: Dict[str, str], traffic_rule_sets: Dict[int, str],
                        activated_traffic_rule_sets: List[str], vehicle_dependent_rules: List[str]) \
            -> List[TrafficRuleMonitor]:
        """
        Initialization of monitor for each MTL rule

        :param traffic_rules: dictionary with traffic rules in temporal logic
        :param traffic_rule_sets: dictionary with sets of related traffic rules
        :param activated_traffic_rule_sets: set of rules which are activated
        :param vehicle_dependent_rules: set of rules which must be evaluated with respect to several vehicles
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
        :returns dictionary containing predicate evaluation
        """
        velocity_predicates = self._velocity_predicates.evaluate_predicates(ego_vehicle, other_vehicles)
        position_predicates = self._position_predicates.evaluate_predicates(ego_vehicle, other_vehicles)
        braking_predicates = self._braking_predicates.evaluate_predicates(ego_vehicle, other_vehicles)
        general_predicates = self._general_predicates.evaluate_predicates(ego_vehicle, other_vehicles)

        combined_predicates = {**velocity_predicates, **position_predicates, **braking_predicates, **general_predicates}
        return combined_predicates

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
