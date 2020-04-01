from typing import List, Dict, Set
import warnings

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from src.monitor.monitor_interface_mtl_forward import TrafficRuleMonitorForward
from src.monitor.monitor_interface_mtl_backward import TrafficRuleMonitorBackward
from src.predicates.velocity_predicates import VelocityPredicateCollection
from src.predicates.position_predicates import PositionPredicateCollection
from src.predicates.braking_predicates import BrakingPredicateCollection
from src.predicates.general_predicates import GeneralPredicateCollection
from src.common.vehicle import Vehicle
from src.common.road_network import RoadNetwork


class TrafficRuleDispatcher:
    """
    Manages the different monitors for each traffic rule
    """
    def __init__(self, traffic_rules_forward: Dict[str, str], traffic_rules_backward: Dict[str, str],
                 traffic_rule_sets: Dict[str, str], road_network: RoadNetwork,
                 simulation_param: Dict, traffic_rule_param: Dict, activated_traffic_rule_sets: List[str],
                 vehicle_dependent_rules: List[str]):
        """
        Constructor

        :param traffic_rules_forward: dictionary with MTL formulas of traffic rules for forward MTL framework
        :param traffic_rules_backward: dictionary with MTL formulas of traffic rules for backward MTL framework
        :param traffic_rule_sets: dictionary with sets of related traffic rules
        :param road_network: road network with lanes based on CommonRoad scenario
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rule_param: dictionary with parameters of traffic rule parameters
        :param activated_traffic_rule_sets: set of rules which are activated
        :param vehicle_dependent_rules: set of rules which must be evaluated with respect to several vehicles
        """
        self._dt = simulation_param.get("dt")
        self._simulation_param = simulation_param
        self._road_network = road_network
        self._monitors_forward = self.create_forward_monitors(traffic_rules_forward, traffic_rule_sets,
                                                              activated_traffic_rule_sets,
                                                              vehicle_dependent_rules)
        necessary_predicates = self.extract_necessary_predicates()
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        self._velocity_predicates = VelocityPredicateCollection(road_network, simulation_param,
                                                                traffic_rule_param, necessary_predicates,
                                                                traffic_sign_interpreter)
        self._position_predicates = PositionPredicateCollection(road_network, simulation_param,
                                                                traffic_rule_param, necessary_predicates,
                                                                traffic_sign_interpreter)
        self._braking_predicates = BrakingPredicateCollection(road_network, simulation_param,
                                                              traffic_rule_param, necessary_predicates,
                                                              traffic_sign_interpreter)
        self._general_predicates = GeneralPredicateCollection(road_network, simulation_param,
                                                              traffic_rule_param, necessary_predicates,
                                                              traffic_sign_interpreter)

        self._monitors_backward = self.create_backward_monitors(traffic_rules_backward, traffic_rule_sets,
                                                                activated_traffic_rule_sets,
                                                                vehicle_dependent_rules)

    def extract_necessary_predicates(self) -> Set[str]:
        """
        Extracts necessary predicates from all active rules
        """
        predicates = set()
        for monitor in self._monitors_forward:
            for pred in monitor.predicates:
                predicates.add(pred)
        return predicates

    @staticmethod
    def create_forward_monitors(traffic_rules: Dict[str, str], traffic_rule_sets: Dict[str, str],
                                activated_traffic_rule_sets: List[str], vehicle_dependent_rules: List[str]) \
            -> List[TrafficRuleMonitorForward]:
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
            if traffic_rule_set_id.startswith("B_"):
                continue
            for rule in traffic_rule_sets.get(traffic_rule_set_id):
                vehicle_dependency = rule in vehicle_dependent_rules
                monitors.append(TrafficRuleMonitorForward((rule, traffic_rules.get(rule)), vehicle_dependency))

        return monitors

    def create_backward_monitors(self, traffic_rules: Dict[str, str], traffic_rule_sets: Dict[str, str],
                                 activated_traffic_rule_sets: List[str], vehicle_dependent_rules: List[str]) \
            -> List[TrafficRuleMonitorBackward]:
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
            if traffic_rule_set_id.startswith("F_"):
                continue
            for rule in traffic_rule_sets.get(traffic_rule_set_id):
                vehicle_dependency = rule in vehicle_dependent_rules
                monitors.append(TrafficRuleMonitorBackward((rule, traffic_rules.get(rule)), vehicle_dependency,
                                                           [self._general_predicates, self._position_predicates,
                                                            self._braking_predicates, self._velocity_predicates]))

        return monitors

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Calls different predicate classes for evaluation predicate of all predicates

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
        Evaluates trajectory using forward and backward framework

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary containing predicate evaluation
        """
        results_forward = self.evaluate_trajectory_forward(ego_vehicle, other_vehicles)
        results_backward = self.evaluate_trajectory_backward(ego_vehicle, other_vehicles)
        result = {}
        result.update(results_forward)
        result.update(results_backward)
        return result

    def evaluate_trajectory_forward(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> Dict[str, bool]:
        """
        Evaluates trajectory for traffic rule compliance with forward MTL framework

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns each rule with boolean indicating satisfaction
        """
        evaluated_predicates = self.evaluate_predicates(ego_vehicle, other_vehicles)
        rule_evaluation = {}
        for rule in self._monitors_forward:
            # evaluate rules which only depend on the ego vehicle and the environment
            if rule.vehicle_dependency is False:
                rule_predicates = {}
                for pred in rule.predicates:
                    trace = []
                    for idx, value in enumerate(evaluated_predicates[pred][ego_vehicle.id].values()):
                        trace.append((idx * self._dt, value))
                    rule_predicates[pred] = trace
                rule_evaluation[rule.name] = rule.evaluate_monitor(rule_predicates)
            else:  # evaluate rules which depend on other vehicles, e.g., safe distance
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
                                and evaluated_predicates[pred].get(ego_vehicle.id) is not None:
                            for idx, value in enumerate(evaluated_predicates[pred][ego_vehicle.id].values()):
                                trace.append((idx * self._dt, value))
                        else:
                            warnings.warn("Predicate cannot be found!")
                            break
                        rule_predicates[pred] = trace
                        rule_evaluated = True
                    rule_evaluation[rule.name + "_veh_" + str(vehicle.id)] = rule.evaluate_monitor(rule_predicates)
                if rule_evaluated is False:
                    rule_evaluation[rule.name] = True
        return rule_evaluation

    def _reset_backward_monitors(self):
        for monitor in self._monitors_backward:
            monitor.reset_monitor()

    def evaluate_trajectory_backward(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> Dict[str, bool]:
        """
        Evaluates trajectory for traffic rule compliance with backward MTL framework

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns each rule with boolean indicating satisfaction
        """
        rule_evaluation = {}
        time_steps = list(ego_vehicle.states_lon.keys())
        time_steps.reverse()
        for rule in self._monitors_backward:
            if rule.vehicle_dependency is False:
                self._reset_backward_monitors()
                rule_evaluation[rule.name] = True
                for time_step in time_steps:
                    predicates = {'time_step': time_step, 'ego_vehicle': ego_vehicle, 'other_vehicles': other_vehicles}
                    result = rule.evaluate_monitor(predicates)
                    if result is False:
                        rule_evaluation[rule.name] = False
                        break
            else:  # evaluate rules which depend on other vehicles, e.g., safe distance
                for vehicle in other_vehicles:
                    self._reset_backward_monitors()
                    rule_evaluation[rule.name + "_veh_" + str(vehicle.id)] = True
                    for time_step in time_steps:
                        predicates = {'time_step': time_step, 'ego_vehicle': ego_vehicle,
                                      'other_vehicles': other_vehicles,
                                      'other_vehicle': vehicle}
                        result = rule.evaluate_monitor(predicates)
                        if result is False:
                            rule_evaluation[rule.name + "_veh_" + str(vehicle.id)] = False
                            break
        return rule_evaluation
