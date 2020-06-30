import unittest
import os

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from src.predicates.braking_predicates import BrakingPredicateCollection
from src.common.helper import *
from src.common.road_network import RoadNetwork


class TestBrakingPredicates(unittest.TestCase):
    def setUp(self):
        config_path = os.path.dirname(os.path.abspath(__file__)) + "/../src/"
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 0.1, 'DEU')
        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._ego_vehicle_param = create_other_vehicles_param(config.get("ego_vehicle_param"))
        self._traffic_rule_param = traffic_rules.get("traffic_rules_param")
        self._road_network_param = config.get("road_network_param")

    def test_keeps_safe_distance(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = False
        exp_sol_constraint_mode_1 = 49.142857142857146
        exp_sol_constraint_mode_2 = 48.28485714285713
        exp_sol_robustness_mode_1 = 4.8491428571428585
        exp_sol_robustness_mode_2 = -0.2428571428571331

        # initialization
        scenario, planning_problem_set = CommonRoadFileReader(os.path.dirname(os.path.abspath(__file__))
                                                              + "/../scenarios/test/DEU_test_safe_distance.xml"). \
            open(lanelet_assignment=True)
        ego_id = 1000
        road_network = RoadNetwork(scenario.lanelet_network, self._road_network_param)
        necessary_predicates = {"keeps_safe_distance_prec__x_ego__x_o"}

        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        braking_predicates = BrakingPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        necessary_predicates, traffic_sign_interpreter)
        ego_vehicle, other_vehicles = create_scenario_vehicles(self._simulation_param.get("dt"),
                                                               scenario.obstacle_by_id(ego_id),
                                                               self._ego_vehicle_param, self._other_vehicles_param,
                                                               road_network, scenario.dynamic_obstacles)
        # Monitor-Mode
        sol_monitor_mode_1 = braking_predicates.keeps_safe_distance_prec(0, ego_vehicle, other_vehicles[0],
                                                                         OperatingMode.MONITOR)
        sol_monitor_mode_2 = braking_predicates.keeps_safe_distance_prec(6, ego_vehicle, other_vehicles[0],
                                                                         OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)

        # Constraint-Mode
        sol_constraint_mode_1 = braking_predicates.keeps_safe_distance_prec(0, ego_vehicle, other_vehicles[0],
                                                                            OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = braking_predicates.keeps_safe_distance_prec(6, ego_vehicle, other_vehicles[0],
                                                                            OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)

        # Robustness-Mode
        sol_robustness_mode_1 = braking_predicates.keeps_safe_distance_prec(0, ego_vehicle, other_vehicles[0],
                                                                            OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = braking_predicates.keeps_safe_distance_prec(6, ego_vehicle, other_vehicles[0],
                                                                            OperatingMode.ROBUSTNESS)

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)

    def test_safe_distance(self):
        self.assertRaises(AssertionError, BrakingPredicateCollection.safe_distance, 0, 0, -1, 0, 1, 0)
        self.assertRaises(AssertionError, BrakingPredicateCollection.safe_distance, 0, 0, 0, -1, 1, 0)
        self.assertRaises(AssertionError, BrakingPredicateCollection.safe_distance, 0, 0, -1, -1, 0, 0)

        exp_sol = 0
        solution = BrakingPredicateCollection.safe_distance(0, 0, -10, -10, 5, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = 0
        solution = BrakingPredicateCollection.safe_distance(0, 0, -10, -10, 5, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = 0
        solution = BrakingPredicateCollection.safe_distance(5, 5, -10, -10, 5, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = 0
        solution = BrakingPredicateCollection.safe_distance(5, 5, -10, -10, 5, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = 50.0
        solution = BrakingPredicateCollection.safe_distance(5, 5, -10, -10, 10, 10)
        self.assertEqual(exp_sol, solution)

        exp_sol = 5.0
        solution = BrakingPredicateCollection.safe_distance(10, 0, -10, -10, 10, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = -5.0
        solution = BrakingPredicateCollection.safe_distance(0, 10, -10, -10, 10, 0)
        self.assertEqual(exp_sol, solution)
