import unittest
import os
import numpy as np

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.scenario.lanelet import LaneletNetwork

from crmonitor.predicates.braking_predicates import BrakingPredicateCollection
from crmonitor.common.helper import *
from crmonitor.common.road_network import RoadNetwork


class TestBrakingPredicates(unittest.TestCase):
    def setUp(self):
        config_path = os.path.dirname(os.path.abspath(__file__)) + "/../crmonitor/"
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 1.0, 'DEU')
        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._ego_vehicle_param = create_other_vehicles_param(config.get("ego_vehicle_param"))
        self._traffic_rule_param = traffic_rules.get("traffic_rules_param")
        self._road_network_param = config.get("road_network_param")

    def test_keeps_safe_distance(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = False
        exp_sol_constraint_mode_1 = 6.0
        exp_sol_constraint_mode_2 = 26.0
        exp_sol_robustness_mode_1 = 9.0
        exp_sol_robustness_mode_2 = -21.0

        state_list_lon_ego = {0: StateLongitudinal(s=0, v=20), 1: StateLongitudinal(s=20, v=20)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(acceleration=-1, time_step=0), 1: State(acceleration=0, time_step=1)}
        lanelet_assignments_ego = {0: {1}, 1: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        state_list_lon_other = {0: StateLongitudinal(s=20, v=20), 1: StateLongitudinal(s=30, v=0)}
        state_list_lat_other = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0)}
        cr_state_list_other = {0: State(acceleration=-1, time_step=0), 1: State(acceleration=0, time_step=1)}
        lanelet_assignments_other = {0: {1}, 1: {1}}
        other_vehicle = Vehicle(state_list_lon_other, state_list_lat_other, Rectangle(5, 2), cr_state_list_other, 0,
                                ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_other, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = BrakingPredicateCollection.keeps_safe_distance_prec(0, ego_vehicle, other_vehicle,
                                                                                 OperatingMode.MONITOR)
        sol_monitor_mode_2 = BrakingPredicateCollection.keeps_safe_distance_prec(1, ego_vehicle, other_vehicle,
                                                                                 OperatingMode.MONITOR)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)

        # Constraint-Mode
        sol_constraint_mode_1 = BrakingPredicateCollection.keeps_safe_distance_prec(0, ego_vehicle, other_vehicle,
                                                                                    OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = BrakingPredicateCollection.keeps_safe_distance_prec(1, ego_vehicle, other_vehicle,
                                                                                    OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1.value)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2.value)

        # Robustness-Mode
        sol_robustness_mode_1 = BrakingPredicateCollection.keeps_safe_distance_prec(0, ego_vehicle, other_vehicle,
                                                                                    OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = BrakingPredicateCollection.keeps_safe_distance_prec(1, ego_vehicle, other_vehicle,
                                                                                    OperatingMode.ROBUSTNESS)

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)

    def test_safe_distance(self):
        # invalid parameters a_min_follow, a_min_lead
        self.assertRaises(AssertionError, BrakingPredicateCollection.safe_distance, 0, 0, -1, 0, 0)
        self.assertRaises(AssertionError, BrakingPredicateCollection.safe_distance, 0, 0, 0, -1, 0)

        exp_sol = 0  # both vehicles standing, no reaction time
        solution = BrakingPredicateCollection.safe_distance(0, 0, -10, -10, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = 0  # both vehicles same velocity, no reaction time
        solution = BrakingPredicateCollection.safe_distance(5, 5, -10, -10, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = 50.0   # both vehicles same velocity, with reaction time
        solution = BrakingPredicateCollection.safe_distance(5, 5, -10, -10, 10)
        self.assertEqual(exp_sol, solution)

        exp_sol = 5.0   # following vehicle higher velocity, no reaction time
        solution = BrakingPredicateCollection.safe_distance(10, 0, -10, -10, 0)
        self.assertEqual(exp_sol, solution)

        exp_sol = -5.0  # leading vehicle higher velocity, no reaction time
        solution = BrakingPredicateCollection.safe_distance(0, 10, -10, -10, 0)
        self.assertEqual(exp_sol, solution)

    def test_brakes_stronger(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = False
        exp_sol_monitor_mode_3 = False
        exp_sol_monitor_mode_4 = False
        exp_sol_constraint_mode_1 = -1
        exp_sol_constraint_mode_2 = -2
        exp_sol_constraint_mode_3 = 0
        exp_sol_constraint_mode_4 = 0
        exp_sol_robustness_mode_1 = 1
        exp_sol_robustness_mode_2 = -1
        exp_sol_robustness_mode_3 = -1
        exp_sol_robustness_mode_4 = -2

        state_list_lon_ego = {0: StateLongitudinal(a=-2), 1: StateLongitudinal(a=-1),
                              2: StateLongitudinal(a=1), 3: StateLongitudinal(a=2)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(acceleration=-1, time_step=0), 1: State(acceleration=0, time_step=1),
                             2: State(acceleration=0, time_step=2), 3: State(acceleration=0, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}

        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        state_list_lon_other = {0: StateLongitudinal(a=-1), 1: StateLongitudinal(a=-2),
                                2: StateLongitudinal(a=1), 3: StateLongitudinal(a=1)}
        state_list_lat_other = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other = {0: State(acceleration=-1, time_step=0), 1: State(acceleration=0, time_step=1),
                               2: State(acceleration=0, time_step=2), 3: State(acceleration=0, time_step=3)}
        lanelet_assignments_other = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle = Vehicle(state_list_lon_other, state_list_lat_other, Rectangle(5, 2), cr_state_list_other, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_other, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = BrakingPredicateCollection.brakes_stronger(0, ego_vehicle, other_vehicle,
                                                                        OperatingMode.MONITOR)
        sol_monitor_mode_2 = BrakingPredicateCollection.brakes_stronger(1, ego_vehicle, other_vehicle,
                                                                        OperatingMode.MONITOR)
        sol_monitor_mode_3 = BrakingPredicateCollection.brakes_stronger(2, ego_vehicle, other_vehicle,
                                                                        OperatingMode.MONITOR)
        sol_monitor_mode_4 = BrakingPredicateCollection.brakes_stronger(3, ego_vehicle, other_vehicle,
                                                                        OperatingMode.MONITOR)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = BrakingPredicateCollection.brakes_stronger(0, ego_vehicle, other_vehicle,
                                                                           OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = BrakingPredicateCollection.brakes_stronger(1, ego_vehicle, other_vehicle,
                                                                           OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = BrakingPredicateCollection.brakes_stronger(2, ego_vehicle, other_vehicle,
                                                                           OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = BrakingPredicateCollection.brakes_stronger(3, ego_vehicle, other_vehicle,
                                                                           OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1.value)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2.value)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3.value)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4.value)

        # Robustness-Mode
        sol_robustness_mode_1 = BrakingPredicateCollection.brakes_stronger(0, ego_vehicle, other_vehicle,
                                                                           OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = BrakingPredicateCollection.brakes_stronger(1, ego_vehicle, other_vehicle,
                                                                           OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = BrakingPredicateCollection.brakes_stronger(2, ego_vehicle, other_vehicle,
                                                                           OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = BrakingPredicateCollection.brakes_stronger(3, ego_vehicle, other_vehicle,
                                                                           OperatingMode.ROBUSTNESS)

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)

    def test_unnecessary_braking(self):
        self._traffic_rule_param["a_abrupt"] = -2

        # expected solutions
        exp_sol_monitor_mode_1 = False  # a_ego > 0
        exp_sol_monitor_mode_2 = True  # a_ego < a_lead - |a_abrupt| for single leading vehicle
        exp_sol_monitor_mode_3 = True  # a_ego < a_lead - |a_abrupt| for all leading vehicles
        exp_sol_monitor_mode_4 = False  # a_ego > a_lead - |a_abrupt| for all leading vehicles
        exp_sol_monitor_mode_5 = True  # a_ego < a_abrupt; no leading vehicle
        exp_sol_monitor_mode_6 = False  # a_ego > a_abrupt; no leading vehicle
        exp_sol_constraint_mode_1 = 1 + self._traffic_rule_param["a_abrupt"]
        exp_sol_constraint_mode_2 = -1 + self._traffic_rule_param["a_abrupt"]
        exp_sol_constraint_mode_3 = -2 + self._traffic_rule_param["a_abrupt"]
        exp_sol_constraint_mode_4 = -1.5 + self._traffic_rule_param["a_abrupt"]
        exp_sol_constraint_mode_5 = self._traffic_rule_param["a_abrupt"]
        exp_sol_constraint_mode_6 = self._traffic_rule_param["a_abrupt"]

        exp_sol_robustness_mode_1 = self._traffic_rule_param["a_abrupt"]
        exp_sol_robustness_mode_2 = 4 + self._traffic_rule_param["a_abrupt"]
        exp_sol_robustness_mode_3 = 5 + self._traffic_rule_param["a_abrupt"]
        exp_sol_robustness_mode_4 = 1.5 + self._traffic_rule_param["a_abrupt"]
        exp_sol_robustness_mode_5 = 8 + self._traffic_rule_param["a_abrupt"]
        exp_sol_robustness_mode_6 = 2

        right_vertices = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0], [80, 0]])
        left_vertices = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4], [80, 4]])
        center_vertices = np.array([[0, 2], [10, 2], [20, 2], [30, 2], [40, 2], [50, 2], [60, 2], [70, 2], [80, 2]])
        lanelet_id = 1
        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(Lanelet(left_vertices, center_vertices, right_vertices, lanelet_id))
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        necessary_predicates = {"unnecessary_braking__x_ego"}
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        braking_predicates = BrakingPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        necessary_predicates, traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10, a=1), 1: StateLongitudinal(s=10, v=10, a=-5),
                              2: StateLongitudinal(s=20, v=10, a=-7), 3: StateLongitudinal(s=30, v=10, a=-3),
                              4: StateLongitudinal(s=40, v=10, a=-8), 5: StateLongitudinal(s=50, v=10, a=2)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4), 5: State(position=50, time_step=5)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}, 5: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {0: StateLongitudinal(s=10, v=10, a=1), 1: StateLongitudinal(s=20, v=10, a=-4),
                                  2: StateLongitudinal(s=30, v=10, a=-2), 3: StateLongitudinal(s=40, v=10, a=-1.5)}
        state_list_lat_other_1 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {0: State(position=10, time_step=0), 1: State(position=20, time_step=1),
                                 2: State(position=30, time_step=1), 3: State(position=40, time_step=1)}
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {0: StateLongitudinal(s=20, v=10, a=1), 1: StateLongitudinal(s=30, v=10, a=-1),
                                  2: StateLongitudinal(s=40, v=10, a=-3), 3: StateLongitudinal(s=50, v=10, a=-2)}
        state_list_lat_other_2 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_2 = {0: State(position=20, time_step=0), 1: State(position=30, time_step=1),
                                 2: State(position=40, time_step=1), 3: State(position=50, time_step=1)}
        lanelet_assignments_other_2 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        other_vehicles = [other_vehicle_1, other_vehicle_2]

        # Monitor-Mode
        sol_monitor_mode_1 = braking_predicates.unnecessary_braking(0, ego_vehicle, other_vehicles,
                                                                    OperatingMode.MONITOR)
        sol_monitor_mode_2 = braking_predicates.unnecessary_braking(1, ego_vehicle, other_vehicles,
                                                                    OperatingMode.MONITOR)
        sol_monitor_mode_3 = braking_predicates.unnecessary_braking(2, ego_vehicle, other_vehicles,
                                                                    OperatingMode.MONITOR)
        sol_monitor_mode_4 = braking_predicates.unnecessary_braking(3, ego_vehicle, other_vehicles,
                                                                    OperatingMode.MONITOR)
        sol_monitor_mode_5 = braking_predicates.unnecessary_braking(4, ego_vehicle, other_vehicles,
                                                                    OperatingMode.MONITOR)
        sol_monitor_mode_6 = braking_predicates.unnecessary_braking(5, ego_vehicle, other_vehicles,
                                                                    OperatingMode.MONITOR)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)

        # Constraint-Mode
        sol_constraint_mode_1 = braking_predicates.unnecessary_braking(0, ego_vehicle, other_vehicles,
                                                                       OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = braking_predicates.unnecessary_braking(1, ego_vehicle, other_vehicles,
                                                                       OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = braking_predicates.unnecessary_braking(2, ego_vehicle, other_vehicles,
                                                                       OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = braking_predicates.unnecessary_braking(3, ego_vehicle, other_vehicles,
                                                                       OperatingMode.CONSTRAINT)
        sol_constraint_mode_5 = braking_predicates.unnecessary_braking(4, ego_vehicle, other_vehicles,
                                                                       OperatingMode.CONSTRAINT)
        sol_constraint_mode_6 = braking_predicates.unnecessary_braking(5, ego_vehicle, other_vehicles,
                                                                       OperatingMode.CONSTRAINT)
        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1.value)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2.value)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3.value)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4.value)
        self.assertEqual(exp_sol_constraint_mode_5, sol_constraint_mode_5.value)
        self.assertEqual(exp_sol_constraint_mode_6, sol_constraint_mode_6.value)

        # Robustness-Mode
        sol_robustness_mode_1 = braking_predicates.unnecessary_braking(0, ego_vehicle, other_vehicles,
                                                                       OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = braking_predicates.unnecessary_braking(1, ego_vehicle, other_vehicles,
                                                                       OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = braking_predicates.unnecessary_braking(2, ego_vehicle, other_vehicles,
                                                                       OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = braking_predicates.unnecessary_braking(3, ego_vehicle, other_vehicles,
                                                                       OperatingMode.ROBUSTNESS)
        sol_robustness_mode_5 = braking_predicates.unnecessary_braking(4, ego_vehicle, other_vehicles,
                                                                       OperatingMode.ROBUSTNESS)
        sol_robustness_mode_6 = braking_predicates.unnecessary_braking(5, ego_vehicle, other_vehicles,
                                                                       OperatingMode.ROBUSTNESS)
        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)
        self.assertEqual(exp_sol_robustness_mode_5, sol_robustness_mode_5)
        self.assertEqual(exp_sol_robustness_mode_6, sol_robustness_mode_6)
