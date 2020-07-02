import unittest
import os
import numpy as np

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.scenario.lanelet import LaneletNetwork

from src.predicates.general_predicates import GeneralPredicateCollection
from src.common.helper import *
from src.common.road_network import RoadNetwork


class TestGeneralPredicates(unittest.TestCase):
    def setUp(self):
        config_path = os.path.dirname(os.path.abspath(__file__)) + "/../src/"
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 0.1, 'DEU')
        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._ego_vehicle_param = create_other_vehicles_param(config.get("ego_vehicle_param"))
        self._traffic_rule_param = traffic_rules.get("traffic_rules_param")
        self._road_network_param = config.get("road_network_param")

    def test_makes_uturn(self):
        self._traffic_rule_param["u_turn"] = 1.57

        # expected solutions
        exp_sol_monitor_mode_1 = False  # theta = 0
        exp_sol_monitor_mode_2 = False  # theta = (1/8) * math.pi
        exp_sol_monitor_mode_3 = True  # theta = (1/2) * math.pi
        exp_sol_monitor_mode_4 = True  # theta = (3/4) * math.pi
        exp_sol_constraint_mode_1 = (-self._traffic_rule_param.get("u_turn"), self._traffic_rule_param.get("u_turn"))
        exp_sol_constraint_mode_2 = (-self._traffic_rule_param.get("u_turn"), self._traffic_rule_param.get("u_turn"))
        exp_sol_constraint_mode_3 = (-self._traffic_rule_param.get("u_turn"), self._traffic_rule_param.get("u_turn"))
        exp_sol_constraint_mode_4 = (-self._traffic_rule_param.get("u_turn"), self._traffic_rule_param.get("u_turn"))
        exp_sol_robustness_mode_1 = self._traffic_rule_param.get("u_turn")
        exp_sol_robustness_mode_2 = self._traffic_rule_param.get("u_turn") - (1/8) * math.pi
        exp_sol_robustness_mode_3 = self._traffic_rule_param.get("u_turn") - (1/2) * math.pi
        exp_sol_robustness_mode_4 = self._traffic_rule_param.get("u_turn") - (3/4) * math.pi

        lanelet_network = LaneletNetwork()
        right_vertices_lane_1 = np.array([[0, 0], [10, 0], [20, 0], [30, .5], [40, 1], [50, 1], [60, 1], [70, 0]])
        left_vertices_lane_1 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4]])
        center_vertices_lane_1 = np.array([[0, 2], [10, 2], [20, 2], [30, 2], [40, 2], [50, 2], [60, 2], [70, 2]])
        lanelet_network.add_lanelet(Lanelet(left_vertices_lane_1, center_vertices_lane_1, right_vertices_lane_1, 1))

        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        necessary_predicates = {"makes_u_turn__x_ego"}
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        necessary_predicates, traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=(1/8) * math.pi),
                              2: StateLateral(d=0, theta=(1/2) * math.pi), 3: StateLateral(d=0, theta=(3/4) * math.pi)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=1), 3: State(position=30, time_step=1)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates.makes_u_turn(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = general_predicates.makes_u_turn(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = general_predicates.makes_u_turn(2, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_4 = general_predicates.makes_u_turn(3, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = general_predicates.makes_u_turn(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = general_predicates.makes_u_turn(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = general_predicates.makes_u_turn(2, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = general_predicates.makes_u_turn(3, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = general_predicates.makes_u_turn(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = general_predicates.makes_u_turn(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = general_predicates.makes_u_turn(2, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = general_predicates.makes_u_turn(3, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)
