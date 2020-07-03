import unittest
import os
import numpy as np

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.scenario.lanelet import LaneletNetwork

from src.predicates.velocity_predicates import VelocityPredicateCollection
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

        right_vertices_lane_1 = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0],
                                          [80, 1], [90, 0]])
        left_vertices_lane_1 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4],
                                         [80, 1], [90, 0]])
        center_vertices_lane_1 = np.array([[0, 2], [10, 2], [20, 2], [30, 2], [40, 2], [50, 2], [60, 2], [70, 2],
                                           [80, 1], [90, 0]])
        self._lanelet_1 = Lanelet(left_vertices_lane_1, center_vertices_lane_1, right_vertices_lane_1, lanelet_id=1,
                                  adjacent_left=2, adjacent_left_same_direction=True)

        right_vertices_lane_2 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4],
                                          [80, 4], [90, 4]])
        left_vertices_lane_2 = np.array([[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8],
                                         [80, 8], [90, 8]])
        center_vertices_lane_2 = np.array([[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12],
                                           [70, 12], [80, 12], [90, 12]])
        self._lanelet_2 =  Lanelet(left_vertices_lane_2, center_vertices_lane_2, right_vertices_lane_2, lanelet_id=2,
                                   adjacent_left=3, adjacent_left_same_direction=True,
                                   adjacent_right=1, adjacent_right_same_direction=True)

        right_vertices_lane_3 = np.array([[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8],
                                          [80, 8], [90, 8]])
        left_vertices_lane_3 = np.array([[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12], [70, 12],
                                         [80, 12], [90, 12]])
        center_vertices_lane_3 = np.array([[0, 10], [10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [60, 10],
                                           [70, 10], [80, 10], [90, 10]])
        self._lanelet_3 = Lanelet(left_vertices_lane_3, center_vertices_lane_3, right_vertices_lane_3, lanelet_id=3,
                                  adjacent_right=2, adjacent_right_same_direction=True)

    def test_in_standstill(self):
        self._traffic_rule_param["standstill_error"] = 0.01

        # expected solutions
        exp_sol_monitor_mode_1 = False
        exp_sol_monitor_mode_2 = True
        exp_sol_monitor_mode_3 = True
        exp_sol_monitor_mode_4 = True

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = VelocityPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                         set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=1), 1: StateLongitudinal(s=1, v=0),
                              2: StateLongitudinal(s=1, v=0.001), 3: StateLongitudinal(s=1.001, v=-0.001)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=1, time_step=1),
                             2: State(position=1, time_step=2), 3: State(position=1.001, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates.in_standstill(0, ego_vehicle)
        sol_monitor_mode_2 = general_predicates.in_standstill(1, ego_vehicle)
        sol_monitor_mode_3 = general_predicates.in_standstill(2, ego_vehicle)
        sol_monitor_mode_4 = general_predicates.in_standstill(3, ego_vehicle)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

    def test_drives_with_slightly_higher_speed(self):
        self._traffic_rule_param["slightly_higher_speed_difference"] = 5.55

        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle has lower velocity
        exp_sol_monitor_mode_2 = False  # ego vehicle has same velocity
        exp_sol_monitor_mode_3 = True  # ego vehicle drives with only slightly higher speed
        exp_sol_monitor_mode_4 = False  # ego vehicle drives too fast
        exp_sol_constraint_mode_1 = 15.55
        exp_sol_constraint_mode_2 = 15.55
        exp_sol_constraint_mode_3 = 15.55
        exp_sol_constraint_mode_4 = 15.55
        exp_sol_robustness_mode_1 = 10.55
        exp_sol_robustness_mode_2 = 5.55
        exp_sol_robustness_mode_3 = 0.55
        exp_sol_robustness_mode_4 = -4.45

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = VelocityPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                         set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=5), 1: StateLongitudinal(s=5, v=10),
                              2: StateLongitudinal(s=15, v=15), 3: StateLongitudinal(s=30, v=20)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {0: StateLongitudinal(s=10, v=10), 1: StateLongitudinal(s=20, v=10),
                                  2: StateLongitudinal(s=30, v=10), 3: StateLongitudinal(s=40, v=10)}
        state_list_lat_other_1 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {0: State(position=10, time_step=0), 1: State(position=20, time_step=1),
                                 2: State(position=30, time_step=2), 3: State(position=40, time_step=3)}
        lanelet_assignments_other_1 = {0: {2}, 1: {2}, 2: {2}, 3: {2}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates.drives_with_slightly_higher_speed(0, ego_vehicle, other_vehicle_1,
                                                                                  OperatingMode.MONITOR)
        sol_monitor_mode_2 = general_predicates.drives_with_slightly_higher_speed(1, ego_vehicle, other_vehicle_1,
                                                                                  OperatingMode.MONITOR)
        sol_monitor_mode_3 = general_predicates.drives_with_slightly_higher_speed(2, ego_vehicle, other_vehicle_1,
                                                                                  OperatingMode.MONITOR)
        sol_monitor_mode_4 = general_predicates.drives_with_slightly_higher_speed(3, ego_vehicle, other_vehicle_1,
                                                                                  OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = general_predicates.drives_with_slightly_higher_speed(0, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = general_predicates.drives_with_slightly_higher_speed(1, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = general_predicates.drives_with_slightly_higher_speed(2, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = general_predicates.drives_with_slightly_higher_speed(3, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = general_predicates.drives_with_slightly_higher_speed(0, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = general_predicates.drives_with_slightly_higher_speed(1, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = general_predicates.drives_with_slightly_higher_speed(2, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = general_predicates.drives_with_slightly_higher_speed(3, ego_vehicle, other_vehicle_1,
                                                                                     OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertAlmostEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)
