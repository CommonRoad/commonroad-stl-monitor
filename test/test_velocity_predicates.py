import unittest
import os
import numpy as np

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.traffic_sign import TrafficSignElement, TrafficSign, TrafficSignIDGermany

from src.predicates.velocity_predicates import VelocityPredicateCollection
from src.common.helper import *
from src.common.road_network import RoadNetwork


class TestVelocityPredicates(unittest.TestCase):
    def setUp(self):
        config_path = os.path.dirname(os.path.abspath(__file__)) + "/../src/"
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 1.0, 'DEU')

        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._ego_vehicle_param = create_other_vehicles_param(config.get("ego_vehicle_param"))
        self._traffic_rules_param = traffic_rules.get("traffic_rules_param")
        self._road_network_param = config.get("road_network_param")

        traffic_sign_min_speed = TrafficSignElement(TrafficSignIDGermany.MIN_SPEED, ["10"])
        self._traffic_sign_1 = TrafficSign(111, [traffic_sign_min_speed], {1}, np.array([0.0, 0.0]))
        traffic_sign_max_speed = TrafficSignElement(TrafficSignIDGermany.MAX_SPEED, ["50"])
        self._traffic_sign_2 = TrafficSign(111, [traffic_sign_max_speed], {1}, np.array([0.0, 0.0]))

        right_vertices_lane_1 = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0],
                                          [80, 1], [90, 0]])
        left_vertices_lane_1 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4],
                                         [80, 1], [90, 0]])
        center_vertices_lane_1 = np.array([[0, 2], [10, 2], [20, 2], [30, 2], [40, 2], [50, 2], [60, 2], [70, 2],
                                           [80, 1], [90, 0]])
        self._lanelet_1 = Lanelet(left_vertices_lane_1, center_vertices_lane_1, right_vertices_lane_1, lanelet_id=1,
                                  adjacent_left=2, adjacent_left_same_direction=True, traffic_signs={111})

        right_vertices_lane_2 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4],
                                          [80, 4], [90, 4]])
        left_vertices_lane_2 = np.array([[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8],
                                         [80, 8], [90, 8]])
        center_vertices_lane_2 = np.array([[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12],
                                           [70, 12], [80, 12], [90, 12]])
        self._lanelet_2 = Lanelet(left_vertices_lane_2, center_vertices_lane_2, right_vertices_lane_2, lanelet_id=2,
                                  adjacent_right=1, adjacent_right_same_direction=True)

    def test_in_standstill(self):
        self._traffic_rules_param["standstill_error"] = 0.01

        # expected solutions
        exp_sol_monitor_mode_1 = False
        exp_sol_monitor_mode_2 = True
        exp_sol_monitor_mode_3 = True
        exp_sol_monitor_mode_4 = True
        exp_sol_constraint_mode_1 = (-0.01, 0.01)
        exp_sol_constraint_mode_2 = (-0.01, 0.01)
        exp_sol_constraint_mode_3 = (-0.01, 0.01)
        exp_sol_constraint_mode_4 = (-0.01, 0.01)
        exp_sol_robustness_mode_1 = -0.99
        exp_sol_robustness_mode_2 = 0.01
        exp_sol_robustness_mode_3 = 0.009
        exp_sol_robustness_mode_4 = 0.009

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

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
        sol_monitor_mode_1 = velocity_predicates.in_standstill(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.in_standstill(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.in_standstill(2, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_4 = velocity_predicates.in_standstill(3, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.in_standstill(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.in_standstill(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.in_standstill(2, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = velocity_predicates.in_standstill(3, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.in_standstill(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.in_standstill(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.in_standstill(2, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = velocity_predicates.in_standstill(3, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertAlmostEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)

    def test_drives_with_slightly_higher_speed(self):
        self._traffic_rules_param["slightly_higher_speed_difference"] = 5.55

        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle has lower velocity
        exp_sol_monitor_mode_2 = False  # ego vehicle has same velocity
        exp_sol_monitor_mode_3 = True  # ego vehicle drives with only slightly higher speed
        exp_sol_monitor_mode_4 = False  # ego vehicle drives too fast
        exp_sol_constraint_mode_1 = 15.55
        exp_sol_constraint_mode_2 = 15.55
        exp_sol_constraint_mode_3 = 15.55
        exp_sol_constraint_mode_4 = 15.55
        exp_sol_robustness_mode_1 = -5
        exp_sol_robustness_mode_2 = -1e-17
        exp_sol_robustness_mode_3 = 0.55
        exp_sol_robustness_mode_4 = -4.45

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

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
        sol_monitor_mode_1 = velocity_predicates.drives_with_slightly_higher_speed(0, ego_vehicle, other_vehicle_1,
                                                                                   OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.drives_with_slightly_higher_speed(1, ego_vehicle, other_vehicle_1,
                                                                                   OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.drives_with_slightly_higher_speed(2, ego_vehicle, other_vehicle_1,
                                                                                   OperatingMode.MONITOR)
        sol_monitor_mode_4 = velocity_predicates.drives_with_slightly_higher_speed(3, ego_vehicle, other_vehicle_1,
                                                                                   OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.drives_with_slightly_higher_speed(0, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.drives_with_slightly_higher_speed(1, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.drives_with_slightly_higher_speed(2, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = velocity_predicates.drives_with_slightly_higher_speed(3, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.drives_with_slightly_higher_speed(0, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.drives_with_slightly_higher_speed(1, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.drives_with_slightly_higher_speed(2, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = velocity_predicates.drives_with_slightly_higher_speed(3, ego_vehicle, other_vehicle_1,
                                                                                      OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertAlmostEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)

    def test_keeps_sign_min_speed_limit(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle drives too slow
        exp_sol_monitor_mode_2 = True  # ego vehicle drives exactly with the min. required speed
        exp_sol_monitor_mode_3 = True  # ego vehicle drives with higher velocity
        exp_sol_monitor_mode_4 = True  # there exists not min. required speed
        exp_sol_constraint_mode_1 = 10.0
        exp_sol_constraint_mode_2 = 10.0
        exp_sol_constraint_mode_3 = 10.0
        exp_sol_constraint_mode_4 = 0
        exp_sol_robustness_mode_1 = -5.0
        exp_sol_robustness_mode_2 = 0
        exp_sol_robustness_mode_3 = 5.0
        exp_sol_robustness_mode_4 = math.inf

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_1, {1})
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=5), 1: StateLongitudinal(s=5, v=10),
                              2: StateLongitudinal(s=15, v=15), 3: StateLongitudinal(s=30, v=5)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=4, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=15, time_step=2), 3: State(position=30, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.keeps_sign_min_speed_limit(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.keeps_sign_min_speed_limit(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.keeps_sign_min_speed_limit(2, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_4 = velocity_predicates.keeps_sign_min_speed_limit(3, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.keeps_sign_min_speed_limit(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.keeps_sign_min_speed_limit(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.keeps_sign_min_speed_limit(2, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = velocity_predicates.keeps_sign_min_speed_limit(3, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.keeps_sign_min_speed_limit(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.keeps_sign_min_speed_limit(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.keeps_sign_min_speed_limit(2, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = velocity_predicates.keeps_sign_min_speed_limit(3, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertAlmostEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)

    def test_drives_faster(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle has lower velocity
        exp_sol_monitor_mode_2 = False  # ego vehicle has same velocity
        exp_sol_monitor_mode_3 = True  # ego vehicle drives with higher speed
        exp_sol_constraint_mode_1 = 10.0
        exp_sol_constraint_mode_2 = 20.0
        exp_sol_constraint_mode_3 = 30.0
        exp_sol_robustness_mode_1 = -5.0
        exp_sol_robustness_mode_2 = -1.e-17
        exp_sol_robustness_mode_3 = 5.0

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=5), 1: StateLongitudinal(s=5, v=20),
                              2: StateLongitudinal(s=25, v=35)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=5, time_step=1),
                             2: State(position=25, time_step=2)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {0: StateLongitudinal(s=10, v=10), 1: StateLongitudinal(s=20, v=20),
                                  2: StateLongitudinal(s=40, v=30)}
        state_list_lat_other_1 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {0: State(position=10, time_step=0), 1: State(position=20, time_step=1),
                                 2: State(position=40, time_step=2)}
        lanelet_assignments_other_1 = {0: {2}, 1: {2}, 2: {2}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.drives_faster(0, ego_vehicle, other_vehicle_1, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.drives_faster(1, ego_vehicle, other_vehicle_1, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.drives_faster(2, ego_vehicle, other_vehicle_1, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.drives_faster(0, ego_vehicle, other_vehicle_1,
                                                                  OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.drives_faster(1, ego_vehicle, other_vehicle_1,
                                                                  OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.drives_faster(2, ego_vehicle, other_vehicle_1,
                                                                  OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.drives_faster(0, ego_vehicle, other_vehicle_1,
                                                                  OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.drives_faster(1, ego_vehicle, other_vehicle_1,
                                                                  OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.drives_faster(2, ego_vehicle, other_vehicle_1,
                                                                  OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)

    def test_keeps_lane_speed_limit(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True  # ego vehicle drives with lower velocity
        exp_sol_monitor_mode_2 = True  # ego vehicle drives exactly with the max speed
        exp_sol_monitor_mode_3 = False  # ego vehicle drives too fast
        exp_sol_monitor_mode_4 = True  # there exists no speed limit
        exp_sol_constraint_mode_1 = 50.0
        exp_sol_constraint_mode_2 = 50.0
        exp_sol_constraint_mode_3 = 50.0
        exp_sol_constraint_mode_4 = math.inf
        exp_sol_robustness_mode_1 = 5.0
        exp_sol_robustness_mode_2 = 0
        exp_sol_robustness_mode_3 = -5.0
        exp_sol_robustness_mode_4 = math.inf

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_2, {1})
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=45), 1: StateLongitudinal(s=45, v=50),
                              2: StateLongitudinal(s=95, v=55), 3: StateLongitudinal(s=150, v=45)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=4, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=45, time_step=1),
                             2: State(position=95, time_step=2), 3: State(position=150, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.keeps_lane_speed_limit(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.keeps_lane_speed_limit(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.keeps_lane_speed_limit(2, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_4 = velocity_predicates.keeps_lane_speed_limit(3, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.keeps_lane_speed_limit(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.keeps_lane_speed_limit(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.keeps_lane_speed_limit(2, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = velocity_predicates.keeps_lane_speed_limit(3, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.keeps_lane_speed_limit(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.keeps_lane_speed_limit(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.keeps_lane_speed_limit(2, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = velocity_predicates.keeps_lane_speed_limit(3, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertAlmostEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)

    def test_speed_limit_suggested(self):
        self._traffic_rules_param["desired_interstate_velocity"] = 55
        # expected solutions
        exp_sol_1 = 50  # speed limit exists
        exp_sol_2 = 55  # there exists no speed limit

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_2, {1})
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=5), 1: StateLongitudinal(s=5, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1)}
        lanelet_assignments_ego = {0: {1}, 1: {2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        sol_1 = velocity_predicates._speed_limit_suggested(0, ego_vehicle)
        sol_2 = velocity_predicates._speed_limit_suggested(1, ego_vehicle)

        self.assertEqual(exp_sol_1, sol_1)
        self.assertEqual(exp_sol_2, sol_2)

    def test_keeps_type_speed_limit(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True  # ego vehicle drives with lower velocity
        exp_sol_monitor_mode_2 = True  # ego vehicle drives exactly with the maximum speed
        exp_sol_monitor_mode_3 = False  # ego vehicle drives too fast
        exp_sol_monitor_mode_4 = True  # there exists no type speed limit
        exp_sol_constraint_mode_1 = 22.22
        exp_sol_constraint_mode_2 = 22.22
        exp_sol_constraint_mode_3 = 22.22
        exp_sol_constraint_mode_4 = math.inf
        exp_sol_robustness_mode_1 = 2.22
        exp_sol_robustness_mode_2 = 0
        exp_sol_robustness_mode_3 = -7.78
        exp_sol_robustness_mode_4 = math.inf

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=20), 1: StateLongitudinal(s=20, v=22.22),
                              2: StateLongitudinal(s=42.22, v=30), 3: StateLongitudinal(s=72.22, v=30)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=4, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=20, time_step=1),
                             2: State(position=42.22, time_step=2), 3: State(position=72.22, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {2}}
        ego_vehicle_1 = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                                ObstacleType.TRUCK, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)
        ego_vehicle_2 = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                                ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.keeps_type_speed_limit(0, ego_vehicle_1, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.keeps_type_speed_limit(1, ego_vehicle_1, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.keeps_type_speed_limit(2, ego_vehicle_1, OperatingMode.MONITOR)
        sol_monitor_mode_4 = velocity_predicates.keeps_type_speed_limit(3, ego_vehicle_2, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.keeps_type_speed_limit(0, ego_vehicle_1, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.keeps_type_speed_limit(1, ego_vehicle_1, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.keeps_type_speed_limit(2, ego_vehicle_1, OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = velocity_predicates.keeps_type_speed_limit(3, ego_vehicle_2, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.keeps_type_speed_limit(0, ego_vehicle_1, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.keeps_type_speed_limit(1, ego_vehicle_1, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.keeps_type_speed_limit(2, ego_vehicle_1, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = velocity_predicates.keeps_type_speed_limit(3, ego_vehicle_2, OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertAlmostEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)

    def test_keeps_braking_speed_limit(self):
        self._ego_vehicle_param["braking_speed_limit"] = 10
        # expected solutions
        exp_sol_monitor_mode_1 = True  # ego vehicle drives with lower velocity
        exp_sol_monitor_mode_2 = True  # ego vehicle drives exactly with the maximum speed
        exp_sol_monitor_mode_3 = False  # ego vehicle drives too fast
        exp_sol_constraint_mode_1 = 10.0
        exp_sol_constraint_mode_2 = 10.0
        exp_sol_constraint_mode_3 = 10.0
        exp_sol_robustness_mode_1 = 5.0
        exp_sol_robustness_mode_2 = 0
        exp_sol_robustness_mode_3 = -5.0

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_2, {1})
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=5), 1: StateLongitudinal(s=5, v=10),
                              2: StateLongitudinal(s=15, v=15)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=15, time_step=2)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.keeps_braking_speed_limit(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.keeps_braking_speed_limit(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.keeps_braking_speed_limit(2, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.keeps_braking_speed_limit(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.keeps_braking_speed_limit(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.keeps_braking_speed_limit(2, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.keeps_braking_speed_limit(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.keeps_braking_speed_limit(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.keeps_braking_speed_limit(2, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)

    def test_keeps_fov_speed_limit(self):
        self._ego_vehicle_param["fov_speed_limit"] = 10
        # expected solutions
        exp_sol_monitor_mode_1 = True  # ego vehicle drives with lower velocity
        exp_sol_monitor_mode_2 = True  # ego vehicle drives exactly with the maximum speed
        exp_sol_monitor_mode_3 = False  # ego vehicle drives too fast
        exp_sol_constraint_mode_1 = 10.0
        exp_sol_constraint_mode_2 = 10.0
        exp_sol_constraint_mode_3 = 10.0
        exp_sol_robustness_mode_1 = 5.0
        exp_sol_robustness_mode_2 = 0
        exp_sol_robustness_mode_3 = -5.0

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_2, {1})
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=5), 1: StateLongitudinal(s=5, v=10),
                              2: StateLongitudinal(s=15, v=15)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=15, time_step=2)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.keeps_fov_speed_limit(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.keeps_fov_speed_limit(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.keeps_fov_speed_limit(2, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.keeps_fov_speed_limit(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.keeps_fov_speed_limit(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.keeps_fov_speed_limit(2, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.keeps_fov_speed_limit(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.keeps_fov_speed_limit(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.keeps_fov_speed_limit(2, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        
    def test_reverses(self):
        self._traffic_rules_param["standstill_error"] = 0.01
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego has velocity of zero
        exp_sol_monitor_mode_2 = False  # ego vehicle has positive velocity
        exp_sol_monitor_mode_3 = False  # ego vehicle has velocity of -standstill_error
        exp_sol_monitor_mode_4 = True  # ego vehicle has velocity smaller than -standstill_error
        exp_sol_constraint_mode_1 = -self._traffic_rules_param["standstill_error"]
        exp_sol_constraint_mode_2 = -self._traffic_rules_param["standstill_error"]
        exp_sol_constraint_mode_3 = -self._traffic_rules_param["standstill_error"]
        exp_sol_constraint_mode_4 = -self._traffic_rules_param["standstill_error"]
        exp_sol_robustness_mode_1 = -self._traffic_rules_param["standstill_error"]
        exp_sol_robustness_mode_2 = -self._traffic_rules_param["standstill_error"] -1
        exp_sol_robustness_mode_3 = -1e-17
        exp_sol_robustness_mode_4 = -self._traffic_rules_param["standstill_error"] - -2

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_2, {1})
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=0), 1: StateLongitudinal(s=0, v=1),
                              2: StateLongitudinal(s=1, v=-self._traffic_rules_param["standstill_error"]),
                              3: StateLongitudinal(s=1-self._traffic_rules_param["standstill_error"], v=-2)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=4, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=0, time_step=1),
                             2: State(position=15, time_step=2), 3: State(position=30, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.reverses(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.reverses(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.reverses(2, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_4 = velocity_predicates.reverses(3, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.reverses(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.reverses(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.reverses(2, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = velocity_predicates.reverses(3, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.reverses(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.reverses(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.reverses(2, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = velocity_predicates.reverses(3, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertAlmostEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)

    def test_exist_standing_leading_vehicle(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle at all
        exp_sol_monitor_mode_2 = False  # two leading vehicles which have velocity > 0
        exp_sol_monitor_mode_3 = True  # first leading vehicle is standing
        exp_sol_monitor_mode_4 = True  # third leading vehicle is standing

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=2), 1: StateLongitudinal(s=2, v=2),
                              2: StateLongitudinal(s=4, v=2), 3: StateLongitudinal(s=6, v=2)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=2, time_step=1),
                             2: State(position=4, time_step=2), 3: State(position=6, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {1: StateLongitudinal(s=12, v=2),
                                  2: StateLongitudinal(s=14, v=0), 3: StateLongitudinal(s=14, v=2)}
        state_list_lat_other_1 = {1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {1: State(position=12, time_step=1),
                                 2: State(position=14, time_step=2), 3: State(position=14, time_step=3)}
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 41, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {1: StateLongitudinal(s=22, v=2),
                                  2: StateLongitudinal(s=24, v=2), 3: StateLongitudinal(s=26, v=2)}
        state_list_lat_other_2 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_2 = {1: State(position=22, time_step=1),
                                 2: State(position=24, time_step=2), 3: State(position=26, time_step=3)}
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 42, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        # other vehicle 3
        state_list_lon_other_3 = {2: StateLongitudinal(s=34, v=2), 3: StateLongitudinal(s=36, v=0)}
        state_list_lat_other_3 = {2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_3 = {2: State(position=34, time_step=2), 3: State(position=36, time_step=3)}
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(state_list_lon_other_3, state_list_lat_other_3, Rectangle(5, 2),
                                  cr_state_list_other_3, 43, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_3, None, None, None)

        # other vehicle 4
        state_list_lon_other_4 = {0: StateLongitudinal(s=-10, v=60), 1: StateLongitudinal(s=50, v=0)}
        state_list_lat_other_4 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=3.5, theta=0)}
        cr_state_list_other_4 = {0: State(position=-10, time_step=0), 1: State(position=50, time_step=1)}
        lanelet_assignments_other_4 = {0: {2}, 1: {2}}
        other_vehicle_4 = Vehicle(state_list_lon_other_4, state_list_lat_other_4, Rectangle(5, 2),
                                  cr_state_list_other_4, 44, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_4, None, None, None)

        other_vehicles = [other_vehicle_1, other_vehicle_2, other_vehicle_3, other_vehicle_4]

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.exist_standing_leading_vehicle(0, ego_vehicle, other_vehicles)
        sol_monitor_mode_2 = velocity_predicates.exist_standing_leading_vehicle(1, ego_vehicle, other_vehicles)
        sol_monitor_mode_3 = velocity_predicates.exist_standing_leading_vehicle(2, ego_vehicle, other_vehicles)
        sol_monitor_mode_4 = velocity_predicates.exist_standing_leading_vehicle(3, ego_vehicle, other_vehicles)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

    def test_preserves_traffic_flow(self):
        self._traffic_rules_param["min_velocity_dif"] = 15
        self._ego_vehicle_param["braking_speed_limit"] = 50
        self._ego_vehicle_param["fov_speed_limit"] = 35
        self._ego_vehicle_param["road_condition_speed_limit"] = 50
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle drives too slow
        exp_sol_monitor_mode_2 = False  # ego vehicle drives at lower limit to
        exp_sol_monitor_mode_3 = True  # ego vehcie drives faster than velocity limit
        exp_sol_constraint_mode_1 = 20
        exp_sol_constraint_mode_2 = 20
        exp_sol_constraint_mode_3 = 20
        exp_sol_robustness_mode_1 = -18
        exp_sol_robustness_mode_2 = -1e-17
        exp_sol_robustness_mode_3 = 30

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_2, {1})
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=2), 1: StateLongitudinal(s=2, v=20),
                              2: StateLongitudinal(s=22, v=50)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=2, time_step=1),
                             2: State(position=22, time_step=2)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.preserves_traffic_flow(0, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_2 = velocity_predicates.preserves_traffic_flow(1, ego_vehicle, OperatingMode.MONITOR)
        sol_monitor_mode_3 = velocity_predicates.preserves_traffic_flow(2, ego_vehicle, OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)

        # Constraint-Mode
        sol_constraint_mode_1 = velocity_predicates.preserves_traffic_flow(0, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = velocity_predicates.preserves_traffic_flow(1, ego_vehicle, OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = velocity_predicates.preserves_traffic_flow(2, ego_vehicle, OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)

        # Robustness-Mode
        sol_robustness_mode_1 = velocity_predicates.preserves_traffic_flow(0, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = velocity_predicates.preserves_traffic_flow(1, ego_vehicle, OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = velocity_predicates.preserves_traffic_flow(2, ego_vehicle, OperatingMode.ROBUSTNESS)

        self.assertAlmostEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertAlmostEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertAlmostEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)

    def test_slow_leading_vehicle(self):
        self._traffic_rules_param["min_velocity_dif"] = 15
        self._ego_vehicle_param["road_condition_speed_limit"] = 50
        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle at all
        exp_sol_monitor_mode_2 = False  # two leading vehicles which drive with speed limit
        exp_sol_monitor_mode_3 = True  # first leading vehicle is drives to slow
        exp_sol_monitor_mode_4 = True  # third leading vehicle drives to slow

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_traffic_sign(self._traffic_sign_2, {1})
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        velocity_predicates = VelocityPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=2), 1: StateLongitudinal(s=2, v=2),
                              2: StateLongitudinal(s=4, v=2), 3: StateLongitudinal(s=6, v=2)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=2, time_step=1),
                             2: State(position=4, time_step=2), 3: State(position=6, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {1: StateLongitudinal(s=12, v=50), 2: StateLongitudinal(s=62, v=12),
                                  3: StateLongitudinal(s=74, v=2)}
        state_list_lat_other_1 = {1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {1: State(position=12, time_step=1),
                                 2: State(position=62, time_step=2), 3: State(position=87, time_step=3)}
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 41, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {1: StateLongitudinal(s=22, v=36),
                                  2: StateLongitudinal(s=58, v=36), 3: StateLongitudinal(s=88, v=36)}
        state_list_lat_other_2 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_2 = {1: State(position=22, time_step=1),
                                 2: State(position=58, time_step=2), 3: State(position=88, time_step=3)}
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 42, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        # other vehicle 3
        state_list_lon_other_3 = {2: StateLongitudinal(s=34, v=50), 3: StateLongitudinal(s=84, v=0)}
        state_list_lat_other_3 = {2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_3 = {2: State(position=34, time_step=2), 3: State(position=84, time_step=3)}
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(state_list_lon_other_3, state_list_lat_other_3, Rectangle(5, 2),
                                  cr_state_list_other_3, 43, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_3, None, None, None)

        # other vehicle 4
        state_list_lon_other_4 = {0: StateLongitudinal(s=-10, v=60), 1: StateLongitudinal(s=50, v=0)}
        state_list_lat_other_4 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=3.5, theta=0)}
        cr_state_list_other_4 = {0: State(position=-10, time_step=0), 1: State(position=50, time_step=1)}
        lanelet_assignments_other_4 = {0: {2}, 1: {2}}
        other_vehicle_4 = Vehicle(state_list_lon_other_4, state_list_lat_other_4, Rectangle(5, 2),
                                  cr_state_list_other_4, 44, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_4, None, None, None)

        other_vehicles = [other_vehicle_1, other_vehicle_2, other_vehicle_3, other_vehicle_4]

        # Monitor-Mode
        sol_monitor_mode_1 = velocity_predicates.slow_leading_vehicle(0, ego_vehicle, other_vehicles)
        sol_monitor_mode_2 = velocity_predicates.slow_leading_vehicle(1, ego_vehicle, other_vehicles)
        sol_monitor_mode_3 = velocity_predicates.slow_leading_vehicle(2, ego_vehicle, other_vehicles)
        sol_monitor_mode_4 = velocity_predicates.slow_leading_vehicle(3, ego_vehicle, other_vehicles)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)