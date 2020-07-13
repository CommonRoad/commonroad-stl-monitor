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
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 1.0, 'DEU')
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

        right_vertices_lane_4 = np.array([[0, 14], [10, 14], [20, 14], [30, 14], [40, 14], [50, 14], [60, 14], [70, 14],
                                          [80, 14], [90, 14]])
        left_vertices_lane_4 = np.array([[0, 18], [10, 18], [20, 18], [30, 18], [40, 18], [50, 18], [60, 18], [70, 18],
                                         [80, 18], [90, 18]])
        center_vertices_lane_4 = np.array([[0, 16], [10, 16], [20, 16], [30, 16], [40, 16], [50, 16], [60, 16],
                                           [70, 16], [80, 16], [90, 16]])
        self._lanelet_4 = Lanelet(left_vertices_lane_4, center_vertices_lane_4, right_vertices_lane_4, lanelet_id=4)

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
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=(1/8) * math.pi),
                              2: StateLateral(d=0, theta=(1/2) * math.pi), 3: StateLateral(d=0, theta=(3/4) * math.pi)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3)}
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

    def test_in_congestion(self):
        self._traffic_rule_param["num_veh_congestion"] = 3
        self._traffic_rule_param["max_congestion_velocity"] = 2.78

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle
        exp_sol_monitor_mode_2 = False  # only two leading vehicles
        exp_sol_monitor_mode_3 = False  # three leading vehicle, but not all drive with required velocity
        exp_sol_monitor_mode_4 = True

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

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
                                  2: StateLongitudinal(s=14, v=2), 3: StateLongitudinal(s=16, v=2)}
        state_list_lat_other_1 = {1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {1: State(position=12, time_step=1),
                                 2: State(position=14, time_step=2), 3: State(position=16, time_step=3)}
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
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
                                  cr_state_list_other_2, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        # other vehicle 3
        state_list_lon_other_3 = {2: StateLongitudinal(s=34, v=5), 3: StateLongitudinal(s=39, v=2)}
        state_list_lat_other_3 = {2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_3 = {2: State(position=34, time_step=2), 3: State(position=36, time_step=3)}
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(state_list_lon_other_3, state_list_lat_other_3, Rectangle(5, 2),
                                  cr_state_list_other_3, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_3, None, None, None)

        other_vehicles = [other_vehicle_1, other_vehicle_2, other_vehicle_3]

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates.in_congestion(0, ego_vehicle, other_vehicles)
        sol_monitor_mode_2 = general_predicates.in_congestion(1, ego_vehicle, other_vehicles)
        sol_monitor_mode_3 = general_predicates.in_congestion(2, ego_vehicle, other_vehicles)
        sol_monitor_mode_4 = general_predicates.in_congestion(3, ego_vehicle, other_vehicles)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

    def test_in_slow_moving_traffic(self):
        self._traffic_rule_param["num_veh_slow_moving_traffic"] = 3
        self._traffic_rule_param["max_slow_moving_traffic_velocity"] = 8.33

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle
        exp_sol_monitor_mode_2 = False  # only two leading vehicles
        exp_sol_monitor_mode_3 = False  # three leading vehicle, but not all drive with required velocity
        exp_sol_monitor_mode_4 = True

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=6), 1: StateLongitudinal(s=6, v=6),
                              2: StateLongitudinal(s=12, v=6), 3: StateLongitudinal(s=18, v=6)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=6, time_step=1),
                             2: State(position=12, time_step=2), 3: State(position=18, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {1: StateLongitudinal(s=16, v=6),
                                  2: StateLongitudinal(s=22, v=6), 3: StateLongitudinal(s=28, v=6)}
        state_list_lat_other_1 = {1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {1: State(position=16, time_step=1),
                                 2: State(position=22, time_step=2), 3: State(position=28, time_step=3)}
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {1: StateLongitudinal(s=26, v=6),
                                  2: StateLongitudinal(s=32, v=6), 3: StateLongitudinal(s=38, v=6)}
        state_list_lat_other_2 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_2 = {1: State(position=22, time_step=1),
                                 2: State(position=24, time_step=2), 3: State(position=26, time_step=3)}
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        # other vehicle 3
        state_list_lon_other_3 = {2: StateLongitudinal(s=42, v=12), 3: StateLongitudinal(s=48, v=6)}
        state_list_lat_other_3 = {2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_3 = {2: State(position=34, time_step=2), 3: State(position=36, time_step=3)}
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(state_list_lon_other_3, state_list_lat_other_3, Rectangle(5, 2),
                                  cr_state_list_other_3, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_3, None, None, None)

        other_vehicles = [other_vehicle_1, other_vehicle_2, other_vehicle_3]

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates.in_slow_moving_traffic(0, ego_vehicle, other_vehicles)
        sol_monitor_mode_2 = general_predicates.in_slow_moving_traffic(1, ego_vehicle, other_vehicles)
        sol_monitor_mode_3 = general_predicates.in_slow_moving_traffic(2, ego_vehicle, other_vehicles)
        sol_monitor_mode_4 = general_predicates.in_slow_moving_traffic(3, ego_vehicle, other_vehicles)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

    def test_in_queue_of_vehicles(self):
        self._traffic_rule_param["num_veh_queue_of_vehicles"] = 3
        self._traffic_rule_param["max_queue_of_vehicles_velocity"] = 16.67

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle
        exp_sol_monitor_mode_2 = False  # only two leading vehicles
        exp_sol_monitor_mode_3 = False  # three leading vehicle, but not all drive with required velocity
        exp_sol_monitor_mode_4 = True

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=15), 1: StateLongitudinal(s=15, v=15),
                              2: StateLongitudinal(s=30, v=15), 3: StateLongitudinal(s=45, v=15)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=15, time_step=1),
                             2: State(position=30, time_step=2), 3: State(position=45, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {1: StateLongitudinal(s=30, v=15),
                                  2: StateLongitudinal(s=45, v=15), 3: StateLongitudinal(s=60, v=15)}
        state_list_lat_other_1 = {1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {1: State(position=30, time_step=1),
                                 2: State(position=45, time_step=2), 3: State(position=60, time_step=3)}
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {1: StateLongitudinal(s=45, v=15),
                                  2: StateLongitudinal(s=60, v=15), 3: StateLongitudinal(s=75, v=15)}
        state_list_lat_other_2 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_2 = {1: State(position=45, time_step=1),
                                 2: State(position=60, time_step=2), 3: State(position=75, time_step=3)}
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        # other vehicle 3
        state_list_lon_other_3 = {2: StateLongitudinal(s=75, v=20), 3: StateLongitudinal(s=80, v=15)}
        state_list_lat_other_3 = {2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_3 = {2: State(position=75, time_step=2), 3: State(position=80, time_step=3)}
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(state_list_lon_other_3, state_list_lat_other_3, Rectangle(5, 2),
                                  cr_state_list_other_3, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_3, None, None, None)

        other_vehicles = [other_vehicle_1, other_vehicle_2, other_vehicle_3]

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates.in_queue_of_vehicles(0, ego_vehicle, other_vehicles)
        sol_monitor_mode_2 = general_predicates.in_queue_of_vehicles(1, ego_vehicle, other_vehicles)
        sol_monitor_mode_3 = general_predicates.in_queue_of_vehicles(2, ego_vehicle, other_vehicles)
        sol_monitor_mode_4 = general_predicates.in_queue_of_vehicles(3, ego_vehicle, other_vehicles)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)

    def test_adjacent_laneelts(self):
        # expected solutions -> number of adjacent lanelts
        exp_sol_1 = 3
        exp_sol_2 = 3
        exp_sol_3 = 3
        exp_sol_4 = 1

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)

        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

        sol_1 = len(general_predicates._adjacent_lanelets(lanelet_network.lanelets[0]))
        sol_2 = len(general_predicates._adjacent_lanelets(lanelet_network.lanelets[1]))
        sol_3 = len(general_predicates._adjacent_lanelets(lanelet_network.lanelets[2]))
        sol_4 = len(general_predicates._adjacent_lanelets(lanelet_network.lanelets[3]))

        self.assertEqual(exp_sol_1, sol_1)
        self.assertEqual(exp_sol_2, sol_2)
        self.assertEqual(exp_sol_3, sol_3)
        self.assertEqual(exp_sol_4, sol_4)

    def test_interstate_broad_enough(self):
        # expected solutions
        exp_sol_1 = True
        exp_sol_2 = True
        exp_sol_3 = False

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)

        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=16, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {4}, 3: {4}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        sol_1 = general_predicates.interstate_broad_enough(0, ego_vehicle)
        sol_2 = general_predicates.interstate_broad_enough(1, ego_vehicle)
        sol_3 = general_predicates.interstate_broad_enough(2, ego_vehicle)

        self.assertEqual(exp_sol_1, sol_1)
        self.assertEqual(exp_sol_2, sol_2)
        self.assertEqual(exp_sol_3, sol_3)

    def test_cut_in(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # before cut-in -> ego vehicle occupies only single lane
        exp_sol_monitor_mode_2 = True  # during cut-in
        exp_sol_monitor_mode_3 = False  # after cut-in
        exp_sol_monitor_mode_4 = False  # driving back to initial lane
        exp_sol_monitor_mode_5 = False  # during cut-in -> but other vehicles is in another lane

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=10, v=10), 1: StateLongitudinal(s=20, v=10),
                              2: StateLongitudinal(s=30, v=10), 3: StateLongitudinal(s=40, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=2, theta=(1/4)*math.pi),
                              2: StateLateral(d=4, theta=0), 3: StateLateral(d=2, theta=-(1/4)*math.pi)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1, 2}, 2: {1, 2}, 3: {1, 2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                                  2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10)}
        state_list_lat_other_1 = {0: StateLateral(d=4, theta=0), 1: StateLateral(d=4, theta=0),
                                  2: StateLateral(d=4, theta=0), 3: StateLateral(d=4, theta=0)}
        cr_state_list_other_1 = {0: State(position=10, time_step=0), 1: State(position=20, time_step=1),
                                 2: State(position=30, time_step=2), 3: State(position=40, time_step=3)}
        lanelet_assignments_other_1 = {0: {2}, 1: {2}, 2: {2}, 3: {2}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {1: StateLongitudinal(s=0, v=10)}
        state_list_lat_other_2 = {1: StateLateral(d=10, theta=0)}
        cr_state_list_other_2 = {1: State(position=10, time_step=1)}
        lanelet_assignments_other_2 = {1: {3}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 0, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates.cut_in(0, ego_vehicle, other_vehicle_1)
        sol_monitor_mode_2 = general_predicates.cut_in(1, ego_vehicle, other_vehicle_1)
        sol_monitor_mode_3 = general_predicates.cut_in(2, ego_vehicle, other_vehicle_1)
        sol_monitor_mode_4 = general_predicates.cut_in(3, ego_vehicle, other_vehicle_1)
        sol_monitor_mode_5 = general_predicates.cut_in(1, ego_vehicle, other_vehicle_2)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

    def test_road_width(self):
        # expected solutions
        exp_sol_monitor_mode_1 = 12
        exp_sol_monitor_mode_2 = 12
        exp_sol_monitor_mode_3 = 12
        exp_sol_monitor_mode_4 = 8
        exp_sol_monitor_mode_5 = 6.200089993250562
        exp_sol_monitor_mode_6 = 8
        exp_sol_monitor_mode_7 = 6.200089993250562
        exp_sol_monitor_mode_8 = 7.800009999250063
        exp_sol_monitor_mode_9 = 6.8000599955003755
        exp_sol_monitor_mode_10 = 7.800009999250063
        exp_sol_monitor_mode_11 = 6.8000599955003755

        lanelet_network = LaneletNetwork()
        right_vertices_lane_4 = np.array([[0, 14], [10, 14], [20, 14], [30, 14], [40, 14], [50, 14], [60, 14], [70, 14],
                                          [80, 14], [90, 14]])
        left_vertices_lane_4 = np.array([[0, 18], [10, 18], [20, 18], [30, 18], [40, 18], [50, 18], [60, 18], [70, 18],
                                         [80, 18], [90, 18]])
        center_vertices_lane_4 = np.array([[0, 16], [10, 16], [20, 16], [30, 16], [40, 16], [50, 16], [60, 16],
                                           [70, 16], [80, 16], [90, 16]])
        lanelet_4 = Lanelet(left_vertices_lane_4, center_vertices_lane_4, right_vertices_lane_4, lanelet_id=4,
                            adjacent_right=5, adjacent_right_same_direction=True)

        right_vertices_lane_5 = np.array([[0, 18], [10, 18], [20, 18], [30, 18], [40, 18], [50, 18], [60, 18], [70, 18],
                                          [80, 18], [90, 18]])
        left_vertices_lane_5 = np.array([[0, 22], [10, 21.8], [20, 21.6], [30, 21.4], [40, 21.2], [50, 21.0],
                                         [60, 20.8], [70, 20.6], [80, 20.4], [90, 20.2]])
        center_vertices_lane_5 = np.array([[0, 20], [10, 19.9], [20, 19.8], [30, 19.7], [40, 19.6], [50, 19.5],
                                           [60, 19.4], [70, 19.3], [80, 19.2], [90, 19.1]])
        lanelet_5 = Lanelet(left_vertices_lane_5, center_vertices_lane_5, right_vertices_lane_5, lanelet_id=5,
                            adjacent_left=4, adjacent_left_same_direction=True)
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(lanelet_4)
        lanelet_network.add_lanelet(lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        general_predicates = GeneralPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        set(), traffic_sign_interpreter)

        # Monitor-Mode
        sol_monitor_mode_1 = general_predicates._road_width(lanelet_network.lanelets[0], 2)
        sol_monitor_mode_2 = general_predicates._road_width(lanelet_network.lanelets[1], 2)
        sol_monitor_mode_3 = general_predicates._road_width(lanelet_network.lanelets[2], 2)
        sol_monitor_mode_4 = general_predicates._road_width(lanelet_network.lanelets[3], 0)
        sol_monitor_mode_5 = general_predicates._road_width(lanelet_network.lanelets[3], 90)
        sol_monitor_mode_6 = general_predicates._road_width(lanelet_network.lanelets[4], 0)
        sol_monitor_mode_7 = general_predicates._road_width(lanelet_network.lanelets[4], 90)
        sol_monitor_mode_8 = general_predicates._road_width(lanelet_network.lanelets[3], 10)
        sol_monitor_mode_9 = general_predicates._road_width(lanelet_network.lanelets[3], 60)
        sol_monitor_mode_10 = general_predicates._road_width(lanelet_network.lanelets[4], 10)
        sol_monitor_mode_11 = general_predicates._road_width(lanelet_network.lanelets[4], 60)


        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode_7)
        self.assertEqual(exp_sol_monitor_mode_8, sol_monitor_mode_8)
        self.assertEqual(exp_sol_monitor_mode_9, sol_monitor_mode_9)
        self.assertEqual(exp_sol_monitor_mode_10, sol_monitor_mode_10)
        self.assertEqual(exp_sol_monitor_mode_11, sol_monitor_mode_11)
