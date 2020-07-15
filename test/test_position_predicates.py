import unittest
import os
import numpy as np

from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.scenario.lanelet import LaneletNetwork

from src.predicates.position_predicates import PositionPredicateCollection
from src.common.helper import *
from src.common.road_network import RoadNetwork


class TestPositionPredicates(unittest.TestCase):
    def setUp(self):
        config_path = os.path.dirname(os.path.abspath(__file__)) + "/../src/"
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 1.0, 'DEU')

        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._ego_vehicle_param = create_other_vehicles_param(config.get("ego_vehicle_param"))
        self._traffic_rules_param = traffic_rules.get("traffic_rules_param")
        self._road_network_param = config.get("road_network_param")

        right_vertices_lane_1 = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0],
                                          [80, 1], [90, 0]])
        left_vertices_lane_1 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4],
                                         [80, 1], [90, 0]])
        center_vertices_lane_1 = np.array([[0, 2], [10, 2], [20, 2], [30, 2], [40, 2], [50, 2], [60, 2], [70, 2],
                                           [80, 1], [90, 0]])
        self._lanelet_1 = Lanelet(left_vertices_lane_1, center_vertices_lane_1, right_vertices_lane_1, lanelet_id=1,
                                  adjacent_left=2, adjacent_left_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.SHOULDER})

        right_vertices_lane_2 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4],
                                          [80, 4], [90, 4]])
        left_vertices_lane_2 = np.array([[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8],
                                         [80, 8], [90, 8]])
        center_vertices_lane_2 = np.array([[0, 6], [10, 6], [20, 6], [30, 6], [40, 6], [50, 6], [60, 6],
                                           [70, 6], [80, 6], [90, 6]])
        self._lanelet_2 = Lanelet(left_vertices_lane_2, center_vertices_lane_2, right_vertices_lane_2, lanelet_id=2,
                                  adjacent_left=3, adjacent_left_same_direction=True,
                                  adjacent_right=1, adjacent_right_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE})

        right_vertices_lane_3 = np.array([[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8],
                                          [80, 8], [90, 8]])
        left_vertices_lane_3 = np.array([[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12], [70, 12],
                                         [80, 12], [90, 12]])
        center_vertices_lane_3 = np.array([[0, 10], [10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [60, 10],
                                           [70, 10], [80, 10], [90, 10]])
        self._lanelet_3 = Lanelet(left_vertices_lane_3, center_vertices_lane_3, right_vertices_lane_3, lanelet_id=3,
                                  adjacent_left=4, adjacent_left_same_direction=True,
                                  adjacent_right=2, adjacent_right_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY})

        right_vertices_lane_4 = np.array([[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12], [70, 12],
                                          [80, 12], [90, 12]])
        left_vertices_lane_4 = np.array([[0, 16], [10, 16], [20, 16], [30, 16], [40, 16], [50, 16], [60, 16], [70, 16],
                                         [80, 16], [90, 16]])
        center_vertices_lane_4 = np.array([[0, 14], [10, 14], [20, 14], [30, 14], [40, 14], [50, 14], [60, 14],
                                           [70, 14], [80, 14], [90, 14]])
        self._lanelet_4 = Lanelet(left_vertices_lane_4, center_vertices_lane_4, right_vertices_lane_4, lanelet_id=4,
                                  adjacent_left=5, adjacent_left_same_direction=True,
                                  adjacent_right=3, adjacent_right_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.EXIT_RAMP})

        right_vertices_lane_5 = np.array([[0, 16], [10, 16], [20, 16], [30, 16], [40, 16], [50, 16], [60, 16], [70, 16],
                                          [80, 16], [90, 16]])
        left_vertices_lane_5 = np.array([[0, 20], [10, 20], [20, 20], [30, 20], [40, 20], [50, 20], [60, 20], [70, 20],
                                         [80, 20], [90, 20]])
        center_vertices_lane_5 = np.array([[0, 18], [10, 18], [20, 18], [30, 18], [40, 18], [50, 18], [60, 18],
                                           [70, 18], [80, 18], [90, 18]])
        self._lanelet_5 = Lanelet(left_vertices_lane_5, center_vertices_lane_5, right_vertices_lane_5, lanelet_id=5,
                                  adjacent_right=4, adjacent_right_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.ACCESS_RAMP})

    def test_in_front_of(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle behind
        exp_sol_monitor_mode_2 = False  # ego vehicle and other vehicle have same occupancy
        exp_sol_monitor_mode_3 = False   # ego vehicle is not completely in front
        exp_sol_monitor_mode_4 = True  # ego vehicle is in front in same lane
        exp_sol_monitor_mode_5 = True  # ego vehicle is in front in another lane
        exp_sol_constraint_mode_1 = 10.5
        exp_sol_constraint_mode_2 = 12.5
        exp_sol_constraint_mode_3 = 14.5
        exp_sol_constraint_mode_4 = 16.5
        exp_sol_constraint_mode_5 = 12.5
        exp_sol_robustness_mode_1 = -13.0
        exp_sol_robustness_mode_2 = -5.0
        exp_sol_robustness_mode_3 = -3.0
        exp_sol_robustness_mode_4 = 5.0
        exp_sol_robustness_mode_5 = 14.0

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=4),
                              2: StateLongitudinal(s=14, v=10), 3: StateLongitudinal(s=24, v=5),
                              4: StateLongitudinal(s=29, v=5)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                              2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0),
                              4: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=14, time_step=2), 3: State(position=24, time_step=3),
                             4: State(position=29, time_step=4)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {0: StateLongitudinal(s=8, v=2),  1: StateLongitudinal(s=10, v=2),
                                  2: StateLongitudinal(s=12, v=2), 3: StateLongitudinal(s=14, v=2)}
        state_list_lat_other_1 = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=0, theta=0),
                                  2: StateLateral(d=0, theta=0), 3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {0: State(position=10, time_step=1), 1: State(position=10, time_step=1),
                                 2: State(position=20, time_step=2), 3: State(position=30, time_step=3)}
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 41, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {4: StateLongitudinal(s=10, v=10)}
        state_list_lat_other_2 = {4: StateLateral(d=4, theta=0)}
        cr_state_list_other_2 = {4: State(position=10, time_step=4)}
        lanelet_assignments_other_2 = {4: {2}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 42, ObstacleType.CAR, self._ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = position_predicates.in_front_of(0, other_vehicle_1, ego_vehicle,
                                                             OperatingMode.MONITOR)
        sol_monitor_mode_2 = position_predicates.in_front_of(1, other_vehicle_1, ego_vehicle,
                                                             OperatingMode.MONITOR)
        sol_monitor_mode_3 = position_predicates.in_front_of(2, other_vehicle_1, ego_vehicle,
                                                             OperatingMode.MONITOR)
        sol_monitor_mode_4 = position_predicates.in_front_of(3, other_vehicle_1, ego_vehicle,
                                                             OperatingMode.MONITOR)
        sol_monitor_mode_5 = position_predicates.in_front_of(4, other_vehicle_2, ego_vehicle,
                                                             OperatingMode.MONITOR)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

        # Constraint-Mode
        sol_constraint_mode_1 = position_predicates.in_front_of(0, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.CONSTRAINT)
        sol_constraint_mode_2 = position_predicates.in_front_of(1, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.CONSTRAINT)
        sol_constraint_mode_3 = position_predicates.in_front_of(2, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.CONSTRAINT)
        sol_constraint_mode_4 = position_predicates.in_front_of(3, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.CONSTRAINT)
        sol_constraint_mode_5 = position_predicates.in_front_of(4, other_vehicle_2, ego_vehicle,
                                                                OperatingMode.CONSTRAINT)

        self.assertEqual(exp_sol_constraint_mode_1, sol_constraint_mode_1)
        self.assertEqual(exp_sol_constraint_mode_2, sol_constraint_mode_2)
        self.assertEqual(exp_sol_constraint_mode_3, sol_constraint_mode_3)
        self.assertEqual(exp_sol_constraint_mode_4, sol_constraint_mode_4)
        self.assertEqual(exp_sol_constraint_mode_5, sol_constraint_mode_5)

        # Robustness-Mode
        sol_robustness_mode_1 = position_predicates.in_front_of(0, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.ROBUSTNESS)
        sol_robustness_mode_2 = position_predicates.in_front_of(1, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.ROBUSTNESS)
        sol_robustness_mode_3 = position_predicates.in_front_of(2, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.ROBUSTNESS)
        sol_robustness_mode_4 = position_predicates.in_front_of(3, other_vehicle_1, ego_vehicle,
                                                                OperatingMode.ROBUSTNESS)
        sol_robustness_mode_5 = position_predicates.in_front_of(4, other_vehicle_2, ego_vehicle,
                                                                OperatingMode.ROBUSTNESS)

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)
        self.assertEqual(exp_sol_robustness_mode_3, sol_robustness_mode_3)
        self.assertEqual(exp_sol_robustness_mode_4, sol_robustness_mode_4)
        self.assertEqual(exp_sol_robustness_mode_5, sol_robustness_mode_5)

    def test_on_shoulder(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = False   # ego vehicle on main carriageway
        exp_sol_monitor_mode_4 = False  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = False  # ego vehicle on access ramp

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = position_predicates.on_shoulder(0, ego_vehicle)
        sol_monitor_mode_2 = position_predicates.on_shoulder(1, ego_vehicle)
        sol_monitor_mode_3 = position_predicates.on_shoulder(2, ego_vehicle)
        sol_monitor_mode_4 = position_predicates.on_shoulder(3, ego_vehicle)
        sol_monitor_mode_5 = position_predicates.on_shoulder(4, ego_vehicle)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

    def test_on_access_ramp(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = False   # ego vehicle on main carriageway
        exp_sol_monitor_mode_4 = False  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = True  # ego vehicle on access ramp

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = position_predicates.on_access_ramp(0, ego_vehicle)
        sol_monitor_mode_2 = position_predicates.on_access_ramp(1, ego_vehicle)
        sol_monitor_mode_3 = position_predicates.on_access_ramp(2, ego_vehicle)
        sol_monitor_mode_4 = position_predicates.on_access_ramp(3, ego_vehicle)
        sol_monitor_mode_5 = position_predicates.on_access_ramp(4, ego_vehicle)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

    def test_on_exit_ramp(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = False   # ego vehicle on main carriageway
        exp_sol_monitor_mode_4 = True  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = False  # ego vehicle on access ramp

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = position_predicates.on_exit_ramp(0, ego_vehicle)
        sol_monitor_mode_2 = position_predicates.on_exit_ramp(1, ego_vehicle)
        sol_monitor_mode_3 = position_predicates.on_exit_ramp(2, ego_vehicle)
        sol_monitor_mode_4 = position_predicates.on_exit_ramp(3, ego_vehicle)
        sol_monitor_mode_5 = position_predicates.on_exit_ramp(4, ego_vehicle)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

    def test_on_main_carriage_way(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = True   # ego vehicle on main carriageway
        exp_sol_monitor_mode_4 = False  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = False  # ego vehicle on access ramp

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = position_predicates.on_main_carriage_way(0, ego_vehicle)
        sol_monitor_mode_2 = position_predicates.on_main_carriage_way(1, ego_vehicle)
        sol_monitor_mode_3 = position_predicates.on_main_carriage_way(2, ego_vehicle)
        sol_monitor_mode_4 = position_predicates.on_main_carriage_way(3, ego_vehicle)
        sol_monitor_mode_5 = position_predicates.on_main_carriage_way(4, ego_vehicle)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

    def test_lanelets_left_of_vehicle(self):
        # expected solutions
        exp_sol_num_lanelets_1 = 4  # 4 lanelets left at t=0
        exp_sol_lanelet_ids_1 = {2, 3, 4, 5}  # IDs of lanelets at t=0
        exp_sol_num_lanelets_2 = 3  # 3 lanelets left at t=1
        exp_sol_lanelet_ids_2 = {3, 4, 5}  # IDs of lanelets at t=1
        exp_sol_num_lanelets_3 = 2   # 2 lanelets left at t=2
        exp_sol_lanelet_ids_3 = {4, 5}  # IDs of lanelets at t=2
        exp_sol_num_lanelets_4 = 1  # 1 lanelets left at t=3
        exp_sol_lanelet_ids_4 = {5}  # IDs of lanelets at t=3
        exp_sol_num_lanelets_5 = 0  # 0 lanelets left at t=4
        exp_sol_lanelet_ids_5 = set()  # IDs of lanelets at t=4
        exp_sol_num_lanelets_6 = 3  # 3 lanelets left at t=5
        exp_sol_lanelet_ids_6 = {3, 4, 5}  # IDs of lanelets at t=5

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10), 5: StateLongitudinal(s=50, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0), 5: StateLateral(d=6, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4), 5: State(position=50, time_step=5)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}, 5: {2, 3}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Evaluation
        lanelets_1 = position_predicates._lanelets_left_of_vehicle(0, ego_vehicle)
        sol_num_lanelets_1 = len(lanelets_1)
        sol_lanelet_ids_1 = {l.lanelet_id for l in lanelets_1}
        lanelets_2 = position_predicates._lanelets_left_of_vehicle(1, ego_vehicle)
        sol_num_lanelets_2 = len(lanelets_2)
        sol_lanelet_ids_2 = {l.lanelet_id for l in lanelets_2}
        lanelets_3 = position_predicates._lanelets_left_of_vehicle(2, ego_vehicle)
        sol_num_lanelets_3 = len(lanelets_3)
        sol_lanelet_ids_3 = {l.lanelet_id for l in lanelets_3}
        lanelets_4 = position_predicates._lanelets_left_of_vehicle(3, ego_vehicle)
        sol_num_lanelets_4 = len(lanelets_4)
        sol_lanelet_ids_4 = {l.lanelet_id for l in lanelets_4}
        lanelets_5 = position_predicates._lanelets_left_of_vehicle(4, ego_vehicle)
        sol_num_lanelets_5 = len(lanelets_5)
        sol_lanelet_ids_5 = {l.lanelet_id for l in lanelets_5}
        lanelets_6 = position_predicates._lanelets_left_of_vehicle(5, ego_vehicle)
        sol_num_lanelets_6 = len(lanelets_6)
        sol_lanelet_ids_6 = {l.lanelet_id for l in lanelets_6}

        self.assertEqual(exp_sol_num_lanelets_1, sol_num_lanelets_1)
        self.assertEqual(exp_sol_lanelet_ids_1, sol_lanelet_ids_1)
        self.assertEqual(exp_sol_num_lanelets_2, sol_num_lanelets_2)
        self.assertEqual(exp_sol_lanelet_ids_2, sol_lanelet_ids_2)
        self.assertEqual(exp_sol_num_lanelets_3, sol_num_lanelets_3)
        self.assertEqual(exp_sol_lanelet_ids_3, sol_lanelet_ids_3)
        self.assertEqual(exp_sol_num_lanelets_4, sol_num_lanelets_4)
        self.assertEqual(exp_sol_lanelet_ids_4, sol_lanelet_ids_4)
        self.assertEqual(exp_sol_num_lanelets_5, sol_num_lanelets_5)
        self.assertEqual(exp_sol_lanelet_ids_5, sol_lanelet_ids_5)
        self.assertEqual(exp_sol_num_lanelets_6, sol_num_lanelets_6)
        self.assertEqual(exp_sol_lanelet_ids_6, sol_lanelet_ids_6)

    def test_lanelets_right_of_vehicle(self):
        # expected solutions
        exp_sol_num_lanelets_1 = 0  # 4 lanelets left at t=0
        exp_sol_lanelet_ids_1 = set()  # IDs of lanelets at t=0
        exp_sol_num_lanelets_2 = 1  # 3 lanelets left at t=1
        exp_sol_lanelet_ids_2 = {1}  # IDs of lanelets at t=1
        exp_sol_num_lanelets_3 = 2   # 2 lanelets left at t=2
        exp_sol_lanelet_ids_3 = {1, 2}  # IDs of lanelets at t=2
        exp_sol_num_lanelets_4 = 3  # 1 lanelets left at t=3
        exp_sol_lanelet_ids_4 = {1, 2, 3}  # IDs of lanelets at t=3
        exp_sol_num_lanelets_5 = 4  # 0 lanelets left at t=4
        exp_sol_lanelet_ids_5 = {1, 2, 3, 4}  # IDs of lanelets at t=4
        exp_sol_num_lanelets_6 = 2  # 3 lanelets left at t=5
        exp_sol_lanelet_ids_6 = {1, 2}  # IDs of lanelets at t=5

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10), 5: StateLongitudinal(s=50, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0), 5: StateLateral(d=6, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4), 5: State(position=50, time_step=5)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}, 5: {2, 3}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Evaluation
        lanelets_1 = position_predicates._lanelets_right_of_vehicle(0, ego_vehicle)
        sol_num_lanelets_1 = len(lanelets_1)
        sol_lanelet_ids_1 = {l.lanelet_id for l in lanelets_1}
        lanelets_2 = position_predicates._lanelets_right_of_vehicle(1, ego_vehicle)
        sol_num_lanelets_2 = len(lanelets_2)
        sol_lanelet_ids_2 = {l.lanelet_id for l in lanelets_2}
        lanelets_3 = position_predicates._lanelets_right_of_vehicle(2, ego_vehicle)
        sol_num_lanelets_3 = len(lanelets_3)
        sol_lanelet_ids_3 = {l.lanelet_id for l in lanelets_3}
        lanelets_4 = position_predicates._lanelets_right_of_vehicle(3, ego_vehicle)
        sol_num_lanelets_4 = len(lanelets_4)
        sol_lanelet_ids_4 = {l.lanelet_id for l in lanelets_4}
        lanelets_5 = position_predicates._lanelets_right_of_vehicle(4, ego_vehicle)
        sol_num_lanelets_5 = len(lanelets_5)
        sol_lanelet_ids_5 = {l.lanelet_id for l in lanelets_5}
        lanelets_6 = position_predicates._lanelets_right_of_vehicle(5, ego_vehicle)
        sol_num_lanelets_6 = len(lanelets_6)
        sol_lanelet_ids_6 = {l.lanelet_id for l in lanelets_6}

        self.assertEqual(exp_sol_num_lanelets_1, sol_num_lanelets_1)
        self.assertEqual(exp_sol_lanelet_ids_1, sol_lanelet_ids_1)
        self.assertEqual(exp_sol_num_lanelets_2, sol_num_lanelets_2)
        self.assertEqual(exp_sol_lanelet_ids_2, sol_lanelet_ids_2)
        self.assertEqual(exp_sol_num_lanelets_3, sol_num_lanelets_3)
        self.assertEqual(exp_sol_lanelet_ids_3, sol_lanelet_ids_3)
        self.assertEqual(exp_sol_num_lanelets_4, sol_num_lanelets_4)
        self.assertEqual(exp_sol_lanelet_ids_4, sol_lanelet_ids_4)
        self.assertEqual(exp_sol_num_lanelets_5, sol_num_lanelets_5)
        self.assertEqual(exp_sol_lanelet_ids_5, sol_lanelet_ids_5)
        self.assertEqual(exp_sol_num_lanelets_6, sol_num_lanelets_6)
        self.assertEqual(exp_sol_lanelet_ids_6, sol_lanelet_ids_6)

    def test_lanelets_right_of_lanelet(self):
        # expected solutions
        exp_sol_num_lanelets_1 = 0  # 4 lanelets left at t=0
        exp_sol_lanelet_ids_1 = set()  # IDs of lanelets at t=0
        exp_sol_num_lanelets_2 = 1  # 3 lanelets left at t=1
        exp_sol_lanelet_ids_2 = {1}  # IDs of lanelets at t=1
        exp_sol_num_lanelets_3 = 2   # 2 lanelets left at t=2
        exp_sol_lanelet_ids_3 = {1, 2}  # IDs of lanelets at t=2
        exp_sol_num_lanelets_4 = 3  # 1 lanelets left at t=3
        exp_sol_lanelet_ids_4 = {1, 2, 3}  # IDs of lanelets at t=3
        exp_sol_num_lanelets_5 = 4  # 0 lanelets left at t=4
        exp_sol_lanelet_ids_5 = {1, 2, 3, 4}  # IDs of lanelets at t=4


        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # Evaluation
        lanelets_1 = position_predicates._lanelets_right_of_lanelet(self._lanelet_1)
        sol_num_lanelets_1 = len(lanelets_1)
        sol_lanelet_ids_1 = {l.lanelet_id for l in lanelets_1}
        lanelets_2 = position_predicates._lanelets_right_of_lanelet(self._lanelet_2)
        sol_num_lanelets_2 = len(lanelets_2)
        sol_lanelet_ids_2 = {l.lanelet_id for l in lanelets_2}
        lanelets_3 = position_predicates._lanelets_right_of_lanelet(self._lanelet_3)
        sol_num_lanelets_3 = len(lanelets_3)
        sol_lanelet_ids_3 = {l.lanelet_id for l in lanelets_3}
        lanelets_4 = position_predicates._lanelets_right_of_lanelet(self._lanelet_4)
        sol_num_lanelets_4 = len(lanelets_4)
        sol_lanelet_ids_4 = {l.lanelet_id for l in lanelets_4}
        lanelets_5 = position_predicates._lanelets_right_of_lanelet(self._lanelet_5)
        sol_num_lanelets_5 = len(lanelets_5)
        sol_lanelet_ids_5 = {l.lanelet_id for l in lanelets_5}

        self.assertEqual(exp_sol_num_lanelets_1, sol_num_lanelets_1)
        self.assertEqual(exp_sol_lanelet_ids_1, sol_lanelet_ids_1)
        self.assertEqual(exp_sol_num_lanelets_2, sol_num_lanelets_2)
        self.assertEqual(exp_sol_lanelet_ids_2, sol_lanelet_ids_2)
        self.assertEqual(exp_sol_num_lanelets_3, sol_num_lanelets_3)
        self.assertEqual(exp_sol_lanelet_ids_3, sol_lanelet_ids_3)
        self.assertEqual(exp_sol_num_lanelets_4, sol_num_lanelets_4)
        self.assertEqual(exp_sol_lanelet_ids_4, sol_lanelet_ids_4)
        self.assertEqual(exp_sol_num_lanelets_5, sol_num_lanelets_5)
        self.assertEqual(exp_sol_lanelet_ids_5, sol_lanelet_ids_5)

    def test_lanelets_left_of_lanelet(self):
        # expected solutions
        exp_sol_num_lanelets_1 = 4  # 4 lanelets left at t=0
        exp_sol_lanelet_ids_1 = {2, 3, 4, 5}  # IDs of lanelets at t=0
        exp_sol_num_lanelets_2 = 3  # 3 lanelets left at t=1
        exp_sol_lanelet_ids_2 = {3, 4, 5}  # IDs of lanelets at t=1
        exp_sol_num_lanelets_3 = 2   # 2 lanelets left at t=2
        exp_sol_lanelet_ids_3 = {4, 5}  # IDs of lanelets at t=2
        exp_sol_num_lanelets_4 = 1  # 1 lanelets left at t=3
        exp_sol_lanelet_ids_4 = {5}  # IDs of lanelets at t=3
        exp_sol_num_lanelets_5 = 0  # 0 lanelets left at t=4
        exp_sol_lanelet_ids_5 = set()  # IDs of lanelets at t=4

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # Evaluation
        lanelets_1 = position_predicates._lanelets_left_of_lanelet(self._lanelet_1)
        sol_num_lanelets_1 = len(lanelets_1)
        sol_lanelet_ids_1 = {l.lanelet_id for l in lanelets_1}
        lanelets_2 = position_predicates._lanelets_left_of_lanelet(self._lanelet_2)
        sol_num_lanelets_2 = len(lanelets_2)
        sol_lanelet_ids_2 = {l.lanelet_id for l in lanelets_2}
        lanelets_3 = position_predicates._lanelets_left_of_lanelet(self._lanelet_3)
        sol_num_lanelets_3 = len(lanelets_3)
        sol_lanelet_ids_3 = {l.lanelet_id for l in lanelets_3}
        lanelets_4 = position_predicates._lanelets_left_of_lanelet(self._lanelet_4)
        sol_num_lanelets_4 = len(lanelets_4)
        sol_lanelet_ids_4 = {l.lanelet_id for l in lanelets_4}
        lanelets_5 = position_predicates._lanelets_left_of_lanelet(self._lanelet_5)
        sol_num_lanelets_5 = len(lanelets_5)
        sol_lanelet_ids_5 = {l.lanelet_id for l in lanelets_5}

        self.assertEqual(exp_sol_num_lanelets_1, sol_num_lanelets_1)
        self.assertEqual(exp_sol_lanelet_ids_1, sol_lanelet_ids_1)
        self.assertEqual(exp_sol_num_lanelets_2, sol_num_lanelets_2)
        self.assertEqual(exp_sol_lanelet_ids_2, sol_lanelet_ids_2)
        self.assertEqual(exp_sol_num_lanelets_3, sol_num_lanelets_3)
        self.assertEqual(exp_sol_lanelet_ids_3, sol_lanelet_ids_3)
        self.assertEqual(exp_sol_num_lanelets_4, sol_num_lanelets_4)
        self.assertEqual(exp_sol_lanelet_ids_4, sol_lanelet_ids_4)
        self.assertEqual(exp_sol_num_lanelets_5, sol_num_lanelets_5)
        self.assertEqual(exp_sol_lanelet_ids_5, sol_lanelet_ids_5)

    def test_in_leftmost_lane(self):  # TODO: add constraint and robustness mode
        # expected solutions
        exp_sol_monitor_mode_1 = False
        exp_sol_monitor_mode_2 = False
        exp_sol_monitor_mode_3 = False
        exp_sol_monitor_mode_4 = False
        exp_sol_monitor_mode_5 = True
        exp_sol_monitor_mode_6 = True
        exp_sol_monitor_mode_7 = False

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10), 5: StateLongitudinal(s=50, v=10),
                              6: StateLongitudinal(s=60, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0), 5: StateLateral(d=14, theta=0),
                              6: StateLateral(d=6, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4), 5: State(position=50, time_step=5),
                             6: State(position=60, time_step=6)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}, 5: {4, 5}, 6: {2, 3}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = position_predicates.in_leftmost_lane(0, ego_vehicle)
        sol_monitor_mode_2 = position_predicates.in_leftmost_lane(1, ego_vehicle)
        sol_monitor_mode_3 = position_predicates.in_leftmost_lane(2, ego_vehicle)
        sol_monitor_mode_4 = position_predicates.in_leftmost_lane(3, ego_vehicle)
        sol_monitor_mode_5 = position_predicates.in_leftmost_lane(4, ego_vehicle)
        sol_monitor_mode_6 = position_predicates.in_leftmost_lane(5, ego_vehicle)
        sol_monitor_mode_7 = position_predicates.in_leftmost_lane(6, ego_vehicle)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode_7)

    def test_in_rightmost_lane(self):  # TODO: add constraint and robustness mode
        # expected solutions
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = False
        exp_sol_monitor_mode_3 = False
        exp_sol_monitor_mode_4 = False
        exp_sol_monitor_mode_5 = False
        exp_sol_monitor_mode_6 = False
        exp_sol_monitor_mode_7 = True

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        road_network = RoadNetwork(lanelet_network, self._road_network_param)
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        position_predicates = PositionPredicateCollection(road_network, self._simulation_param,
                                                          self._traffic_rules_param, set(), traffic_sign_interpreter)

        # ego vehicle
        state_list_lon_ego = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
                              2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10),
                              4: StateLongitudinal(s=40, v=10), 5: StateLongitudinal(s=50, v=10),
                              6: StateLongitudinal(s=60, v=10)}
        state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=4, theta=0),
                              2: StateLateral(d=8, theta=0), 3: StateLateral(d=12, theta=0),
                              4: StateLateral(d=16, theta=0), 5: StateLateral(d=10, theta=0),
                              6: StateLateral(d=2, theta=0)}
        cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
                             2: State(position=20, time_step=2), 3: State(position=30, time_step=3),
                             4: State(position=40, time_step=4), 5: State(position=50, time_step=5),
                             6: State(position=60, time_step=6)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}, 5: {3, 4}, 6: {1, 2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)

        # Monitor-Mode
        sol_monitor_mode_1 = position_predicates.in_rightmost_lane(0, ego_vehicle)
        sol_monitor_mode_2 = position_predicates.in_rightmost_lane(1, ego_vehicle)
        sol_monitor_mode_3 = position_predicates.in_rightmost_lane(2, ego_vehicle)
        sol_monitor_mode_4 = position_predicates.in_rightmost_lane(3, ego_vehicle)
        sol_monitor_mode_5 = position_predicates.in_rightmost_lane(4, ego_vehicle)
        sol_monitor_mode_6 = position_predicates.in_rightmost_lane(5, ego_vehicle)
        sol_monitor_mode_7 = position_predicates.in_rightmost_lane(6, ego_vehicle)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode_7)
