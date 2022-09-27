import math
import unittest
from pathlib import Path

import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import (TrafficSign, TrafficSignIDGermany, TrafficSignElement, )
from commonroad.scenario.trajectory import State
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, LineMarking, Lanelet, LaneletType
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import StateLongitudinal, StateLateral, Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.position import PredInSameLane, PredSingleLane, PredPreceding, PredSafeDistPrec, PredInFrontOf
from crmonitor.predicates.velocity import PredLaneSpeedLimit
from crmonitor.predicates.general import PredCutIn
from crmonitor.common.world import World
from crmonitor.predicates.position import (PredInSameLane, PredSingleLane, PredPreceding, PredSafeDistPrec,
                                           PredInFrontOf, PredRightOfBroadLaneMarking, PredLeftOfBroadLaneMarking,
                                           PredOnAccessRamp, PredOnShoulder, PredOnMainCarriageway, PredInRightmostLane,
                                           PredInLeftmostLane)
from crmonitor.predicates.velocity import PredLaneSpeedLimit
from crmonitor.predicates.general import PredCutIn
from tests.util import parallel_lanes


class TestPositionPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = False

        right_vertices_lane_1 = np.array(
                [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0], [70, 0], [80, 1], [90, 0], [100, 0],
                 [110, 0]])
        left_vertices_lane_1 = np.array(
                [[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4], [80, 1], [90, 4], [100, 4],
                 [110, 4]])
        center_vertices_lane_1 = np.array(
                [[0, 2], [10, 2], [20, 2], [30, 2], [40, 2], [50, 2], [60, 2], [70, 2], [80, 1], [90, 2], [100, 2],
                 [110, 2]])
        self._lanelet_1 = Lanelet(left_vertices_lane_1, center_vertices_lane_1, right_vertices_lane_1, lanelet_id=1,
                                  adjacent_left=2, adjacent_left_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.SHOULDER})

        right_vertices_lane_2 = np.array(
                [[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4], [80, 4], [90, 4], [100, 4],
                 [110, 4]])
        left_vertices_lane_2 = np.array(
                [[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8], [80, 8], [90, 8], [100, 8],
                 [110, 8]])
        center_vertices_lane_2 = np.array(
                [[0, 6], [10, 6], [20, 6], [30, 6], [40, 6], [50, 6], [60, 6], [70, 6], [80, 6], [90, 6], [100, 6],
                 [110, 6]])
        self._lanelet_2 = Lanelet(left_vertices_lane_2, center_vertices_lane_2, right_vertices_lane_2, lanelet_id=2,
                                  adjacent_left=3, adjacent_left_same_direction=True, adjacent_right=1,
                                  adjacent_right_same_direction=True, lanelet_type={LaneletType.INTERSTATE},
                                  line_marking_left_vertices=LineMarking.BROAD_DASHED)

        right_vertices_lane_3 = np.array(
                [[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8], [80, 8], [90, 8], [100, 8],
                 [110, 8]])
        left_vertices_lane_3 = np.array(
                [[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12], [70, 12], [80, 12], [90, 12],
                 [100, 12], [110, 12]])
        center_vertices_lane_3 = np.array(
                [[0, 10], [10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [60, 10], [70, 10], [80, 10], [90, 10],
                 [100, 10], [110, 10]])
        self._lanelet_3 = Lanelet(left_vertices_lane_3, center_vertices_lane_3, right_vertices_lane_3, lanelet_id=3,
                                  adjacent_left=4, adjacent_left_same_direction=True, adjacent_right=2,
                                  adjacent_right_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY},
                                  line_marking_right_vertices=LineMarking.BROAD_DASHED)

        right_vertices_lane_4 = np.array(
                [[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12], [70, 12], [80, 12], [90, 12],
                 [100, 12], [110, 12]])
        left_vertices_lane_4 = np.array(
                [[0, 16], [10, 16], [20, 16], [30, 16], [40, 16], [50, 16], [60, 16], [70, 16], [80, 16], [90, 16],
                 [100, 16], [110, 16]])
        center_vertices_lane_4 = np.array(
                [[0, 14], [10, 14], [20, 14], [30, 14], [40, 14], [50, 14], [60, 14], [70, 14], [80, 14], [90, 14],
                 [100, 14], [110, 14]])
        self._lanelet_4 = Lanelet(left_vertices_lane_4, center_vertices_lane_4, right_vertices_lane_4, lanelet_id=4,
                                  adjacent_left=5, adjacent_left_same_direction=True, adjacent_right=3,
                                  adjacent_right_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.EXIT_RAMP})

        self._lanelet_4_2 = Lanelet(left_vertices_lane_4, center_vertices_lane_4, right_vertices_lane_4, lanelet_id=4,
                                    adjacent_left=5, adjacent_left_same_direction=True, adjacent_right=3,
                                    adjacent_right_same_direction=True,
                                    lanelet_type={LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY})

        right_vertices_lane_5 = np.array(
                [[0, 16], [10, 16], [20, 16], [30, 16], [40, 16], [50, 16], [60, 16], [70, 16], [80, 16], [90, 16],
                 [100, 16], [110, 16]])
        left_vertices_lane_5 = np.array(
                [[0, 20], [10, 20], [20, 20], [30, 20], [40, 20], [50, 20], [60, 20], [70, 20], [80, 20], [90, 20],
                 [100, 20], [110, 20]])
        center_vertices_lane_5 = np.array(
                [[0, 18], [10, 18], [20, 18], [30, 18], [40, 18], [50, 18], [60, 18], [70, 18], [80, 18], [90, 18],
                 [00, 18], [110, 18]])
        self._lanelet_5 = Lanelet(left_vertices_lane_5, center_vertices_lane_5, right_vertices_lane_5, lanelet_id=5,
                                  adjacent_right=4, adjacent_right_same_direction=True,
                                  lanelet_type={LaneletType.INTERSTATE, LaneletType.ACCESS_RAMP})

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        self.road_network = RoadNetwork(lanelet_network, self.config.get("road_network_param"))

        # ego vehicle
        cr_state_list_ego = {0: State(position=[0, 0], time_step=0, orientation=0),
                             1: State(position=[10, 4], time_step=1, orientation=0),
                             2: State(position=[20, 8], time_step=2, orientation=0),
                             3: State(position=[30, 12], time_step=3, orientation=0),
                             4: State(position=[40, 16], time_step=4, orientation=0),
                             5: State(position=[50, -2], time_step=5, orientation=0),
                             6: State(position=[60, -2], time_step=6, orientation=0)}
        lanelet_assignments_ego = {0: {1}, 1: {2}, 2: {3}, 3: {4}, 4: {5}, 5: {2, 3}, 6: {2, 3}}
        ego_vehicle_param = self.config.get("ego_vehicle_param")
        self.ego_vehicle = Vehicle(0, ObstacleType.CAR, ego_vehicle_param, Rectangle(5, 2), cr_state_list_ego, None,
                                   CurvilinearStateManager(self.road_network), lanelet_assignments_ego)


    def test_Left_right_of_broad_lane_marking(self):
        # expected solutions
        exp_sol_monitor_mode_1_left = False
        exp_sol_monitor_mode_2_left = False
        exp_sol_monitor_mode_3_left = True
        exp_sol_monitor_mode_4_left = True
        exp_sol_monitor_mode_5_left = True
        exp_sol_monitor_mode_6_left = False
        exp_sol_monitor_mode_7_left = False

        exp_sol_monitor_mode_1_right = True
        exp_sol_monitor_mode_2_right = True
        exp_sol_monitor_mode_3_right = False
        exp_sol_monitor_mode_4_right = False
        exp_sol_monitor_mode_5_right = False
        exp_sol_monitor_mode_6_right = False
        exp_sol_monitor_mode_7_right = False

        world = World({self.ego_vehicle}, self.road_network)

        # Right of broad lane markings
        pred = PredRightOfBroadLaneMarking(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1_right, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1_right, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2_right, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2_right, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3_right, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3_right, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4_right, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4_right, sol_robustness_monitor_mode_4 > 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5_right, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5_right, sol_robustness_monitor_mode_5 > 0)

        sol_monitor_mode_6 = pred.evaluate_boolean(world, 5, vehicle_ids)
        sol_robustness_monitor_mode_6 = pred.evaluate_robustness(world, 5, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_6_right, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_6_right, sol_robustness_monitor_mode_6 > 0)

        sol_monitor_mode_7 = pred.evaluate_boolean(world, 6, vehicle_ids)
        sol_robustness_monitor_mode_7 = pred.evaluate_robustness(world, 6, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_7_right, sol_monitor_mode_7)
        self.assertEqual(exp_sol_monitor_mode_7_right, sol_robustness_monitor_mode_7 > 0)

        # Left of broad lane markings
        pred = PredLeftOfBroadLaneMarking(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1_left, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1_left, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2_left, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2_left, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3_left, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3_left, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4_left, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4_left, sol_robustness_monitor_mode_4 > 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5_left, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5_left, sol_robustness_monitor_mode_5 > 0)

        sol_monitor_mode_6 = pred.evaluate_boolean(world, 5, vehicle_ids)
        sol_robustness_monitor_mode_6 = pred.evaluate_robustness(world, 5, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_6_left, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_6_left, sol_robustness_monitor_mode_6 > 0)

        sol_monitor_mode_7 = pred.evaluate_boolean(world, 6, vehicle_ids)
        sol_robustness_monitor_mode_7 = pred.evaluate_robustness(world, 6, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_7_left, sol_monitor_mode_7)
        self.assertEqual(exp_sol_monitor_mode_7_left, sol_robustness_monitor_mode_7 > 0)

    def test_on_access_ramp(self):
        exp_sol_monitor_mode_1 = False  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = False  # ego vehicle on main carriageway
        exp_sol_monitor_mode_4 = False  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = True  # ego vehicle on access ramp

        world = World({self.ego_vehicle}, self.road_network)

        # Left of broad lane markings
        pred = PredOnAccessRamp(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 > 0)

    def test_on_shoulder(self):
        exp_sol_monitor_mode_1 = True  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = False  # ego vehicle on main carriageway
        exp_sol_monitor_mode_4 = False  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = False  # ego vehicle on access ramp

        world = World({self.ego_vehicle}, self.road_network)

        # Left of broad lane markings
        pred = PredOnShoulder(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 > 0)

    def test_on_main_carriage_way(self):
        exp_sol_monitor_mode_1 = False  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = True  # ego vehicle on main carriageway
        exp_sol_monitor_mode_4 = False  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = False  # ego vehicle on access ramp

        world = World({self.ego_vehicle}, self.road_network)

        # Left of broad lane markings
        pred = PredOnMainCarriageway(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 > 0)

    def test_in_rightmost_lane(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = True
        exp_sol_monitor_mode_3 = False
        exp_sol_monitor_mode_4 = False
        exp_sol_monitor_mode_5 = False

        world = World({self.ego_vehicle}, self.road_network)

        pred = PredInRightmostLane(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        # fix the lanelet assignment
        for time, lanelet in self.ego_vehicle.lanelet_assignment.items():
            shape = self.ego_vehicle.shape
            state = self.ego_vehicle.states_cr[time]

            self.ego_vehicle.lanelet_assignment[time] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation))

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 > 0)

    def test_in_leftmost_lane(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False
        exp_sol_monitor_mode_2 = False
        exp_sol_monitor_mode_3 = False
        exp_sol_monitor_mode_4 = False
        exp_sol_monitor_mode_5 = True

        world = World({self.ego_vehicle}, self.road_network)

        pred = PredInLeftmostLane(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        # fix the lanelet assignment
        for time, lanelet in self.ego_vehicle.lanelet_assignment.items():
            shape = self.ego_vehicle.shape
            state = self.ego_vehicle.states_cr[time]

            self.ego_vehicle.lanelet_assignment[time] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation))

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 > 0)
