import unittest
from pathlib import Path

import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import (
    Lanelet,
    LaneletNetwork,
    LaneletType,
    LineMarking,
)
from commonroad.scenario.obstacle import ObstacleType, State

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import CurvilinearStateManager, Vehicle
from crmonitor.common.world import World
from crmonitor.predicates.position import (
    PredDrivesLeftmost,
    PredDrivesRightmost,
    PredInLeftmostLane,
    PredInRightmostLane,
    PredLeftOf,
    PredLeftOfBroadLaneMarking,
    PredMainCarriageWayRightLane,
    PredOnAccessRamp,
    PredOnMainCarriageway,
    PredOnShoulder,
    PredRightOfBroadLaneMarking,
)


class TestPositionPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = False

        right_vertices_lane_1 = np.array(
            [
                [0, 0],
                [10, 0],
                [20, 0],
                [30, 0],
                [40, 0],
                [50, 0],
                [60, 0],
                [70, 0],
                [80, 1],
                [90, 0],
                [100, 0],
                [110, 0],
            ]
        )
        left_vertices_lane_1 = np.array(
            [
                [0, 4],
                [10, 4],
                [20, 4],
                [30, 4],
                [40, 4],
                [50, 4],
                [60, 4],
                [70, 4],
                [80, 1],
                [90, 4],
                [100, 4],
                [110, 4],
            ]
        )
        center_vertices_lane_1 = np.array(
            [
                [0, 2],
                [10, 2],
                [20, 2],
                [30, 2],
                [40, 2],
                [50, 2],
                [60, 2],
                [70, 2],
                [80, 1],
                [90, 2],
                [100, 2],
                [110, 2],
            ]
        )
        self._lanelet_1 = Lanelet(
            left_vertices_lane_1,
            center_vertices_lane_1,
            right_vertices_lane_1,
            lanelet_id=1,
            adjacent_left=2,
            adjacent_left_same_direction=True,
            lanelet_type={LaneletType.INTERSTATE, LaneletType.SHOULDER},
        )

        right_vertices_lane_2 = np.array(
            [
                [0, 4],
                [10, 4],
                [20, 4],
                [30, 4],
                [40, 4],
                [50, 4],
                [60, 4],
                [70, 4],
                [80, 4],
                [90, 4],
                [100, 4],
                [110, 4],
            ]
        )
        left_vertices_lane_2 = np.array(
            [
                [0, 8],
                [10, 8],
                [20, 8],
                [30, 8],
                [40, 8],
                [50, 8],
                [60, 8],
                [70, 8],
                [80, 8],
                [90, 8],
                [100, 8],
                [110, 8],
            ]
        )
        center_vertices_lane_2 = np.array(
            [
                [0, 6],
                [10, 6],
                [20, 6],
                [30, 6],
                [40, 6],
                [50, 6],
                [60, 6],
                [70, 6],
                [80, 6],
                [90, 6],
                [100, 6],
                [110, 6],
            ]
        )
        self._lanelet_2 = Lanelet(
            left_vertices_lane_2,
            center_vertices_lane_2,
            right_vertices_lane_2,
            lanelet_id=2,
            adjacent_left=3,
            adjacent_left_same_direction=True,
            adjacent_right=1,
            adjacent_right_same_direction=True,
            lanelet_type={LaneletType.INTERSTATE},
            line_marking_left_vertices=LineMarking.BROAD_DASHED,
        )

        right_vertices_lane_3 = np.array(
            [
                [0, 8],
                [10, 8],
                [20, 8],
                [30, 8],
                [40, 8],
                [50, 8],
                [60, 8],
                [70, 8],
                [80, 8],
                [90, 8],
                [100, 8],
                [110, 8],
            ]
        )
        left_vertices_lane_3 = np.array(
            [
                [0, 12],
                [10, 12],
                [20, 12],
                [30, 12],
                [40, 12],
                [50, 12],
                [60, 12],
                [70, 12],
                [80, 12],
                [90, 12],
                [100, 12],
                [110, 12],
            ]
        )
        center_vertices_lane_3 = np.array(
            [
                [0, 10],
                [10, 10],
                [20, 10],
                [30, 10],
                [40, 10],
                [50, 10],
                [60, 10],
                [70, 10],
                [80, 10],
                [90, 10],
                [100, 10],
                [110, 10],
            ]
        )
        self._lanelet_3 = Lanelet(
            left_vertices_lane_3,
            center_vertices_lane_3,
            right_vertices_lane_3,
            lanelet_id=3,
            adjacent_left=4,
            adjacent_left_same_direction=True,
            adjacent_right=2,
            adjacent_right_same_direction=True,
            lanelet_type={LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY},
            line_marking_right_vertices=LineMarking.BROAD_DASHED,
        )

        right_vertices_lane_4 = np.array(
            [
                [0, 12],
                [10, 12],
                [20, 12],
                [30, 12],
                [40, 12],
                [50, 12],
                [60, 12],
                [70, 12],
                [80, 12],
                [90, 12],
                [100, 12],
                [110, 12],
            ]
        )
        left_vertices_lane_4 = np.array(
            [
                [0, 16],
                [10, 16],
                [20, 16],
                [30, 16],
                [40, 16],
                [50, 16],
                [60, 16],
                [70, 16],
                [80, 16],
                [90, 16],
                [100, 16],
                [110, 16],
            ]
        )
        center_vertices_lane_4 = np.array(
            [
                [0, 14],
                [10, 14],
                [20, 14],
                [30, 14],
                [40, 14],
                [50, 14],
                [60, 14],
                [70, 14],
                [80, 14],
                [90, 14],
                [100, 14],
                [110, 14],
            ]
        )
        self._lanelet_4 = Lanelet(
            left_vertices_lane_4,
            center_vertices_lane_4,
            right_vertices_lane_4,
            lanelet_id=4,
            adjacent_left=5,
            adjacent_left_same_direction=True,
            adjacent_right=3,
            adjacent_right_same_direction=True,
            lanelet_type={LaneletType.INTERSTATE, LaneletType.EXIT_RAMP},
        )

        self._lanelet_4_2 = Lanelet(
            left_vertices_lane_4,
            center_vertices_lane_4,
            right_vertices_lane_4,
            lanelet_id=4,
            adjacent_left=5,
            adjacent_left_same_direction=True,
            adjacent_right=3,
            adjacent_right_same_direction=True,
            lanelet_type={LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY},
        )

        right_vertices_lane_5 = np.array(
            [
                [0, 16],
                [10, 16],
                [20, 16],
                [30, 16],
                [40, 16],
                [50, 16],
                [60, 16],
                [70, 16],
                [80, 16],
                [90, 16],
                [100, 16],
                [110, 16],
            ]
        )
        left_vertices_lane_5 = np.array(
            [
                [0, 20],
                [10, 20],
                [20, 20],
                [30, 20],
                [40, 20],
                [50, 20],
                [60, 20],
                [70, 20],
                [80, 20],
                [90, 20],
                [100, 20],
                [110, 20],
            ]
        )
        center_vertices_lane_5 = np.array(
            [
                [0, 18],
                [10, 18],
                [20, 18],
                [30, 18],
                [40, 18],
                [50, 18],
                [60, 18],
                [70, 18],
                [80, 18],
                [90, 18],
                [00, 18],
                [110, 18],
            ]
        )
        self._lanelet_5 = Lanelet(
            left_vertices_lane_5,
            center_vertices_lane_5,
            right_vertices_lane_5,
            lanelet_id=5,
            adjacent_right=4,
            adjacent_right_same_direction=True,
            lanelet_type={LaneletType.INTERSTATE, LaneletType.ACCESS_RAMP},
        )

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        lanelet_network.add_lanelet(self._lanelet_5)
        self.road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param")
        )

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 0], time_step=0, orientation=0),
            1: State(position=[10, 4], time_step=1, orientation=0),
            2: State(position=[20, 8], time_step=2, orientation=0),
            3: State(position=[30, 12], time_step=3, orientation=0),
            4: State(position=[40, 16], time_step=4, orientation=0),
            5: State(position=[50, -2], time_step=5, orientation=0),
            6: State(position=[60, -2], time_step=6, orientation=0),
        }
        lanelet_assignments_ego = {
            0: {1},
            1: {2},
            2: {3},
            3: {4},
            4: {5},
            5: {2, 3},
            6: {2, 3},
        }
        ego_vehicle_param = self.config.get("ego_vehicle_param")
        self.ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_ego,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_ego,
        )

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
        self.assertEqual(
            exp_sol_monitor_mode_1_right, sol_robustness_monitor_mode_1 > 0
        )

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2_right, sol_monitor_mode_2)
        self.assertEqual(
            exp_sol_monitor_mode_2_right, sol_robustness_monitor_mode_2 > 0
        )

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3_right, sol_monitor_mode_3)
        self.assertEqual(
            exp_sol_monitor_mode_3_right, sol_robustness_monitor_mode_3 > 0
        )

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4_right, sol_monitor_mode_4)
        self.assertEqual(
            exp_sol_monitor_mode_4_right, sol_robustness_monitor_mode_4 > 0
        )

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(world, 4, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_5_right, sol_monitor_mode_5)
        self.assertEqual(
            exp_sol_monitor_mode_5_right, sol_robustness_monitor_mode_5 > 0
        )

        sol_monitor_mode_6 = pred.evaluate_boolean(world, 5, vehicle_ids)
        sol_robustness_monitor_mode_6 = pred.evaluate_robustness(world, 5, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_6_right, sol_monitor_mode_6)
        self.assertEqual(
            exp_sol_monitor_mode_6_right, sol_robustness_monitor_mode_6 > 0
        )

        sol_monitor_mode_7 = pred.evaluate_boolean(world, 6, vehicle_ids)
        sol_robustness_monitor_mode_7 = pred.evaluate_robustness(world, 6, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_7_right, sol_monitor_mode_7)
        self.assertEqual(
            exp_sol_monitor_mode_7_right, sol_robustness_monitor_mode_7 > 0
        )

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

    def PredMainCarriageWayRightLane(self):
        exp_sol_monitor_mode_1 = False  # ego vehicle on shoulder
        exp_sol_monitor_mode_2 = False  # no specific type
        exp_sol_monitor_mode_3 = True  # ego vehicle on main carriageway (the right one)
        exp_sol_monitor_mode_4 = False  # ego vehicle on exit ramp
        exp_sol_monitor_mode_5 = False  # ego vehicle on access ramp

        world = World({self.ego_vehicle}, self.road_network)

        pred = PredMainCarriageWayRightLane(self.config)
        vehicle_ids = [self.ego_vehicle.id]

        # fix the lanelet assignment
        for time, lanelet in self.ego_vehicle.lanelet_assignment.items():
            shape = self.ego_vehicle.shape
            state = self.ego_vehicle.states_cr[time]

            self.ego_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

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

            self.ego_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

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

            self.ego_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

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

    def test_left_of(self):
        exp_sol_monitor_mode_1 = False  # other vehicle in left lane but not adjacent
        exp_sol_monitor_mode_2 = False  # other vehicle exactly left of
        exp_sol_monitor_mode_3 = False  # other vehicle partially left of in front
        exp_sol_monitor_mode_4 = False  # other vehicle partially left of in behind
        exp_sol_monitor_mode_5 = False  # other vehicle left of in front and behind
        exp_sol_monitor_mode_6 = False  # other vehicle in same lane in front
        exp_sol_monitor_mode_7 = True  # other vehicle in right lane but not adjacent
        exp_sol_monitor_mode_8 = True  # other vehicle exactly right of
        exp_sol_monitor_mode_9 = True  # other vehicle partially right of in front
        exp_sol_monitor_mode_10 = True  # other vehicle partially right of behind
        exp_sol_monitor_mode_11 = (
            True  # other vehicle partially right of in front and behind
        )

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 0], time_step=0, orientation=0, velocity=10),
            1: State(position=[10, 0], time_step=1, orientation=0, velocity=10),
            2: State(position=[20, 0], time_step=2, orientation=0, velocity=10),
            3: State(position=[30, 0], time_step=3, orientation=0, velocity=10),
            4: State(position=[40, 0], time_step=4, orientation=0, velocity=10),
            5: State(position=[50, 0], time_step=5, orientation=0, velocity=10),
            6: State(position=[60, 0], time_step=6, orientation=0, velocity=10),
            7: State(position=[70, 0], time_step=7, orientation=0, velocity=10),
            8: State(position=[80, 0], time_step=8, orientation=0, velocity=10),
            9: State(position=[90, 0], time_step=9, orientation=0, velocity=10),
            10: State(position=[100, 0], time_step=10, orientation=0, velocity=10),
        }
        lanelet_assignments_ego = {
            0: {3},
            1: {3},
            2: {3},
            3: {3},
            4: {3},
            5: {3},
            6: {3},
            7: {3},
            8: {3},
            9: {3},
            10: {3},
        }
        ego_vehicle_param = self.config.get("ego_vehicle_param")
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_ego,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_ego,
        )

        # other vehicle 1
        cr_state_list_other_1 = {
            0: State(position=[5, 4], time_step=0, orientation=0, velocity=10),
            1: State(position=[10, 4], time_step=1, orientation=0, velocity=10),
            2: State(position=[21, 4], time_step=2, orientation=0, velocity=10),
            3: State(position=[29, 4], time_step=3, orientation=0, velocity=10),
            5: State(position=[55, 0], time_step=5, orientation=0, velocity=10),
            6: State(position=[65, -4], time_step=6, orientation=0, velocity=10),
            7: State(position=[70, -4], time_step=7, orientation=0, velocity=10),
            8: State(position=[81, -4], time_step=8, orientation=0, velocity=10),
            9: State(position=[89, -4], time_step=9, orientation=0, velocity=10),
        }
        lanelet_assignments_other_1 = {
            0: {4},
            1: {4},
            2: {4},
            3: {4},
            5: {3},
            6: {2},
            7: {2},
            8: {2},
            9: {2},
        }
        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_1,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_other_1,
        )

        # other vehicle 2
        cr_state_list_other_2 = {
            4: State(position=[40, 4], time_step=0, orientation=0, velocity=10),
            10: State(position=[100, -4], time_step=1, orientation=0, velocity=10),
        }
        lanelet_assignments_other_2 = {4: {4}, 10: {2}}
        other_vehicle_2 = Vehicle(
            2,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_2,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_other_2,
        )

        pred = PredLeftOf(self.config)
        vehicle_ids_1 = [ego_vehicle.id, other_vehicle_1.id]
        vehicle_ids_2 = [ego_vehicle.id, other_vehicle_2.id]

        world = World(
            {ego_vehicle, other_vehicle_1, other_vehicle_2}, self.road_network
        )

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids_1)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(
            world, 0, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 >= 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids_1)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(
            world, 1, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 >= 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids_1)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(
            world, 2, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 >= 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids_1)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(
            world, 3, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 >= 0)

        sol_monitor_mode_5 = pred.evaluate_boolean(world, 4, vehicle_ids_2)
        sol_robustness_monitor_mode_5 = pred.evaluate_robustness(
            world, 4, vehicle_ids_2
        )
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 >= 0)

        sol_monitor_mode_6 = pred.evaluate_boolean(world, 5, vehicle_ids_1)
        sol_robustness_monitor_mode_6 = pred.evaluate_robustness(
            world, 5, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_6, sol_robustness_monitor_mode_6 >= 0)

        sol_monitor_mode_7 = pred.evaluate_boolean(world, 6, vehicle_ids_1)
        sol_robustness_monitor_mode_7 = pred.evaluate_robustness(
            world, 6, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode_7)
        self.assertEqual(exp_sol_monitor_mode_7, sol_robustness_monitor_mode_7 >= 0)

        sol_monitor_mode_8 = pred.evaluate_boolean(world, 7, vehicle_ids_1)
        sol_robustness_monitor_mode_8 = pred.evaluate_robustness(
            world, 7, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_8, sol_monitor_mode_8)
        self.assertEqual(exp_sol_monitor_mode_8, sol_robustness_monitor_mode_8 >= 0)

        sol_monitor_mode_9 = pred.evaluate_boolean(world, 8, vehicle_ids_1)
        sol_robustness_monitor_mode_9 = pred.evaluate_robustness(
            world, 8, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_9, sol_monitor_mode_9)
        self.assertEqual(exp_sol_monitor_mode_9, sol_robustness_monitor_mode_9 >= 0)

        sol_monitor_mode_10 = pred.evaluate_boolean(world, 9, vehicle_ids_1)
        sol_robustness_monitor_mode_10 = pred.evaluate_robustness(
            world, 9, vehicle_ids_1
        )
        self.assertEqual(exp_sol_monitor_mode_10, sol_monitor_mode_10)
        self.assertEqual(exp_sol_monitor_mode_10, sol_robustness_monitor_mode_10 >= 0)

        sol_monitor_mode_11 = pred.evaluate_boolean(world, 10, vehicle_ids_2)
        sol_robustness_monitor_mode_11 = pred.evaluate_robustness(
            world, 10, vehicle_ids_2
        )
        self.assertEqual(exp_sol_monitor_mode_11, sol_monitor_mode_11)
        self.assertEqual(exp_sol_monitor_mode_11, sol_robustness_monitor_mode_11 >= 0)

    def test_drives_leftmost(self):
        self.config["close_to_lane_border"] = 0.2
        self.config["close_to_other_vehicle"] = 0.5

        # expected solutions for leftmost
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = False
        exp_sol_monitor_mode_3 = True
        exp_sol_monitor_mode_4 = False
        exp_sol_monitor_mode_5 = False

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param")
        )

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 8], time_step=0, orientation=0, velocity=10),
            1: State(position=[10, 6.5], time_step=1, orientation=0, velocity=10),
            2: State(position=[20, 4.6], time_step=2, orientation=0, velocity=10),
            3: State(position=[30, 4.6], time_step=3, orientation=0, velocity=10),
            4: State(position=[40, 2], time_step=4, orientation=0, velocity=10),
        }
        lanelet_assignments_ego = {0: {2}, 1: {2}, 2: {1, 2}, 3: {1, 2}, 4: {1}}
        ego_vehicle_param = self.config.get("ego_vehicle_param")
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_ego,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_ego,
        )

        # fix the lanelet assignment
        for time, lanelet in ego_vehicle.lanelet_assignment.items():
            shape = ego_vehicle.shape
            state = ego_vehicle.states_cr[time]

            ego_vehicle.lanelet_assignment[
                time
            ] = road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        # other vehicle 1
        cr_state_list_other_1 = {
            2: State(position=[20, 6.7], time_step=2, orientation=0, velocity=10),
            3: State(position=[30, 6], time_step=3, orientation=0, velocity=10),
        }
        lanelet_assignments_other_1 = {2: {2}, 3: {2}}

        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_1,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_other_1,
        )

        for time, lanelet in other_vehicle_1.lanelet_assignment.items():
            shape = other_vehicle_1.shape
            state = other_vehicle_1.states_cr[time]

            other_vehicle_1.lanelet_assignment[
                time
            ] = road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        pred = PredDrivesLeftmost(self.config)
        vehicle_ids = [ego_vehicle.id]

        world = World({ego_vehicle, other_vehicle_1}, road_network)
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

    def test_drives_rightmost(self):
        self.config["close_to_lane_border"] = 0.2
        self.config["close_to_other_vehicle"] = 0.5

        # expected solutions for rightmost
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = False
        exp_sol_monitor_mode_3 = True
        exp_sol_monitor_mode_4 = False
        exp_sol_monitor_mode_5 = False

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param")
        )

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 1.0], time_step=0, orientation=0, velocity=10),
            1: State(position=[10, 2.5], time_step=1, orientation=0, velocity=10),
            2: State(position=[20, 4.6], time_step=2, orientation=0, velocity=10),
            3: State(position=[30, 4.6], time_step=3, orientation=0, velocity=10),
            4: State(position=[40, 7], time_step=4, orientation=0, velocity=10),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1, 2}, 3: {1, 2}, 4: {2}}
        ego_vehicle_param = self.config.get("ego_vehicle_param")
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_ego,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_ego,
        )

        # fix the lanelet assignment
        for time, lanelet in ego_vehicle.lanelet_assignment.items():
            shape = ego_vehicle.shape
            state = ego_vehicle.states_cr[time]

            ego_vehicle.lanelet_assignment[
                time
            ] = road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        # other vehicle 1
        cr_state_list_other_1 = {
            2: State(position=[20, 2.4], time_step=2, orientation=0, velocity=10),
            3: State(position=[30, 2.0], time_step=3, orientation=0, velocity=10),
        }
        lanelet_assignments_other_1 = {2: {1}, 3: {1}}

        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_1,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_other_1,
        )

        for time, lanelet in other_vehicle_1.lanelet_assignment.items():
            shape = other_vehicle_1.shape
            state = other_vehicle_1.states_cr[time]

            other_vehicle_1.lanelet_assignment[
                time
            ] = road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        pred = PredDrivesRightmost(self.config)
        vehicle_ids = [ego_vehicle.id]

        world = World({ego_vehicle, other_vehicle_1}, road_network)
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
