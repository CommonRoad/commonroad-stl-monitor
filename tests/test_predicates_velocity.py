import math
import unittest
from pathlib import Path
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import State, ObstacleType

from crmonitor.common.helper import load_yaml
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.velocity import (
    PredReverses,
    PredSlowLeadingVehicle,
    PredPreservesTrafficFlow,
    PredInStandStill,
    PredExistStandingLeadingVehicle,
    PredDrivesFaster,
    PredDrivesWithSlightlyHigherSpeed,
)
from crmonitor.predicates.acceleration import PredCausesBrakingIntersection


class TestVelocityPredicates(unittest.TestCase):
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
                [90, 0],
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
                [90, 0],
            ]
        )
        self._lanelet_1 = Lanelet(
            left_vertices_lane_1,
            center_vertices_lane_1,
            right_vertices_lane_1,
            lanelet_id=1,
            adjacent_left=2,
            adjacent_left_same_direction=True,
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
            ]
        )
        center_vertices_lane_2 = np.array(
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
            ]
        )
        self._lanelet_3 = Lanelet(
            left_vertices_lane_3,
            center_vertices_lane_3,
            right_vertices_lane_3,
            lanelet_id=3,
            adjacent_right=2,
            adjacent_right_same_direction=True,
        )

        right_vertices_lane_4 = np.array(
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
            ]
        )
        left_vertices_lane_4 = np.array(
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
            ]
        )
        center_vertices_lane_4 = np.array(
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
            ]
        )
        self._lanelet_4 = Lanelet(
            left_vertices_lane_4,
            center_vertices_lane_4,
            right_vertices_lane_4,
            lanelet_id=4,
        )

        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        self.road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param")
        )

    # def test_reverses(self):
    #     self.config["standstill_error"] = 0.01
    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False  # ego has velocity of zero
    #     exp_sol_monitor_mode_2 = False  # ego vehicle has positive velocity
    #     exp_sol_monitor_mode_3 = False  # ego vehicle has velocity of -standstill_error
    #     exp_sol_monitor_mode_4 = (
    #         True  # ego vehicle has velocity smaller than -standstill_error
    #     )

    #     # ego vehicle
    #     cr_state_list_ego = {
    #         0: State(position=[0, 0], time_step=0, orientation=0, velocity=0),
    #         1: State(position=[0, 0], time_step=1, orientation=0, velocity=1),
    #         2: State(
    #             position=[1, 0],
    #             time_step=2,
    #             orientation=0,
    #             velocity=-self.config["standstill_error"],
    #         ),
    #         3: State(
    #             position=[1 - self.config["standstill_error"], 0],
    #             time_step=3,
    #             orientation=0,
    #             velocity=-2,
    #         ),
    #     }
    #     lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {2}}
    #     ego_vehicle_param = self.config.get("ego_vehicle_param")
    #     ego_vehicle = Vehicle(
    #         0,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_ego,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_ego,
    #     )

    #     pred = PredReverses(self.config)
    #     vehicle_ids = [ego_vehicle.id]

    #     world = World({ego_vehicle}, self.road_network)

    #     sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
    #     sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

    #     sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
    #     sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

    #     sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
    #     sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    #     sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
    #     sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

    # def test_slow_leading_vehicle(self):
    #     self.config["min_velocity_dif"] = 15
    #     self.config["road_condition_speed_limit"] = 50
    #     self.config["country"] = "DEU"
    #     self.config["desired_interstate_velocity"] = 36.11

    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False  # no leading vehicle at all
    #     exp_sol_monitor_mode_2 = (
    #         False  # two leading vehicles which drive with speed limit
    #     )
    #     exp_sol_monitor_mode_3 = True  # first leading vehicle is drives to slow
    #     exp_sol_monitor_mode_4 = True  # third leading vehicle drives to slow

    #     # ego vehicle
    #     cr_state_list_ego = {
    #         0: State(position=[5, 0], time_step=0, orientation=0, velocity=2),
    #         1: State(position=[7, 0], time_step=1, orientation=0, velocity=2),
    #         2: State(position=[9, 0], time_step=2, orientation=0, velocity=2),
    #         3: State(position=[11, 0], time_step=3, orientation=0, velocity=2),
    #     }
    #     lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
    #     ego_vehicle_param = self.config.get("ego_vehicle_param")
    #     ego_vehicle = Vehicle(
    #         0,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_ego,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_ego,
    #     )

    #     # other vehicle 1
    #     cr_state_list_other_1 = {
    #         1: State(position=[12, 0], time_step=1, orientation=0, velocity=50),
    #         2: State(position=[62, 0], time_step=2, orientation=0, velocity=12),
    #         3: State(position=[74, 0], time_step=3, orientation=0, velocity=2),
    #     }
    #     lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
    #     other_vehicle_1 = Vehicle(
    #         1,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_1,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_1,
    #     )

    #     # other vehicle 2
    #     cr_state_list_other_2 = {
    #         1: State(position=[22, 0], time_step=1, orientation=0, velocity=36),
    #         2: State(position=[58, 0], time_step=2, orientation=0, velocity=36),
    #         3: State(position=[88, 0], time_step=3, orientation=0, velocity=36),
    #     }
    #     lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
    #     other_vehicle_2 = Vehicle(
    #         2,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_2,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_2,
    #     )

    #     # other vehicle 3
    #     cr_state_list_other_3 = {
    #         2: State(position=[34, 0], time_step=2, orientation=0, velocity=50),
    #         3: State(position=[84, 0], time_step=3, orientation=0, velocity=0),
    #     }
    #     lanelet_assignments_other_3 = {2: {1}, 3: {1}}
    #     other_vehicle_3 = Vehicle(
    #         3,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_3,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_3,
    #     )

    #     # other vehicle 4
    #     cr_state_list_other_4 = {
    #         0: State(position=[0, 0], time_step=0, orientation=0, velocity=60),
    #         1: State(position=[50, 3.5], time_step=1, orientation=0, velocity=0),
    #     }
    #     lanelet_assignments_other_4 = {0: {1}, 1: {2}}
    #     other_vehicle_4 = Vehicle(
    #         4,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_4,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_4,
    #     )

    #     pred = PredSlowLeadingVehicle(self.config)
    #     vehicle_ids = [ego_vehicle.id]

    #     world = World(
    #         {
    #             ego_vehicle,
    #             other_vehicle_1,
    #             other_vehicle_2,
    #             other_vehicle_3,
    #             other_vehicle_4,
    #         },
    #         self.road_network,
    #     )

    #     sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
    #     sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

    #     sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
    #     sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

    #     sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
    #     sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    #     sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
    #     sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

    # def test_preserves_traffic_flow(self):
    #     self.config["min_velocity_dif"] = 15
    #     self.config["braking_speed_limit"] = 50
    #     self.config["fov_speed_limit"] = 35
    #     self.config["road_condition_speed_limit"] = 50
    #     self.config["desired_interstate_velocity"] = 36.11
    #     self.config["country"] = "DEU"

    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False  # ego vehicle drives too slow
    #     exp_sol_monitor_mode_2 = False  # ego vehicle drives at lower limit to
    #     exp_sol_monitor_mode_3 = True  # ego vehcie drives faster than velocity limit

    #     # ego vehicle
    #     cr_state_list_ego = {
    #         0: State(position=[0, 0], time_step=0, orientation=0, velocity=2),
    #         1: State(position=[2, 0], time_step=1, orientation=0, velocity=20),
    #         2: State(position=[22, 0], time_step=2, orientation=0, velocity=50),
    #     }
    #     lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
    #     ego_vehicle_param = self.config.get("ego_vehicle_param")
    #     ego_vehicle = Vehicle(
    #         0,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_ego,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_ego,
    #     )

    #     pred = PredPreservesTrafficFlow(self.config)
    #     vehicle_ids = [ego_vehicle.id]

    #     world = World({ego_vehicle}, self.road_network)

    #     sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
    #     sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

    #     sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
    #     sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

    #     sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
    #     sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    # def test_in_standstill(self):
    #     self.config["standstill_error"] = 0.01
    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False
    #     exp_sol_monitor_mode_2 = True
    #     exp_sol_monitor_mode_3 = True
    #     exp_sol_monitor_mode_4 = True

    #     # ego vehicle
    #     cr_state_list_ego = {
    #         0: State(position=[0, 0], time_step=0, orientation=0, velocity=1),
    #         1: State(position=[1, 0], time_step=1, orientation=0, velocity=0),
    #         2: State(position=[1, 0], time_step=2, orientation=0, velocity=0.001),
    #         3: State(position=[1, 0], time_step=3, orientation=0, velocity=-0.001),
    #     }
    #     lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
    #     ego_vehicle_param = self.config.get("ego_vehicle_param")
    #     ego_vehicle = Vehicle(
    #         0,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_ego,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_ego,
    #     )

    #     pred = PredInStandStill(self.config)
    #     vehicle_ids = [ego_vehicle.id]

    #     world = World({ego_vehicle}, self.road_network)

    #     sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
    #     sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

    #     sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
    #     sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

    #     sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
    #     sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    #     sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
    #     sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

    # def test_exists_standing_leading_vehicle(self):
    #     self.config["standstill_error"] = 0.01

    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False  # no leading vehicle at all
    #     exp_sol_monitor_mode_2 = False  # two leading vehicles which have velocity > 0
    #     exp_sol_monitor_mode_3 = True  # first leading vehicle is standing
    #     exp_sol_monitor_mode_4 = True  # third leading vehicle is standing

    #     # ego vehicle
    #     cr_state_list_ego = {
    #         0: State(position=[5, 0], time_step=0, orientation=0, velocity=2),
    #         1: State(position=[7, 0], time_step=1, orientation=0, velocity=2),
    #         2: State(position=[9, 0], time_step=2, orientation=0, velocity=2),
    #         3: State(position=[11, 0], time_step=3, orientation=0, velocity=2),
    #     }
    #     lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
    #     ego_vehicle_param = self.config.get("ego_vehicle_param")
    #     ego_vehicle = Vehicle(
    #         0,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_ego,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_ego,
    #     )

    #     # other vehicle 1
    #     cr_state_list_other_1 = {
    #         1: State(position=[12, 0], time_step=1, orientation=0, velocity=2),
    #         2: State(position=[14, 0], time_step=2, orientation=0, velocity=0),
    #         3: State(position=[14, 0], time_step=3, orientation=0, velocity=2),
    #     }
    #     lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
    #     other_vehicle_1 = Vehicle(
    #         1,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_1,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_1,
    #     )

    #     # other vehicle 2
    #     cr_state_list_other_2 = {
    #         1: State(position=[22, 0], time_step=1, orientation=0, velocity=2),
    #         2: State(position=[24, 0], time_step=2, orientation=0, velocity=2),
    #         3: State(position=[26, 0], time_step=3, orientation=0, velocity=2),
    #     }
    #     lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
    #     other_vehicle_2 = Vehicle(
    #         2,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_2,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_2,
    #     )

    #     # other vehicle 3
    #     cr_state_list_other_3 = {
    #         2: State(position=[34, 0], time_step=2, orientation=0, velocity=2),
    #         3: State(position=[36, 0], time_step=3, orientation=0, velocity=0),
    #     }
    #     lanelet_assignments_other_3 = {2: {1}, 3: {1}}
    #     other_vehicle_3 = Vehicle(
    #         3,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_3,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_3,
    #     )

    #     # other vehicle 4
    #     cr_state_list_other_4 = {
    #         0: State(position=[0, 0], time_step=0, orientation=0, velocity=60),
    #         1: State(position=[50, 3.5], time_step=1, orientation=0, velocity=0),
    #     }
    #     lanelet_assignments_other_4 = {0: {1}, 1: {2}}
    #     other_vehicle_4 = Vehicle(
    #         4,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_4,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_4,
    #     )

    #     pred = PredExistStandingLeadingVehicle(self.config)
    #     vehicle_ids = [ego_vehicle.id]

    #     world = World(
    #         {
    #             ego_vehicle,
    #             other_vehicle_1,
    #             other_vehicle_2,
    #             other_vehicle_3,
    #             other_vehicle_4,
    #         },
    #         self.road_network,
    #     )

    #     sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
    #     sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

    #     sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
    #     sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

    #     sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
    #     sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    #     sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
    #     sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

    # def test_drives_faster(self):
    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False  # ego vehicle has lower velocity
    #     exp_sol_monitor_mode_2 = False  # ego vehicle has same velocity
    #     exp_sol_monitor_mode_3 = True  # ego vehicle drives with higher speed

    #     # ego vehicle
    #     cr_state_list_ego = {
    #         0: State(position=[0, 0], time_step=0, orientation=0, velocity=5),
    #         1: State(position=[5, 0], time_step=1, orientation=0, velocity=20),
    #         2: State(position=[25, 0], time_step=2, orientation=0, velocity=35),
    #     }
    #     lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
    #     ego_vehicle_param = self.config.get("ego_vehicle_param")
    #     ego_vehicle = Vehicle(
    #         0,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_ego,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_ego,
    #     )

    #     # other vehicle 1
    #     cr_state_list_other_1 = {
    #         0: State(position=[10, 0], time_step=0, orientation=0, velocity=10),
    #         1: State(position=[20, 0], time_step=1, orientation=0, velocity=20),
    #         2: State(position=[40, 0], time_step=2, orientation=0, velocity=30),
    #     }
    #     lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}}
    #     other_vehicle_1 = Vehicle(
    #         1,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_1,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_1,
    #     )

    #     pred = PredDrivesFaster(self.config)
    #     vehicle_ids = [ego_vehicle.id, other_vehicle_1.id]

    #     world = World({ego_vehicle, other_vehicle_1}, self.road_network)

    #     sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
    #     sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

    #     sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
    #     sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

    #     sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
    #     sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    # def test_drives_with_slightly_higher_speed(self):
    #     self.config["slightly_higher_speed_difference"] = 5.55

    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False  # ego vehicle has lower velocity
    #     exp_sol_monitor_mode_2 = False  # ego vehicle has same velocity
    #     exp_sol_monitor_mode_3 = (
    #         True  # ego vehicle drives with only slightly higher speed
    #     )
    #     exp_sol_monitor_mode_4 = False  # ego vehicle drives too fast

    #     # ego vehicle
    #     cr_state_list_ego = {
    #         0: State(position=[0, 0], time_step=0, orientation=0, velocity=5),
    #         1: State(position=[5, 0], time_step=1, orientation=0, velocity=10),
    #         2: State(position=[15, 0], time_step=2, orientation=0, velocity=15),
    #         3: State(position=[30, 0], time_step=3, orientation=0, velocity=20),
    #     }
    #     lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
    #     ego_vehicle_param = self.config.get("ego_vehicle_param")
    #     ego_vehicle = Vehicle(
    #         0,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_ego,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_ego,
    #     )

    #     # other vehicle 1
    #     cr_state_list_other_1 = {
    #         0: State(position=[10, 0], time_step=0, orientation=0, velocity=10),
    #         1: State(position=[20, 0], time_step=1, orientation=0, velocity=10),
    #         2: State(position=[30, 0], time_step=2, orientation=0, velocity=10),
    #         3: State(position=[40, 0], time_step=3, orientation=0, velocity=10),
    #     }
    #     lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
    #     other_vehicle_1 = Vehicle(
    #         1,
    #         ObstacleType.CAR,
    #         ego_vehicle_param,
    #         Rectangle(5, 2),
    #         cr_state_list_other_1,
    #         None,
    #         CurvilinearStateManager(self.road_network),
    #         lanelet_assignments_other_1,
    #     )

    #     pred = PredDrivesWithSlightlyHigherSpeed(self.config)
    #     vehicle_ids = [ego_vehicle.id, other_vehicle_1.id]

    #     world = World({ego_vehicle, other_vehicle_1}, self.road_network)

    #     sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
    #     sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

    #     sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
    #     sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

    #     sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
    #     sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    #     sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
    #     sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)

    # TODO
    def test_causes_braking_intersection(self):
        scenario, _ = CommonRoadFileReader(
            str(
                "scenarios/test_intersection/DEU_Intersectionwithlightsandsigns-1_1_T-1.xml"
            )
        ).open(True)

        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )

        world = World.create_from_scenario(scenario)

        exp_sol_monitor_mode_1 = True  # all conditions met
        exp_sol_monitor_mode_2 = False  # a_p > a_br
        exp_sol_monitor_mode_3 = False  # d > d_br
        exp_sol_monitor_mode_4 = False  # d < 0

        # ego vehicle
        cr_state_list_p = {
            0: State(
                position=[4, 0],
                time_step=0,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=-2,
            ),
            1: State(
                position=[4, 0],
                time_step=1,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=1,
            ),
            2: State(
                position=[4, 0],
                time_step=2,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=-2,
            ),
            3: State(
                position=[6, 0],
                time_step=3,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=-2,
            ),
        }

        cr_state_list_k = {
            0: State(
                position=[6, 0],
                time_step=0,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=-2,
            ),
            1: State(
                position=[6, 0],
                time_step=1,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=1,
            ),
            2: State(
                position=[15, 0],
                time_step=2,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=-2,
            ),
            3: State(
                position=[4, 0],
                time_step=3,
                orientation=(0) * math.pi,
                velocity = 10,
                acceleration=-2,
            ),
        }

        lanelet_assignments_p = {0: {0}, 1: {0}, 2: {0}, 3: {0}}
        lanelet_assignments_k = {0: {0}, 1: {0}, 2: {0}, 3: {0}}

        # TODO: Params
        # ego_vehicle_param = self.config.get("ego_vehicle_param")

        vehicle_p = Vehicle(
            0,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_p,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_p,
        )
        vehicle_k = Vehicle(
            1,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_k,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_k,
        )

        vehicles = [0, 1]

        pred = PredCausesBrakingIntersection(self.config)

        world.add_vehicle(vehicle_p)
        world.add_vehicle(vehicle_k)
        # world = World({ego_vehicle}, self.road_network)

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicles)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicles)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 < 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicles)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicles)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        # ts = 2 : outside incoming => false
        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicles)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicles)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 < 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicles)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicles)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 < 0)
