import math
import unittest
from pathlib import Path
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import State, ObstacleType

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.general import (
    PredInterstateBroadEnough,
    PredInCongestion,
    PredInSlowMovingTraffic,
    PredInQueueOfVehicles,
    PredMakesUTurn,
    # PredTurningLeft,
    # PredTurningRight,
    # PredGoingStraight,
)


class TestGeneralPredicates(unittest.TestCase):
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

    def test_interstate_broad_enough(self):
        self.config["min_interstate_width"] = 7.0

        # expected solutions
        exp_sol_1 = True
        exp_sol_2 = True
        exp_sol_3 = False

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 0], time_step=0, orientation=0, velocity=1),
            1: State(position=[10, 0], time_step=1, orientation=0, velocity=1),
            2: State(position=[20, 0], time_step=2, orientation=0, velocity=1),
            3: State(position=[30, 16], time_step=3, orientation=0, velocity=1),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {4}, 3: {4}}
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

        world = World({ego_vehicle}, self.road_network)

        # Right of broad lane markings
        pred = PredInterstateBroadEnough(self.config)
        vehicle_ids = [ego_vehicle.id]

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_1, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_2, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_3, sol_robustness_monitor_mode_3 > 0)

    def test_in_congestion(self):
        self.config["num_veh_congestion"] = 3
        self.config["max_congestion_velocity"] = 2.78

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle
        exp_sol_monitor_mode_2 = False  # only two leading vehicles
        exp_sol_monitor_mode_3 = (
            False  # three leading vehicle, but not all drive with required velocity
        )
        exp_sol_monitor_mode_4 = True

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 0], time_step=0, orientation=0, velocity=2),
            1: State(position=[2, 0], time_step=1, orientation=0, velocity=2),
            2: State(position=[4, 0], time_step=2, orientation=0, velocity=2),
            3: State(position=[6, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
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
            1: State(position=[12, 0], time_step=1, orientation=0, velocity=2),
            2: State(position=[14, 0], time_step=2, orientation=0, velocity=2),
            3: State(position=[16, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
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
            1: State(position=[22, 0], time_step=1, orientation=0, velocity=2),
            2: State(position=[24, 0], time_step=2, orientation=0, velocity=2),
            3: State(position=[26, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
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

        # other vehicle 3
        cr_state_list_other_3 = {
            2: State(position=[34, 0], time_step=2, orientation=0, velocity=5),
            3: State(position=[39, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(
            3,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_3,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_other_3,
        )

        pred = PredInCongestion(self.config)
        vehicle_ids = [ego_vehicle.id]

        world = World(
            {ego_vehicle, other_vehicle_1, other_vehicle_2, other_vehicle_3},
            self.road_network,
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

    def test_in_slow_moving_traffic(self):
        self.config["num_veh_slow_moving_traffic"] = 3
        self.config["max_slow_moving_traffic_velocity"] = 8.33

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle
        exp_sol_monitor_mode_2 = False  # only two leading vehicles
        exp_sol_monitor_mode_3 = (
            False  # three leading vehicle, but not all drive with required velocity
        )
        exp_sol_monitor_mode_4 = True

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 0], time_step=0, orientation=0, velocity=6),
            1: State(position=[2, 0], time_step=1, orientation=0, velocity=6),
            2: State(position=[4, 0], time_step=2, orientation=0, velocity=6),
            3: State(position=[6, 0], time_step=3, orientation=0, velocity=6),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
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
            1: State(position=[16, 0], time_step=1, orientation=0, velocity=6),
            2: State(position=[22, 0], time_step=2, orientation=0, velocity=6),
            3: State(position=[28, 0], time_step=3, orientation=0, velocity=6),
        }
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
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
            1: State(position=[26, 0], time_step=1, orientation=0, velocity=6),
            2: State(position=[32, 0], time_step=2, orientation=0, velocity=6),
            3: State(position=[38, 0], time_step=3, orientation=0, velocity=6),
        }
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
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

        # other vehicle 3
        cr_state_list_other_3 = {
            2: State(position=[42, 0], time_step=2, orientation=0, velocity=12),
            3: State(position=[48, 0], time_step=3, orientation=0, velocity=6),
        }
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(
            3,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_3,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_other_3,
        )

        pred = PredInSlowMovingTraffic(self.config)
        vehicle_ids = [ego_vehicle.id]

        world = World(
            {ego_vehicle, other_vehicle_1, other_vehicle_2, other_vehicle_3},
            self.road_network,
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

    def test_in_queue_of_vehicles(self):
        self.config["num_veh_queue_of_vehicles"] = 3
        self.config["max_queue_of_vehicles_velocity"] = 16.67

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle
        exp_sol_monitor_mode_2 = False  # only two leading vehicles
        exp_sol_monitor_mode_3 = (
            False  # three leading vehicle, but not all drive with required velocity
        )
        exp_sol_monitor_mode_4 = True

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 0], time_step=0, orientation=0, velocity=15),
            1: State(position=[15, 0], time_step=1, orientation=0, velocity=15),
            2: State(position=[30, 0], time_step=2, orientation=0, velocity=15),
            3: State(position=[45, 0], time_step=3, orientation=0, velocity=15),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
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
            1: State(position=[30, 0], time_step=1, orientation=0, velocity=15),
            2: State(position=[45, 0], time_step=2, orientation=0, velocity=15),
            3: State(position=[60, 0], time_step=3, orientation=0, velocity=15),
        }
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
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
            1: State(position=[45, 0], time_step=1, orientation=0, velocity=15),
            2: State(position=[60, 0], time_step=2, orientation=0, velocity=15),
            3: State(position=[75, 0], time_step=3, orientation=0, velocity=15),
        }
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
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

        # other vehicle 3
        cr_state_list_other_3 = {
            2: State(position=[75, 0], time_step=2, orientation=0, velocity=20),
            3: State(position=[80, 0], time_step=3, orientation=0, velocity=15),
        }
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(
            3,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_3,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_other_3,
        )

        pred = PredInQueueOfVehicles(self.config)
        vehicle_ids = [ego_vehicle.id]

        world = World(
            {ego_vehicle, other_vehicle_1, other_vehicle_2, other_vehicle_3},
            self.road_network,
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

    def test_makes_u_turn(self):
        self.config["u_turn"] = 1.57

        exp_sol_monitor_mode_1 = False  # theta = 0
        exp_sol_monitor_mode_2 = False  # theta = (1/8) * math.pi
        exp_sol_monitor_mode_3 = True  # theta = (1/2) * math.pi
        exp_sol_monitor_mode_4 = True  # theta = (3/4) * math.pi

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=[0, 0], time_step=0, orientation=0, velocity=15),
            1: State(
                position=[10, 0],
                time_step=1,
                orientation=(1 / 8) * math.pi,
                velocity=15,
            ),
            2: State(
                position=[20, 0],
                time_step=2,
                orientation=(1 / 2) * math.pi,
                velocity=15,
            ),
            3: State(
                position=[30, 0],
                time_step=3,
                orientation=(3 / 4) * math.pi,
                velocity=15,
            ),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
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

        pred = PredMakesUTurn(self.config)
        vehicle_ids = [ego_vehicle.id]

        world = World({ego_vehicle}, self.road_network)

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

    # TODO
    def test_going_straight(self):
        self.assertEqual(1, 1)

    # TODO
    def test_turning_right(self):

        self.assertEqual(1, 1)

    # TODO
    def test_turning_left(self):
        self.assertEqual(1, 1)
