import unittest

import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.state import CustomState
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, VehicleParameters
from crmonitor.common.world import World
from crmonitor.predicates.base import PredicateConfig
from crmonitor.predicates.velocity import (
    PredDrivesFaster,
    PredDrivesWithSlightlyHigherSpeed,
    PredExistStandingLeadingVehicle,
    PredInStandStill,
    PredPreservesTrafficFlow,
    PredReverses,
    PredSlowLeadingVehicle,
)


class TestInterstateVelocityPredicates(unittest.TestCase):
    def setUp(self) -> None:
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
        self.road_network = RoadNetwork(lanelet_network)

    def test_reverses(self):
        predicate_config = PredicateConfig(scale_rob=False, standstill_error=0.001)
        dt = 0.1
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego has velocity of zero
        exp_sol_monitor_mode_2 = False  # ego vehicle has positive velocity
        exp_sol_monitor_mode_3 = False  # ego vehicle has velocity of -standstill_error
        exp_sol_monitor_mode_4 = True  # ego vehicle has velocity smaller than -standstill_error

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=[0, 0], time_step=0, orientation=0, velocity=0),
            1: CustomState(position=[0, 0], time_step=1, orientation=0, velocity=1),
            2: CustomState(
                position=[1, 0],
                time_step=2,
                orientation=0,
                velocity=-predicate_config.standstill_error,
            ),
            3: CustomState(
                position=[1 - predicate_config.standstill_error, 0],
                time_step=3,
                orientation=0,
                velocity=-2,
            ),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {2}}
        ego_vehicle = Vehicle(
            id=0,
            obstacle_type=ObstacleType.CAR,
            vehicle_param=VehicleParameters.create_for_ego_vehicle(dt),
            dt=0.1,
            road_network=self.road_network,
            shape=Rectangle(5, 2),
            states_cr=cr_state_list_ego,
            lanelet_assignment=lanelet_assignments_ego,
        )

        pred = PredReverses(predicate_config)
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

    def test_slow_leading_vehicle(self):
        predicate_config = PredicateConfig(
            scale_rob=False,
            min_velocity_diff=15,
            desired_interstate_velocity=36.11,
            country="DEU",
        )
        ego_vehicle_params = VehicleParameters.create_for_ego_vehicle(
            dt=0.1, road_condition_speed_limit=50.0
        )

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle at all
        exp_sol_monitor_mode_2 = False  # two leading vehicles which drive with speed limit
        exp_sol_monitor_mode_3 = True  # first leading vehicle is drives to slow
        exp_sol_monitor_mode_4 = True  # third leading vehicle drives to slow

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=[5, 0], time_step=0, orientation=0, velocity=2),
            1: CustomState(position=[7, 0], time_step=1, orientation=0, velocity=2),
            2: CustomState(position=[9, 0], time_step=2, orientation=0, velocity=2),
            3: CustomState(position=[11, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_ego,
            lanelet_assignments_ego,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_params,
        )

        # other vehicle 1
        cr_state_list_other_1 = {
            1: CustomState(position=[12, 0], time_step=1, orientation=0, velocity=50),
            2: CustomState(position=[62, 0], time_step=2, orientation=0, velocity=12),
            3: CustomState(position=[74, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_1,
            lanelet_assignments_other_1,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_params,
        )

        # other vehicle 2
        cr_state_list_other_2 = {
            1: CustomState(position=[22, 0], time_step=1, orientation=0, velocity=36),
            2: CustomState(position=[58, 0], time_step=2, orientation=0, velocity=36),
            3: CustomState(position=[88, 0], time_step=3, orientation=0, velocity=36),
        }
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_2 = Vehicle(
            2,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_2,
            lanelet_assignments_other_2,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_params,
        )

        # other vehicle 3
        cr_state_list_other_3 = {
            2: CustomState(position=[34, 0], time_step=2, orientation=0, velocity=50),
            3: CustomState(position=[84, 0], time_step=3, orientation=0, velocity=0),
        }
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(
            3,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_3,
            lanelet_assignments_other_3,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_params,
        )

        # other vehicle 4
        cr_state_list_other_4 = {
            0: CustomState(position=[0, 0], time_step=0, orientation=0, velocity=60),
            1: CustomState(position=[50, 3.5], time_step=1, orientation=0, velocity=0),
        }
        lanelet_assignments_other_4 = {0: {1}, 1: {2}}
        other_vehicle_4 = Vehicle(
            4,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_4,
            lanelet_assignments_other_4,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_params,
        )

        pred = PredSlowLeadingVehicle(predicate_config)
        vehicle_ids = [ego_vehicle.id]

        world = World(
            {
                ego_vehicle,
                other_vehicle_1,
                other_vehicle_2,
                other_vehicle_3,
                other_vehicle_4,
            },
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

    def test_preserves_traffic_flow(self):
        predicate_config = PredicateConfig(
            scale_rob=False,
            min_velocity_diff=15,
            country="DEU",
            desired_interstate_velocity=36.11,
        )

        ego_vehicle_params = VehicleParameters.create_for_ego_vehicle(
            dt=0.1, braking_speed_limit=50, fov_speed_limit=35, road_condition_speed_limit=50
        )

        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle drives too slow
        exp_sol_monitor_mode_2 = False  # ego vehicle drives at lower limit to
        exp_sol_monitor_mode_3 = True  # ego vehcie drives faster than velocity limit

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=[0, 0], time_step=0, orientation=0, velocity=2),
            1: CustomState(position=[2, 0], time_step=1, orientation=0, velocity=20),
            2: CustomState(position=[22, 0], time_step=2, orientation=0, velocity=50),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_ego,
            lanelet_assignments_ego,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_params,
        )

        pred = PredPreservesTrafficFlow(predicate_config)
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

    def test_in_standstill(self):
        predicate_config = PredicateConfig(scale_rob=False, standstill_error=0.01)
        # expected solutions
        exp_sol_monitor_mode_1 = False
        exp_sol_monitor_mode_2 = True
        exp_sol_monitor_mode_3 = True
        exp_sol_monitor_mode_4 = True

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=[0, 0], time_step=0, orientation=0, velocity=1),
            1: CustomState(position=[1, 0], time_step=1, orientation=0, velocity=0),
            2: CustomState(position=[1, 0], time_step=2, orientation=0, velocity=0.001),
            3: CustomState(position=[1, 0], time_step=3, orientation=0, velocity=-0.001),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle_param = VehicleParameters.create_for_ego_vehicle(dt=0.1)
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_ego,
            lanelet_assignments_ego,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        pred = PredInStandStill(predicate_config)
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

    def test_exists_standing_leading_vehicle(self):
        predicate_config = PredicateConfig(scale_rob=False, standstill_error=0.01)

        # expected solutions
        exp_sol_monitor_mode_1 = False  # no leading vehicle at all
        exp_sol_monitor_mode_2 = False  # two leading vehicles which have velocity > 0
        exp_sol_monitor_mode_3 = True  # first leading vehicle is standing
        exp_sol_monitor_mode_4 = True  # third leading vehicle is standing

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=[5, 0], time_step=0, orientation=0, velocity=2),
            1: CustomState(position=[7, 0], time_step=1, orientation=0, velocity=2),
            2: CustomState(position=[9, 0], time_step=2, orientation=0, velocity=2),
            3: CustomState(position=[11, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle_param = VehicleParameters.create_for_ego_vehicle(dt=0.1)
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_ego,
            lanelet_assignments_ego,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        # other vehicle 1
        cr_state_list_other_1 = {
            1: CustomState(position=[12, 0], time_step=1, orientation=0, velocity=2),
            2: CustomState(position=[14, 0], time_step=2, orientation=0, velocity=0),
            3: CustomState(position=[14, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_other_1 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_1,
            lanelet_assignments_other_1,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        # other vehicle 2
        cr_state_list_other_2 = {
            1: CustomState(position=[22, 0], time_step=1, orientation=0, velocity=2),
            2: CustomState(position=[24, 0], time_step=2, orientation=0, velocity=2),
            3: CustomState(position=[26, 0], time_step=3, orientation=0, velocity=2),
        }
        lanelet_assignments_other_2 = {1: {1}, 2: {1}, 3: {1}}
        other_vehicle_2 = Vehicle(
            2,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_2,
            lanelet_assignments_other_2,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        # other vehicle 3
        cr_state_list_other_3 = {
            2: CustomState(position=[34, 0], time_step=2, orientation=0, velocity=2),
            3: CustomState(position=[36, 0], time_step=3, orientation=0, velocity=0),
        }
        lanelet_assignments_other_3 = {2: {1}, 3: {1}}
        other_vehicle_3 = Vehicle(
            3,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_3,
            lanelet_assignments_other_3,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        # other vehicle 4
        cr_state_list_other_4 = {
            0: CustomState(position=[0, 0], time_step=0, orientation=0, velocity=60),
            1: CustomState(position=[50, 3.5], time_step=1, orientation=0, velocity=0),
        }
        lanelet_assignments_other_4 = {0: {1}, 1: {2}}
        other_vehicle_4 = Vehicle(
            4,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_4,
            lanelet_assignments_other_4,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        pred = PredExistStandingLeadingVehicle(predicate_config)
        vehicle_ids = [ego_vehicle.id]

        world = World(
            {
                ego_vehicle,
                other_vehicle_1,
                other_vehicle_2,
                other_vehicle_3,
                other_vehicle_4,
            },
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

    def test_drives_faster(self):
        predicate_config = PredicateConfig(scale_rob=False, standstill_error=0.001)
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle has lower velocity
        exp_sol_monitor_mode_2 = False  # ego vehicle has same velocity
        exp_sol_monitor_mode_3 = True  # ego vehicle drives with higher speed

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=[0, 0], time_step=0, orientation=0, velocity=5),
            1: CustomState(position=[5, 0], time_step=1, orientation=0, velocity=20),
            2: CustomState(position=[25, 0], time_step=2, orientation=0, velocity=35),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}}
        ego_vehicle_param = VehicleParameters.create_for_ego_vehicle(dt=0.1)
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_ego,
            lanelet_assignments_ego,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        # other vehicle 1
        cr_state_list_other_1 = {
            0: CustomState(position=[10, 0], time_step=0, orientation=0, velocity=10),
            1: CustomState(position=[20, 0], time_step=1, orientation=0, velocity=20),
            2: CustomState(position=[40, 0], time_step=2, orientation=0, velocity=30),
        }
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}}
        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_1,
            lanelet_assignments_other_1,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        pred = PredDrivesFaster(predicate_config)
        vehicle_ids = [ego_vehicle.id, other_vehicle_1.id]

        world = World({ego_vehicle, other_vehicle_1}, self.road_network)

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

    def test_drives_with_slightly_higher_speed(self):
        predicate_config = PredicateConfig(
            scale_rob=False,
            standstill_error=0.001,
            slightly_higher_speed_difference=5.55,
        )

        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle has lower velocity
        exp_sol_monitor_mode_2 = False  # ego vehicle has same velocity
        exp_sol_monitor_mode_3 = True  # ego vehicle drives with only slightly higher speed
        exp_sol_monitor_mode_4 = False  # ego vehicle drives too fast

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=[0, 0], time_step=0, orientation=0, velocity=5),
            1: CustomState(position=[5, 0], time_step=1, orientation=0, velocity=10),
            2: CustomState(position=[15, 0], time_step=2, orientation=0, velocity=15),
            3: CustomState(position=[30, 0], time_step=3, orientation=0, velocity=20),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        ego_vehicle_param = VehicleParameters.create_for_ego_vehicle(dt=0.1)
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_ego,
            lanelet_assignments_ego,
            self.road_network,
            0.1,
            vehicle_param=ego_vehicle_param,
        )

        # other vehicle 1
        cr_state_list_other_1 = {
            0: CustomState(position=[10, 0], time_step=0, orientation=0, velocity=10),
            1: CustomState(position=[20, 0], time_step=1, orientation=0, velocity=10),
            2: CustomState(position=[30, 0], time_step=2, orientation=0, velocity=10),
            3: CustomState(position=[40, 0], time_step=3, orientation=0, velocity=10),
        }
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            Rectangle(5, 2),
            cr_state_list_other_1,
            lanelet_assignments_other_1,
            self.road_network,
            dt=0.1,
            vehicle_param=ego_vehicle_param,
        )

        pred = PredDrivesWithSlightlyHigherSpeed(predicate_config)
        vehicle_ids = [ego_vehicle.id, other_vehicle_1.id]

        world = World({ego_vehicle, other_vehicle_1}, self.road_network)

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


class TestPredSlowLeadingVehicle:
    def test_evaluate_robustness(self): ...
