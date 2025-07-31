import os
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.trajectory import Trajectory
from crmonitor.common import ScenarioType, World, WorldConfig
from crmonitor.predicates.base import PredicateConfig
from crmonitor.predicates.priority import (
    PredAtTrafficSignStop,  # not covered
    PredRelevantTrafficLight,  # not covered
    PredSamePriorityRightStraight,  # not covered
)
from tests.resources import IntersectionScenarios


class TestIntersectionPriorityPredicates(unittest.TestCase):
    def testAtTrafficSign(self):
        world = IntersectionScenarios.R_IN1.get_world()
        ego_vehicle = world.vehicle_by_id(31)

        pred = PredAtTrafficSignStop()
        for time in range(ego_vehicle.end_time + 1):
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id])
            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id])

            assert sol_monitor_1 == (sol_monitor_2 >= 0)

    def testRelevantTrafficLight(self):
        scenario = IntersectionScenarios.R_IN1.get_commonroad_scenario()
        dynamic_obstacle_initial_state = InitialState(
            position=np.array([5, 0.0]),
            velocity=15,
            orientation=0.0,
            time_step=0,
            yaw_rate=0.0,
            slip_angle=0.0,
        )
        state_list_ego = []
        state_list_ego.append(
            CustomState(position=np.array([17, 0.0]), velocity=15, orientation=0.0, time_step=1)
        )
        state_list_ego.append(
            CustomState(position=np.array([20, 0.0]), velocity=15, orientation=0.0, time_step=2)
        )
        state_list_ego.append(
            CustomState(position=np.array([21, 0.0]), velocity=10, orientation=0.0, time_step=3)
        )
        state_list_ego.append(
            CustomState(position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=4)
        )
        state_list_ego.append(
            CustomState(position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=5)
        )
        state_list_ego.append(
            CustomState(position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=6)
        )
        dynamic_obstacle_trajectory = Trajectory(1, state_list_ego)
        dynamic_obstacle_shape = Rectangle(width=2, length=5)
        dynamic_obstacle_prediction = TrajectoryPrediction(
            dynamic_obstacle_trajectory, dynamic_obstacle_shape
        )
        dynamic_obstacle_prediction.shape_lanelet_assignment = {
            0: {1},
            1: {1},
            2: {1, 3},
            3: {1, 3},
            4: {3},
            5: {3},
            6: {3},
        }
        dynamic_obstacle_prediction.center_lanelet_assignment = {
            0: {1},
            1: {1},
            2: {1, 3},
            3: {1, 3},
            4: {3},
            5: {3},
            6: {3},
        }
        dynamic_obstacle_id = scenario.generate_object_id()
        dynamic_obstacle_type = ObstacleType.CAR
        dynamic_obstacle = DynamicObstacle(
            dynamic_obstacle_id,
            dynamic_obstacle_type,
            dynamic_obstacle_shape,
            dynamic_obstacle_initial_state,
            dynamic_obstacle_prediction,
            initial_center_lanelet_ids={1},
            initial_shape_lanelet_ids={1},
        )
        scenario.add_objects(dynamic_obstacle)
        world = World.create_from_scenario(scenario, WorldConfig(ScenarioType.INTERSECTION))
        ego_vehicle = world.vehicle_by_id(dynamic_obstacle_id)

        pred = PredRelevantTrafficLight()
        for time in range(ego_vehicle.end_time + 1):
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id])

            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id])

            self.assertEqual(sol_monitor_1, sol_monitor_2 >= 0)

    def testSamePriority(self):
        world = IntersectionScenarios.R_IN3.get_world()
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)

        pred = PredSamePriorityRightStraight()
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id, target_vehicle.id])

            sol_monitor_2 = pred.evaluate_robustness(
                world, time, [ego_vehicle.id, target_vehicle.id]
            )

            self.assertEqual(sol_monitor_1, sol_monitor_2 >= 0)
