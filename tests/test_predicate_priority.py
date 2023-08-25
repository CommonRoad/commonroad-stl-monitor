import math
import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import State, ObstacleType

import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.priority import (  # not covered
    PredRelevantTrafficLight,  # not covered
    PredHasPriorityRightRight,
    PredHasPriorityRightLeft,
    PredHasPriorityRightStraight,  # not covered
    # not covered
    PredHasPriorityLeftStraight,
    PredHasPriorityStraightRight,  # not covered
    PredHasPriorityStraightStraight,
    PredSamePriorityRightRight,
    PredSamePriorityRightLeft,
    PredSamePriorityRightStraight,  # not covered
    PredSamePriorityLeftStraight,
    PredSamePriorityStraightRight,  # not covered
    PredSamePriorityStraightStraight,
    PredAtTrafficSignStop,
)


class TestPriorityPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True

    def testAtTrafficSign(self):
        scenario, _ = CommonRoadFileReader(
            str("../scenarios/test_intersection/DEU_TestRIN1-1_1_T-1.xml")
        ).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        # scenario, _ = CommonRoadFileReader(
        #         str("../scenarios/test_intersection/DEU_TestStopLineInFront-1_1_T-1.xml")).open(True)
        # dynamic_obstacle_initial_state = CustomState(position=np.array([5, 0.0]),
        #                                              velocity=15,
        #                                              orientation=0.0,
        #                                              time_step=0)
        # state_list_ego = []
        # state_list_ego.append(CustomState(position=np.array([17, 0.0]), velocity=15, orientation=0.0, time_step=1))
        # state_list_ego.append(CustomState(position=np.array([20, 0.0]), velocity=15, orientation=0.0, time_step=2))
        # state_list_ego.append(CustomState(position=np.array([21, 0.0]), velocity=10, orientation=0.0, time_step=3))
        # state_list_ego.append(CustomState(position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=4))
        # state_list_ego.append(CustomState(position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=5))
        # state_list_ego.append(CustomState(position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=6))
        # dynamic_obstacle_trajectory = Trajectory(1, state_list_ego)
        # dynamic_obstacle_shape = Rectangle(width=2, length=5)
        # dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)
        # dynamic_obstacle_prediction.shape_lanelet_assignment = {0: {1}, 1: {1}, 2: {1, 3}, 3: {1, 3}, 4: {3}, 5: {3},
        #                                                         6: {3}}
        # dynamic_obstacle_prediction.center_lanelet_assignment = {0: {1}, 1: {1}, 2: {1, 3}, 3: {1, 3}, 4: {3}, 5: {3},
        #                                                          6: {3}}
        # dynamic_obstacle_id = scenario.generate_object_id()
        # dynamic_obstacle_type = ObstacleType.CAR
        # dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id, dynamic_obstacle_type, dynamic_obstacle_shape,
        #                                    dynamic_obstacle_initial_state, dynamic_obstacle_prediction,
        #                                    initial_center_lanelet_ids={1}, initial_shape_lanelet_ids={1})
        # scenario.add_objects(
        #     dynamic_obstacle)
        # # plt.figure(figsize=(25, 10))
        # # rnd = MPRenderer()
        # # rnd.draw_params.time_begin = 2
        # # scenario.draw(rnd)
        # # rnd.render()
        # # plt.show()
        #
        # world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        ego_vehicle = world.vehicle_by_id(31)

        pred = PredAtTrafficSignStop(self.config)
        for time in range(ego_vehicle.end_time + 1):
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_1)
            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_2)

    def testRelevantTrafficLight(self):
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestRelevantTrafficLight-1_1_T-1.xml"
            )
        ).open(True)
        # str("../scenarios/test_intersection/DEU_TestStopLineInFront-1_1_T-1.xml")).open(True)
        dynamic_obstacle_initial_state = CustomState(
            position=np.array([5, 0.0]), velocity=15, orientation=0.0, time_step=0
        )
        state_list_ego = []
        state_list_ego.append(
            CustomState(
                position=np.array([17, 0.0]), velocity=15, orientation=0.0, time_step=1
            )
        )
        state_list_ego.append(
            CustomState(
                position=np.array([20, 0.0]), velocity=15, orientation=0.0, time_step=2
            )
        )
        state_list_ego.append(
            CustomState(
                position=np.array([21, 0.0]), velocity=10, orientation=0.0, time_step=3
            )
        )
        state_list_ego.append(
            CustomState(
                position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=4
            )
        )
        state_list_ego.append(
            CustomState(
                position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=5
            )
        )
        state_list_ego.append(
            CustomState(
                position=np.array([23, 0.0]), velocity=0, orientation=0.0, time_step=6
            )
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
        # plt.figure(figsize=(25, 10))
        # rnd = MPRenderer()
        # rnd.draw_params.time_begin = 2
        # scenario.draw(rnd)
        # rnd.render()
        # plt.show()

        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        ego_vehicle = world.vehicle_by_id(dynamic_obstacle_id)

        pred = PredRelevantTrafficLight(self.config)
        for time in range(ego_vehicle.end_time + 1):
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_1)

        for time in range(ego_vehicle.end_time + 1):
            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_2)

    def testSamePriority(self):
        # scenario, _ = CommonRoadFileReader(
        #         str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(True)
        # scenario, _ = CommonRoadFileReader(
        #     str("../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestIntersectionInteract-3_1_T-1.xml"
            )
        ).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        # ego_vehicle = world.vehicle_by_id(32)
        # target_vehicle = world.vehicle_by_id(31)
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)

        pred = PredSamePriorityRightStraight(self.config)
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            sol_monitor_1 = pred.evaluate_boolean(
                world, time, [ego_vehicle.id, target_vehicle.id]
            )
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(
                world, time, [ego_vehicle.id, target_vehicle.id]
            )
            print(sol_monitor_2)
