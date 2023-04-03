import math
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import (LaneletNetwork, LineMarking, Lanelet, LaneletType, )
from commonroad.scenario.obstacle import State, ObstacleType

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
from crmonitor.predicates.position import (PredRightOfBroadLaneMarking, PredLeftOfBroadLaneMarking,
                                           PredOnLaneletWithTypeIntersection, PredInIntersectionConflictArea,
                                           PredOnIncomingLeftOf, PredOnOncomOf, PredStopLineInFront)


class TestPositionPredicates(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_sl"] = 1.0

    def testStopLineInFront(self):
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestStopLineInFront-1_1_T-1.xml"
            )
        ).open(True)
        dynamic_obstacle_initial_state = CustomState(position=np.array([5, 0.0]),
                                                     velocity=15,
                                                     orientation=0.0,
                                                     time_step=0)
        state_list_ego = []
        state_list_ego.append(CustomState(position=np.array([17, 0.0]),
                                          velocity=15,
                                          orientation=0.0,
                                          time_step=1))
        state_list_ego.append(CustomState(position=np.array([23, 0.0]),
                                          velocity=15,
                                          orientation=0.0,
                                          time_step=2))
        dynamic_obstacle_trajectory = Trajectory(1, state_list_ego)
        dynamic_obstacle_shape = Rectangle(width=2, length=5)
        dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)
        dynamic_obstacle_prediction.shape_lanelet_assignment = {0: {1}, 1: {1}, 2: {3}}
        dynamic_obstacle_prediction.center_lanelet_assignment = {0: {1}, 1: {1}, 2: {3}}
        dynamic_obstacle_id = scenario.generate_object_id()
        dynamic_obstacle_type = ObstacleType.CAR
        dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                           dynamic_obstacle_type,
                                           dynamic_obstacle_shape,
                                           dynamic_obstacle_initial_state,
                                           dynamic_obstacle_prediction,
                                           initial_center_lanelet_ids={1},
                                           initial_shape_lanelet_ids={1})
        scenario.add_objects(dynamic_obstacle)
        # plt.figure(figsize=(25, 10))
        # rnd = MPRenderer()
        # rnd.draw_params.time_begin = 2
        # scenario.draw(rnd)
        # rnd.render()
        # plt.show()
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(dynamic_obstacle_id)

        for time in range(ego_vehicle.end_time + 1):
            pred = PredStopLineInFront(self.config)
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_2)


