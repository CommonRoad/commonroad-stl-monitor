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
        scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-1_1_T-1.xml")).open(
            lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurnRight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(31)

        for time in range(ego_vehicle.end_time + 1):
            vehicle_states = ego_vehicle.state_list_cr
            vehicle_state = vehicle_states[time]
            test = ego_vehicle.state_list_cr[time]
            vehicle_position = test.position
            pred = PredStopLineInFront(self.config)
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_2)

    def testOnIncomingLeftOf(self):
        scenario, _ = CommonRoadFileReader(
                str("../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml")).open(
                lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            print('------------------------------------')
            print(time)
            pred = PredOnIncomingLeftOf(self.config)
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id, target_vehicle.id])
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id, target_vehicle.id])
            print(sol_monitor_2)

    def testInIntersectionConflictArea(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(
            str("../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml")).open(
                lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        # ego_vehicle = world.vehicle_by_id(30)
        # target_vehicle = world.vehicle_by_id(32)
        ego_vehicle = world.vehicle_by_id(31)
        target_vehicle = world.vehicle_by_id(30)
        rob = list()
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            print('---------------------------------')
            print(time)
            pred = PredInIntersectionConflictArea(self.config)
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id, target_vehicle.id])
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id, target_vehicle.id])
            rob.append(sol_monitor_2)
            print(sol_monitor_2)
        # fig = plt.figure()
        # ax = fig.gca()
        # ax.plot(range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1), rob, 'b-')
        # for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
        #     if rob[time] <= 0:
        #         ax.plot(time, rob[time], "rx")
        #     else:
        #         ax.plot(time, rob[time], "g.")
        # ax.plot([24, 24], [-1, 1], 'black')
        # ax.set_ylim([-1, 1])
        # ax.grid(True)
        # ax.set_title('in_intersection_conflict_area__a1_a0')
        # plt.show()

    def testOnLaneletWithTypeIntersection(self):
        scenario, _ = CommonRoadFileReader(
            str("../scenarios/test_intersection/DEU_TestTurnRight-1_1_T-1.xml")).open(
                lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(31)
        for time in range(ego_vehicle.end_time + 1):
            pred = PredOnLaneletWithTypeIntersection(self.config)
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_2)


    def testOnOncomOf(self):
        scenario, _ = CommonRoadFileReader(
            str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(
            lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(
        #         str("../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(32)
        target_vehicle = world.vehicle_by_id(31)
        # ego_vehicle = world.vehicle_by_id(30)
        # target_vehicle = world.vehicle_by_id(31)
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            pred = PredOnOncomOf(self.config)
            sol_monitor_1 = pred.evaluate_boolean(world, time, [target_vehicle.id, ego_vehicle.id])
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(world, time, [target_vehicle.id, ego_vehicle.id])
            print(sol_monitor_2)
