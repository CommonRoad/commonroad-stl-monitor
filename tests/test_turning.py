import math
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.general import (PredInterstateBroadEnough, PredTurningLeft, PredTurningRight,
                                          PredGoingStraight)


class TestTurning(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_sl"] = 1.0

    def testTurningRight(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurning-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestGoingStraight-1_1_T-1.xml")).open(
                lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(26)

        for time in range(ego_vehicle.end_time + 1):
            print(time)
            pred_test = PredTurningRight(self.config)
            sol_monitor_5 = pred_test.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_5)

            sol_monitor_6 = pred_test.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_6)

            print('-----------------------------------------')

    def testTurningLeft(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurning-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestGoingStraight-1_1_T-1.xml")).open(
                lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(26)

        for time in range(ego_vehicle.end_time + 1):
            print(time)
            pred_test = PredTurningLeft(self.config)
            sol_monitor_5 = pred_test.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_5)

            sol_monitor_6 = pred_test.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_6)

            print('-----------------------------------------')

    def testGoingStraight(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurning-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestGoingStraight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestIntersectionInteract-3_1_T-1.xml")).open(
                lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(30)

        for time in range(ego_vehicle.end_time + 1):
            print(time)
            pred_test = PredGoingStraight(self.config)
            sol_monitor_5 = pred_test.evaluate_boolean(world, time, [ego_vehicle.id])
            print(sol_monitor_5)

            sol_monitor_6 = pred_test.evaluate_robustness(world, time, [ego_vehicle.id])
            print(sol_monitor_6)

            print('-----------------------------------------')


