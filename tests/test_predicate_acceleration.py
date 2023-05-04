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
from crmonitor.predicates.acceleration import (PredCausesBrakingIntersection)


class TestPriorityPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True

    def testCausesBrakingIntersection(self):
        scenario, _ = CommonRoadFileReader(
                str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(32)
        target_vehicle = world.vehicle_by_id(30)

        pred = PredCausesBrakingIntersection(self.config)
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id, target_vehicle.id])
            print(sol_monitor_1)

            sol_monitor_2 = pred.evaluate_robustness(world, time, [ego_vehicle.id, target_vehicle.id])
            print(sol_monitor_2)

