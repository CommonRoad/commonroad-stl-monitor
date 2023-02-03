import math
import unittest
from pathlib import Path
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.common.file_reader import CommonRoadFileReader


from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.priority import (
    PredRelevantTrafficLight,
    PredSamePriority,
    PredHasPriority,
    PredRelevantTrafficLight,
)


class TestPriorityPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = False

        scenario, _ = CommonRoadFileReader(str("scenarios/test_intersection/DEU_Intersectionwithlightsandsigns-1_1_T-1.xml")).open(True)
        
        self.scenario = scenario
        self.road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))



    # TODO
    def test_same_priority(self):
        self.assertEqual(1, 1)

    # TODO
    def test_relevant_traffic_light(self):
        world = World.create_from_scenario(self.scenario)
        
        exp_sol_monitor_mode_1 = False  # traffic light inactive
        exp_sol_monitor_mode_2 = True
        exp_sol_monitor_mode_3 = False  # no traffic light

        # ego vehicle
        cr_state_list_ego = {
            0: State(
                position=[42, 3],
                time_step=0,
                orientation=(1) * math.pi,
                velocity=15,
            ),
            1: State(
                position=[13, 0],
                time_step=1,
                # orientation=(1 / 2) * math.pi, #straight
                # orientation=(3 / 4) * math.pi, #left
                orientation=(0) * math.pi,  # right
                velocity=15,
            ),
            2: State(
                position=[30, 0],
                time_step=2,
                orientation=(0) * math.pi,
                velocity=15,
            ),
        }

        lanelet_assignments_ego = {0: {10}, 1: {1}, 2: {9}}

        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_ego,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_ego,
        )

        pred = PredRelevantTrafficLight(self.config)

        world.add_vehicle(ego_vehicle)
        # world = World({ego_vehicle}, self.road_network)

        # ts = 0 : still in incoming => false
        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, [0])
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, [0])
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 < 0)

        # ts = 1 : inside the intersection on a lanelet going right => true
        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, [0])
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, [0])
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        # ts = 2 : outside incoming => false
        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, [0])
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, [0])
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 < 0)

    # TODO
    def test_has_priority(self):
        self.assertEqual(1, 1)


