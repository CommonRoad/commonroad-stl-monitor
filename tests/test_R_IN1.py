import logging
import os
import unittest
from pathlib import Path

import numpy as np
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.state import CustomState
from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.monitor.rule import parse_rule, RuleNode, PredicateNode, ExistNode, AllNode
from tests.util import parallel_lanes

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

class RuleTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = root_path / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_sl"] = 1.0

        rules_path = root_path / "traffic_rules_rtamt.yaml"
        self.traffic_rules = load_yaml(str(rules_path))
        self.scenario_root_path = root_path.parent / "scenarios"

    def test_R_IN1(self):
        scenario, _ = CommonRoadFileReader(
                str("../scenarios/test_intersection/DEU_TestRIN1-1_1_T-1.xml")).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)

        # scenario, _ = CommonRoadFileReader(
        #         str("../scenarios/test_intersection/DEU_TestStopLineInFront-1_1_T-1.xml")).open(True)
        # dynamic_obstacle_initial_state = CustomState(position=np.array([5, 0.0]), velocity=15, orientation=0.0,
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
        # dynamic_obstacle_prediction.shape_lanelet_assignment = {0: {1}, 1: {1}, 2: {1, 3}, 3: {1, 3}, 4: {3}, 5: {3}, 6: {3}}
        # dynamic_obstacle_prediction.center_lanelet_assignment = {0: {1}, 1: {1}, 2: {1, 3}, 3: {1, 3}, 4: {3}, 5: {3}, 6: {3}}
        # dynamic_obstacle_id = scenario.generate_object_id()
        # dynamic_obstacle_type = ObstacleType.CAR
        # dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id, dynamic_obstacle_type, dynamic_obstacle_shape,
        #                                    dynamic_obstacle_initial_state, dynamic_obstacle_prediction,
        #                                    initial_center_lanelet_ids={1}, initial_shape_lanelet_ids={1})
        # scenario.add_objects(dynamic_obstacle)
        # # plt.figure(figsize=(25, 10))
        # # rnd = MPRenderer()
        # # rnd.draw_params.time_begin = 3
        # # scenario.draw(rnd)
        # # rnd.render()
        # # plt.show()
        #
        # world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(31)
        rule_eval = RuleEvaluator.create_from_config(world, ego_vehicle, "R_IN1")
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, RuleNode))
        self.assertEqual(len(rule.children), 4)
        self.assertTrue(all([isinstance(c, PredicateNode) for c in rule.children]))
        rule_robustness = []
        for i in range(ego_vehicle.end_time + 1):
            rob = rule_eval.update()
            rule_robustness.append(rob)
        rule_robustness = np.array(rule_robustness)
        print(rule_robustness)
