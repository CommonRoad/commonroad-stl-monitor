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
from crmonitor.evaluation.visualization import plot_rule_visualization


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
from crmonitor.monitor.rule import (
    parse_rule,
    RuleNode,
    PredicateNode,
    ExistNode,
    AllNode,
)
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
                str("../scenarios/test_intersection/DEU_TestRIN1-3_1_T-1.xml")).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
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
            print("-------------------------")
            print(i)
            print(rob)
        rule_robustness = np.array(rule_robustness)
        print(rule_robustness)

        # visualization_config = {'cut_in': {'show_non_effective_predicate_instances_for_vehicles': [(1004, 1000)]
        #                                    # we want to visualize the cut-in predicate always for the vehicle-pair (1004, 1000),
        #                                    # even if the predicate instance is not effective.
        #                                    }}
        # # For a more rectangular scenario:
        # scenario_fig_size = (25., 3.)
        # scenario_scale_compared_to_other_plots = 6
        # # For evaluating all rules together
        # flag_rule_conjunction = True
        #
        # current_time_step = 0
        # while current_time_step < 50:
        #     plot_rule_visualization(scenario, ego_vehicle.id, current_time_step, [rule_eval],
        #                             visualization_config=visualization_config, scenario_fig_size=scenario_fig_size,
        #                             bar_chart_plot_limits=(-1.0, 1.0), rule_robustness_course_plot_limits=(-1.0, 1.0),
        #                             flag_plot_predicate_bar_chart=True, flat_plot_rule_robustness_course=True,
        #                             scenario_plot_limits=None, flag_rule_conjunction=flag_rule_conjunction)
        #     current_time_step += 1
        #     plt.show()

    def test_META_1(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(
        #     lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml"
            )
        ).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        # ego_vehicle = world.vehicle_by_id(32)
        # target_vehicle = world.vehicle_by_id(30)
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)
        rule_eval = RuleEvaluator.create_from_config(world, ego_vehicle, "META_1")
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, AllNode))
        rule_robustness = []
        for i in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            rob = rule_eval.update()
            pred_rob = rule_eval.get_predicates()
            rule_robustness.append(rob)
            print("-------------------------------")
            print(i)
            print(rob)
        rule_robustness = np.array(rule_robustness)
        print(rule_robustness)

        # visualization_config = {'cut_in': {'show_non_effective_predicate_instances_for_vehicles': [(1004, 1000)]
        #                                    # we want to visualize the cut-in predicate always for the vehicle-pair (1004, 1000),
        #                                    # even if the predicate instance is not effective.
        #                                    }}
        # # For a more rectangular scenario:
        # scenario_fig_size = (15., 15.)
        # scenario_scale_compared_to_other_plots = 4
        # # For evaluating all rules together
        # flag_rule_conjunction = True
        #
        # current_time_step = 0
        # while current_time_step < 50:
        #     plot_rule_visualization(scenario, ego_vehicle.id, current_time_step, [rule_eval],
        #                             visualization_config=visualization_config, scenario_fig_size=scenario_fig_size,
        #                             bar_chart_plot_limits=(-1.0, 1.0), rule_robustness_course_plot_limits=(-1.0, 1.0),
        #                             flag_plot_predicate_bar_chart=True, flat_plot_rule_robustness_course=True,
        #                             scenario_plot_limits=[40, 80, -15, 15], flag_rule_conjunction=flag_rule_conjunction)
        #     current_time_step += 1
        #     plt.show()

    def test_R_IN3(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(
        #     lanelet_assignment=True)
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
        # target_vehicle = world.vehicle_by_id(30)
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)
        rule_eval = RuleEvaluator.create_from_config(world, ego_vehicle, "R_IN3")
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, AllNode))
        rule_robustness = []
        pred_robs = list()
        for i in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            rob = rule_eval.update()
            pred_rob = rule_eval.get_predicates()
            pred_robs.append(pred_rob)
            prob_rob = rule_eval.get_propositions()
            rule_robustness.append(rob)
            print("-------------------------------")
            print(i)
            print(rob)
        rule_robustness = np.array(rule_robustness)
        print(rule_robustness)

        # visualization_config = {'cut_in': {'show_non_effective_predicate_instances_for_vehicles': [(1004, 1000)]
        #                                    # we want to visualize the cut-in predicate always for the vehicle-pair (1004, 1000),
        #                                    # even if the predicate instance is not effective.
        #                                    }}
        # # For a more rectangular scenario:
        # scenario_fig_size = (15., 15.)
        # scenario_scale_compared_to_other_plots = 4
        # # For evaluating all rules together
        # flag_rule_conjunction = True
        #
        # current_time_step = 0
        # while current_time_step < 50:
        #     plot_rule_visualization(scenario, ego_vehicle.id, current_time_step, [rule_eval],
        #                             visualization_config=visualization_config, scenario_fig_size=scenario_fig_size,
        #                             bar_chart_plot_limits=(-1.0, 1.0), rule_robustness_course_plot_limits=(-1.0, 1.0),
        #                             flag_plot_predicate_bar_chart=True, flat_plot_rule_robustness_course=True,
        #                             scenario_plot_limits=[40, 80, -15, 15], flag_rule_conjunction=flag_rule_conjunction)
        #     current_time_step += 1
        #     plt.show()

    def test_R_IN4(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(
        #     lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml"
            )
        ).open(lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(
        #         str("../scenarios/test_intersection/DEU_TestIntersectionInteract-3_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        # ego_vehicle = world.vehicle_by_id(32)
        # target_vehicle = world.vehicle_by_id(30)
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)
        rule_eval = RuleEvaluator.create_from_config(world, ego_vehicle, "R_IN4")
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, AllNode))
        rule_robustness = []
        for i in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            rob = rule_eval.update()
            pred_rob = rule_eval.get_predicates()
            rule_robustness.append(rob)
            print("-------------------------------")
            print(i)
            print(rob)
        rule_robustness = np.array(rule_robustness)
        print(rule_robustness)

        # visualization_config = {'cut_in': {'show_non_effective_predicate_instances_for_vehicles': [(1004, 1000)]
        #                                    # we want to visualize the cut-in predicate always for the vehicle-pair (1004, 1000),
        #                                    # even if the predicate instance is not effective.
        #                                    }}
        # # For a more rectangular scenario:
        # scenario_fig_size = (15., 15.)
        # scenario_scale_compared_to_other_plots = 4
        # # For evaluating all rules together
        # flag_rule_conjunction = True
        #
        # current_time_step = 0
        # while current_time_step < 50:
        #     plot_rule_visualization(scenario, ego_vehicle.id, current_time_step, [rule_eval],
        #                             visualization_config=visualization_config, scenario_fig_size=scenario_fig_size,
        #                             bar_chart_plot_limits=(-1.0, 1.0), rule_robustness_course_plot_limits=(-1.0, 1.0),
        #                             flag_plot_predicate_bar_chart=True, flat_plot_rule_robustness_course=True,
        #                             scenario_plot_limits=[40, 80, -15, 15], flag_rule_conjunction=flag_rule_conjunction)
        #     current_time_step += 1
        #     plt.show()
