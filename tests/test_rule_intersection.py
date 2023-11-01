import os
import logging
import unittest
from pathlib import Path

import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from crmonitor.evaluation.proposition_evaluation import PropositionRuleEvaluator
from crmonitor.rule.rule_node import (
    RuleNode,
    PredicateNode,
    ExistNode,
    AllNode,
)

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
        self.config["scenario"] = "intersection"

        rules_path = root_path / "traffic_rules_rtamt.yaml"
        self.traffic_rules = load_yaml(str(rules_path))
        self.traffic_rules["traffic_rules_param"]["use_mpr"] = False
        self.scenario_root_path = root_path.parent / "scenarios"

    def test_R_IN1(self):
        exp_violation_time_step = 24
        self.config["intersection_road_network_param"]["map_type"] = "hand_draft"
        scenario_file = os.path.join(
            self.scenario_root_path, "test_intersection/DEU_TestRIN1-3_1_T-1.xml"
        )
        scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(31)
        rule_eval = PropositionRuleEvaluator.create_from_config(
            world, ego_vehicle, "R_IN1", self.traffic_rules
        )
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, RuleNode))
        self.assertEqual(len(rule.children), 4)
        self.assertTrue(all([isinstance(c, PredicateNode) for c in rule.children]))
        rule_robustness = list()
        prob_robs = list()
        for _ in range(
            rule_eval.ego_vehicle.start_time, rule_eval.ego_vehicle.end_time + 1
        ):
            rob = rule_eval.update()
            prob_rob = rule_eval.get_propositions()
            prob_robs.append(prob_rob)
            rule_robustness.append(rob)
        rule_robustness = np.array(rule_robustness)
        self.assertTrue(rule_robustness[exp_violation_time_step - 1] >= 0)
        self.assertTrue(rule_robustness[exp_violation_time_step] < 0)

    def test_R_IN3(self):
        self.config["intersection_road_network_param"]["map_type"] = "hand_draft"
        rtamt_further_time_range = 10
        exp_violation_time_step = 20 + rtamt_further_time_range
        exp_violation_end_time_step = 27 + rtamt_further_time_range
        scenario_file = os.path.join(
            self.scenario_root_path,
            "test_intersection/DEU_TestIntersectionRIN3.xml",
        )
        scenario, _ = CommonRoadFileReader(scenario_file).open(True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(30)
        rule_eval = PropositionRuleEvaluator.create_from_config(
            world, ego_vehicle, "R_IN3_hand_draft", self.traffic_rules
        )
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, AllNode))
        rule_robustness = list()
        pred_robs = list()
        prob_robs = list()
        for _ in range(
            rule_eval.ego_vehicle.start_time, rule_eval.ego_vehicle.end_time + 1
        ):
            rob = rule_eval.update()
            pred_rob = rule_eval.get_predicates()
            pred_robs.append(pred_rob)
            prob_rob, _, _ = rule_eval.get_propositions()
            prob_robs.append([prob_rob[prop_name] for prop_name in prob_rob.keys()])
            rule_robustness.append(rob)
        rule_robustness = np.array(rule_robustness)
        self.assertTrue(rule_robustness[exp_violation_time_step - 1] >= 0)
        self.assertTrue(rule_robustness[exp_violation_time_step] < 0)
        self.assertTrue(rule_robustness[exp_violation_end_time_step - 1] < 0)
        self.assertTrue(rule_robustness[exp_violation_end_time_step] >= 0)

    def test_R_IN4(self):
        self.config["intersection_road_network_param"]["map_type"] = "dataset"
        exp_violation_time_step = 133
        scenario_file = os.path.join(
            self.scenario_root_path,
            "test_intersection/DEU_AAH1-2_176000_T-6149.xml",
        )
        scenario, _ = CommonRoadFileReader(scenario_file).open(True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(10093)
        rule_eval = PropositionRuleEvaluator.create_from_config(
            world, ego_vehicle, "R_IN4", self.traffic_rules
        )
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, AllNode))
        rule_robustness = list()
        for _ in range(
            rule_eval.ego_vehicle.start_time, rule_eval.ego_vehicle.end_time + 1
        ):
            rob = rule_eval.update()
            rule_robustness.append(rob)
        rule_robustness = np.array(rule_robustness)
        self.assertTrue(rule_robustness[exp_violation_time_step - 1] >= 0)
        self.assertTrue(rule_robustness[exp_violation_time_step] < 0)

    def test_R_IN5(self):
        self.config["intersection_road_network_param"]["map_type"] = "dataset"
        exp_violation_time_step = 68
        scenario_file = os.path.join(
            self.scenario_root_path,
            "test_intersection/DEU_AAH1-2_7900_T-1049.xml",
        )
        scenario, _ = CommonRoadFileReader(scenario_file).open(True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(10020)
        rule_eval = PropositionRuleEvaluator.create_from_config(
            world, ego_vehicle, "R_IN5", traffic_rules_config=self.traffic_rules
        )
        rule = rule_eval._rule
        self.assertTrue(isinstance(rule, AllNode))
        rule_robustness = list()
        prob_robs = list()
        for _ in range(
            rule_eval.ego_vehicle.start_time, rule_eval.ego_vehicle.end_time + 1
        ):
            rob = rule_eval.update()
            rule_robustness.append(rob)
            prob_rob, _, _ = rule_eval.get_propositions()
            prob_robs.append([prob_rob[prop_name] for prop_name in prob_rob.keys()])
        rule_robustness = np.array(rule_robustness)
        self.assertTrue(rule_robustness[exp_violation_time_step - 1] >= 0)
        self.assertTrue(rule_robustness[exp_violation_time_step] < 0)
