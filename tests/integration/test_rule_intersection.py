import logging
import unittest

import numpy as np
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.predicates.base import PredicateConfig
from tests.resources import IntersectionScenarios

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

_PREDICATE_CONFIG = PredicateConfig(scale_rob=True, d_sl=1.0)


class RuleTest(unittest.TestCase):
    def test_R_IN1(self):
        exp_violation_time_step = 24
        world = IntersectionScenarios.R_IN1.get_world()
        ego_vehicle = world.vehicle_by_id(31)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN1", world.scenario.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        self.assertTrue(rule_robustness[exp_violation_time_step - 1] >= 0)
        self.assertTrue(rule_robustness[exp_violation_time_step] < 0)

    def test_R_IN3(self):
        rtamt_further_time_range = 10
        exp_violation_time_step = 20 + rtamt_further_time_range
        exp_violation_end_time_step = 27 + rtamt_further_time_range
        world = IntersectionScenarios.R_IN3.get_world()
        ego_vehicle = world.vehicle_by_id(30)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN3_hand_draft", world.scenario.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0
        assert rule_robustness[exp_violation_end_time_step - 1] < 0
        assert rule_robustness[exp_violation_end_time_step] >= 0

    def test_R_IN4(self):
        exp_violation_time_step = 133
        world = IntersectionScenarios.R_IN4.get_world()
        ego_vehicle = world.vehicle_by_id(10093)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN4", world.scenario.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0

    def test_R_IN5(self):
        exp_violation_time_step = 68
        world = IntersectionScenarios.R_IN5.get_world()
        ego_vehicle = world.vehicle_by_id(10020)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN5", world.scenario.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0
