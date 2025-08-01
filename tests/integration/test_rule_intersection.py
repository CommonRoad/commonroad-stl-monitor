import logging

import numpy as np
from crmonitor.common import (
    World,
)
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from tests.resources import IntersectionScenarios

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


class TestRulesIntersection:
    def _base_test_rule(
        self, rule_name: str, world: World, ego_id: int, exp_violation: bool
    ) -> None:
        rule_eval = OfflineRuleEvaluator.create_for_rule(rule_name, world.dt)
        rule_robustness = np.array(rule_eval.evaluate(world, ego_id))
        bool_value = rule_robustness >= 0.0

        assert exp_violation == np.all(bool_value), (
            f"expected violation {exp_violation} but got violation {np.all(bool_value)} for robustness trace {rule_robustness}."
        )

    def test_R_IN1(self):
        exp_violation_time_step = 24
        world = IntersectionScenarios.R_IN1.get_world()
        ego_vehicle = world.vehicle_by_id(31)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN1", world.scenario.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0

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
