import numpy as np
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from tests.resources import IntersectionScenarios


class TestRulesIntersection:
    def test_R_IN1(self):
        exp_violation_time_step = 22
        world = IntersectionScenarios.R_IN1.get_world()
        ego_vehicle = world.vehicle_by_id(31)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN1", world.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0

    def test_R_IN3(self):
        exp_violation_time_step = 24
        exp_violation_end_time_step = 27
        world = IntersectionScenarios.R_IN3.get_world()
        ego_vehicle = world.vehicle_by_id(30)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN3_hand_draft", world.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0
        assert rule_robustness[exp_violation_end_time_step - 1] < 0
        assert rule_robustness[exp_violation_end_time_step] >= 0

    def test_R_IN4(self):
        exp_violation_time_step = 108
        world = IntersectionScenarios.R_IN4.get_world()
        ego_vehicle = world.vehicle_by_id(10093)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN4", world.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0

    def test_R_IN5(self):
        exp_violation_time_step = 43
        world = IntersectionScenarios.R_IN5.get_world()
        ego_vehicle = world.vehicle_by_id(10020)
        rule_eval = OfflineRuleEvaluator.create_for_rule("R_IN5", world.dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        rule_robustness = np.array(rule_robustness)
        assert rule_robustness[exp_violation_time_step - 1] >= 0
        assert rule_robustness[exp_violation_time_step] < 0
