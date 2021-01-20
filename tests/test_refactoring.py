import logging
import unittest
from typing import List, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.evaluation import RuleSetEvaluator
from crmonitor.common.helper import load_yaml
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.rule import Rule

logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)


def check_violation(rob_values: List[Tuple[float, float]]):
    bool_values = [r[1] >= 0.0 for r in rob_values]
    return all(bool_values)


class RefactoringTests(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = "crmonitor/config.yaml"
        self.config = load_yaml(config_path)
        rules_path = "crmonitor/traffic_rules.yaml"
        self.traffic_rules = load_yaml(rules_path)

    def test_safe_distance(self):
        # one vehicles which has no leading vehicle (1001)
        # two vehicles which violate safe distance to directly leading vehicle (1003, 1004)
        # one vehicle which violates safe distance to two leading vehicles (1002)
        # one vehicle which violates safe distance partially (1000)
        # one vehicle which always keeps safe distance (1005)
        # one vehicle which keeps safe distance to vehicle which minimally occupies lane (1006)
        # one vehicle which has no leading vehicles and drives in two lanes (1007)
        # one vehicle which leaves lane (1009)
        # one vehicle which violates safe distance to leading vehicle which leaves lane and
        #   recaptures safe distance to vehicle which enters lane (1008)
        # one vehicle which performs illegal cut-in (1010)
        exp_result = [(1000, {
            1001: False,
            1002: True,
            1003: True,
            1004: True,
            1005: True,
            1006: True,
            1007: True,
            1008: True,
            1009: True,
            1010: True}), (1001, {
            1000: True,
            1002: True,
            1003: True,
            1004: True,
            1005: True,
            1006: True,
            1007: True,
            1008: True,
            1009: True,
            1010: True}), (1002, {
            1000: True,
            1001: True,
            1003: False,
            1004: False,
            1005: True,
            1006: True,
            1007: False,
            1008: True,
            1009: True,
            1010: True}), (1003, {
            1000: True,
            1001: True,
            1002: True,
            1004: False,
            1005: True,
            1006: True,
            1007: False,
            1008: True,
            1009: True,
            1010: True}), (1004, {
            1000: True,
            1001: True,
            1002: True,
            1003: True,
            1005: True,
            1006: True,
            1007: False,
            1008: True,
            1009: True,
            1010: True}), (1005, {
            1000: True,
            1001: True,
            1002: True,
            1003: True,
            1004: True,
            1006: True,
            1007: True,
            1008: True,
            1009: True,
            1010: True}), (1006, {
            1000: True,
            1001: True,
            1002: True,
            1003: True,
            1004: True,
            1005: True,
            1007: True,
            1008: True,
            1009: True,
            1010: True}), (1007, {
            1000: True,
            1001: True,
            1002: True,
            1003: True,
            1004: True,
            1005: True,
            1006: True,
            1008: True,
            1009: True,
            1010: True}), (1008, {
            1000: True,
            1001: True,
            1002: True,
            1003: True,
            1004: True,
            1005: True,
            1006: True,
            1007: True,
            1009: False,
            1010: True}), (1009, {
            1000: True,
            1001: True,
            1002: True,
            1003: True,
            1004: True,
            1005: True,
            1006: True,
            1007: True,
            1008: True,
            1010: True}), (1010, {
            1000: True,
            1001: True,
            1002: True,
            1003: True,
            1004: True,
            1005: True,
            1006: True,
            1007: True,
            1008: True,
            1009: True})]

        scenario_file = "scenarios/test_interstate/DEU_test_safe_distance.xml"
        rule_str = "always((in_front_of__a0_a1 and in_same_lane__a0_a1 and " \
                   "!once[0, 30](cut_in__a1_a0 and prev(not cut_in__a1_a0)))" \
                   " implies keeps_safe_distance_prec__a0_a1)"

        scenario, _ = CommonRoadFileReader(scenario_file).open(
                lanelet_assignment=True)

        rule = Rule(rule_str, self.traffic_rules)

        rule_eval = RuleSetEvaluator([rule])

        for ego_id, o_ids in exp_result:
            world_state = WorldState.create_from_scenario(scenario, ego_id,
                                                          self.config)
            for o_id, exp_violation in o_ids.items():
                rob_values, _ = rule_eval.evaluate_all_rules_all_timesteps(
                    world_state, (o_id,))
                self.assertEqual(exp_violation, rob_values[0][-1][1] >= 0.0,
                                 f"Test failed for ego_id={ego_id} and o_id={o_id}")

    def test_unnecessary_braking(self):
        # one vehicle accelerates (1000)
        # one vehicle drives with constant velocity (1001)
        # two leading vehicle which brake only minimal (1005, 1007)
        # one vehicle following another vehicle which brakes normal (1006)
        # one vehicle which has no leading vehicle violates acceleration constraint (1002)
        scenario, planning_problem_set = CommonRoadFileReader(
            "scenarios/test_interstate/DEU_test_unnecessary_braking.xml").open(
            lanelet_assignment=True)
        exp_result = {
            1000: True,
            1001: True,
            1002: False,
            1005: True,
            1006: True,
            1007: True}
        rule_str = "always(not unnecessary_braking__a0)"
        rule = Rule(rule_str, self.traffic_rules)
        rule_eval = RuleSetEvaluator([rule])
        for ego_id, exp_violation in exp_result.items():
            world_state = WorldState.create_from_scenario(scenario, ego_id,
                                                          self.config)
            rob_values, _ = rule_eval.evaluate_all_rules_all_timesteps(
                world_state, tuple())
            self.assertEqual(exp_violation, rob_values[0][-1][1] >= 0.0,
                             f"Test failed for ego_id={ego_id}")


if __name__ == "__main__":
    unittest.main()
