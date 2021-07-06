import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from ruamel.yaml import YAML

from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import Rule
from crmonitor.predicates.rule_evaluator import RuleEvaluator


class TestRuleEvaluator(unittest.TestCase):

    def test_rule(self):
        config = YAML().load(Path("../traffic_rules.yaml"))
        rule = Rule.from_string("A a1: (in_front_of__a0_a1) and single_lane__a0", config)
        scenario,_ = CommonRoadFileReader("../../scenarios/test_interstate/DEU_test_safe_distance_lane_change.xml").open(True)
        eval = RuleEvaluator(rule)
        ws = WorldState.create_from_scenario(scenario, 1001)
        eval.evaluate_robustness_incremental(ws)
        # evaluator = RuleEvaluator(rule)


if __name__ == '__main__':
    unittest.main()