import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from ruamel.yaml import YAML

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.monitor.rule import (
    parse_rule,
    AllNode,
    RuleNode,
    ExistNode,
    PredicateNode,
    IOType,
)


class TestRuleEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = root_path / "config.yaml"
        self.config = YAML().load(config_path)
        rules_path = root_path / "traffic_rules_rtamt.yaml"
        self.traffic_rule_params = YAML().load(rules_path)
        self.scenario_root_path = root_path.parent / "scenarios/test_interstate"

    def test_smoke(self):
         rules = [
             "A a1: (in_front_of__a0_a1 and cut_in__a0_a1)",
             "A a1: (in_front_of__a0_a1) and single_lane__a0",
             "E a1: (in_front_of__a0_a1 and cut_in__a0_a1)",
             "E a1: (in_front_of__a0_a1) and single_lane__a0",
             "single_lane__a0",
             "single_lane__a0 and single_lane__a0",
             "A a1: (in_front_of__a0_a1)",
         ]

         scenario, _ = CommonRoadFileReader(
             str(self.scenario_root_path / "DEU_test_safe_distance_lane_change.xml")
         ).open(True)
         for r in rules:
             rule = parse_rule(r, self.traffic_rule_params)
             ws = World.create_from_scenario(scenario)
             ego_vehicle = ws.vehicle_by_id(1001)
             evaluator = RuleEvaluator(rule, ego_vehicle, ws)
             rob = evaluator.update()

         evaluator.reset(ego_vehicle, ws)

    def test_parsing(self):
        rule = parse_rule(
            "A a1: (in_front_of__a0_a1 and cut_in__a0_a1)", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, AllNode))
        rule = parse_rule(
            "A a1: (in_front_of__a0_a1) and single_lane__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))
        rule = parse_rule(
            "E a1: (in_front_of__a0_a1 and cut_in__a0_a1)", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, ExistNode))
        rule = parse_rule(
            "E a1: (in_front_of__a0_a1) and single_lane__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))
        rule = parse_rule("single_lane__a0", self.traffic_rule_params)
        self.assertTrue(isinstance(rule, RuleNode))
        
        rule = parse_rule(
            "single_lane__a0 and single_lane__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))

        rule = parse_rule(
            "A a1: (in_front_of__a0_a1) and single_lane_i__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))
        
        self.assertTrue(isinstance(rule.children[1], PredicateNode))
        self.assertEqual(rule.children[1].io_type, IOType.INPUT)
        self.assertEqual(rule.children[0].children[0].children[0].io_type, IOType.OUTPUT)


if __name__ == "__main__":
    unittest.main()
