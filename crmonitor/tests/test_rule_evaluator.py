import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from ruamel.yaml import YAML

from crmonitor.common.evaluation import RuleSetEvaluator
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import parse_rule, ExistNode, RuleNode, \
    PredicateNode
from crmonitor.predicates.visitor import CreateEvaluatorVisitor, \
    EvaluationVisitor


class TestRuleEvaluator(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parent.parent
        config_path = root_path / "config.yaml"
        self.config = YAML().load(config_path)
        rules_path = root_path / "traffic_rules_rtamt.yaml"
        self.traffic_rule_params = YAML().load(rules_path)
        self.scenario_root_path = root_path.parent / "scenarios/test_interstate"

    @unittest.SkipTest
    def test_smoke(self):
        rules = [
            "A a1: (in_front_of__a0_a1 and cut_in__a0_a1)",
            "A a1: (in_front_of__a0_a1) and single_lane__a0",
            "E a1: (in_front_of__a0_a1 and cut_in__a0_a1)",
            "E a1: (in_front_of__a0_a1) and single_lane__a0",
            "single_lane__a0",
            "single_lane__a0 and single_lane__a0",
        ]
        config = YAML().load(Path("../traffic_rules.yaml"))
        scenario, _ = CommonRoadFileReader(self.scenario_root_path / "DEU_test_safe_distance_lane_change.xml").open(True)
        for r in rules:
            rule = parse_rule(r, config)
            eval = rule.visit(CreateEvaluatorVisitor())
            eval_visitor = EvaluationVisitor()
            ws = WorldState.create_from_scenario(scenario, 1001)
            ws.time_step = ws.ego_vehicle.start_time
            rob = eval.visit(eval_visitor, ws, (1001,), False)

    def test_unnecessary_braking(self):
        # one vehicle accelerates (1000)
        # one vehicle drives with constant velocity (1001)
        # one vehicle which has no leading vehicle violates acceleration constraint (1002)
        # two leading vehicle which brake only minimal (1005, 1007)
        # one vehicle following another vehicle which brakes normal (1006)
        scenario_file = self.scenario_root_path / "DEU_test_unnecessary_braking.xml"
        scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        exp_result = {
            1000: True,
            1001: True,
            1002: False,
            1005: True,
            1006: True,
            1007: True,
        }
        rule_str = self.traffic_rule_params["traffic_rules_forward"]["R_G2"]
        self.traffic_rule_params["scale_rob"] = False
        rule = parse_rule(
            rule_str,
            self.traffic_rule_params,
            name="UnnecessaryBraking"
        )
        self.assertTrue(isinstance(rule, RuleNode))
        self.assertEqual(len(rule.children), 2)
        self.assertTrue(any([isinstance(c, PredicateNode) for c in rule.children]))
        self.assertTrue(any([isinstance(c, ExistNode) for c in rule.children]))
        rule_eval = RuleSetEvaluator([rule])
        for ego_id, exp_violation in exp_result.items():
            world_state = WorldState.create_from_scenario(scenario, ego_id, self.config)
            df_rule, _ = rule_eval.evaluate_incremental(
                world_state
            )
            rob_value = all([r >= 0.0 for r in df_rule["robustness"].values])
            self.assertEqual(
                exp_violation, rob_value, f"Test failed for ego_id={ego_id}"
            )


if __name__ == "__main__":
    unittest.main()
