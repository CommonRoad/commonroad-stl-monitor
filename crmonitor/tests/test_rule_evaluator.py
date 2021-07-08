import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from ruamel.yaml import YAML

from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import parse_rule
from crmonitor.evaluation.visitor import MonitorCreationRuleTreeVisitor, \
    EvaluationMonitorTreeVisitor


class TestRuleEvaluator(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parent.parent
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
        ]
        config = YAML().load(Path("../traffic_rules.yaml"))
        scenario, _ = CommonRoadFileReader(self.scenario_root_path / "DEU_test_safe_distance_lane_change.xml").open(True)
        for r in rules:
            rule = parse_rule(r, config)
            eval = rule.visit(MonitorCreationRuleTreeVisitor(scenario.dt))
            eval_visitor = EvaluationMonitorTreeVisitor()
            ws = WorldState.create_from_scenario(scenario, 1001)
            ws.time_step = ws.ego_vehicle.start_time
            rob = eval.visit(eval_visitor, ws, (1001,), False)

if __name__ == "__main__":
    unittest.main()
