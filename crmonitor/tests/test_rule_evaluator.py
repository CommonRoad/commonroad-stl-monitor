mport unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from ruamel.yaml import YAML

from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import parse_rule
from crmonitor.predicates.visitor import CreateEvaluatorVisitor, EvaluationVisitor

import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from ruamel.yaml import YAML

from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import parse_rule
from crmonitor.predicates.visitor import CreateEvaluatorVisitor, \
    EvaluationVisitor


class TestRuleEvaluator(unittest.TestCase):
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
        scenario, _ = CommonRoadFileReader(
            "../../scenarios/test_interstate" "/DEU_test_safe_distance_lane_change.xml"
        ).open(True)
        for r in rules:
            rule = parse_rule(r, config)
            eval = rule.visit(CreateEvaluatorVisitor())
            eval_visitor = EvaluationVisitor()
            ws = WorldState.create_from_scenario(scenario, 1001)
            ws.time_step = ws.ego_vehicle.start_time
            rob = eval.visit(eval_visitor, ws, (1001,), False)


if __name__ == "__main__":
    unittest.main()
