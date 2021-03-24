import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.evaluation import RuleSetEvaluator
from crmonitor.common.helper import load_yaml
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.rule import Rule


class TestMtlStlEquivalence(unittest.TestCase):
    def setUp(self):
        self.config = load_yaml("crmonitor/config.yaml")

    def test_reference(self):
        # False -> Violation
        scenarios = [
            ("DEU_LocationEUpper-16_7_T-1.xml", 62, False),
            ("DEU_LocationFUpper-58_1_T-1.xml", 108, True),
            ("DEU_LocationFUpper-58_1_T-1.xml", 69, True),
            ("DEU_LocationFUpper-58_1_T-1.xml", 101, False),
            ("DEU_LocationFUpper-58_1_T-1.xml", 115, True),
        ]
        path = Path("scenarios/test_highd")

        for scn_name, ego_id, exp in scenarios:
            scenario, _ = CommonRoadFileReader(path / scn_name).open(True)

            world_state = WorldState.create_from_scenario(scenario, ego_id, self.config)

            rule = Rule(
                "((in_front_of__a0_a1 and in_same_lane__a0_a1 and !once["
                "0, 15](cut_in__a1_a0 and prev(not cut_in__a1_a0))) "
                "implies keeps_safe_distance_prec__a0_a1)",
                config=load_yaml("crmonitor/traffic_rules.yaml"),
                name="G1",
            )

            rule_evaluator = RuleSetEvaluator(rules=[rule])

            rob, pred = rule_evaluator.evaluate_all_rules_all_timesteps_floating(
                world_state
            )
            pred["sig"] = pred["value"] >= 0.0
            self.assertEqual(
                rob["rob"].min() >= 0,
                exp,
                f"Failed for scenario {scenario}, vehicle {ego_id}!",
            )


if __name__ == "__main__":
    unittest.main()
