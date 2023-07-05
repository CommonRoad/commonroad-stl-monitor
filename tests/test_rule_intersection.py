import os
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.predicates.predicate_factory import PredicateFactory
from crmonitor.rule.rule_factory import RuleFactory


class TestIntersectionRules(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = root_path / "config.yaml"
        self.config = load_yaml(str(config_path))
        rules_path = root_path / "traffic_rules_rtamt.yaml"
        self.traffic_rules = load_yaml(str(rules_path))
        self.scenario_root_path = root_path.parent / "scenarios"

    def test_red_light(self):
        scenario_file = os.path.join(
            self.scenario_root_path, "test_intersection/ZAM_TrafficLight-1_1_T-1.xml"
        )
        scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(
            lanelet_assignment=True
        )
        exp_result = {
            30: True,  # Turning Right on green
            33: True,  # Turning Right on green
            38: True,  # Turning Right on green
            35: True,  # Straight on green
            36: True,  # Waiting on green
            37: True,  # Waiting on red
            31: True,  # Turning right on red
            32: False,  # Turning left on red
            34: False,  # Turning left on red
        }
        rule_str = self.traffic_rules["traffic_rules"]["R_IN2_TOR"]
        self.traffic_rules["scale_rob"] = False
        rule = RuleFactory(
            PredicateFactory(self.traffic_rules["traffic_rules_param"])
        ).parse_rule(rule_str, name="RedLightRunning")
        world = World.create_from_scenario(scenario)
        for ego_id, exp_violation in exp_result.items():
            ego_vehicle = world.vehicle_by_id(ego_id)
            rule_eval = RuleEvaluator(rule, ego_vehicle, world)
            rule_robustness = []
            for rob in rule_eval:
                rule_robustness.append(rob)
            rule_robustness = np.array(rule_robustness)
            bool_value = rule_robustness >= 0.0
            self.assertEqual(
                exp_violation, np.all(bool_value), f"Test failed for ego_id={ego_id}"
            )

    def test_stop_line(self):
        scenario_file = (
            self.scenario_root_path / "test_intersection/DEU_test_stop_line.xml"
        )
        scenario, planning_problem_set = CommonRoadFileReader(str(scenario_file)).open(
            lanelet_assignment=True
        )
        world = World.create_from_scenario(scenario)

        expected = {
            1000: True,  # Ignored stop line
            1001: False,  # Waited forever
            1002: False,  # Waited long enough
            1003: True,  # Waited too far away (and not long enough)
        }

        for ego in scenario.dynamic_obstacles:
            ego_vehicle = world.vehicle_by_id(ego.obstacle_id)
            rule_eval = RuleEvaluator.create_from_config(
                world, ego_vehicle.id, "R_IN1_past"
            )
            values = rule_eval.evaluate()
            violated = np.any(values < 0.0)
            self.assertEqual(
                expected[ego.obstacle_id],
                violated,
                f"Test failed for ID={ego.obstacle_id}, {expected[ego.obstacle_id]} != {violated}",
            )


if __name__ == "__main__":
    unittest.main()
