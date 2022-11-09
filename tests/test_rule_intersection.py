import logging
import os
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.lanelet import LaneletType

from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.monitor.rule import (parse_rule, )

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


class RuleTest(unittest.TestCase):
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
            self.scenario_root_path, "test_intersection/traffic_light_test_2.xml"
        )
        scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(
            lanelet_assignment=True
        )
        # TODO: Fix lanelet network completely!
        scenario.lanelet_network.intersections[0].incomings[3]._successors_right.add(
            203
        )
        scenario.lanelet_network.intersections[0].incomings[1]._successors_right.add(
            201
        )
        scenario.lanelet_network.intersections[0].incomings[3]._successors_straight.add(
            103
        )
        scenario.lanelet_network.intersections[0].incomings[3]._successors_left.add(303)
        scenario.lanelet_network.find_lanelet_by_id(203)._lanelet_type.add(
            LaneletType.INTERSECTION
        )
        scenario.lanelet_network.find_lanelet_by_id(103)._lanelet_type.add(
            LaneletType.INTERSECTION
        )
        scenario.lanelet_network.find_lanelet_by_id(303)._lanelet_type.add(
            LaneletType.INTERSECTION
        )
        scenario.lanelet_network.find_lanelet_by_id(201)._lanelet_type.add(
            LaneletType.INTERSECTION
        )
        scenario.lanelet_network.find_lanelet_by_id(403)._predecessor.append(203)
        scenario.lanelet_network.find_lanelet_by_id(400)._predecessor.append(201)
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
        rule_str = self.traffic_rules["traffic_rules"]["TL"]
        self.traffic_rules["scale_rob"] = False
        rule = parse_rule(rule_str, self.traffic_rules, name="RedLightRunning")
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


if __name__ == "__main__":
    unittest.main()
