import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator


class TestStopLine(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        self.scenario_root_path = root_path.parent / "scenarios"

    def test_r_in_1(self):
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
                world, ego_vehicle, "R_IN1_past"
            )
            values = rule_eval.evaluate()
            violated = np.any(values < 0.0)
            self.assertEqual(
                expected[ego.obstacle_id],
                violated,
                f"Test failed for ID={ego.obstacle_id}, {expected[ego.obstacle_id]} != {violated}",
            )
