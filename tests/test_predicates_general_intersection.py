import os
import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from crmonitor.predicates.general import (
    PredTurningLeft,
    PredTurningRight,
    PredGoingStraight,
)


class TestTurning(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.scenario_root_path = root_path.parent / "scenarios"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_sl"] = 1.0
        self.config["scenario"] = "intersection"

    def testTurningRight(self):
        scenario_file = os.path.join(
            self.scenario_root_path, "test_intersection/DEU_TestIntersectionInteract-3_1_T-1.xml"
        )
        scenario, _ = CommonRoadFileReader(
            scenario_file
        ).open(True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(30)

        for time in range(ego_vehicle.end_time + 1):
            pred_test = PredTurningRight(self.config)
            sol_monitor_1 = pred_test.evaluate_boolean(world, time, [ego_vehicle.id])

            sol_monitor_2 = pred_test.evaluate_robustness(world, time, [ego_vehicle.id])

            self.assertEqual(sol_monitor_1, sol_monitor_2 >= 0)

    def testTurningLeft(self):
        scenario_file = os.path.join(
            self.scenario_root_path, "test_intersection/DEU_TestIntersectionInteract-3_1_T-1.xml"
        )
        scenario, _ = CommonRoadFileReader(
            scenario_file
        ).open(True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(30)

        for time in range(ego_vehicle.end_time + 1):
            pred_test = PredTurningLeft(self.config)
            sol_monitor_1 = pred_test.evaluate_boolean(world, time, [ego_vehicle.id])

            sol_monitor_2 = pred_test.evaluate_robustness(world, time, [ego_vehicle.id])

            self.assertEqual(sol_monitor_1, sol_monitor_2 >= 0)

    def testGoingStraight(self):
        scenario_file = os.path.join(
            self.scenario_root_path, "test_intersection/DEU_TestIntersectionInteract-3_1_T-1.xml"
        )
        scenario, _ = CommonRoadFileReader(
            scenario_file
        ).open(True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(30)

        for time in range(ego_vehicle.end_time + 1):
            pred_test = PredGoingStraight(self.config)
            sol_monitor_1 = pred_test.evaluate_boolean(world, time, [ego_vehicle.id])

            sol_monitor_2 = pred_test.evaluate_robustness(world, time, [ego_vehicle.id])

            self.assertEqual(sol_monitor_1, sol_monitor_2 >= 0)
