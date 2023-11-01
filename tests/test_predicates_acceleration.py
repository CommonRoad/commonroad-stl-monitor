import os
import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from crmonitor.predicates.acceleration import PredCausesBrakingIntersection


class TestIntersectionAccelerationPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.scenario_root_path = root_path.parent / "scenarios"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_br"] = 15.0
        self.config["a_br"] = -1.0
        self.config["scenario"] = "intersection"
        self.config["use_mpr"] = False

    def testCausesBrakingIntersection(self):
        scenario_file = os.path.join(
            self.scenario_root_path,
            "test_intersection/DEU_TestIntersectionRIN3.xml",
        )
        scenario, _ = CommonRoadFileReader(scenario_file).open(True)
        world = World.create_from_scenario(scenario, self.config)
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)

        pred = PredCausesBrakingIntersection(self.config)
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            sol_monitor_1 = pred.evaluate_boolean(
                world, time, [ego_vehicle.id, target_vehicle.id]
            )
            sol_monitor_2 = pred.evaluate_robustness(
                world, time, [ego_vehicle.id, target_vehicle.id]
            )

            self.assertEqual(sol_monitor_1, sol_monitor_2 >= 0)


class TestInterstateAccelerationPredicates(unittest.TestCase):
    def setUp(self) -> None:
        pass
