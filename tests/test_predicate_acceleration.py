import unittest
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader


from crmonitor.common.helper import load_yaml
from crmonitor.common.world import World
from crmonitor.predicates.acceleration import PredCausesBrakingIntersection


class TestPriorityPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_br"] = 15.0
        self.config["a_br"] = -1.0

    def testCausesBrakingIntersection(self):
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestIntersectionInteract-3_1_T-1.xml"
            )
        ).open(True)
        world = World.create_from_scenario(scenario)
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
