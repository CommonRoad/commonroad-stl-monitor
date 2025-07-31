import unittest

from crmonitor.predicates.acceleration import PredCausesBrakingIntersection
from crmonitor.predicates.base import PredicateConfig
from tests.resources import IntersectionScenarios


class TestIntersectionAccelerationPredicates:
    def test_causes_braking_intersection(self):
        config = PredicateConfig(d_br=15.0, a_br=-1.0)
        world = IntersectionScenarios.R_IN3.get_world()
        ego_vehicle = world.vehicle_by_id(30)
        target_vehicle = world.vehicle_by_id(31)

        pred = PredCausesBrakingIntersection(config)
        for time in range(min(ego_vehicle.end_time, target_vehicle.end_time) + 1):
            sol_monitor_1 = pred.evaluate_boolean(world, time, [ego_vehicle.id, target_vehicle.id])
            sol_monitor_2 = pred.evaluate_robustness(
                world, time, [ego_vehicle.id, target_vehicle.id]
            )

            assert sol_monitor_1 == (sol_monitor_2 >= 0)
