from enum import Enum
import logging
from typing import List

from crmonitor.common.world import World
from crmonitor.predicates.base import BasePredicateEvaluator

logger = logging.getLogger(__name__)


class AccelerationPredicates(str, Enum):
    BrakesAbruptly = "brakes_abruptly"
    RelBrakesAbruptly = "rel_brakes_abruptly"
    CausesBrakingIntersection = "causes_braking_intersection"


class PredAbruptBreaking(BasePredicateEvaluator):
    predicate_name = AccelerationPredicates.BrakesAbruptly
    arity = 1

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        accel = world.vehicle_by_id(vehicle_ids[0]).states_cr[time_step].acceleration
        rob = self.config["a_abrupt"] - accel
        return self._scale_acc(rob)


class PredRelAbruptBreaking(BasePredicateEvaluator):
    predicate_name = AccelerationPredicates.RelBrakesAbruptly
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        accel_k = world.vehicle_by_id(vehicle_ids[0]).states_cr[time_step].acceleration
        accel_p = world.vehicle_by_id(vehicle_ids[1]).states_cr[time_step].acceleration
        rob = -accel_k + accel_p + self.config["a_abrupt"]
        return self._scale_acc(rob)


class PredCausesBrakingIntersection(BasePredicateEvaluator):
    """
    evaluates if the first vehicle causes the braking of the second vehicle.
    """
    predicate_name = AccelerationPredicates.CausesBrakingIntersection
    arity = 2

    # TODO
    # def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:

    # TODO
    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        return self._scale_lat_dist(100)