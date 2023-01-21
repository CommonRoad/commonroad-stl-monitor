from enum import Enum
import logging
import numpy as np

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
        # TODO: Thresholds : yaml file ?
        d_br = 10
        a_br = 10
        rob = 0
        vehicle_k = vehicle_ids[0]
        vehicle_p = vehicle_ids[1]
        rear_k = vehicle_k.rear_s(time_step)
        front_p = vehicle_p.front_s(time_step)
        d = rear_k - front_p
        a = vehicle_p.get_lon_state(time_step).a
        rob = np.minimum((d - d_br), (a - a_br))
        # TODO: does rob need scaling ?
        return rob
