from enum import Enum
import logging
import numpy as np

from typing import List


from crmonitor.common.world import World
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.predicates import utils


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

    # TODO:
    # describe robustness
    #

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        d_br = self.config["d_br"]
        a_br = self.config["a_br"]
        rob = 0
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lane_k = vehicle_k.get_lane(time_step)
        lane_p = vehicle_p.get_lane(time_step)

        rear_k = vehicle_k.rear_s(time_step)
        front_p = vehicle_p.front_s(time_step)

        # TODO: does d need scaling ?
        d = rear_k - front_p
        a = vehicle_p.get_lon_state(time_step).a

        # d_br - d : how well d is far from the threshold
        # a_br - a : how well a is far from the threshold
        # d : d shouldn't be negative
        rob = np.minimum(
            np.minimum(self._scale_lon_dist(d_br - d), self._scale_acc(a_br - a)),
            self._scale_lon_dist(d),
        )

        return rob
