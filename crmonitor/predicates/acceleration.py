import logging
from enum import Enum
from typing import List

import numpy as np

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

# ------------------------------------------------------------------------ #
# new predicate from Mahdi Bayouli
# class PredCausesBrakingIntersection1212(BasePredicateEvaluator):
#     """
#     evaluates if the first vehicle causes the braking of the second vehicle.
#     """
#
#     predicate_name = AccelerationPredicates.CausesBrakingIntersection
#     arity = 2
#
#     # TODO:
#     # describe robustness
#     #
#
#     def evaluate_robustness(
#         self, world: World, time_step, vehicle_ids: List[int]
#     ) -> float:
#         d_br = self.config["d_br"]
#         a_br = self.config["a_br"]
#         rob = 0
#         vehicle_k = world.vehicle_by_id(vehicle_ids[0])
#         vehicle_p = world.vehicle_by_id(vehicle_ids[1])
#
#         lane_k = vehicle_k.get_lane(time_step)
#         lane_p = vehicle_p.get_lane(time_step)
#
#         rear_k = vehicle_k.rear_s(time_step)
#         front_p = vehicle_p.front_s(time_step)
#
#         # TODO: does d need scaling ?
#         d = rear_k - front_p
#         a = vehicle_p.get_lon_state(time_step).a
#
#         # d_br - d : how well d is far from the threshold
#         # a_br - a : how well a is far from the threshold
#         # d : d shouldn't be negative
#         rob = np.minimum(
#             np.minimum(self._scale_lon_dist(d_br - d), self._scale_acc(a_br - a)),
#             self._scale_lon_dist(d),
#         )
#
#         return rob


class PredCausesBrakingIntersection(BasePredicateEvaluator):
    """
    evaluates if the k-th vehicle causes the braking of the p-th vehicle.
    """
    predicate_name = AccelerationPredicates.CausesBrakingIntersection
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        # TODO: fix config problem
        # d_br = self.config['traffic_rules_param']["d_br"]
        # a_br = self.config['traffic_rules_param']["a_br"]
        d_br = 1.0
        a_br = -1.0
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        rear_k_s = vehicle_k.rear_s(time_step, vehicle_p.ref_path_lane)
        front_p_s = vehicle_p.front_s(time_step, vehicle_p.ref_path_lane)
        if rear_k_s is None:
            return -1
        distance_vehicle = rear_k_s - front_p_s
        rob_distance = np.min([distance_vehicle, d_br - distance_vehicle])
        a_p = vehicle_p.get_lon_state(time_step, vehicle_p.ref_path_lane).a
        rob_a = a_br - a_p
        rob = np.min([self._scale_lon_dist(rob_distance), self._scale_acc(rob_a)])
        return rob

