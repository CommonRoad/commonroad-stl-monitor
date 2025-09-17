import logging
from typing import List

import numpy as np

from crmonitor.common.world import World
from crmonitor.predicates.base import AbstractPredicate, PredicateName

_LOGGER = logging.getLogger(__name__)


class AccelerationPredicates(PredicateName):
    BrakesAbruptly = "brakes_abruptly"
    BrakesAbruptlyRelative = "brakes_abruptly_relative"
    CausesBrakingIntersection = "causes_braking_intersection"


class PredAbruptBreaking(AbstractPredicate):
    predicate_name = AccelerationPredicates.BrakesAbruptly
    arity = 1

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        accel = ego_vehicle.get_lon_state(time_step).a
        rob = self.config.a_abrupt - accel
        return self._scaler.scale_acc(rob)


class PredAbruptBreakingRelative(AbstractPredicate):
    predicate_name = AccelerationPredicates.BrakesAbruptlyRelative
    arity = 2

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        accel_k = world.vehicle_by_id(vehicle_ids[0]).get_lon_state(time_step).a
        accel_p = world.vehicle_by_id(vehicle_ids[1]).get_lon_state(time_step).a
        rob = -accel_k + accel_p + self.config.a_abrupt
        return self._scaler.scale_acc(rob)


class PredCausesBrakingIntersection(AbstractPredicate):
    """
    evaluates if the k-th vehicle causes the braking of the p-th vehicle.

    If the distance between the frontmost point of the p-th vehicle and the rearmost point of the k-th vehicle along
    the reference lane of the p-th one is smaller than a threshold (d_br) and the acceleration of the p-th vehicle is
    lower or equal to a threshold (a_br), the k-th vehicle causes the braking of the p-th vehicle.
    """

    predicate_name = AccelerationPredicates.CausesBrakingIntersection
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        d_br = self.config.d_br
        a_br = self.config.a_br
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        # rearmost point of the k-th vehicle along the reference lane of p-th one
        rear_k_s = vehicle_k.rear_s(time_step, vehicle_p.ref_path_lane)
        # frontmost point of the p-th vehicle along the reference lane of p-th one
        front_p_s = vehicle_p.front_s(time_step, vehicle_p.ref_path_lane)
        # if the k-th vehicle is far away from the reference lane of the p-th vehicle, return -1
        if rear_k_s is None:
            return False
        distance_vehicle = rear_k_s - front_p_s
        # calculate the longitudinal acceleration of the p-th vehicle
        a_p = vehicle_p.get_lon_state(time_step, vehicle_p.ref_path_lane).a
        return (0 <= distance_vehicle <= d_br) and (a_p <= a_br)

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        d_br = self.config.d_br
        a_br = self.config.a_br
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        # rearmost point of the k-th vehicle along the reference lane of p-th one
        rear_k_s = vehicle_k.rear_s(time_step, vehicle_p.ref_path_lane)
        # frontmost point of the p-th vehicle along the reference lane of p-th one
        front_p_s = vehicle_p.front_s(time_step, vehicle_p.ref_path_lane)
        # if the k-th vehicle is far away from the reference lane of the p-th vehicle, return -1
        if rear_k_s is None:
            return -1
        distance_vehicle = rear_k_s - front_p_s
        rob_distance = np.min([distance_vehicle, d_br - distance_vehicle])
        # calculate the longitudinal acceleration of the p-th vehicle
        a_p = vehicle_p.get_lon_state(time_step, vehicle_p.ref_path_lane).a
        rob_a = a_br - a_p
        robustness = np.min([self._scale_lon_dist(rob_distance), self._scale_acc(rob_a)])
        return robustness
