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
        veh_state = world.vehicle_by_id(vehicle_ids[0]).states_cr[time_step]
        if veh_state.has_value("acceleration_y"):
            # todo: if state has acceleration_y, we assume that acceleration and acceleration_y
            #  are components on the x- and y-axes  in the Cartesian coordinate system.
            accel = veh_state.acceleration * np.cos(
                veh_state.orientation
            ) + veh_state.acceleration_y * np.sin(veh_state.orientation)
        else:
            accel = veh_state.acceleration
        rob = self.config["a_abrupt"] - accel
        return self._scale_acc(rob)


class PredRelAbruptBreaking(BasePredicateEvaluator):
    predicate_name = AccelerationPredicates.RelBrakesAbruptly
    arity = 2

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        state_k = world.vehicle_by_id(vehicle_ids[0]).states_cr[time_step]
        state_p = world.vehicle_by_id(vehicle_ids[1]).states_cr[time_step]
        if state_k.has_value("acceleration_y"):
            accel_k = state_k.acceleration * np.cos(
                state_k.orientation
            ) + state_k.acceleration_y * np.sin(state_k.orientation)
        else:
            accel_k = state_k.acceleration
        if state_p.has_value("acceleration_y"):
            accel_p = state_p.acceleration * np.cos(
                state_p.orientation
            ) + state_p.acceleration_y * np.sin(state_p.orientation)
        else:
            accel_p = state_p.acceleration
        rob = -accel_k + accel_p + self.config["a_abrupt"]
        return self._scale_acc(rob)


class PredCausesBrakingIntersection(BasePredicateEvaluator):
    """
    evaluates if the k-th vehicle causes the braking of the p-th vehicle.

    If the distance between the frontmost point of the p-th vehicle and the rearmost point of the k-th vehicle along
    the reference lane of the p-th one is smaller than a threshold (d_br) and the acceleration of the p-th vehicle is
    lower or equal to a threshold (a_br), the k-th vehicle causes the braking of the p-th vehicle.
    """

    predicate_name = AccelerationPredicates.CausesBrakingIntersection
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        d_br = self.config["d_br"]
        a_br = self.config["a_br"]
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
        state_p = vehicle_p.states_cr[time_step]
        if state_p.has_value("acceleration_y"):
            a_p = state_p.acceleration * np.cos(
                state_p.orientation
            ) + state_p.acceleration_y * np.sin(state_p.orientation)
        else:
            a_p = state_p.acceleration
        return (0 <= distance_vehicle <= d_br) and (a_p <= a_br)

    def evaluate_robustness(
        self, world: World, time_step, vehicle_ids: List[int]
    ) -> float:
        d_br = self.config["d_br"]
        a_br = self.config["a_br"]
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
        state_p = vehicle_p.states_cr[time_step]
        if state_p.has_value("acceleration_y"):
            a_p = state_p.acceleration * np.cos(
                state_p.orientation
            ) + state_p.acceleration_y * np.sin(state_p.orientation)
        else:
            a_p = state_p.acceleration
        rob_a = a_br - a_p
        robustness = np.min(
            [self._scale_lon_dist(rob_distance), self._scale_acc(rob_a)]
        )
        return robustness
