import logging
import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSignInterpreter
from typing_extensions import override

from crmonitor.common.world import World
from crmonitor.predicates.base import (
    AbstractPredicate,
    PredicateConfig,
    PredicateName,
)
from crmonitor.predicates.position import PredInFrontOf, PredInSameLane

_LOGGER = logging.getLogger(__name__)


class VelocityPredicates(PredicateName):
    KeepsLaneSpeedLimit = "keeps_lane_speed_limit"
    KeepsTypeSpeedLimit = "keeps_type_speed_limit"
    KeepsLaneSpeedLimitStar = "keeps_lane_speed_limit_star"
    KeepsFovSpeedLimit = "keeps_fov_speed_limit"
    KeepsBrakeSpeedLimit = "keeps_brake_speed_limit"
    Reverses = "reverses"
    HasCongestionVelocity = "has_congestion_velocity"
    HasSlowMovingVelocity = "has_slow_moving_velocity"
    HasQueueVelocity = "has_queue_velocity"
    SlowLeadingVehicle = "slow_leading_vehicle"
    SlowAsLeadingVehicle = "slow_as_leading_vehicle"
    PreservesTrafficFlow = "preserves_traffic_flow"
    ExistStandingLeadingVehicle = "exist_standing_leading_vehicle"
    InStandstill = "in_standstill"
    DrivesFaster = "drives_faster"
    DrivesWithSlightlyHigherSpeed = "drives_with_slightly_higher_speed"
    VelocityBelow2 = "velocity_below_2"
    VelocityBelow5 = "velocity_below_five"
    VelocityBelow15 = "velocity_below_15"
    VelocityBelow20 = "velocity_below_20"


class GenericSpeedLimit(AbstractPredicate, ABC):
    @abstractmethod
    def get_speed_limit(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float | None: ...

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        time_step = time_step
        speed_limit = self.get_speed_limit(world, time_step, vehicle_ids)
        if speed_limit is None:
            rob = math.inf
        else:
            rob = speed_limit + self.config.eps - vehicle.get_lon_state(time_step).v
        rob = self._scaler.scale_speed(rob)
        return rob


class PredLaneSpeedLimit(GenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsLaneSpeedLimit
    arity = 1

    def __init__(self, config: PredicateConfig | None = None):
        super().__init__(config)
        self.country = SupportedTrafficSignCountry(self.config.country)

    @override
    def get_speed_limit(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float | None:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids = vehicle.lanelet_ids_at_time_step(time_step)
        ts_interpreter = TrafficSignInterpreter(self.country, world.road_network.lanelet_network)
        speed_limit = ts_interpreter.speed_limit(frozenset(lanelet_ids))
        return speed_limit


class PredTypeSpeedLimit(GenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsTypeSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle_type = world.vehicle_by_id(vehicle_ids[0]).obstacle_type
        if vehicle_type is ObstacleType.TRUCK:
            return self.config.max_interstate_speed_truck
        else:
            return None


class PredFovSpeedLimit(GenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsFovSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.fov_speed_limit


class PredBrSpeedLimit(GenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsBrakeSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.braking_speed_limit


class PredLaneSpeedLimitStar(PredLaneSpeedLimit):
    predicate_name = VelocityPredicates.KeepsLaneSpeedLimitStar
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        speed_limit = super(PredLaneSpeedLimitStar, self).get_speed_limit(
            world, time_step, vehicle_ids
        )
        if speed_limit is None:
            speed_limit = self.config.desired_interstate_velocity
        return speed_limit


class PredHasSlowMovingVelocity(PredLaneSpeedLimit):
    predicate_name = VelocityPredicates.HasSlowMovingVelocity
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        return self.config.max_slow_moving_traffic_velocity


class PredHasCongestionVelocity(PredLaneSpeedLimit):
    predicate_name = VelocityPredicates.HasCongestionVelocity
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids) -> float:
        return self.config.max_congestion_velocity


class PredHasQueueVelocity(PredLaneSpeedLimit):
    predicate_name = VelocityPredicates.HasQueueVelocity
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        return self.config.max_queue_of_vehicles_velocity


class PredVelocityBelow2(GenericSpeedLimit):
    predicate_name = VelocityPredicates.VelocityBelow2
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        # Convert 2 km/h to m/s
        return 2.0 / 3.6


class PredVelocityBelow5(GenericSpeedLimit):
    predicate_name = VelocityPredicates.VelocityBelow5
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        # Convert 5 km/h to m/s
        return 5.0 / 3.6


class PredVelocityBelow15(GenericSpeedLimit):
    predicate_name = VelocityPredicates.VelocityBelow15
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        # Convert 15 km/h to m/s
        return 15.0 / 3.6


class PredVelocityBelow20(GenericSpeedLimit):
    predicate_name = VelocityPredicates.VelocityBelow20
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        # Convert 20 km/h to m/s
        return 20.0 / 3.6


class PredReverses(AbstractPredicate):
    """
    Evaluates if a vehicle drives backwards
    """

    predicate_name = VelocityPredicates.Reverses
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        vel = vehicle.get_lon_state(time_step).v
        val = vel < -self.config.standstill_error
        return val

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return self._scale_speed(
            -self.config.standstill_error - vehicle.get_lon_state(time_step).v - self.config.eps,
        )


class PredSlowAsLeadingVehicle(AbstractPredicate):
    """
    Predicate to evaluate whether a vehicle is 'slow' when driving as a leading vehicle of an ego vehicle.
    """

    predicate_name = VelocityPredicates.SlowAsLeadingVehicle
    arity = 1

    def __init__(self, config) -> None:
        super().__init__(config)
        self._lane_speed_limit_evaluator = PredLaneSpeedLimitStar(config)
        self._type_speed_limit_evaluator = PredTypeSpeedLimit(config)

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        veh_id = vehicle_ids[0]
        vehicle = world.vehicle_by_id(veh_id)
        v_type = self._type_speed_limit_evaluator.get_speed_limit(world, time_step, [veh_id])
        v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(world, time_step, [veh_id])
        v_max = min(v for v in (v_type, v_max_lane) if v is not None)
        return self._scale_speed(
            v_max
            - vehicle.get_lon_state(time_step).v
            - self.config.min_velocity_diff
            - self.config.eps
        )


class PredSlowLeadingVehicle(AbstractPredicate):
    """
    Predicate which evaluates if a slow leading vehicle exists if front of a vehicle
    """

    predicate_name = VelocityPredicates.SlowLeadingVehicle
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._in_front_of_evaluator = PredInFrontOf(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._lane_speed_limit_evaluator = PredLaneSpeedLimitStar(config)
        self._type_speed_limit_evaluator = PredTypeSpeedLimit(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if not self._in_front_of_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ) or not self._same_lane_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ):
                continue
            v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(
                world, time_step, [veh_o.id]
            )
            v_type = self._type_speed_limit_evaluator.get_speed_limit(world, time_step, [veh_o.id])
            v_list = [
                vehicle.vehicle_param.road_condition_speed_limit,
                v_max_lane,
                v_type,
            ]
            v_max = min(v for v in v_list if v is not None)
            if v_max - veh_o.get_lon_state(time_step).v >= self.config.min_velocity_diff:
                return True
        return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        rob_slow_leading_list = [self._scale_speed(-np.inf)]
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if not self._in_front_of_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ) or not self._same_lane_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ):
                continue
            v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(
                world, time_step, [veh_o.id]
            )
            v_type = self._type_speed_limit_evaluator.get_speed_limit(world, time_step, [veh_o.id])
            v_list = [
                vehicle.vehicle_param.road_condition_speed_limit,
                v_max_lane,
                v_type,
            ]
            v_max = min(v for v in v_list if v is not None)
            rob_slow_leading_list.append(
                self._scale_speed(
                    v_max
                    - veh_o.get_lon_state(time_step).v
                    - self.config.min_velocity_diff
                    - self.config.eps,
                )
            )
        return max(rob_slow_leading_list)


class PredPreservesTrafficFlow(AbstractPredicate):
    """
    Predicate for minimum speed limit evaluation
    """

    predicate_name = VelocityPredicates.PreservesTrafficFlow
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._lane_speed_limit_evaluator = PredLaneSpeedLimitStar(config)
        self._type_speed_limit_evaluator = PredTypeSpeedLimit(config)
        self._fov_speed_limit_evaluator = PredFovSpeedLimit(config)
        self._breaking_speed_limit_evaluator = PredBrSpeedLimit(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(world, time_step, vehicle_ids)
        v_type = self._type_speed_limit_evaluator.get_speed_limit(world, time_step, vehicle_ids)
        v_fov = self._fov_speed_limit_evaluator.get_speed_limit(world, time_step, vehicle_ids)
        v_breaking = self._breaking_speed_limit_evaluator.get_speed_limit(
            world, time_step, vehicle_ids
        )
        v_list = [
            vehicle.vehicle_param.road_condition_speed_limit,
            v_fov,
            v_breaking,
            v_max_lane,
            v_type,
        ]
        v_max = min(v for v in v_list if v is not None)
        if v_max - vehicle.get_lon_state(time_step).v < self.config.min_velocity_diff:
            return True
        else:
            return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        v_max_lane = self._lane_speed_limit_evaluator.get_speed_limit(world, time_step, vehicle_ids)
        v_type = self._type_speed_limit_evaluator.get_speed_limit(world, time_step, vehicle_ids)
        v_fov = self._fov_speed_limit_evaluator.get_speed_limit(world, time_step, vehicle_ids)
        v_breaking = self._breaking_speed_limit_evaluator.get_speed_limit(
            world, time_step, vehicle_ids
        )
        v_list = [
            vehicle.vehicle_param.road_condition_speed_limit,
            v_fov,
            v_breaking,
            v_max_lane,
            v_type,
        ]
        v_max = min(v for v in v_list if v is not None)
        return self._scale_speed(
            self.config.min_velocity_diff
            - v_max
            + vehicle.get_lon_state(time_step).v
            - self.config.eps,
        )


class PredInStandStill(AbstractPredicate):
    """
    Evaluation if vehicle is standing
    """

    predicate_name = VelocityPredicates.InStandstill
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        # ---------------------------------------------------

        if (
            -self.config.standstill_error
            < vehicle.get_lon_state(time_step=time_step, lane=vehicle.ref_path_lane).v
            < self.config.standstill_error
        ):
            return True
        else:
            return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        # avoid getting None of velocity
        ref_path = vehicle.ref_path_lane
        # ---------------------------------------------------

        return self._scale_speed(
            min(
                vehicle.get_lon_state(time_step=time_step, lane=ref_path).v
                + self.config.standstill_error,
                self.config.standstill_error
                - vehicle.get_lon_state(time_step=time_step, lane=ref_path).v
                - self.config.eps,  # TODO why eps only here and not above?
            )
        )


class PredExistStandingLeadingVehicle(AbstractPredicate):
    """
    Predicate which checks if a standing leading vehicle exist in front of a vehicle
    """

    predicate_name = VelocityPredicates.ExistStandingLeadingVehicle
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._in_front_of_evaluator = PredInFrontOf(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._in_standstill_evaluator = PredInStandStill(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if not self._in_front_of_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ) or not self._same_lane_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ):
                continue
            if self._in_standstill_evaluator.evaluate_boolean(world, time_step, [veh_o.id]):
                return True
        return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        rob_standstill_list = [self._scale_speed(-np.inf)]
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        for veh_o in other_vehicles:
            if veh_o.get_lon_state(time_step) is None:
                continue
            if not self._in_front_of_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ) or not self._same_lane_evaluator.evaluate_boolean(
                world, time_step, [vehicle_ids[0], veh_o.id]
            ):
                continue

            rob_standstill_list.append(
                self._in_standstill_evaluator.evaluate_robustness(world, time_step, [veh_o.id])
            )
        return max(rob_standstill_list)


class PredDrivesFaster(AbstractPredicate):
    """
    Predicate which checks if the kth vehicle drives faster than the pth vehicle
    """

    predicate_name = VelocityPredicates.DrivesFaster
    arity = 2

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        return self._scale_speed(
            vehicle_k.get_lon_state(time_step).v
            - vehicle_p.get_lon_state(time_step).v
            - self.config.eps,
        )


class PredDrivesWithSlightlyHigherSpeed(AbstractPredicate):
    """
    Predicate which checks if the kth vehicle drives maximum with slightly higher speed than the pth vehicle
    """

    predicate_name = VelocityPredicates.DrivesWithSlightlyHigherSpeed
    arity = 2

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        return self._scale_speed(
            min(
                vehicle_k.get_lon_state(time_step).v
                - vehicle_p.get_lon_state(time_step).v
                - self.config.eps,
                self.config.slightly_higher_speed_difference
                - vehicle_k.get_lon_state(time_step).v
                + vehicle_p.get_lon_state(time_step).v
                - self.config.eps,
            )
        )
