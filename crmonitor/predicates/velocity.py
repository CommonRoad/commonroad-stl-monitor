from enum import Enum
import logging
import math
from typing import List

from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from ruamel.yaml.comments import CommentedMap

from crmonitor.common.world import World
from crmonitor.predicates.base import BasePredicateEvaluator

logger = logging.getLogger(__name__)


class VelocityPredicates(str, Enum):
    KeepsLaneSpeedLimit = "keeps_lane_speed_limit"
    KeepsTypeSpeedLimit = "keeps_type_speed_limit"
    KeepsFovSpeedLimit = "keeps_fov_speed_limit"
    KeepsBrakeSpeedLimit = "keeps_brake_speed_limit"


class PredGenericSpeedLimit(BasePredicateEvaluator):
    def __init__(self, config: CommentedMap):
        super().__init__(config)

    def get_speed_limit(self, world, time_step, vehicle_ids):
        raise NotImplementedError

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        time_step = time_step
        speed_limit = self.get_speed_limit(world, time_step, vehicle_ids)
        if speed_limit is None:
            rob = math.inf
        else:
            rob = speed_limit + self.eps - vehicle.states_cr[time_step].velocity
        rob = self._scale_speed(rob)
        return rob


class PredLaneSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsLaneSpeedLimit
    arity = 1

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self.country = SupportedTrafficSignCountry(config.get("country"))

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        ts_interpreter = TrafficSigInterpreter(
            self.country, world.road_network.lanelet_network
        )
        speed_limit = ts_interpreter.speed_limit(frozenset(lanelet_ids))
        return speed_limit


class PredTypeSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsTypeSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle_type = world.vehicle_by_id(vehicle_ids[0]).obstacle_type
        if vehicle_type is ObstacleType.TRUCK:
            return self.config["max_interstate_speed_truck"]
        else:
            return None


class PredFovSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsFovSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("fov_speed_limit")


class PredBrSpeedLimit(PredGenericSpeedLimit):
    predicate_name = VelocityPredicates.KeepsBrakeSpeedLimit
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("braking_speed_limit")


class PredLaneSpeedLimitStar(PredLaneSpeedLimit):
    predicate_name = "keeps_lane_speed_limit_star"
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        speed_limit = super(PredLaneSpeedLimitStar, self).get_speed_limit(world, time_step, vehicle_ids)
        if speed_limit is None:
            speed_limit = self.config["desired_interstate_velocity"]
        return speed_limit