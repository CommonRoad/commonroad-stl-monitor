import abc
import logging
import math
from functools import partial
from typing import List, Tuple

import numpy as np
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from ruamel.yaml.comments import CommentedMap


def norm(x, min_val, max_val):
    if math.isinf(x):
        x = max_val
    normed_val = ((x - min_val) / (max_val - min_val))
    if normed_val > 1.0:
        logging.debug("Value to normalize exceeded maximum!")
    elif normed_val < 0.0:
        logging.debug("Value to normalize exceeded minimum!")
    return normed_val


def scale_clip(x, min_val, max_val, new_min=0.0, new_max=1.0, copysign=False):
    n = norm(math.fabs(x), min_val, max_val)
    rescaled = n * (new_max - new_min) + new_min
    # Clip
    rescaled = min(max(rescaled, new_min), new_max)
    if copysign:
        rescaled = math.copysign(rescaled, x)
    return rescaled


def get_preceding_vehicles(world_state: WorldState, vehicle_rear: Vehicle) -> \
        List[Tuple[float, Vehicle]]:
    """
    Returns a list of preceding vehicles in ascending order of distance
    :param world_state: Current world state
    :param vehicle_rear: Reference vehicle
    :return: Sorted list of tuples of distance and vehicle object
    """
    veh = []
    rear_lanes = vehicle_rear.robust_lanelet_assignment[world_state.time_step]
    for vehicle_lead in world_state.other_vehicles + [world_state.ego_vehicle]:
        if not vehicle_lead.is_valid(
                world_state.time_step) or vehicle_lead is vehicle_rear:
            continue
        lead_lanes = vehicle_lead.robust_lanelet_assignment[world_state.time_step]
        intersecting_lanes = lead_lanes.intersection(rear_lanes)
        if len(intersecting_lanes) > 0:
            dist = vehicle_lead.rear_s(
                    world_state.time_step) - vehicle_rear.front_s(
                    world_state.time_step)
            if dist >= 0.0:
                veh.append((dist, vehicle_lead))
    return sorted(veh, key=lambda d: d[0])


class LazyValue:
    def __init__(self, function):
        self._function = function
        self._value = None

    @property
    def value(self):
        if self._value is None:
            self._value = self._function()
        return self._value


class IPredicateEvaluator(abc.ABC):
    predicate_name = "interface"

    def __init__(self, config: CommentedMap):
        self.config = config
        self.scale = config.setdefault("scale_rob", True)

    def _scale(self, x, *args, **kwargs):
        if self.scale:
            return scale_clip(x, *args, **kwargs)
        else:
            return x

    def _scale_speed(self, x):
        return self._scale(x, 0.0, 250.0 / 3.6, copysign=True)

    def _scale_acc(self, x):
        return self._scale(x, 0, 10.5, copysign=True)

    def _scale_dist(self, x):
        return self._scale(x, 0.0, 200.0, copysign=True)

    def _scale_angle(self, x):
        # angle = x - (math.ceil((x + math.pi) / (2 * math.pi)) - 1) * 2 * math.pi
        # TODO: Might be slow
        # angle = math.asin(math.sin(x))
        return self._scale(x, 0, math.pi, copysign=True)

    def evaluate_boolean_lazy(self, world_state: WorldState,
                              vehicle_ids: List[int]):
        fun = partial(self.evaluate_boolean, world_state=world_state,
                      vehicles=vehicle_ids)
        lazy_value = LazyValue(fun)
        return lazy_value

    def evaluate_boolean(self, world_state: WorldState,
                         vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world_state, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        pass


class PredInSameLane(IPredicateEvaluator):
    predicate_name = "in_same_lane"
    arity = 2

    def evaluate_boolean(self, world_state: WorldState,
                         vehicle_ids: List[int]) -> bool:
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world_state.vehicle_by_id(vehicle_ids[1])
        lane_ids_k = world_state.road_network.find_lanes_by_lanelets(
                vehicle_k.lanelet_assignment[world_state.time_step])
        lane_ids_p = world_state.road_network.find_lanes_by_lanelets(
                vehicle_p.lanelet_assignment[world_state.time_step])
        intersecting_lanes = lane_ids_p.intersection(lane_ids_k)
        return len(intersecting_lanes) > 0

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        if self.evaluate_boolean(world_state, vehicle_ids):
            return self._scale_dist(math.inf)

        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world_state.vehicle_by_id(vehicle_ids[1])
        min_dist_k_to_p_lanes = math.inf
        k_occ = vehicle_k.occupancy_at_time_step(
                world_state.time_step).shapely_object
        # TODO: could be also reused from binary computation
        lane_ids_k = world_state.road_network.find_lanes_by_lanelets(
                vehicle_k.lanelet_assignment[world_state.time_step])
        lane_ids_p = world_state.road_network.find_lanes_by_lanelets(
                vehicle_p.lanelet_assignment[world_state.time_step])
        for lane_p in lane_ids_p:
            dist = lane_p.lanelet.convert_to_polygon().shapely_object.distance(
                    k_occ)
            min_dist_k_to_p_lanes = min(min_dist_k_to_p_lanes, dist)

        min_dist_p_to_k_lanes = math.inf
        p_occ = vehicle_p.occupancy_at_time_step(
                world_state.time_step).shapely_object
        for lane_k in lane_ids_k:
            dist = lane_k.lanelet.convert_to_polygon().shapely_object.distance(
                    p_occ)
            min_dist_p_to_k_lanes = min(min_dist_p_to_k_lanes, dist)

        return -min(self._scale_dist(min_dist_k_to_p_lanes),
                    self._scale_dist(min_dist_p_to_k_lanes))


class PredInFrontOf(IPredicateEvaluator):
    predicate_name = "in_front_of"
    arity = 2

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        rear = world_state.vehicle_by_id(vehicle_ids[0])
        front = world_state.vehicle_by_id(vehicle_ids[1])
        return self._scale_dist(
                front.rear_s(world_state.time_step) - rear.front_s(
                        world_state.time_step))


class PredSingleLane(IPredicateEvaluator):
    predicate_name = "single_lane"
    arity = 1

    def evaluate_boolean(self, world_state: WorldState,
                         vehicle_ids: List[int]) -> bool:
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        k_lanes = world_state.road_network.find_lanes_by_lanelets(
                vehicle_k.lanelet_assignment[world_state.time_step])
        single_lane = False
        if len(k_lanes) == 1:  # single_lane
            single_lane = True
        return single_lane

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        single_lane_boolean = self.evaluate_boolean(world_state, vehicle_ids)
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        if single_lane_boolean:
            k_lanes = world_state.road_network.find_lanes_by_lanelets(
                    vehicle_k.lanelet_assignment[world_state.time_step])
            k_lane = k_lanes.pop()
            k_occ = vehicle_k.occupancy_at_time_step(
                    world_state.time_step).shapely_object
            lane_poly = k_lane.lanelet.convert_to_polygon().shapely_object
            distance_to_boundary = lane_poly.boundary.distance(k_occ)
        else:
            distance_to_boundary = -math.inf
        return self._scale_dist(distance_to_boundary)


class PredCutIn(IPredicateEvaluator):
    predicate_name = "cut_in"
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._single_lane_evaluator = PredSingleLane(config)

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        cutting_vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world_state.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_robustness(
                world_state, [vehicle_ids[0]])
        same_lane = self._same_lane_evaluator.evaluate_robustness(world_state,
                                                                  vehicle_ids)

        l_dist = cutted_vehicle.states_lat[world_state.time_step].d - \
                 cutting_vehicle.states_lat[world_state.time_step].d
        l_orient = cutting_vehicle.states_lat[world_state.time_step].theta
        r_dist = cutting_vehicle.states_lat[world_state.time_step].d - \
                 cutted_vehicle.states_lat[world_state.time_step].d
        r_orient = -cutting_vehicle.states_lat[world_state.time_step].theta + .0

        l_dist = self._scale_dist(l_dist)
        r_dist = self._scale_dist(r_dist)
        l_orient = self._scale_angle(l_orient)
        r_orient = self._scale_angle(r_orient)

        rob = min(-single_lane, same_lane,
                  max(min(l_dist, l_orient), min(r_dist, r_orient)))
        if rob >= 0.0:
            rob = 1.0
        return rob


class PredSafeDistPrec(IPredicateEvaluator):
    predicate_name = "keeps_safe_distance_prec"
    arity = 2

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def calculate_safe_distance(cls, v_follow, v_lead, a_min_lead, a_min_follow,
                                t_react_follow):
        d_safe = ((v_lead ** 2) / (-2 * np.abs(a_min_lead)) - (
                v_follow ** 2) / (-2 * np.abs(
                a_min_follow)) + v_follow * t_react_follow)

        return d_safe

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        vehicle_follow = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_lead = world_state.vehicle_by_id(vehicle_ids[1])
        time_step = world_state.time_step

        if vehicle_lead.states_lon.get(time_step) is None:
            return self._scale_dist(math.inf)
        a_min_follow = vehicle_follow.vehicle_param.get("a_min")
        a_min_lead = vehicle_lead.vehicle_param.get("a_min")
        t_react_follow = vehicle_follow.vehicle_param.get("t_react")
        safe_distance = self.calculate_safe_distance(
                vehicle_follow.states_lon[time_step].v,
                vehicle_lead.states_lon[time_step].v, a_min_lead, a_min_follow,
                t_react_follow)

        delta_s = vehicle_lead.rear_s(time_step) - vehicle_follow.front_s(
                time_step)
        rob = self._scale_dist(delta_s - safe_distance)
        return rob


class PredUnnecessaryBraking(IPredicateEvaluator):
    predicate_name = "unnecessary_braking"
    arity = 1

    def __init__(self, config):
        super().__init__(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._front_evaluator = PredInFrontOf(config)
        self._safe_distance_evaluator = PredSafeDistPrec(config)
        self.a_abrupt = config["a_abrupt"]

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        other_ids = [veh.id for veh in world_state.other_vehicles] + [
            world_state.ego_vehicle.id]
        other_ids.remove(vehicle_ids[0])
        ego_acc = world_state.vehicle_by_id(vehicle_ids[0]).states_lon[
            world_state.time_step].a
        # # Short circuit: If we are not breaking, no unnecessary braking
        # if ego_acc >= 0.0:
        #     return -ego_acc + 0.0
        excemption_a = [math.inf]
        excemption_b = [-math.inf]
        for o_id in other_ids:
            if not world_state.vehicle_by_id(o_id).is_valid(
                    world_state.time_step):
                continue
            ids = [vehicle_ids[0], o_id]
            same_lane = self._same_lane_evaluator.evaluate_robustness(
                    world_state, ids)
            front_of = self._front_evaluator.evaluate_robustness(world_state,
                                                                 ids)
            safe_dist = self._safe_distance_evaluator.evaluate_robustness(
                    world_state, ids)
            excemption_a.append(-min(front_of, same_lane))
            other_acc = world_state.vehicle_by_id(o_id).states_lon[
                world_state.time_step].a
            acc_diff = self._scale_acc(self.a_abrupt + other_acc - ego_acc)
            excemption_b.append(min(safe_dist, front_of, same_lane, acc_diff))

        min_excempt_a = min(min(excemption_a),
                            self._scale_acc(self.a_abrupt - ego_acc))
        max_excempt_b = max(excemption_b)

        rob = min(self._scale_acc(-ego_acc), max(min_excempt_a, max_excempt_b))
        return rob


class PredGenericSpeedLimit(IPredicateEvaluator):
    def __init__(self, config: CommentedMap):
        super().__init__(config)

    def get_speed_limit(self, world_state, vehicle_ids):
        raise NotImplementedError

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        time_step = world_state.time_step
        speed_limit = self.get_speed_limit(world_state, vehicle_ids)
        if speed_limit is None:
            rob = math.inf
        else:
            rob = speed_limit - vehicle.states_lon[time_step].v
        rob = self._scale_speed(rob)
        return rob


class PredLaneSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_lane_speed_limit"
    arity = 1

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self.country = SupportedTrafficSignCountry(config.get("country"))

    def get_speed_limit(self, world_state, vehicle_ids):
        vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        time_step = world_state.time_step
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        ts_interpreter = TrafficSigInterpreter(self.country,
                                               world_state.road_network.lanelet_network)
        speed_limit = ts_interpreter.speed_limit(frozenset(lanelet_ids))
        return speed_limit


class PredTypeSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_type_speed_limit"
    arity = 1

    def get_speed_limit(self, world_state, vehicle_ids):
        vehicle_type = world_state.vehicle_by_id(vehicle_ids[0]).obstacle_type
        if vehicle_type is ObstacleType.TRUCK:
            return 22.22
        else:
            return None


class PredFovSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_fov_speed_limit"
    arity = 1

    def get_speed_limit(self, world_state, vehicle_ids):
        vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("fov_speed_limit")


class PredLaneSpeedLimitStar(PredLaneSpeedLimit):
    predicate_name = "keeps_lane_speed_limit_star"
    arity = 1

    def get_speed_limit(self, world_state, vehicle_ids):
        speed_limit = super(PredLaneSpeedLimitStar, self).get_speed_limit(
            world_state, vehicle_ids)
        if speed_limit is None:
            speed_limit = 130.0 / 3.6
        return speed_limit


class PredLeadingVehicle(IPredicateEvaluator):
    predicate_name = "has_leading_vehicle"
    arity = 1

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._front_evaluator = PredInFrontOf(config)

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        other_ids = [veh.id for veh in world_state.other_vehicles] + [
            world_state.ego_vehicle.id]
        other_ids.remove(vehicle_ids[0])
        rob_values = []
        for o_id in other_ids:
            if not world_state.vehicle_by_id(o_id).is_valid(
                    world_state.time_step):
                continue
            ids = [vehicle_ids[0], o_id]
            same_lane = self._same_lane_evaluator.evaluate_robustness(
                    world_state, ids)
            front_of = self._front_evaluator.evaluate_robustness(world_state,
                                                                 ids)
            rob_values.append(min(same_lane, front_of))
        if len(rob_values) == 0:
            rob = self._scale_dist(-math.inf)
        else:
            rob = min(rob_values)
        return rob


class PredPrecedes(IPredicateEvaluator):
    predicate_name = "precedes"
    arity = 2

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        prec_veh = get_preceding_vehicles(world_state, world_state.vehicle_by_id(vehicle_ids[0]))
        if len(prec_veh) > 0 and prec_veh[0][1].id == vehicle_ids[1]:
            return self._scale_dist(math.inf)
        else:
            return self._scale_dist(-math.inf)


class PredAcceleration(IPredicateEvaluator):
    predicate_name = "accel"
    arity = 1

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        accel = world_state.vehicle_by_id(vehicle_ids[0]).states_lon[
            world_state.time_step].a
        return accel


class PredMaxLeadAcceleration(IPredicateEvaluator):
    predicate_name = "max_lead_accel"
    arity = 1

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        lead_veh = get_preceding_vehicles(world_state,
                                          world_state.vehicle_by_id(
                                                  vehicle_ids[0]))
        if len(lead_veh) == 0:
            return 0.0
        else:
            accel = [v.states_lon[world_state.time_step].a for _, v in lead_veh]
            return max(accel)
