import abc
import math
from typing import List
from functools import partial
import warnings

from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState


def norm(x, min_val, max_val):
    normed_val = ((x - min_val) / (max_val - min_val))
    if normed_val > 1.0:
        warnings.warn("Value to normalize exceeded maximum!")
    elif normed_val < 0.0:
        warnings.warn("Value to normalize exceeded minimum!")
    return normed_val


def scale(x, min_val, max_val, new_min=-1.0, new_max=1.0):
    n = norm(x, min_val, max_val)
    return n * (new_max - new_min) + new_min


def get_preceding_vehicle(world_state: WorldState) -> Vehicle:
    veh = None
    min_dist = math.inf
    vehicle_k = world_state.ego_vehicle
    lane_ids_k = world_state.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[world_state.time_step])
    for vehicle_p in world_state.other_vehicles:
        lane_ids_p = world_state.road_network.find_lanes_by_lanelets(
                vehicle_p.lanelet_assignment[world_state.time_step])
        intersecting_lanes = lane_ids_p.intersection(lane_ids_k)
        if len(intersecting_lanes) > 0:
            dist = vehicle_p.front_s(world_state.time_step) - vehicle_k.rear_s(
                    world_state.time_step)
            if dist < min_dist:
                min_dist = dist
                veh = vehicle_p
    return veh


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

    def __init__(self, config):
        self.config = config

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
            return math.inf

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

        return -min(min_dist_k_to_p_lanes, min_dist_p_to_k_lanes)


class PredInFrontOf(IPredicateEvaluator):
    predicate_name = "in_front_of"
    arity = 2

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        rear = world_state.vehicle_by_id(vehicle_ids[0])
        front = world_state.vehicle_by_id(vehicle_ids[1])
        return front.rear_s(world_state.time_step) - rear.front_s(
            world_state.time_step)


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
        return distance_to_boundary


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

        rob = min(-single_lane, same_lane,
                  max(min(l_dist, l_orient), min(r_dist, r_orient)))
        if rob >= 0.0:
            rob = math.inf
        return rob


class PredSafeDistPrec(IPredicateEvaluator):
    predicate_name = "keeps_safe_distance_prec"
    arity = 2

    def __init__(self, config):
        super().__init__(
                config)  # self._a_min_follow = config["ego_vehicle_param"]["a_min"]  # self._a_min_lead = config["other_vehicles_param"]["a_min"]  # self._t_react_follow = config["ego_vehicle_param"]["t_react"]  # assert (  #             self._a_min_follow and 0 > self._a_min_lead), "<BrakingPredicateCollection/safe_distance>: acceleration is not valid"

    def _calculate_safe_distance(self, v_follow, v_lead, a_min_lead,
                                 a_min_follow, t_react_follow):
        d_safe = ((v_lead ** 2) / (-2 * abs(a_min_lead)) - (v_follow ** 2) / (
                -2 * abs(a_min_follow)) + v_follow * t_react_follow)

        return d_safe

    def evaluate_robustness(self, world_state: WorldState,
                            vehicle_ids: List[int]) -> float:
        vehicle_follow = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_lead = world_state.vehicle_by_id(vehicle_ids[1])
        time_step = world_state.time_step

        if vehicle_lead.states_lon.get(time_step) is None:
            return math.inf
        a_min_follow = vehicle_follow.vehicle_param.get("a_min")
        a_min_lead = vehicle_lead.vehicle_param.get("a_min")
        t_react_follow = vehicle_follow.vehicle_param.get("t_react")
        safe_distance = self._calculate_safe_distance(
                vehicle_follow.states_lon[time_step].v,
                vehicle_lead.states_lon[time_step].v, a_min_lead, a_min_follow,
                t_react_follow)

        delta_s = vehicle_lead.rear_s(time_step) - vehicle_follow.front_s(
                time_step)
        return delta_s - safe_distance
