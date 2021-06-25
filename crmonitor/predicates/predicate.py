import abc
import logging
import math
from functools import reduce
from typing import List, Tuple, Set, Iterable

import numpy as np
from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from ruamel.yaml.comments import CommentedMap

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState

logger = logging.getLogger(__name__)


def union_set(s: Iterable):
    return reduce(lambda agg, e: agg.union(e), s, set())


def scale_clip(x, min_val, max_val, new_min=0.0, new_max=1.0, copysign=False):
    abs_x = np.abs(x)
    if np.isinf(abs_x):
        rescaled = new_max
    else:
        rescaled = np.interp(abs_x, [min_val, max_val], [new_min, new_max])
    if copysign:
        rescaled = np.copysign(rescaled, x)
    return rescaled


def get_succeeding_vehicles(
    world_state: WorldState, vehicle_front: Vehicle
) -> List[Tuple[float, Vehicle]]:
    """
    Returns a list of preceding vehicles in ascending order of distance
    :param world_state: Current world state
    :param vehicle_front: Reference vehicle
    :return: Sorted list of tuples of distance and vehicle object
    """
    veh = []
    front_lanes = vehicle_front.lanelet_assignment[world_state.time_step]
    for vehicle_rear in world_state.other_vehicles + [world_state.ego_vehicle]:
        if (
            not vehicle_rear.is_valid(world_state.time_step)
            or vehicle_rear is vehicle_front
        ):
            continue
        rear_lanes = vehicle_rear.lanelet_assignment[world_state.time_step]
        intersecting_lanes = rear_lanes.intersection(front_lanes)
        if len(intersecting_lanes) > 0:
            dist = vehicle_front.rear_s(world_state.time_step) - vehicle_rear.front_s(
                world_state.time_step
            )
            if dist >= 0.0:
                veh.append((dist, vehicle_rear))
    return sorted(veh, key=lambda d: d[0])


def distance_to_bounds(
    vehicle_i: Vehicle, lanelet_ids: Iterable[int], world_state: WorldState
):
    state = vehicle_i.states_cr[world_state.time_step]
    occ_points = list(
        rotate_translate(
            vehicle_i.shape.vertices[:-1], state.position, state.orientation
        )
    )
    lanelets = [
        world_state.road_network.lanelet_network.find_lanelet_by_id(i)
        for i in lanelet_ids
    ]
    left_bounds = [
        l for l in lanelets if l.adj_left is not None and l.adj_left not in lanelet_ids
    ]
    right_bounds = [
        l
        for l in lanelets
        if l.adj_right is not None and l.adj_right not in lanelet_ids
    ]
    d_left = [np.array([])]
    for l in left_bounds:
        # For performance reasons, we find a lane that contains the lanelet
        # so that
        # the curvilinear coordinate system stored in the lane can be reused.
        lane = world_state.road_network.find_lanes_by_lanelets([l.lanelet_id]).pop()
        start_s = lane.clcs_left.convert_to_curvilinear_coords(*l.left_vertices[0])[0]
        end_s = lane.clcs_left.convert_to_curvilinear_coords(*l.left_vertices[-1])[0]
        corner_points = np.array(
            lane.clcs_left.convert_list_of_points_to_curvilinear_coords(occ_points, 1)
        )
        # Only consider points within the projection domain of the lanelet
        points_in_proj_domain = corner_points[
            (corner_points[:, 0] >= start_s) & (corner_points[:, 0] <= end_s)
        ]
        if points_in_proj_domain.size > 0:
            d_left.append(points_in_proj_domain[:, 1])

    d_right = [np.array([])]
    for l in right_bounds:
        lane = world_state.road_network.find_lanes_by_lanelets([l.lanelet_id]).pop()
        start_s = lane.clcs_right.convert_to_curvilinear_coords(*l.right_vertices[0])[0]
        end_s = lane.clcs_right.convert_to_curvilinear_coords(*l.right_vertices[-1])[0]
        corner_points = np.array(
            lane.clcs_right.convert_list_of_points_to_curvilinear_coords(occ_points, 1)
        )
        points_in_proj_domain = corner_points[
            (corner_points[:, 0] >= start_s) & (corner_points[:, 0] <= end_s)
        ]
        if points_in_proj_domain.size > 0:
            d_right.append(points_in_proj_domain[:, 1])
    return np.concatenate(d_left), np.concatenate(d_right)


class BasePredicateEvaluator(abc.ABC):
    predicate_name = "interface"

    def __init__(self, config: CommentedMap):
        self.config = config
        self.scale = config.setdefault("scale_rob", True)
        self.eps = 1e-5

    def _scale(self, x, *args, **kwargs):
        if self.scale:
            return scale_clip(x, *args, **kwargs)
        else:
            return x

    def _scale_speed(self, x):
        return self._scale(x, 0.0, 250.0 / 3.6, copysign=True)

    def _scale_acc(self, x):
        return self._scale(x, 0, 10.5, copysign=True)

    def _scale_lon_dist(self, x):
        return self._scale(x, 0.0, 200.0, copysign=True)

    def _scale_lat_dist(self, x):
        return self._scale(x, 0.0, 20.0, copysign=True)

    def _scale_angle(self, x):
        # angle = x - (math.ceil((x + math.pi) / (2 * math.pi)) - 1) * 2 *
        # math.pi
        # TODO: Might be slow
        # angle = math.asin(math.sin(x))
        return self._scale(x, 0, math.pi, copysign=True)

    def evaluate_boolean(self, world_state: WorldState, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world_state, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        pass

    def evaluate_robustness_with_cache(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        vehicle_ids_tuple = tuple(vehicle_ids)
        value = world_state.predicate_values[world_state.time_step][
            self.predicate_name
        ].get(vehicle_ids_tuple)
        if value is None:
            logger.debug(
                "Evaluating predicate %s , t=%d, ids=%s",
                self.predicate_name,
                world_state.time_step,
                vehicle_ids,
            )
            value = self.evaluate_robustness(world_state, vehicle_ids)
            world_state.predicate_values[world_state.time_step][self.predicate_name][
                vehicle_ids_tuple
            ] = value
        return value


class PredInSameLane(BasePredicateEvaluator):
    predicate_name = "in_same_lane"
    arity = 2

    def evaluate_boolean(self, world_state: WorldState, vehicle_ids: List[int]) -> bool:
        intersecting_lanes = self.get_same_lanes(world_state, vehicle_ids)
        return len(intersecting_lanes) > 0

    def get_same_lanes(self, world_state, vehicle_ids) -> Set[Lane]:
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world_state.vehicle_by_id(vehicle_ids[1])
        lanes_k = world_state.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[world_state.time_step]
        )
        lanes_p = world_state.road_network.find_lanes_by_lanelets(
            vehicle_p.lanelet_assignment[world_state.time_step]
        )
        intersecting_lanes = lanes_p.intersection(lanes_k)
        return intersecting_lanes

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        """
        If boolean is
        True: Minimum lateral displacement to not be in the same lane anymore
        False: Minimum distance to lanes of other
        :param world_state:
        :param vehicle_ids:
        :return:
        """
        # Predicate is symmetric
        vehicle_ids_tuple = tuple(reversed(vehicle_ids))
        value = world_state.predicate_values[world_state.time_step][
            self.predicate_name
        ].get(vehicle_ids_tuple)
        if value is not None:
            return value

        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world_state.vehicle_by_id(vehicle_ids[1])

        def distance_to_lanes(vehicle_i: Vehicle, lanelet_ids: Iterable[int]):
            d_left, d_right = distance_to_bounds(vehicle_i, lanelet_ids, world_state)
            d_left = -np.min(d_left) if d_left.size > 0 else np.inf
            d_right = np.max(d_right) if d_right.size > 0 else np.inf
            return np.fmin(d_left, d_right)

        lanelet_ids_k = union_set(
            [l.contained_lanelets for l in vehicle_k.lanes_at_state(world_state)]
        )
        lanelet_ids_p = union_set(
            [l.contained_lanelets for l in vehicle_p.lanes_at_state(world_state)]
        )
        rob = np.fmin(
            distance_to_lanes(vehicle_k, lanelet_ids_p),
            distance_to_lanes(vehicle_p, lanelet_ids_k),
        )
        return self._scale_lat_dist(rob)


class PredInFrontOf(BasePredicateEvaluator):
    predicate_name = "in_front_of"
    arity = 2

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        rear = world_state.vehicle_by_id(vehicle_ids[0])
        front = world_state.vehicle_by_id(vehicle_ids[1])
        return self._scale_lon_dist(
            front.rear_s(world_state.time_step) - rear.front_s(world_state.time_step)
        )


class PredSingleLane(BasePredicateEvaluator):
    predicate_name = "single_lane"
    arity = 1

    def evaluate_boolean(self, world_state: WorldState, vehicle_ids: List[int]) -> bool:
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        k_lanes = world_state.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[world_state.time_step]
        )
        return len(k_lanes) == 1

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        """
        If false: 1 - largest fractional overlap with occupied lanes
        If true: Distance to lane polygon boundary
        :param world_state:
        :param vehicle_ids:
        :return:
        """
        # single_lane_boolean = self.evaluate_boolean(world_state, vehicle_ids)
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        k_lanes = list(
            world_state.road_network.find_lanes_by_lanelets(
                vehicle_k.lanelet_assignment[world_state.time_step]
            )
        )
        assert (
            len(k_lanes) > 0
        ), f"Vehicle must be assigned to at least one lane! {str(world_state.scenario.scenario_id)}, id={vehicle_ids[0]}, t={world_state.time_step}, ego={world_state.ego_vehicle.id}"

        shape_k = vehicle_k.shapely_occupancy_at_time_step(world_state.time_step)
        overlap_areas = [
            lane.lanelet.convert_to_polygon().shapely_object.intersection(shape_k).area
            for lane in k_lanes
        ]
        max_overlap_lane = k_lanes[np.argmax(overlap_areas)]

        d_left, d_right = distance_to_bounds(
            vehicle_k, max_overlap_lane.contained_lanelets, world_state
        )
        d_left = -np.max(d_left) if d_left.size > 0 else np.inf
        d_right = np.min(d_right) if d_right.size > 0 else np.inf
        rob = np.fmin(d_left, d_right)
        return self._scale_lat_dist(rob)


class PredCutIn(BasePredicateEvaluator):
    predicate_name = "cut_in"
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._single_lane_evaluator = PredSingleLane(config)

    def evaluate_boolean(self, world_state: WorldState, vehicle_ids: List[int]) -> bool:
        cutting_vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world_state.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_boolean(
            world_state, [vehicle_ids[0]]
        )
        if single_lane:
            return False
        same_lane = self._same_lane_evaluator.evaluate_boolean(world_state, vehicle_ids)
        if not same_lane:
            return False
        d_p = cutted_vehicle.states_lat[world_state.time_step].d
        d_k = cutting_vehicle.states_lat[world_state.time_step].d
        orient_k = cutting_vehicle.states_lat[world_state.time_step].theta

        result = (d_k < d_p and orient_k > self.eps) or (
            d_k > d_p and orient_k < -self.eps
        )
        return result

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        cutting_vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world_state.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_robustness_with_cache(
            world_state,
            (vehicle_ids[0],),
        )
        same_lane = self._same_lane_evaluator.evaluate_robustness_with_cache(
            world_state, vehicle_ids
        )

        r_l_dist = (
            cutted_vehicle.states_lat[world_state.time_step].d
            - cutting_vehicle.states_lat[world_state.time_step].d
        )
        r_l_orient = cutting_vehicle.states_lat[world_state.time_step].theta - self.eps
        l_r_dist = (
            cutting_vehicle.states_lat[world_state.time_step].d
            - cutted_vehicle.states_lat[world_state.time_step].d
        )
        l_r_orient = -self.eps - cutting_vehicle.states_lat[world_state.time_step].theta

        r_l_dist = self._scale_lat_dist(r_l_dist)
        l_r_dist = self._scale_lat_dist(l_r_dist)
        r_l_orient = self._scale_angle(r_l_orient)
        l_r_orient = self._scale_angle(l_r_orient)

        rob = min(
            -single_lane,
            same_lane,
            max(min(r_l_dist, r_l_orient), min(l_r_dist, l_r_orient)),
        )
        return rob


class PredSafeDistPrec(BasePredicateEvaluator):
    predicate_name = "keeps_safe_distance_prec"
    arity = 2

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def calculate_safe_distance(
        cls, v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow
    ):
        d_safe = (
            (v_lead ** 2) / (-2 * np.abs(a_min_lead))
            - (v_follow ** 2) / (-2 * np.abs(a_min_follow))
            + v_follow * t_react_follow
        )

        return d_safe

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        vehicle_follow = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_lead = world_state.vehicle_by_id(vehicle_ids[1])
        time_step = world_state.time_step

        if vehicle_lead.states_lon.get(time_step) is None:
            return self._scale_lon_dist(math.inf)
        a_min_follow = vehicle_follow.vehicle_param.get("a_min")
        a_min_lead = vehicle_lead.vehicle_param.get("a_min")
        t_react_follow = vehicle_follow.vehicle_param.get("t_react")
        safe_distance = self.calculate_safe_distance(
            vehicle_follow.states_lon[time_step].v,
            vehicle_lead.states_lon[time_step].v,
            a_min_lead,
            a_min_follow,
            t_react_follow,
        )

        delta_s = vehicle_lead.rear_s(time_step) - vehicle_follow.front_s(time_step)
        rob = self._scale_lon_dist(delta_s - safe_distance)
        return rob


class PredGenericSpeedLimit(BasePredicateEvaluator):
    def __init__(self, config: CommentedMap):
        super().__init__(config)

    def get_speed_limit(self, world_state, vehicle_ids):
        raise NotImplementedError

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        time_step = world_state.time_step
        speed_limit = self.get_speed_limit(world_state, vehicle_ids)
        if speed_limit is None:
            rob = math.inf
        else:
            rob = speed_limit + self.eps - vehicle.states_lon[time_step].v
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
        ts_interpreter = TrafficSigInterpreter(
            self.country, world_state.road_network.lanelet_network
        )
        speed_limit = ts_interpreter.speed_limit(frozenset(lanelet_ids))
        return speed_limit


class PredTypeSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_type_speed_limit"
    arity = 1

    def get_speed_limit(self, world_state, vehicle_ids):
        vehicle_type = world_state.vehicle_by_id(vehicle_ids[0]).obstacle_type
        if vehicle_type is ObstacleType.TRUCK:
            return self.config["max_interstate_speed_truck"]
        else:
            return None


class PredFovSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_fov_speed_limit"
    arity = 1

    def get_speed_limit(self, world_state, vehicle_ids):
        vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("fov_speed_limit")


class PredBrSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_brake_speed_limit"
    arity = 1

    def get_speed_limit(self, world_state, vehicle_ids):
        vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("braking_speed_limit")


class PredLaneSpeedLimitStar(PredLaneSpeedLimit):
    predicate_name = "keeps_lane_speed_limit_star"
    arity = 1

    def get_speed_limit(self, world_state, vehicle_ids):
        speed_limit = super(PredLaneSpeedLimitStar, self).get_speed_limit(
            world_state, vehicle_ids
        )
        if speed_limit is None:
            speed_limit = self.config["desired_interstate_velocity"]
        return speed_limit


class PredSucceeds(BasePredicateEvaluator):
    predicate_name = "succeeds"
    arity = 2

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self.same_lane = PredInSameLane(config)

    def evaluate_boolean(self, world_state: WorldState, vehicle_ids: List[int]) -> bool:
        succeeding_vehicle_id = vehicle_ids[0]
        other_vehicle_id = vehicle_ids[1]
        other_vehicle = world_state.vehicle_by_id(other_vehicle_id)
        succ_veh = get_succeeding_vehicles(world_state, other_vehicle)
        return len(succ_veh) > 0 and succ_veh[0][1].id == succeeding_vehicle_id

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        ego_vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world_state.vehicle_by_id(vehicle_ids[1])
        succ_veh = get_succeeding_vehicles(world_state, other_vehicle)
        if len(succ_veh) > 0 and succ_veh[0][1].id == vehicle_ids[0]:
            same_lane = self.same_lane.evaluate_robustness_with_cache(
                world_state, vehicle_ids
            )
            assert same_lane >= 0.0
            overtake = other_vehicle.rear_s(
                world_state.time_step
            ) - ego_vehicle.front_s(world_state.time_step)
            assert overtake >= 0.0
            if len(succ_veh) >= 2:
                fallback = ego_vehicle.front_s(world_state.time_step) - succ_veh[1][
                    1
                ].front_s(world_state.time_step)
                assert fallback >= 0.0
            else:
                fallback = math.inf
            return min(
                same_lane,
                self._scale_lon_dist(overtake),
                self._scale_lon_dist(fallback),
            )
        else:
            # if other_vehicle.rear_s(world_state.time_step) < ego_vehicle.front_s(world_state.time_step):
            #     # Other vehicle is behind
            #     v = ego_vehicle.front_s(world_state.time_step) - other_vehicle.rear_s(world_state.time_step)
            # else:
            #     # Other vehicle is in front
            #     suc_veh = get_succeeding_vehicles(world_state, other_vehicle)
            #     if len(suc_veh) > 0:
            #         # Other vehicle has a successor
            #         v = suc_veh[0][1].front_s(world_state.time_step) - ego_vehicle.front_s(world_state.time_step)
            #     else:
            #         # Should only happen if same_lane < 0.0
            #         # as otherwise ego should be the successor -> precedes
            #         v = 0.0
            # if same_lane < 0.0:
            #     return -self._scale_lon_dist(np.sqrt(same_lane * same_lane + v * v))
            # else:
            #     assert v != 0.0
            #     return -self._scale_lon_dist(v)
            return -1.0


class PredAbruptBreaking(BasePredicateEvaluator):
    predicate_name = "brakes_abruptly"
    arity = 1

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        accel = (
            world_state.vehicle_by_id(vehicle_ids[0])
            .states_lon[world_state.time_step]
            .a
        )
        rob = self.config["a_abrupt"] - accel
        return self._scale_acc(rob)


class PredRelAbruptBreaking(BasePredicateEvaluator):
    predicate_name = "rel_brakes_abruptly"
    arity = 2

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        accel_k = (
            world_state.vehicle_by_id(vehicle_ids[0])
            .states_lon[world_state.time_step]
            .a
        )
        accel_p = (
            world_state.vehicle_by_id(vehicle_ids[1])
            .states_lon[world_state.time_step]
            .a
        )
        rob = -accel_k + accel_p + self.config["a_abrupt"]
        return self._scale_acc(rob)
