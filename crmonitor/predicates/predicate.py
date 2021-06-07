import abc
import math
from functools import reduce
from typing import List, Tuple, Set

import numpy as np
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from crmonitor.common.helper import min_max
from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from ruamel.yaml.comments import CommentedMap


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


def get_adjacent_lanelets(lanelet_ids, lanelet_network):
    lanelets = [lanelet_network.find_lanelet_by_id(l_id) for l_id in lanelet_ids]
    left_adj = [
        lanelet.adj_left
        for lanelet in lanelets
        if lanelet.adj_left is not None and lanelet.adj_left not in lanelet_ids
    ]
    right_adj = [
        lanelet.adj_right
        for lanelet in lanelets
        if lanelet.adj_right is not None and lanelet.adj_right not in lanelet_ids
    ]
    left_adj = [lanelet_network.find_lanelet_by_id(l_id) for l_id in left_adj]
    right_adj = [lanelet_network.find_lanelet_by_id(l_id) for l_id in right_adj]
    return left_adj, right_adj


class IPredicateEvaluator(abc.ABC):
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
        return self.evaluate_robustness(world_state, vehicle_ids) > 0.0

    @abc.abstractmethod
    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        pass


class PredInSameLane(IPredicateEvaluator):
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

    @staticmethod
    def distance_to_lane_bound(vehicle_k, intersecting_lanes, world_state):
        intersecting_lanelets = [
            l.contained_lanelets.intersection(
                vehicle_k.lanelet_assignment[world_state.time_step]
            )
            for l in intersecting_lanes
        ]
        intersecting_lanelets = reduce(
            lambda x, agg: agg.union(), intersecting_lanelets, set()
        )
        left_adj, right_adj = get_adjacent_lanelets(
            intersecting_lanelets, world_state.road_network.lanelet_network
        )
        has_left_adj = len(left_adj) > 0
        has_right_adj = len(right_adj) > 0
        if not has_left_adj and not has_right_adj:
            # No adjacent lanes
            return np.inf

        occ = vehicle_k.occupancy_at_time_step(world_state.time_step)
        vert = list(occ.vertices)
        if has_left_adj:
            lane = intersecting_lanes[
                np.argmax([l.lanelet.left_vertices[0, 1] for l in intersecting_lanes])
            ]
            _, dist_left = zip(
                *lane.clcs_left.convert_list_of_points_to_curvilinear_coords(vert, 1)
            )
            dist_left = [d for d in dist_left if d < 0]
            d_l_min, d_l_max = min_max(dist_left)
            d_l_min = np.abs(d_l_min)
            d_l_max = np.abs(d_l_max)
        else:
            d_l_min = np.inf

        if has_right_adj:
            lane = intersecting_lanes[
                np.argmin([l.lanelet.right_vertices[0, 1] for l in intersecting_lanes])
            ]
            _, dist_right = zip(
                *lane.clcs_right.convert_list_of_points_to_curvilinear_coords(vert, 1)
            )
            dist_right = [d for d in dist_right if d > 0]
            d_r_min, d_r_max = min_max(dist_right)
        else:
            d_r_min = np.inf

        if d_l_min < d_r_min:
            res = d_l_min, d_l_max
        elif d_r_min < d_l_min:
            res = d_r_min, d_r_max
        else:
            res = d_l_min, np.fmin(d_l_max, d_r_max)
        return res

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
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world_state.vehicle_by_id(vehicle_ids[1])
        if self.evaluate_boolean(world_state, vehicle_ids):
            intersecting_lanes = list(self.get_same_lanes(world_state, vehicle_ids))
            rob_k = self.distance_to_lane_bound(
                vehicle_k, intersecting_lanes, world_state
            )[1]
            rob_p = self.distance_to_lane_bound(
                vehicle_p, intersecting_lanes, world_state
            )[1]
            return self._scale_lat_dist(np.fmin(rob_k, rob_p))
        else:
            lanes_p = world_state.road_network.find_lanes_by_lanelets(
                vehicle_p.lanelet_assignment[world_state.time_step]
            )
            k_occ = list(
                vehicle_k.occupancy_at_time_step(world_state.time_step).vertices
            )
            k_to_p_lanes = np.min(
                [
                    np.min(
                        np.abs(
                            list(
                                zip(
                                    *lane.clcs_right.convert_list_of_points_to_curvilinear_coords(
                                        k_occ, 1
                                    )
                                )
                            )[1]
                        )
                    )
                    for lane in lanes_p
                ]
                + [
                    np.min(
                        np.abs(
                            list(
                                zip(
                                    *lane.clcs_left.convert_list_of_points_to_curvilinear_coords(
                                        k_occ, 1
                                    )
                                )
                            )[1]
                        )
                    )
                    for lane in lanes_p
                ]
            )

            lanes_k = world_state.road_network.find_lanes_by_lanelets(
                vehicle_k.lanelet_assignment[world_state.time_step]
            )
            p_occ = list(
                vehicle_p.occupancy_at_time_step(world_state.time_step).vertices
            )
            p_to_k_lanes = np.min(
                [
                    np.min(
                        np.abs(
                            list(
                                zip(
                                    *lane.clcs_right.convert_list_of_points_to_curvilinear_coords(
                                        p_occ, 1
                                    )
                                )
                            )[1]
                        )
                    )
                    for lane in lanes_k
                ]
                + [
                    np.min(
                        np.abs(
                            list(
                                zip(
                                    *lane.clcs_left.convert_list_of_points_to_curvilinear_coords(
                                        p_occ, 1
                                    )
                                )
                            )[1]
                        )
                    )
                    for lane in lanes_k
                ]
            )
            return -self._scale_lat_dist(np.fmin(k_to_p_lanes, p_to_k_lanes))


class PredInFrontOf(IPredicateEvaluator):
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


class PredSingleLane(IPredicateEvaluator):
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
        # k_lanes = list(k_lanes)
        shape_k = vehicle_k.shapely_occupancy_at_time_step(world_state.time_step)
        overlap_areas = [
            lane.lanelet.convert_to_polygon().shapely_object.intersection(shape_k).area
            for lane in k_lanes
        ]
        assert len(overlap_areas) == len(
            k_lanes
        ), f"No intersection found for some lanes. Found {len(overlap_areas)} instead of {len(k_lanes)}"
        max_overlap_lane = k_lanes[np.argmax(overlap_areas)]
        intersecting_lanelets = max_overlap_lane.contained_lanelets.intersection(
            vehicle_k.lanelet_assignment[world_state.time_step]
        )
        left_adj, right_adj = get_adjacent_lanelets(
            intersecting_lanelets, world_state.road_network.lanelet_network
        )
        has_left_adj = len(left_adj) > 0
        has_right_adj = len(right_adj) > 0
        if not has_left_adj and not has_right_adj:
            # No adjacent lanes
            return self._scale_lat_dist(np.inf)
        occ = vehicle_k.occupancy_at_time_step(world_state.time_step)
        vert = list(occ.vertices)
        if has_left_adj:
            _, dist_left = zip(
                *max_overlap_lane.clcs_left.convert_list_of_points_to_curvilinear_coords(
                    vert, 1
                )
            )
            d_l = -np.max(dist_left)
        else:
            d_l = np.inf

        if has_right_adj:
            _, dist_right = zip(
                *max_overlap_lane.clcs_right.convert_list_of_points_to_curvilinear_coords(
                    vert, 1
                )
            )
            d_r = np.min(dist_right)
        else:
            d_r = np.inf
        return self._scale_lat_dist(np.fmin(d_l, d_r))


class PredCutIn(IPredicateEvaluator):
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

        result = (d_k < d_p and orient_k > 0) or (d_k > d_p and orient_k < 0)
        return result

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        cutting_vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world_state.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_robustness(
            world_state, [vehicle_ids[0]]
        )
        same_lane = self._same_lane_evaluator.evaluate_robustness(
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


class PredSafeDistPrec(IPredicateEvaluator):
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


class PredGenericSpeedLimit(IPredicateEvaluator):
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


class PredSucceeds(IPredicateEvaluator):
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
            same_lane = self.same_lane.evaluate_robustness(world_state, vehicle_ids)
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


class PredAbruptBreaking(IPredicateEvaluator):
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
        return rob


class PredRelAbruptBreaking(IPredicateEvaluator):
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
