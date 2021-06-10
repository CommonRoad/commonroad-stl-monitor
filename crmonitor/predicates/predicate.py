import abc
import logging
import math
from functools import reduce
from typing import List, Tuple, Set

import numpy as np
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from ruamel.yaml.comments import CommentedMap
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def scale_clip(x, min_val, max_val, new_min=0.0, new_max=1.0, copysign=False):
    abs_x = np.abs(x)
    if np.isinf(abs_x):
        rescaled = new_max
    else:
        rescaled = np.interp(abs_x, [min_val, max_val], [new_min, new_max])
    if copysign:
        rescaled = np.copysign(rescaled, x)
    return rescaled


def get_preceding_vehicles(
    world_state: WorldState, vehicle_rear: Vehicle
) -> List[Tuple[float, Vehicle]]:
    """
    Returns a list of preceding vehicles in ascending order of distance
    :param world_state: Current world state
    :param vehicle_rear: Reference vehicle
    :return: Sorted list of tuples of distance and vehicle object
    """
    veh = []
    rear_lanes = vehicle_rear.lanelet_assignment[world_state.time_step]
    for vehicle_lead in world_state.other_vehicles + [world_state.ego_vehicle]:
        if (
            not vehicle_lead.is_valid(world_state.time_step)
            or vehicle_lead is vehicle_rear
        ):
            continue
        lead_lanes = vehicle_lead.lanelet_assignment[world_state.time_step]
        intersecting_lanes = lead_lanes.intersection(rear_lanes)
        if len(intersecting_lanes) > 0:
            dist = vehicle_lead.rear_s(world_state.time_step) - vehicle_rear.front_s(
                world_state.time_step
            )
            if dist >= 0.0:
                veh.append((dist, vehicle_lead))
    return sorted(veh, key=lambda d: d[0])


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
        return self.evaluate_robustness(world_state, vehicle_ids) > 0.0

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
        lane_ids_k = world_state.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[world_state.time_step]
        )
        lane_ids_p = world_state.road_network.find_lanes_by_lanelets(
            vehicle_p.lanelet_assignment[world_state.time_step]
        )
        intersecting_lanes = lane_ids_p.intersection(lane_ids_k)
        intersecting_lanelets = reduce(
            lambda x, y: x.union(y),
            [l.contained_lanelets for l in intersecting_lanes],
            set(),
        )
        return intersecting_lanelets

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
            intersecting_lanes = self.get_same_lanes(world_state, vehicle_ids)
            assert len(intersecting_lanes) > 0
            lanelets = [
                world_state.road_network.lanelet_network.find_lanelet_by_id(l_id)
                for l_id in intersecting_lanes
            ]
            left_adj = [
                lanelet.adj_left
                for lanelet in lanelets
                if lanelet.adj_left is not None
                and lanelet.adj_left not in intersecting_lanes
            ]
            right_adj = [
                lanelet.adj_right
                for lanelet in lanelets
                if lanelet.adj_right is not None
                and lanelet.adj_right not in intersecting_lanes
            ]
            adj = left_adj + right_adj
            if len(adj) == 0:
                # No adjacent lanes
                return self._scale_lat_dist(np.inf)
            adj = [
                world_state.road_network.lanelet_network.find_lanelet_by_id(l_id)
                for l_id in adj
            ]
            occ = world_state.vehicle_by_id(vehicle_ids[0]).occupancy_at_time_step(
                world_state.time_step
            )
            shapley_occ = occ.shapely_object
            polys = [l.convert_to_polygon().shapely_object for l in adj]
            dist = [shapley_occ.distance(l) for l in polys]
            poly = polys[np.argmin(dist)]
            # Directed Hausdorff distance
            max_dist = np.max([poly.distance(Point(p)) for p in occ.vertices])
            return self._scale_lat_dist(max_dist)
        else:
            lanes_p = world_state.road_network.find_lanes_by_lanelets(
                vehicle_p.lanelet_assignment[world_state.time_step]
            )
            min_dist_k_to_p_lanes = math.inf
            k_occ = vehicle_k.occupancy_at_time_step(
                world_state.time_step
            ).shapely_object
            for lane_p in lanes_p:
                dist = k_occ.distance(
                    lane_p.lanelet.convert_to_polygon().shapely_object
                )
                min_dist_k_to_p_lanes = min(min_dist_k_to_p_lanes, dist)
            return -self._scale_lat_dist(min_dist_k_to_p_lanes)


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
        single_lane = False
        if len(k_lanes) == 1:  # single_lane
            single_lane = True
        return single_lane

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
        single_lane_boolean = self.evaluate_boolean(world_state, vehicle_ids)
        vehicle_k = world_state.vehicle_by_id(vehicle_ids[0])
        k_lanes = world_state.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[world_state.time_step]
        )
        assert (
            len(k_lanes) > 0
        ), f"Vehicle must be assigned to at least one lane! {str(world_state.scenario.scenario_id)}, id={vehicle_ids[0]}, t={world_state.time_step}, ego={world_state.ego_vehicle.id}"
        if single_lane_boolean:
            assert len(k_lanes) == 1
            k_lane = k_lanes.pop()
            k_occ = vehicle_k.occupancy_at_time_step(
                world_state.time_step
            ).shapely_object
            lane_poly = k_lane.lanelet.convert_to_polygon().shapely_object
            distance_to_boundary = lane_poly.boundary.distance(k_occ)
            return self._scale_lon_dist(distance_to_boundary)
        else:
            k_lanes = list(k_lanes)
            shape_k = vehicle_k.occupancy_at_time_step(
                world_state.time_step
            ).shapely_object
            overlap_areas = [
                lane.lanelet.convert_to_polygon()
                .shapely_object.intersection(shape_k)
                .area
                for lane in k_lanes
            ]
            assert len(overlap_areas) == len(
                k_lanes
            ), f"No intersection found for some lanes. Found {len(overlap_areas)} instead of {len(k_lanes)}"
            max_overlap_lane = (
                k_lanes[np.argmax(overlap_areas)]
                .lanelet.convert_to_polygon()
                .shapely_object
            )
            diff = shape_k.boundary.coords
            dist = [Point(*p).distance(max_overlap_lane) for p in diff]
            return -max(dist)


class PredCutIn(BasePredicateEvaluator):
    predicate_name = "cut_in"
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._single_lane_evaluator = PredSingleLane(config)

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        cutting_vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world_state.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_robustness_with_cache(
            world_state, [vehicle_ids[0]]
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
            return 22.22
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
            speed_limit = 130.0 / 3.6
        return speed_limit


class PredPrecedes(BasePredicateEvaluator):
    predicate_name = "precedes"
    arity = 2

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self.same_lane = PredInSameLane(config)
        self.same_lane.scale = False

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        ego_vehicle = world_state.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world_state.vehicle_by_id(vehicle_ids[1])
        succ_veh = get_succeeding_vehicles(world_state, other_vehicle)
        if len(succ_veh) > 0 and succ_veh[0][1].id == vehicle_ids[0]:
            same_lane = self.same_lane.evaluate_robustness_with_cache(world_state, vehicle_ids)
            assert same_lane >= 0.0
            overtake = other_vehicle.rear_s(world_state.time_step) - ego_vehicle.front_s(world_state.time_step)
            assert overtake >= 0.0
            if len(succ_veh) >= 2:
                fallback = ego_vehicle.front_s(world_state.time_step) - succ_veh[1][1].front_s(world_state.time_step)
                assert fallback >= 0.0
            else:
                fallback = math.inf
            return min(same_lane, overtake, fallback)
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


class PredAcceleration(BasePredicateEvaluator):
    predicate_name = "accel"
    arity = 1

    def evaluate_boolean(self, world_state: WorldState, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world_state, vehicle_ids) > -2.

    def evaluate_robustness(
        self, world_state: WorldState, vehicle_ids: List[int]
    ) -> float:
        accel = (world_state.vehicle_by_id(vehicle_ids[0]).states_lon[world_state.time_step].a)
        return accel
