import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import shapely.ops
from commonroad.scenario.lanelet import LaneletType, LineMarking
from shapely.geometry.polygon import Polygon
from typing_extensions import override

from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.errors import PredicateEvaluationError
from crmonitor.predicates import utils
from crmonitor.predicates.base import (
    AbstractPredicate,
    PredicateConfig,
    PredicateName,
)
from crmonitor.predicates.utils import (
    distance_to_bounds,
    distance_to_lanes,
    distance_to_left_bounds_clcs,
    distance_to_right_bounds_clcs,
    lanelets_left_of_vehicle,
    lanelets_right_of_vehicle,
    vehicle_directly_left,
    vehicle_directly_right,
)

_LOGGER = logging.getLogger(__name__)


class PositionPredicates(PredicateName):
    InSameLane = "in_same_lane"
    InFrontOf = "in_front_of"
    SingleLane = "single_lane"
    KeepsSafeDistancePrec = "keeps_safe_distance_prec"
    Precedes = "precedes"
    # newly added besides the ones for R_G1-R_G3
    RightOfBroadLaneMarking = "right_of_broad_lane_marking"
    LeftOfBroadLaneMarking = "left_of_broad_lane_marking"
    OnAccessRamp = "on_access_ramp"
    OnShoulder = "on_shoulder"
    OnMainCarriageway = "on_main_carriage_way"
    OnExitRamp = "on_exit_ramp"  # not used
    InRightmostLane = "in_rightmost_lane"
    InLeftmostLane = "in_leftmost_lane"
    CloseToLeftBound = "close_to_left_bound"
    CloseToRightBound = "close_to_right_bound"
    CloseToVehicleLeft = "close_to_vehicle_left"
    CloseToVehicleRight = "close_to_vehicle_right"
    MainCarriagewayRightLane = "main_carriageway_right_lane"
    LeftOf = "left_of"
    DrivesLeftmost = "drives_leftmost"
    DrivesRightmost = "drives_rightmost"
    OnLaneletWithTypeIntersection = "on_lanelet_with_type_intersection"
    OnIncomingLeftOf = "on_incoming_left_of"
    OnOncomOf = "on_oncom_of"
    InIntersectionConflictArea = "in_intersection_conflict_area"
    StopLineInFront = "stop_line_in_front"
    LatLeftOf = "lat_left_of"
    HeadingRight = "heading_right"
    LatLeftOfVehicle = "lat_left_of_vehicle"
    RearBehindFront = "rear_behind_front"
    LatCloseToVehicleLeft = "lat_close_to_vehicle_left"
    LatCloseToVehicleRight = "lat_close_to_vehicle_right"


class PredInSameLane(AbstractPredicate):
    predicate_name = PositionPredicates.InSameLane
    arity = 2

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        lane_ids_k = vehicle_k.lane_ids_at_time_step(time_step)
        lane_ids_p = vehicle_p.lane_ids_at_time_step(time_step)
        intersecting_lanes = lane_ids_p.intersection(lane_ids_k)
        return len(intersecting_lanes) > 0

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        """
        If boolean is
        True: Minimum lateral displacement to not be in the same lane anymore
        False: Minimum distance to lanes of other
        """

        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        lanelet_ids_k = vehicle_k.lanelet_ids_at_time_step(time_step)
        lanelet_ids_p = vehicle_p.lanelet_ids_at_time_step(time_step)

        if len(lanelet_ids_k) == 0 or len(lanelet_ids_p) == 0:
            # Vehicles outside the road network are currently not supported.
            raise PredicateEvaluationError(
                self.predicate_name,
                world,
                time_step,
                vehicle_ids,
                "Vehicles are outside the road network",
            )

        rob = np.fmin(
            distance_to_lanes(vehicle_k, lanelet_ids_p, world, time_step),
            distance_to_lanes(vehicle_p, lanelet_ids_k, world, time_step),
        )
        return self._scale_lat_dist(rob)


class PredInFrontOf(AbstractPredicate):
    predicate_name = PositionPredicates.InFrontOf
    arity = 2

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        rear = world.vehicle_by_id(vehicle_ids[0])
        front = world.vehicle_by_id(vehicle_ids[1])
        ref_lane = rear.lane_at_time_step(time_step)
        return self._scale_lon_dist(
            front.rear_s(time_step, ref_lane) - rear.front_s(time_step, ref_lane)
        )


class PredSingleLane(AbstractPredicate):
    predicate_name = PositionPredicates.SingleLane
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        k_lane_ids = vehicle_k.lanelet_ids_at_time_step(time_step)
        return len(k_lane_ids) == 1

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        """
        If false: 1 - the largest fractional overlap with occupied lanes
        If true: Distance to lane polygon boundary
        """
        vehicle = world.vehicle_by_id(vehicle_ids[0])

        ref_lane = vehicle.lane_at_time_step(time_step)
        if ref_lane is None:
            raise PredicateEvaluationError(
                self.predicate_name,
                world,
                time_step,
                vehicle_ids,
                "Vehicle is outside of road network",
            )

        d_left, d_right = distance_to_bounds(vehicle, ref_lane.contained_lanelets, world, time_step)
        d_left = -np.max(d_left) if d_left.size > 0 else np.inf
        d_right = np.min(d_right) if d_right.size > 0 else np.inf
        rob = np.fmin(d_left, d_right)

        return self._scale_lat_dist(rob)


class PredSafeDistPrec(AbstractPredicate):
    predicate_name = PositionPredicates.KeepsSafeDistancePrec
    arity = 2

    @classmethod
    def calculate_safe_distance(cls, v_follow, v_lead, a_min_lead, a_min_follow, t_react_follow):
        d_safe = (
            (v_lead**2) / (-2 * np.abs(a_min_lead))
            - (v_follow**2) / (-2 * np.abs(a_min_follow))
            + v_follow * t_react_follow
        )

        return d_safe

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle_follow = world.vehicle_by_id(vehicle_ids[0])
        vehicle_lead = world.vehicle_by_id(vehicle_ids[1])

        if vehicle_lead.lane_at_time_step(time_step) is None:
            return self._scale_lon_dist(math.inf)
        a_min_follow = vehicle_follow.vehicle_param.a_min
        a_min_lead = vehicle_lead.vehicle_param.a_min
        t_react_follow = vehicle_follow.vehicle_param.t_react
        ref_lane = vehicle_follow.lane_at_time_step(time_step)

        safe_distance = self.calculate_safe_distance(
            vehicle_follow.get_lon_state(time_step, ref_lane).v,
            vehicle_lead.get_lon_state(time_step, ref_lane).v,
            a_min_lead,
            a_min_follow,
            t_react_follow,
        )

        delta_s = vehicle_lead.rear_s(time_step, ref_lane) - vehicle_follow.front_s(
            time_step, ref_lane
        )
        rob = self._scale_lon_dist(delta_s - safe_distance)
        return rob

    @staticmethod
    def _plot_red_arrow(ax, x, y, size=1.0):
        ax.plot(x, y, linewidth=2, color="r", zorder=25)
        ax.arrow(
            x[-2],
            y[-2],
            x[-1] - x[-2],
            y[-1] - y[-2],
            lw=0,
            length_includes_head=True,
            head_width=size,
            head_length=size,
            zorder=25,
            color="r",
        )

    def visualize_unsafe_region(self, ax, time_step: int, unsafe_s: float, vehicle_lead: Vehicle):
        """
        Plots the unsafe region starting from the rear of the front vehicle
        """
        # the ids of lanes are increasing together with the d-coordinate
        vehicle_lanes = list(vehicle_lead.lanes_at_time_step(time_step))
        # the upper the lane in the road network is, the smaller the index in the list as
        if (
            vehicle_lanes[0].lanelet.center_vertices[0][1]
            < vehicle_lanes[-1].lanelet.center_vertices[0][1]
        ):
            vehicle_lanes = vehicle_lanes[::-1]
        reference_lane = vehicle_lead.lane_at_time_step(time_step)
        # get the Cartesian coordinate of the safe distance
        safe_pos_cart = reference_lane.convert_to_cartesian_coords(unsafe_s, 0)
        lead_rear_cart = reference_lane.convert_to_cartesian_coords(
            vehicle_lead.rear_s(time_step), 0.0
        )
        # left vertices
        front_rear_left_cart = vehicle_lanes[0].clcs_left.convert_to_cartesian_coords(
            vehicle_lead.rear_s(time_step), 0.0
        )
        safe_pos_left_cart = vehicle_lanes[0].clcs_left.convert_to_cartesian_coords(unsafe_s, 0)
        reference_left = np.vstack(vehicle_lanes[0].clcs_left.clcs.reference_path())
        vertices_left = reference_left[
            (reference_left[:, 0] > safe_pos_left_cart[0])
            & (reference_left[:, 0] < front_rear_left_cart[0]),
            :,
        ]
        vertices_left = np.concatenate(
            ([safe_pos_left_cart], vertices_left, [front_rear_left_cart])
        )
        # right vertices
        lead_rear_right_cart = vehicle_lanes[-1].clcs_right.convert_to_cartesian_coords(
            vehicle_lead.rear_s(time_step), 0.0
        )
        safe_pos_right_cart = vehicle_lanes[-1].clcs_right.convert_to_cartesian_coords(unsafe_s, 0)
        reference_right = np.vstack(vehicle_lanes[-1].clcs_right.clcs.reference_path())
        vertices_right = reference_right[
            (reference_right[:, 0] > safe_pos_left_cart[0])
            & (reference_right[:, 0] < front_rear_left_cart[0]),
            :,
        ]
        vertices_right = np.concatenate(
            ([safe_pos_right_cart], vertices_right, [lead_rear_right_cart])
        )
        # concatenate vertices
        vertices_total = list(
            np.concatenate(
                (
                    [safe_pos_cart],
                    vertices_left,
                    [lead_rear_cart],
                    vertices_right,
                    [safe_pos_cart],
                )
            )
        )
        # compute centroid
        cent = (
            sum([v[0] for v in vertices_total]) / len(vertices_total),
            sum([v[1] for v in vertices_total]) / len(vertices_total),
        )
        # sort by polar angle
        vertices_total.sort(key=lambda v: math.atan2(v[1] - cent[1], v[0] - cent[0]))

        unsafe_region = Polygon(vertices_total)
        ax.fill(
            *unsafe_region.exterior.xy,
            zorder=30,
            alpha=0.2,
            facecolor="red",
            edgecolor=None,
        )

    def visualize(
        self,
        vehicle_ids: List[int],
        add_vehicle_draw_params: Callable[[int, any], None],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ):
        self._gather_predicate_values_to_plot(
            vehicle_ids, world, time_step, predicate_names2vehicle_ids2values
        )
        latest_value = self.evaluate_robustness(world, time_step, vehicle_ids)
        latest_value_unscaled = (
            latest_value * self._scaler._scale_constants.MAX_LONG_DIST
        )  # un-scale to actual range and make positive
        vehicle_follow = world.vehicle_by_id(vehicle_ids[0])

        lane_clcs = vehicle_follow.lane_at_time_step(
            time_step
        ).clcs  # center curvilinear coordinate system
        sampling_step_size = 1.0

        s_start = vehicle_follow.front_s(time_step)
        num_points = max(2, abs(int(latest_value_unscaled / sampling_step_size)))
        points_s = np.linspace(0, latest_value_unscaled, num_points) + s_start
        points_s = points_s[:, None]
        points_l = np.zeros((points_s.shape[0], 1))
        points_curvi = np.concatenate((points_s, points_l), axis=1)
        points_cartesian = np.stack(
            [lane_clcs.convert_to_cartesian_coords(*p) for p in points_curvi], axis=0
        )

        def fun(renderer):
            self._plot_red_arrow(renderer.ax, points_cartesian[:, 0], points_cartesian[:, 1])
            unsafe_s = latest_value_unscaled + s_start
            self.visualize_unsafe_region(
                renderer.ax, time_step, unsafe_s, world.vehicle_by_id(vehicle_ids[1])
            )

        return (fun,)

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((0, 1))
        ax.plot(0, 0.5, color="r")
        PredSafeDistPrec._plot_red_arrow(ax, [0, 1], [0.5, 0.5], size=0.1)
        PredSafeDistPrec._plot_red_arrow(ax, [0, -1], [0.5, 0.5], size=0.1)


class PredPreceding(AbstractPredicate):
    predicate_name = PositionPredicates.Precedes
    arity = 2

    def __init__(self, config: PredicateConfig | None = None):
        super().__init__(config)
        self.same_lane = PredInSameLane(config)

    @staticmethod
    def _get_candidates(
        world: World, time_step, vehicle_rear: Vehicle
    ) -> List[Tuple[float, Vehicle, Lane, bool]]:
        """
        Returns a list of vehicles in ascending order of distance
        :param time_step:
        :param world: Current world state
        :param vehicle_rear: Reference vehicle
        :return: Sorted list of tuples of distance and vehicle object
        """
        veh = []
        rear_lanes = vehicle_rear.lanes_at_time_step(time_step)
        for vehicle_front in world.vehicles:
            if not vehicle_front.is_valid(time_step) or vehicle_front is vehicle_rear:
                continue
            front_lanes = vehicle_front.lanes_at_time_step(time_step)
            intersecting_lanes = rear_lanes.intersection(front_lanes)
            same_lane = len(intersecting_lanes) > 0
            lane = (
                list(intersecting_lanes)[0]
                if len(intersecting_lanes) > 0
                else vehicle_rear.lane_at_time_step(time_step)
            )
            dist = vehicle_front.rear_s(time_step, lane) - vehicle_rear.front_s(time_step, lane)
            veh.append((dist, vehicle_front, lane, same_lane))
        return sorted(veh, key=lambda d: d[0])

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        rear_vehicle_id = vehicle_ids[0]
        front_vehicle_id = vehicle_ids[1]
        rear_vehicle = world.vehicle_by_id(rear_vehicle_id)
        candidates = self._get_candidates(world, time_step, rear_vehicle)
        pred_veh = [elem for elem in candidates if elem[0] >= 0.0 and elem[3]]
        return len(pred_veh) > 0 and pred_veh[0][1].id == front_vehicle_id

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        rear_veh = world.vehicle_by_id(vehicle_ids[0])
        front_veh = world.vehicle_by_id(vehicle_ids[1])
        veh_lon_dist = self._get_candidates(world, time_step, rear_veh)
        veh_front_dist = [_ for _ in veh_lon_dist if _[0] >= 0 and _[3]]
        bool_val = len(veh_front_dist) > 0 and veh_front_dist[0][1].id == vehicle_ids[1]
        same_lane = self.same_lane.evaluate_robustness(world, time_step, vehicle_ids)
        if bool_val:
            assert same_lane >= -self.config.eps
            same_lane = max(same_lane, 0.0)

        for dist, veh, _, __ in veh_lon_dist:
            if veh == front_veh:
                dist_front = dist
                break
        else:
            # Should never happen
            assert False

        pred_wo_other = [v for v in veh_front_dist if v[1] is not front_veh]
        if len(pred_wo_other) > 0:
            _, pred_wo_other, lane, __ = pred_wo_other[0]
            dist_pred = pred_wo_other.rear_s(time_step, lane) - front_veh.rear_s(time_step)
        else:
            dist_pred = math.inf

        rob = min(
            same_lane,
            self._scale_lon_dist(dist_front),
            self._scale_lon_dist(dist_pred),
        )
        return rob


class PredRightOfBroadLaneMarking(AbstractPredicate):
    predicate_name = PositionPredicates.RightOfBroadLaneMarking
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                return False

        lanelets_left_of_veh = lanelets_left_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_left_of_veh:
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                return True
        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                _, d_right = distance_to_bounds(vehicle, [l_id], world, time_step)
                d_right = np.min(d_right) if d_right.size > 0 else np.inf
                # the abs function is to prevent the distance to be positive but the robustness is violated
                return self._scale_lat_dist(-abs(d_right))
        lanelets_left_of_veh = lanelets_left_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_left_of_veh:
            if (
                lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID
            ):
                d_left, _ = distance_to_bounds(vehicle, [lanelet.lanelet_id], world, time_step)
                d_left = -np.max(d_left) if d_left.size > 0 else np.inf
                return self._scale_lat_dist(d_left)
        return self._scale_lat_dist(-np.inf)


class PredLeftOfBroadLaneMarking(AbstractPredicate):
    predicate_name = PositionPredicates.LeftOfBroadLaneMarking
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                return False

        lanelets_right_of_veh = lanelets_right_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_right_of_veh:
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                return True
        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                d_left, _ = distance_to_bounds(vehicle, [l_id], world, time_step)
                d_left = -np.max(d_left) if d_left.size > 0 else np.inf
                # the abs function is to prevent the distance to be positive but the robustness is violated
                return self._scale_lat_dist(-abs(d_left))
        lanelets_right_of_veh = lanelets_right_of_vehicle(
            time_step, vehicle, world.road_network.lanelet_network
        )
        for lanelet in lanelets_right_of_veh:
            if (
                lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED
                or lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID
            ):
                _, d_right = distance_to_bounds(vehicle, [lanelet.lanelet_id], world, time_step)
                d_right = np.min(d_right) if d_right.size > 0 else np.inf
                return self._scale_lat_dist(d_right)
        return self._scale_lat_dist(-np.inf)


class _AbstractPredOnLaneletType(AbstractPredicate, ABC):
    @abstractmethod
    def get_lanelet_type(self) -> LaneletType: ...

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        if vehicle is None:
            raise ValueError(
                f"Failed to evaluate predicate {self.predicate_name} at time step {time_step}: Vehicle {vehicle_ids[0]} could not be found in world"
            )

        lanelet_type = self.get_lanelet_type()

        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet_type in lanelet.lanelet_type:
                return True

        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])

        lanelet_type = self.get_lanelet_type()

        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        matching_lanelet_ids = [
            l_id
            for l_id in lanelet_ids_occ
            if lanelet_type
            in world.road_network.lanelet_network.find_lanelet_by_id(l_id).lanelet_type
        ]
        if len(matching_lanelet_ids) > 0:
            return self._scale_lat_dist(
                distance_to_lanes(vehicle, matching_lanelet_ids, world, time_step)
            )
        else:
            return self._scale_lat_dist(-np.inf)


class PredOnAccessRamp(_AbstractPredOnLaneletType):
    """
    Evaluates if a vehicle is on an access ramp.
    """

    predicate_name = PositionPredicates.OnAccessRamp
    arity = 1

    @override
    def get_lanelet_type(self) -> LaneletType:
        return LaneletType.ACCESS_RAMP


class PredOnShoulder(_AbstractPredOnLaneletType):
    """
    Evaluates if a vehicle is on a shoulder lane.
    """

    predicate_name = PositionPredicates.OnShoulder
    arity = 1

    @override
    def get_lanelet_type(self) -> LaneletType:
        return LaneletType.SHOULDER


class PredOnMainCarriageway(_AbstractPredOnLaneletType):
    """
    Evaluates if a vehicle is on a main carriage way.
    """

    predicate_name = PositionPredicates.OnMainCarriageway
    arity = 1

    @override
    def get_lanelet_type(self) -> LaneletType:
        return LaneletType.MAIN_CARRIAGE_WAY


class PredInRightmostLane(AbstractPredicate):
    """
    check if any assigned lanelet of ego vehicle is in rightmost lane
    """

    predicate_name = PositionPredicates.InRightmostLane
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])

        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if (
                LaneletType.SHOULDER
                not in lanelet.lanelet_type  # Shoulder is not considered rightmost lane
                and (
                    lanelet.adj_right is None
                    or lanelet.adj_right_same_direction is False
                    or (
                        LaneletType.SHOULDER
                        in world.road_network.lanelet_network.find_lanelet_by_id(
                            lanelet.adj_right
                        ).lanelet_type
                    )
                )
            ):
                return True
        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        rightmost_lanelet_ids = [
            lanelet.lanelet_id
            for lanelet in world.road_network.lanelet_network.lanelets
            if LaneletType.SHOULDER
            not in lanelet.lanelet_type  # Shoulder is not considered rightmost lane
            and (
                lanelet.adj_right is None
                or lanelet.adj_right_same_direction is False
                or (
                    LaneletType.SHOULDER
                    in world.road_network.lanelet_network.find_lanelet_by_id(
                        lanelet.adj_right
                    ).lanelet_type
                )
            )
        ]

        if len(rightmost_lanelet_ids) == 0:
            return self._scaler.scale_lat_dist(-np.inf)

        dis_to_lane = distance_to_lanes(vehicle, rightmost_lanelet_ids, world, time_step)
        return self._scale_lat_dist(dis_to_lane)


class PredInLeftmostLane(AbstractPredicate):
    """
    check if any assigned lanelet of ego vehicle is in leftmost lane
    """

    predicate_name = PositionPredicates.InLeftmostLane
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])

        lanelet_ids_occ = vehicle.lanelet_ids_at_time_step(time_step)
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.adj_left_same_direction is None:
                return True
        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        leftmost_lanelet_ids = [
            lanelet.lanelet_id
            for lanelet in world.road_network.lanelet_network.lanelets
            if lanelet.adj_left_same_direction is None
        ]

        if len(leftmost_lanelet_ids) == 0:
            return self._scaler.scale_lat_dist(-np.inf)

        dis_to_lane = distance_to_lanes(vehicle, leftmost_lanelet_ids, world, time_step)
        return self._scaler.scale_lat_dist(dis_to_lane)


class PredMainCarriageWayRightLane(AbstractPredicate):
    """
    Evaluates if a vehicle occupies the rightmost main carriageway lane.
    """

    predicate_name = PositionPredicates.MainCarriagewayRightLane
    arity = 1

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type and (
                not lanelet.adj_right_same_direction
                or LaneletType.MAIN_CARRIAGE_WAY
                not in world.road_network.lanelet_network.find_lanelet_by_id(
                    lanelet.adj_right
                ).lanelet_type
            ):
                return True
        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        main_carriageway_right_lanelet_ids = [
            lanelet.lanelet_id
            for lanelet in world.road_network.lanelet_network.lanelets
            if LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type
            and (
                not lanelet.adj_right_same_direction
                or LaneletType.MAIN_CARRIAGE_WAY
                not in world.road_network.lanelet_network.find_lanelet_by_id(
                    lanelet.adj_right
                ).lanelet_type
            )
        ]
        if len(main_carriageway_right_lanelet_ids) == 0:
            return self._scaler.scale_lat_dist(-np.inf)

        dis_to_lane = distance_to_lanes(
            vehicle, main_carriageway_right_lanelet_ids, world, time_step
        )
        return self._scaler.scale_lat_dist(dis_to_lane)


class PredLeftOf(AbstractPredicate):
    predicate_name = PositionPredicates.LeftOf
    arity = 2

    @override
    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        """
        Evaluates if the kth vehicle is left of the pth vehicle
        """
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        # share the same lane as the reference, otherwise the comparison does not make sense
        lane_share = list(
            world.road_network.find_lanes_by_lanelets(vehicle_k.lanelet_assignment[time_step])
        )[0]
        if not vehicle_p.left_d(time_step, lane_share) < vehicle_k.right_d(time_step, lane_share):
            return False
        else:
            if (
                vehicle_p.rear_s(time_step, lane_share)
                <= vehicle_k.front_s(time_step, lane_share)
                <= vehicle_p.front_s(time_step, lane_share)
            ):
                return True
            if (
                vehicle_p.rear_s(time_step, lane_share)
                <= vehicle_k.rear_s(time_step, lane_share)
                <= vehicle_p.front_s(time_step, lane_share)
            ):
                return True
            if vehicle_k.rear_s(time_step, lane_share) < vehicle_p.rear_s(
                time_step, lane_share
            ) and vehicle_p.front_s(time_step, lane_share) < vehicle_k.front_s(
                time_step, lane_share
            ):
                return True
            else:
                return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        # share the same lane as the reference, otherwise the comparison does not make sense
        lane_share = list(
            world.road_network.find_lanes_by_lanelets(vehicle_k.lanelet_assignment[time_step])
        )[0]
        left_p = vehicle_p.left_d(time_step, lane_share)
        right_k = vehicle_k.right_d(time_step, lane_share)
        rear_p = vehicle_p.rear_s(time_step, lane_share)
        rear_k = vehicle_k.rear_s(time_step, lane_share)
        front_p = vehicle_p.front_s(time_step, lane_share)
        front_k = vehicle_k.front_s(time_step, lane_share)

        #  pred = (left_p < right_k) and (
        #         (rear_p <= rear_k and rear_k <= front_p) or (front_p < front_k and rear_k < rear_p) or (
        #         rear_p <= front_k and front_k <= front_p))

        left_p_less_than_right_k = self._scale_lat_dist(right_k - left_p)
        rear_p_less_than_rear_k = self._scale_lon_dist(rear_k - rear_p)
        rear_k_less_than_front_p = self._scale_lon_dist(front_p - rear_k)
        front_p_less_than_front_k = self._scale_lon_dist(front_k - front_p)
        rear_k_less_than_rear_p = self._scale_lon_dist(rear_p - rear_k)
        rear_p_less_than_front_k = self._scale_lon_dist(front_k - rear_p)
        front_k_less_than_front_p = self._scale_lon_dist(front_p - front_k)

        rob = min(
            left_p_less_than_right_k,
            max(
                min(rear_p_less_than_rear_k, rear_k_less_than_front_p),
                min(front_p_less_than_front_k, rear_k_less_than_rear_p),
                min(rear_p_less_than_front_k, front_k_less_than_front_p),
            ),
        )
        return rob


class PredDrivesLeftmost(AbstractPredicate):
    """
    Evaluates if a vehicle drives leftmost within its occupied lanes.
    """

    predicate_name = PositionPredicates.DrivesLeftmost
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        veh_dir_l = vehicle_directly_left(time_step, vehicle, other_vehicles)
        if veh_dir_l is not None:
            share_lane = vehicle.lane_at_time_step(time_step)
            if (
                veh_dir_l.right_d(time_step, share_lane) - vehicle.left_d(time_step, share_lane)
                < self.config.close_to_other_vehicle
            ):
                return True

        lanes = world.road_network.find_lanes_by_lanelets(lanelet_ids_occ)
        for lane in lanes:
            left_position = vehicle.left_d(time_step, lane)
            s_ego = vehicle.get_lon_state(time_step, lane).s
            if 0.5 * lane.width(s_ego) - left_position < self.config.close_to_lane_border:
                return True

        return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        lanes = world.road_network.find_lanes_by_lanelets(lanelet_ids_occ)

        lane_bound_dist = np.inf
        for lane in lanes:
            dists_to_left_bounds = map(abs, distance_to_left_bounds_clcs(vehicle, lane, time_step))
            lane_bound_dist = min(min(dists_to_left_bounds), lane_bound_dist)
        lane_bound_dist = self.config.close_to_lane_border - lane_bound_dist

        veh_dir_l = vehicle_directly_left(time_step, vehicle, other_vehicles)
        veh_dir_l_dist = -np.inf
        if veh_dir_l is not None:
            share_lane = vehicle.lane_at_time_step(time_step)
            veh_dir_l_dist = self.config.close_to_other_vehicle - abs(
                veh_dir_l.right_d(time_step, share_lane) + vehicle.left_d(time_step, share_lane)
            )

        return self._scale_lat_dist(max(veh_dir_l_dist, lane_bound_dist))


class PredDrivesRightmost(AbstractPredicate):
    """
    Evaluates if a vehicle drives rightmost within its occupied lanes.
    """

    predicate_name = PositionPredicates.DrivesRightmost
    arity = 1

    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != ego_vehicle.id
        ]
        veh_dir_r = vehicle_directly_right(time_step, ego_vehicle, other_vehicles)
        if veh_dir_r is not None:
            share_lane = ego_vehicle.lane_at_time_step(time_step)
            if (
                -veh_dir_r.left_d(time_step, share_lane)
                + ego_vehicle.right_d(time_step, share_lane)
                < self.config.close_to_other_vehicle
            ):
                return True

        lanes = ego_vehicle.lanes_at_time_step(time_step)
        for lane in lanes:
            right_position = ego_vehicle.right_d(time_step, lane)
            s_ego = ego_vehicle.get_lon_state(time_step, lane).s
            if 0.5 * lane.width(s_ego) + right_position < self.config.close_to_lane_border:
                return True

        return False

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicles = [
            world.vehicle_by_id(v_id)
            for v_id in world.vehicle_ids_for_time_step(time_step)
            if v_id != vehicle.id
        ]
        lanes = vehicle.lanes_at_time_step(time_step)
        lane_bound_dist = np.inf
        for lane in lanes:
            dists_to_right_bounds = map(
                abs, distance_to_right_bounds_clcs(vehicle, lane, time_step)
            )

            lane_bound_dist = min(min(dists_to_right_bounds), lane_bound_dist)
        lane_bound_dist = self.config.close_to_lane_border - lane_bound_dist

        veh_dir_r = vehicle_directly_right(time_step, vehicle, other_vehicles)
        veh_dir_r_dist = -np.inf
        if veh_dir_r is not None:
            share_lane = vehicle.lane_at_time_step(time_step)
            veh_dir_r_dist = self.config.close_to_other_vehicle - abs(
                veh_dir_r.left_d(time_step, share_lane) - vehicle.right_d(time_step, share_lane)
            )
        return self._scale_lat_dist(max(veh_dir_r_dist, lane_bound_dist))


class PredCloseToLeftBound(AbstractPredicate):
    predicate_name = PositionPredicates.CloseToLeftBound
    arity = 1

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        assert ego_vehicle is not None
        lanes = ego_vehicle.lanes_at_time_step(time_step)

        dist = np.inf
        for lane in lanes:
            dists_to_left_bounds = map(
                abs, distance_to_left_bounds_clcs(ego_vehicle, lane, time_step)
            )
            dist = min(min(dists_to_left_bounds), dist)
        return self._scale_lat_dist(self.config.close_to_lane_border - dist)


class PredCloseToRightBound(AbstractPredicate):
    predicate_name = PositionPredicates.CloseToRightBound
    arity = 1

    @override
    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        assert ego_vehicle is not None

        lanes = ego_vehicle.lanes_at_time_step(time_step)

        dist = np.inf
        for lane in lanes:
            dists_to_right_bounds = map(
                abs, distance_to_right_bounds_clcs(ego_vehicle, lane, time_step)
            )
            dist = min(min(dists_to_right_bounds), dist)
        return self._scale_lat_dist(self.config.close_to_lane_border - dist)


class PredCloseToVehicleLeft(AbstractPredicate):
    predicate_name = PositionPredicates.CloseToVehicleLeft
    arity = 2

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world.vehicle_by_id(vehicle_ids[1])
        share_lane = ego_vehicle.lane_at_time_step(time_step)
        lat_dist = self._scale_lat_dist(
            self.config.close_to_other_vehicle
            - abs(
                other_vehicle.right_d(time_step, share_lane)
                - ego_vehicle.left_d(time_step, share_lane)
            )
        )
        lon_dist = self._scale_lon_dist(
            ((ego_vehicle.shape.length + other_vehicle.shape.length) / 2)
            - abs(
                other_vehicle.get_lon_state(time_step, share_lane).s
                - ego_vehicle.get_lon_state(time_step, share_lane).s
            )
        )
        return min(lat_dist, lon_dist)


class PredCloseToVehicleRight(AbstractPredicate):
    predicate_name = PositionPredicates.CloseToVehicleRight
    arity = 2

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world.vehicle_by_id(vehicle_ids[1])
        share_lane = ego_vehicle.lane_at_time_step(time_step)
        lat_dist = self._scale_lat_dist(
            self.config.close_to_other_vehicle
            - abs(
                other_vehicle.left_d(time_step, share_lane)
                - ego_vehicle.right_d(time_step, share_lane)
            )
        )
        lon_dist = self._scale_lon_dist(
            ((ego_vehicle.shape.length + other_vehicle.shape.length) / 2)
            - abs(
                other_vehicle.get_lon_state(time_step, share_lane).s
                - ego_vehicle.get_lon_state(time_step, share_lane).s
            )
        )
        return min(lat_dist, lon_dist)


class PredLatLeftOf(AbstractPredicate):
    predicate_name = PositionPredicates.LatLeftOf
    arity = 2

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world.vehicle_by_id(vehicle_ids[1])
        share_lane = ego_vehicle.lane_at_time_step(time_step)

        return self._scale_lat_dist(
            ego_vehicle.get_lat_state(time_step, share_lane).d
            - other_vehicle.get_lat_state(time_step, share_lane).d
        )


class PredHeadingRight(AbstractPredicate):
    predicate_name = PositionPredicates.HeadingRight
    arity = 1

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        lane = ego_vehicle.lane_at_time_step(time_step)

        return self._scale_angle(-ego_vehicle.get_lat_state(time_step, lane).theta)


class PredLatLeftOfVehicle(AbstractPredicate):
    predicate_name = PositionPredicates.LatLeftOfVehicle
    arity = 2

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world.vehicle_by_id(vehicle_ids[1])
        share_lane = ego_vehicle.lane_at_time_step(time_step)

        return self._scale_lat_dist(
            ego_vehicle.right_d(time_step, share_lane) - other_vehicle.left_d(time_step, share_lane)
        )


class PredRearBehindFront(AbstractPredicate):
    predicate_name = PositionPredicates.RearBehindFront
    arity = 2

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world.vehicle_by_id(vehicle_ids[1])
        share_lane = ego_vehicle.lane_at_time_step(time_step)

        return self._scale_lon_dist(
            other_vehicle.front_s(time_step, share_lane) - ego_vehicle.rear_s(time_step, share_lane)
        )


class PredLatCloseToVehicleLeft(AbstractPredicate):
    predicate_name = PositionPredicates.LatCloseToVehicleLeft
    arity = 2

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world.vehicle_by_id(vehicle_ids[1])
        share_lane = ego_vehicle.lane_at_time_step(time_step)
        lat_dist = self._scale_lat_dist(
            self.config.close_to_other_vehicle
            - abs(
                other_vehicle.right_d(time_step, share_lane)
                - ego_vehicle.left_d(time_step, share_lane)
            )
        )
        return lat_dist


class PredLatCloseToVehicleRight(AbstractPredicate):
    predicate_name = PositionPredicates.LatCloseToVehicleRight
    arity = 2

    def evaluate_robustness(self, world: World, time_step: int, vehicle_ids: List[int]) -> float:
        ego_vehicle = world.vehicle_by_id(vehicle_ids[0])
        other_vehicle = world.vehicle_by_id(vehicle_ids[1])
        share_lane = ego_vehicle.lane_at_time_step(time_step)
        lat_dist = self._scale_lat_dist(
            self.config.close_to_other_vehicle
            - abs(
                other_vehicle.left_d(time_step, share_lane)
                - ego_vehicle.right_d(time_step, share_lane)
            )
        )
        return lat_dist


##################
#  intersection  #
##################
class PredStopLineInFront(AbstractPredicate):
    predicate_name = PositionPredicates.StopLineInFront
    arity = 1

    def __init__(self, config: PredicateConfig | None = None):
        super().__init__(config)
        self._dict_veh_id_stop_line_s = defaultdict(lambda: None)
        self._dict_veh_id_intersection_lanelets = defaultdict(lambda: None)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        """
        A stop line is in front of a vehicle if any occupied lanelet references a stop line within a distance d_sl
        """
        if self._dict_veh_id_stop_line_s[vehicle_ids[0]]:
            vehicle = world.vehicle_by_id(vehicle_ids[0])
            stop_line_s = self._dict_veh_id_stop_line_s[vehicle_ids[0]]
            intersection_lanelets = self._dict_veh_id_intersection_lanelets[vehicle_ids[0]]
            if not intersection_lanelets:
                return False
        else:
            vehicle = world.vehicle_by_id(vehicle_ids[0])
            # Find all lanelets in the map that have a stop line
            lanelets_with_stop_line = [
                lanelet.lanelet_id
                for lanelet in world.road_network.lanelet_network.lanelets
                if lanelet.stop_line is not None
            ]
            # Get the set of lanelets in the current path, that have a stop line
            intersection_lanelets = list(
                vehicle.ref_path_lane.contained_lanelets.intersection(lanelets_with_stop_line)
            )
            # If there is no stop line in current reference path, return False
            if len(intersection_lanelets) == 0:
                return False
            # get the longitudinal position of stop line based on ego reference path
            # find the closest vertices
            stop_line_s = np.array(
                [
                    min(
                        vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                            *world.road_network.lanelet_network.find_lanelet_by_id(
                                lanelet
                            ).stop_line.start
                        )[0],
                        vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                            *world.road_network.lanelet_network.find_lanelet_by_id(
                                lanelet
                            ).stop_line.end
                        )[0],
                    )
                    for lanelet in intersection_lanelets
                ]
            )
            self._dict_veh_id_intersection_lanelets[vehicle_ids[0]] = intersection_lanelets
            self._dict_veh_id_stop_line_s[vehicle_ids[0]] = stop_line_s
        # Get the front longitudinal value of the vehicle
        front_s = vehicle.front_s(time_step, vehicle.ref_path_lane) or -np.inf
        for i in range(stop_line_s.shape[0]):
            # check if vehicle in this lanelet in lateral horizon
            if not vehicle.lanelet_assignment[time_step].intersection([intersection_lanelets[i]]):
                continue
            # Get the distance to the stop lines
            stop_line_distance = stop_line_s[i] - front_s
            if 0 <= stop_line_distance <= self.config.d_sl:
                return True
        return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        """
        If there is no stop line along the reference path, return -1.
        If there is a stop line along the reference path return minimum difference between d_sl and
        the distance to stop line.
        """
        robustness = -np.inf
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        # Find all lanelets in the map that have a stop line
        lanelets_with_stop_line = [
            lanelet.lanelet_id
            for lanelet in world.road_network.lanelet_network.lanelets
            if lanelet.stop_line is not None
        ]
        # Get the set of lanelets in the current path, that have a stop line
        intersection_lanelets = list(
            vehicle.ref_path_lane.contained_lanelets.intersection(lanelets_with_stop_line)
        )
        # If there is no stop line in current reference path, return -1
        if len(intersection_lanelets) == 0:
            return self._scale_lon_dist(float(robustness))
        # Get the front longitudinal value of the vehicle
        front_s = vehicle.front_s(time_step, vehicle.ref_path_lane) or -np.inf
        # get the longitudinal position of stop line based on ego reference path
        # find the closest vertices
        stop_line_s = np.array(
            [
                min(
                    vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *world.road_network.lanelet_network.find_lanelet_by_id(
                            lanelet
                        ).stop_line.start
                    )[0],
                    vehicle.ref_path_lane.clcs.convert_to_curvilinear_coords(
                        *world.road_network.lanelet_network.find_lanelet_by_id(
                            lanelet
                        ).stop_line.end
                    )[0],
                )
                for lanelet in intersection_lanelets
            ]
        )
        for i in range(stop_line_s.shape[0]):
            # check if vehicle in this lanelet in lateral horizon
            d_lane = utils.distance_to_lanes(vehicle, [intersection_lanelets[i]], world, time_step)
            if d_lane < 0:
                continue
            # Get the distance to the stop lines
            stop_line_distance = stop_line_s[i] - front_s
            stop_line_robustness = np.fmin(
                self.config.d_sl - stop_line_distance, stop_line_distance
            )
            robustness = max(robustness, stop_line_robustness)
        return self._scale_lon_dist(float(robustness))


class PredOnIncomingLeftOf(AbstractPredicate):
    """
    evaluate if the k-th vehicle occupies a lane that is left of the lane of the p-th vehicle in terms of incoming
    consider in multiple intersections
    """

    predicate_name = PositionPredicates.OnIncomingLeftOf
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        # consider scenario with multiple intersections
        incomings_k, dis_to_incomings_k = utils.get_incoming_multi_intersections(
            vehicle_k, time_step, road_network
        )
        incomings_p, dis_to_incomings_p = utils.get_incoming_multi_intersections(
            vehicle_p, time_step, road_network
        )
        for i in range(len(incomings_k)):
            inc_left_of_k_id = incomings_k[i].left_of
            # check if k-th incoming is left of p-th incoming
            if (
                (incomings_p[i].incoming_id == inc_left_of_k_id)
                and (dis_to_incomings_k[i] >= 0)
                and (dis_to_incomings_p[i] >= 0)
            ):
                return True
            else:
                return False

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        rob = -1
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        # consider scenario with multiple intersections
        incomings_k, dis_to_incomings_k = utils.get_incoming_multi_intersections(
            vehicle_k, time_step, road_network
        )
        incomings_p, dis_to_incomings_p = utils.get_incoming_multi_intersections(
            vehicle_p, time_step, road_network
        )
        for i in range(len(incomings_k)):
            inc_left_of_k_id = incomings_k[i].left_of
            # check if k-th incoming is left of p-th incoming
            if incomings_p[i].incoming_id == inc_left_of_k_id:
                rob = max(
                    rob,
                    min(
                        self._scale_lon_dist(dis_to_incomings_k[i]),
                        self._scale_lon_dist(dis_to_incomings_p[i]),
                    ),
                )
            else:
                rob = max(rob, -1)
        return rob


class PredInIntersectionConflictArea(AbstractPredicate):
    predicate_name = PositionPredicates.InIntersectionConflictArea
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        incoming_k = vehicle_k.incoming_intersection
        incoming_p = vehicle_p.incoming_intersection
        # check whether two lanelets are part of the same intersection incoming
        adj_inc_k = road_network.adjacent_lanelets(incoming_k.incoming_lanelets)
        if len(adj_inc_k.intersection(incoming_p.incoming_lanelets)) != 0:
            return False
        lanelets_assignment_k = vehicle_k.lanelet_assignment[time_step]
        # find lanelets of assignment with type intersection
        lanelets_k_intersection = [
            la_id
            for la_id in lanelets_assignment_k
            if LaneletType.INTERSECTION
            in road_network.lanelet_network.find_lanelet_by_id(la_id).lanelet_type
        ]
        # find lanelets of reference path of p-th vehicle conflicting lanelets_k_intersection
        conflict_lanelet = vehicle_p.ref_path_lane.contained_lanelets.intersection(
            lanelets_k_intersection
        )
        # exclude lanelets_dir of k-th vehicle (conflict lanelet must exclude lanelets_dir of k-th vehicle)
        conflict_lanelet = conflict_lanelet.difference(vehicle_k.lanelets_dir)
        if len(conflict_lanelet) == 0:
            return False
        else:
            return True

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        road_network = world.road_network
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        incoming_k = vehicle_k.incoming_intersection
        incoming_p = vehicle_p.incoming_intersection
        # check whether two lanelets are part of the same intersection incoming
        adj_inc_k = road_network.adjacent_lanelets(incoming_k.incoming_lanelets)
        if len(adj_inc_k.intersection(incoming_p.incoming_lanelets)) != 0:
            return -1
        lanelets_assignment_k = vehicle_k.lanelet_assignment[time_step]
        # find lanelets of assignment with type intersection
        lanelets_k_intersection = [
            la_id
            for la_id in lanelets_assignment_k
            if LaneletType.INTERSECTION
            in road_network.lanelet_network.find_lanelet_by_id(la_id).lanelet_type
        ]
        # find lanelets of reference path of p-th vehicle conflicting lanelets_k_intersection
        conflict_lanelet = vehicle_p.ref_path_lane.contained_lanelets.intersection(
            lanelets_k_intersection
        )
        # exclude lanelets_dir of k-th vehicle (conflict lanelet must exclude lanelets_dir of k-th vehicle)
        conflict_lanelet = conflict_lanelet.difference(vehicle_k.lanelets_dir)
        if len(conflict_lanelet) != 0:
            current_lanelet_k = list(lanelets_assignment_k.intersection(vehicle_k.lanelets_dir))
            # get center vertices of current lanelets of k-th vehicle
            center_vertices_k = None
            for lanelet_id_k in current_lanelet_k:
                lanelet_item = road_network.lanelet_network.find_lanelet_by_id(lanelet_id_k)
                if center_vertices_k is None:
                    center_vertices_k = lanelet_item.center_vertices
                else:
                    center_vertices_k = np.append(
                        center_vertices_k, lanelet_item.center_vertices, axis=0
                    )
            # get polygon of conflict lanelets
            conflict_polygon = None
            for lanelet_id_conflict in conflict_lanelet:
                lanelet_item = road_network.lanelet_network.find_lanelet_by_id(lanelet_id_conflict)
                if conflict_polygon is None:
                    conflict_polygon = lanelet_item.polygon.shapely_object
                else:
                    conflict_polygon = conflict_polygon.union(lanelet_item.polygon.shapely_object)
            # find conflict points between center certices of current lanelets of k-th vehicle and conflict lanelets
            conflict_points = utils.find_conflict_points(center_vertices_k, conflict_polygon)
            if conflict_points is None:
                return 0.001
            # get front- and rear-most points along reference path of k-th vehicle
            front_s_k = vehicle_k.front_s(time_step, vehicle_k.ref_path_lane)
            rear_s_k = vehicle_k.rear_s(time_step, vehicle_k.ref_path_lane)
            # get longitudinal positions of conflict points along reference path of k-th vehicle
            start_conflict_s = vehicle_k.ref_path_lane.clcs.convert_to_curvilinear_coords(
                *conflict_points[0]
            )[0]
            end_conflict_s = vehicle_k.ref_path_lane.clcs.convert_to_curvilinear_coords(
                *conflict_points[1]
            )[0]
            rob = min(front_s_k - start_conflict_s, end_conflict_s - rear_s_k)
            # TODO: fix threshold
            rob = max(0.001, self._scale_lon_dist(rob))
        else:
            # if there is no conflict lanelet currently, search the whole reference path of p-th vehicle
            all_conflict_points = list()
            for lanelet_id in vehicle_p.ref_path_lane.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                if LaneletType.INTERSECTION in lanelet.lanelet_type:
                    # find conflict points between center vertices of lanelets_dir of k-th vehicle and reference path
                    # lanelets of p-th vehicle
                    conflict_points = utils.find_conflict_points(
                        vehicle_k.lanelets_dir_center_vertices,
                        lanelet.polygon.shapely_object,
                    )
                    if conflict_points is not None:
                        all_conflict_points.append(conflict_points)
            if len(all_conflict_points) == 0:
                return -1
            else:
                # get front- and rear-most points along reference path of k-th vehicle
                front_s_k = vehicle_k.front_s(time_step, vehicle_k.ref_path_lane)
                rear_s_k = vehicle_k.rear_s(time_step, vehicle_k.ref_path_lane)
                # get longitudinal positions of conflict points along reference path of k-th vehicle
                start_conflict_s = vehicle_k.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *all_conflict_points[0][0]
                )[0]
                end_conflict_s = vehicle_k.ref_path_lane.clcs.convert_to_curvilinear_coords(
                    *all_conflict_points[-1][-1]
                )[0]
                rob = min(front_s_k - start_conflict_s, end_conflict_s - rear_s_k)
                # TODO: fix wrong result of route planner
                rob = min(-0.01, self._scale_lon_dist(rob))
        return rob


class PredOnLaneletWithTypeIntersection(AbstractPredicate):
    """
    evaluates if a vehicle is on a lanelet with a specific type.
    """

    predicate_name = PositionPredicates.OnLaneletWithTypeIntersection
    arity = 1

    def evaluate_boolean(self, world: World, time_step: int, vehicle_ids: tuple[int, ...]) -> bool:
        road_network = world.road_network
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelets_assignment = vehicle.lanelet_assignment[time_step]
        for lanelet_id in lanelets_assignment:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if LaneletType.INTERSECTION in lanelet.lanelet_type:
                return True
        return False

    def evaluate_robustness(
        self, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        rob = -np.inf
        road_network = world.road_network
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        lanelets_assignment = vehicle.lanelet_assignment[time_step]
        lanelets_occ_intersection = list()
        lane_occ_intersection = list()
        # find occupied lanelet at intersection and find corresponding lane
        for lanelet_id in lanelets_assignment:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            if LaneletType.INTERSECTION in lanelet.lanelet_type:
                lanelets_occ_intersection.append(lanelet)
                lane_occ_intersection.append(
                    utils.find_longest_lane_by_intersection_lanelet(lanelet_id, road_network)
                )
        # if current occupied lanelets are not at intersection, use lanelets_dir to find previous and future
        if len(lanelets_occ_intersection) == 0:
            for lanelet_id in vehicle.lanelets_dir:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                if LaneletType.INTERSECTION in lanelet.lanelet_type:
                    lanelets_occ_intersection.append(lanelet)
                    lane_occ_intersection.append(vehicle.ref_path_lane)
        for i in range(len(lanelets_occ_intersection)):
            front_s = vehicle.front_s(time_step, lane_occ_intersection[i])
            rear_s = vehicle.rear_s(time_step, lane_occ_intersection[i])

            lanelet_occ = lanelets_occ_intersection[i]
            lanelet_start_s = lane_occ_intersection[i].convert_to_curvilinear_coords(
                *lanelet_occ.center_vertices[0]
            )[0]
            lanelet_end_s = lane_occ_intersection[i].convert_to_curvilinear_coords(
                *lanelet_occ.center_vertices[-1]
            )[0]
            # lanelet in front of vehicle
            if (front_s - lanelet_start_s) < 0 < (lanelet_end_s - rear_s):
                rob = max(rob, front_s - lanelet_start_s)
            # vehicle in front of lanelet
            elif (lanelet_end_s - rear_s) <= 0 <= (front_s - lanelet_start_s):
                rob = max(rob, lanelet_end_s - rear_s)
            # vehicle inside lanelet
            else:
                distance_robustness = min(front_s - lanelet_start_s, lanelet_end_s - rear_s)
                rob = max(rob, distance_robustness)
        return self._scale_lon_dist(rob)


class PredOnOncomOf(AbstractPredicate):
    predicate_name = PositionPredicates.OnOncomOf
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        road_network = world.road_network
        vehicle_target = world.vehicle_by_id(vehicle_ids[0])
        vehicle_ego = world.vehicle_by_id(vehicle_ids[1])
        incoming_target = vehicle_target.incoming_intersection
        incoming_ego = vehicle_ego.incoming_intersection
        target_polygon = list()
        for straight_suc in incoming_target.successors_straight:
            lanelet_straight = road_network.lanelet_network.find_lanelet_by_id(straight_suc)
            target_polygon.append(lanelet_straight.polygon.shapely_object)
        target_polygon = shapely.ops.unary_union(target_polygon)
        ego_polygon = list()
        for straight_suc in incoming_ego.successors_straight:
            lanelet_straight = road_network.lanelet_network.find_lanelet_by_id(straight_suc)
            ego_polygon.append(lanelet_straight.polygon.shapely_object)
        ego_polygon = shapely.ops.unary_union(ego_polygon)
        intersection_polygon = target_polygon.intersection(ego_polygon)
        if intersection_polygon.geom_type == "Polygon" and not intersection_polygon.is_empty:
            rob = -1.0
        else:
            rob = 1.0
        return rob
