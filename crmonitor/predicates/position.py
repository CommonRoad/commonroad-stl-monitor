from enum import Enum
import logging
import math
import operator
from typing import List, Tuple, Set, Iterable, Dict, Callable
from shapely.geometry.polygon import Polygon
import numpy as np

from ruamel.yaml.comments import CommentedMap

from crmonitor.common.helper import union_set
from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

from crmonitor.predicates.base import BasePredicateEvaluator, MAX_LONG_DIST
from crmonitor.predicates.utils import distance_to_bounds

logger = logging.getLogger(__name__)


class PositionPredicates(str, Enum):
    InSameLane = "in_same_lane"
    InFrontOf = "in_front_of"
    SingleLane = "single_lane"
    KeepsSafeDistancePrec = "keeps_safe_distance_prec"
    Precedes = "precedes"


class PredInSameLane(BasePredicateEvaluator):
    predicate_name = PositionPredicates.InSameLane
    arity = 2

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        intersecting_lanes = self.get_same_lanes(world, time_step, vehicle_ids)
        return len(intersecting_lanes) > 0

    def get_same_lanes(self, world, time_step, vehicle_ids) -> Set[Lane]:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])
        lanes_k = world.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[time_step]
        )
        lanes_p = world.road_network.find_lanes_by_lanelets(
            vehicle_p.lanelet_assignment[time_step]
        )
        intersecting_lanes = lanes_p.intersection(lanes_k)
        return intersecting_lanes

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        """
        If boolean is
        True: Minimum lateral displacement to not be in the same lane anymore
        False: Minimum distance to lanes of other
        """
        # Predicate is symmetric
        vehicle_ids_tuple = tuple(reversed(vehicle_ids))
        value = world.vehicle_by_id(vehicle_ids_tuple[0]).predicate_cache[time_step, self.predicate_name, vehicle_ids_tuple[1:]]
        if value is not None:
            return value

        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        vehicle_p = world.vehicle_by_id(vehicle_ids[1])

        def distance_to_lanes(vehicle_i: Vehicle, lanelet_ids: Iterable[int]):
            d_left, d_right = distance_to_bounds(vehicle_i, lanelet_ids, world, time_step)
            d_left = -np.min(d_left) if d_left.size > 0 else np.inf
            d_right = np.max(d_right) if d_right.size > 0 else np.inf
            return np.fmin(d_left, d_right)

        lanelet_ids_k = union_set(
            [l.contained_lanelets for l in vehicle_k.lanes_at_state(time_step)]
        )
        lanelet_ids_p = union_set(
            [l.contained_lanelets for l in vehicle_p.lanes_at_state(time_step)]
        )
        rob = np.fmin(
            distance_to_lanes(vehicle_k, lanelet_ids_p),
            distance_to_lanes(vehicle_p, lanelet_ids_k),
        )
        return self._scale_lat_dist(rob)


class PredInFrontOf(BasePredicateEvaluator):
    predicate_name = PositionPredicates.InFrontOf
    arity = 2

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        rear = world.vehicle_by_id(vehicle_ids[0])
        front = world.vehicle_by_id(vehicle_ids[1])
        return self._scale_lon_dist(
                front.rear_s(time_step, rear.get_lane(time_step)) - rear.front_s(time_step)
        )


class PredSingleLane(BasePredicateEvaluator):
    predicate_name = PositionPredicates.SingleLane
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        k_lanes = world.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[time_step]
        )
        return len(k_lanes) == 1

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        """
        If false: 1 - the largest fractional overlap with occupied lanes
        If true: Distance to lane polygon boundary
        """
        # single_lane_boolean = self.evaluate_boolean(world, vehicle_ids)
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        k_lanes = sorted(vehicle_k.lanes_at_state(time_step))
        assert (
            len(k_lanes) > 0
        ), f"Vehicle must be assigned to at least one lane! {str(world.scenario.scenario_id)}, id={vehicle_ids[0]}, t={time_step}"

        ref_point = np.array(vehicle_k.states_cr[time_step].position)
        ref_lanes = [
            l
            for l in k_lanes
            if l.lanelet.polygon.contains_point(ref_point)
        ]
        ref_lane = ref_lanes[0] if len(ref_lanes) > 0 else k_lanes[0]

        d_left, d_right = distance_to_bounds(vehicle_k, ref_lane.contained_lanelets, world, time_step)
        d_left = -np.max(d_left) if d_left.size > 0 else np.inf
        d_right = np.min(d_right) if d_right.size > 0 else np.inf
        rob = np.fmin(d_left, d_right)
        return self._scale_lat_dist(rob)


class PredSafeDistPrec(BasePredicateEvaluator):
    predicate_name = PositionPredicates.KeepsSafeDistancePrec
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

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle_follow = world.vehicle_by_id(vehicle_ids[0])
        vehicle_lead = world.vehicle_by_id(vehicle_ids[1])
        time_step = time_step

        if vehicle_lead.get_lane(time_step) is None:
            return self._scale_lon_dist(math.inf)
        a_min_follow = vehicle_follow.vehicle_param.get("a_min")
        a_min_lead = vehicle_lead.vehicle_param.get("a_min")
        t_react_follow = vehicle_follow.vehicle_param.get("t_react")
        safe_distance = self.calculate_safe_distance(
            vehicle_follow.states_cr[time_step].velocity,
            vehicle_lead.states_cr[time_step].velocity,
            a_min_lead,
            a_min_follow,
            t_react_follow,
        )

        delta_s = vehicle_lead.rear_s(time_step) - vehicle_follow.front_s(time_step)
        rob = self._scale_lon_dist(delta_s - safe_distance)
        return rob

    @staticmethod
    def _plot_red_arrow(ax, x, y, size=1.):
        ax.plot(x, y, linewidth=2, color='r', zorder=25)
        ax.arrow(x[-2], y[-2], x[-1] - x[-2], y[-1] - y[-2], lw=0, length_includes_head=True, head_width=size, head_length=size, zorder=25, color='r')

    def visualize_unsafe_region(self,
                                ax,
                                time_step: int,
                                unsafe_s: float,
                                vehicle_lead: Vehicle):
        """
        Plots the unsafe region starting from the rear of the front vehicle
        """
        # the ids of lanes are increasing together with the d-coordinate
        vehicle_lanes = list(sorted(vehicle_lead.lanes_at_state(time_step),
                                    key=operator.attrgetter('lane_id'),
                                    reverse=True))
        reference_lane = vehicle_lead.get_lane(time_step)
        # get the Cartesian coordinate of the safe distance
        safe_pos_cart = reference_lane.clcs.convert_to_cartesian_coords(unsafe_s, 0)
        lead_rear_cart = reference_lane.clcs.\
            convert_to_cartesian_coords(vehicle_lead.rear_s(time_step), 0.0)
        # left vertices
        front_rear_left_cart = vehicle_lanes[0].clcs_left.\
            convert_to_cartesian_coords(vehicle_lead.rear_s(time_step), 0.0)
        safe_pos_left_cart = vehicle_lanes[0].clcs_left.convert_to_cartesian_coords(unsafe_s, 0)
        reference_left = np.vstack(vehicle_lanes[0].clcs_left.reference_path())
        vertices_left = reference_left[(reference_left[:, 0] > safe_pos_left_cart[0]) & (
                    reference_left[:, 0] < front_rear_left_cart[0]), :]
        vertices_left = np.concatenate(([safe_pos_left_cart], vertices_left, [front_rear_left_cart]))
        # right vertices
        lead_rear_right_cart = vehicle_lanes[-1].clcs_right.convert_to_cartesian_coords(
            vehicle_lead.rear_s(time_step), 0.0)
        safe_pos_right_cart = vehicle_lanes[-1].clcs_right.convert_to_cartesian_coords(unsafe_s, 0)
        reference_right = np.vstack(vehicle_lanes[-1].clcs_right.reference_path())
        vertices_right = reference_right[(reference_right[:, 0] > safe_pos_left_cart[0]) & (
                    reference_right[:, 0] < front_rear_left_cart[0]), :]
        vertices_right = np.concatenate(([safe_pos_right_cart], vertices_right, [lead_rear_right_cart]))
        # concatenate vertices
        vertices_total = np.concatenate(([safe_pos_cart],
                                         vertices_left,
                                         [lead_rear_cart],
                                         np.flip(vertices_right, 0),
                                         [safe_pos_cart])).tolist()
        unsafe_region = Polygon(vertices_total)
        ax.fill(*unsafe_region.exterior.xy, zorder=30, alpha=0.2, facecolor='red', edgecolor=None)

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
        latest_value = self.evaluate_robustness_with_cache(
            world, time_step, vehicle_ids
        )
        latest_value_unscaled = (
            latest_value * MAX_LONG_DIST
        )  # un-scale to actual range and make positive
        vehicle_follow = world.vehicle_by_id(vehicle_ids[0])

        lane_clcs = vehicle_follow.get_lane(time_step).clcs  # center curvilinear coordinate system
        sampling_step_size = 1.

        s_start = vehicle_follow.front_s(time_step)
        num_points = max(2, abs(int(latest_value_unscaled / sampling_step_size)))
        points_s = np.linspace(0, latest_value_unscaled, num_points) + s_start
        points_s = points_s[:, None]
        points_l = np.zeros((points_s.shape[0], 1))
        points_curvi = np.concatenate((points_s, points_l), axis=1)
        points_cartesian = np.stack([lane_clcs.convert_to_cartesian_coords(*p) for p in points_curvi], axis=0)

        def fun(renderer):
            self._plot_red_arrow(renderer.ax, points_cartesian[:,0], points_cartesian[:,1])
            unsafe_s = latest_value_unscaled + s_start
            self.visualize_unsafe_region(renderer.ax, time_step, unsafe_s, world.vehicle_by_id(vehicle_ids[1]))
        return (fun,)

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((0, 1))
        ax.plot(0, 0.5, color="r")
        PredSafeDistPrec._plot_red_arrow(ax, [0, 1], [0.5, 0.5], size=0.1)
        PredSafeDistPrec._plot_red_arrow(ax, [0, -1], [0.5, 0.5], size=0.1)


class PredPreceding(BasePredicateEvaluator):
    predicate_name = PositionPredicates.Precedes
    arity = 2

    def __init__(self, config: CommentedMap):
        super().__init__(config)
        self.same_lane = PredInSameLane(config)

    @staticmethod
    def get_predecessors(world: World, time_step, vehicle_rear: Vehicle) -> List[Tuple[float, Vehicle, Lane, bool]]:
        """
        Returns a list of preceding vehicles in ascending order of distance
        :param time_step:
        :param world: Current world state
        :param vehicle_rear: Reference vehicle
        :return: Sorted list of tuples of distance and vehicle object
        """
        veh = []
        rear_lanes = vehicle_rear.lanes_at_state(time_step)
        for vehicle_front in world.vehicles:
            if (
                not vehicle_front.is_valid(time_step)
                or vehicle_front is vehicle_rear
            ):
                continue
            front_lanes = vehicle_front.lanes_at_state(time_step)
            intersecting_lanes = rear_lanes.intersection(front_lanes)
            same_lane = len(intersecting_lanes) > 0
            lane = list(intersecting_lanes)[0] if len(intersecting_lanes) > 0 else vehicle_rear.get_lane(time_step)
            dist = vehicle_front.rear_s(
                time_step, lane
            ) - vehicle_rear.front_s(time_step, lane)
            veh.append((dist, vehicle_front, lane, same_lane))
        return sorted(veh, key=lambda d: d[0])

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        rear_vehicle_id = vehicle_ids[0]
        front_vehicle_id = vehicle_ids[1]
        rear_vehicle = world.vehicle_by_id(rear_vehicle_id)
        pred_veh = self.get_predecessors(world, time_step, rear_vehicle)
        return len(pred_veh) > 0 and pred_veh[0][1].id == front_vehicle_id

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        rear_veh = world.vehicle_by_id(vehicle_ids[0])
        front_veh = world.vehicle_by_id(vehicle_ids[1])
        veh_lon_dist = self.get_predecessors(world, time_step, rear_veh)
        veh_front_dist = [_ for _ in veh_lon_dist if _[0] >= 0 and _[3]]
        bool_val = len(veh_front_dist) > 0 and veh_front_dist[0][1].id == vehicle_ids[1]
        same_lane = self.same_lane.evaluate_robustness_with_cache(world, time_step, vehicle_ids)
        if bool_val:
            assert same_lane >= -self.eps
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