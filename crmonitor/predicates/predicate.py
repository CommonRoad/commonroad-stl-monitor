import abc
import logging
import math
import operator
from typing import List, Tuple, Set, Iterable, Dict, Callable
from shapely.geometry.polygon import Polygon

import matplotlib.colors
import numpy as np
from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.visualization.renderer import IRenderer
from matplotlib import pyplot as plt
from ruamel.yaml.comments import CommentedMap

from crmonitor.common.helper import union_set, cartesian_to_curvilinear
from crmonitor.common.road_network import Lane
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

logger = logging.getLogger(__name__)


def distance_to_bounds(vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step):
    state = vehicle_i.states_cr[time_step]
    occ_points = rotate_translate(
            vehicle_i.shape.vertices[:-1], state.position, state.orientation
        )
    lanelets = [
        world.road_network.lanelet_network.find_lanelet_by_id(i)
        for i in lanelet_ids
    ]
    left_bounds = tuple([
        l.left_vertices for l in lanelets if l.adj_left is not None and l.adj_left not in lanelet_ids
    ])
    right_bounds = tuple([
        l.right_vertices
        for l in lanelets
        if l.adj_right is not None and l.adj_right not in lanelet_ids
    ])
    if len(left_bounds) > 0:
        d_left = np.array(cartesian_to_curvilinear(left_bounds, occ_points))[..., 1].ravel()
        d_left = d_left[~np.isnan(d_left)]
    else:
        d_left = np.array([])
    if len(right_bounds) > 0:
        d_right = np.array(cartesian_to_curvilinear(right_bounds, occ_points))[..., 1].ravel()
        d_right = d_right[~np.isnan(d_right)]
    else:
        d_right = np.array([])

    return d_left, d_right


MAX_LONG_DIST = 200.0
MAX_LAT_DIST = 20.0


class BasePredicateEvaluator(abc.ABC):
    predicate_name = "interface"

    def __init__(self, config: CommentedMap):
        self.config = config
        self.scale = config.setdefault("scale_rob", True)
        self.eps = 1e-5

    def _scale(self, x, max_value):
        return np.clip(x / max_value, -1.0, 1.0) if self.scale else x

    def _scale_speed(self, x):
        return self._scale(x, 250.0 / 3.6)

    def _scale_acc(self, x):
        return self._scale(x, 10.5)

    def _scale_lon_dist(self, x):
        return self._scale(x, MAX_LONG_DIST)

    def _scale_lat_dist(self, x):
        return self._scale(x, MAX_LAT_DIST)

    def _scale_angle(self, x):
        return self._scale(x, math.pi)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        return self.evaluate_robustness(world, time_step, vehicle_ids) >= 0.0

    @abc.abstractmethod
    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        pass

    def evaluate_robustness_with_cache(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        vehicle_ids_tuple = tuple(vehicle_ids)
        value = vehicle.predicate_cache.get_robustness(time_step, self.predicate_name, vehicle_ids_tuple[1:])
        if value is None:
            logger.debug(
                "Evaluating predicate %s , t=%d, ids=%s",
                self.predicate_name,
                time_step,
                vehicle_ids_tuple,
            )
            value = self.evaluate_robustness(world, time_step, vehicle_ids)
            vehicle.predicate_cache.set_robustness(time_step, self.predicate_name, vehicle_ids_tuple[1:], value)
        return value

    def visualize(
        self,
        vehicle_ids: List[int],
        add_vehicle_draw_params: Callable[[int, any], None],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ) -> Tuple[Callable[[IRenderer], None], ...]:
        """
        Overwrite this function for visualizing a predicate in a certain way within the scenario plot.
        """
        self._gather_predicate_values_to_plot(
            vehicle_ids, world, time_step, predicate_names2vehicle_ids2values
        )
        return ()

    def _gather_predicate_values_to_plot(
        self,
        vehicle_ids: List[int],
        world: World,
        time_step: int,
        predicate_names2vehicle_ids2values: Dict[str, Dict[Tuple[int, ...], float]],
    ):
        predicate_names2vehicle_ids2values[self.predicate_name][
            tuple(vehicle_ids)
        ] = self.evaluate_robustness_with_cache(world, time_step, vehicle_ids)

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        ax.axis("off")
        ax.text(0.1, 0.5, "[not visualized]", fontsize=12)


class PredInSameLane(BasePredicateEvaluator):
    predicate_name = "in_same_lane"
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
        :param time_step:
        :param world:
        :param vehicle_ids:
        :return:
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
    predicate_name = "in_front_of"
    arity = 2

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        rear = world.vehicle_by_id(vehicle_ids[0])
        front = world.vehicle_by_id(vehicle_ids[1])
        return self._scale_lon_dist(
                front.rear_s(time_step, rear.get_lane(time_step)) - rear.front_s(time_step)
        )


class PredSingleLane(BasePredicateEvaluator):
    predicate_name = "single_lane"
    arity = 1

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        vehicle_k = world.vehicle_by_id(vehicle_ids[0])
        k_lanes = world.road_network.find_lanes_by_lanelets(
            vehicle_k.lanelet_assignment[time_step]
        )
        return len(k_lanes) == 1

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        """
        If false: 1 - largest fractional overlap with occupied lanes
        If true: Distance to lane polygon boundary
        :param time_step:
        :param world:
        :param vehicle_ids:
        :return:
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


class PredCutIn(BasePredicateEvaluator):
    predicate_name = "cut_in"
    arity = 2

    def __init__(self, config):
        super().__init__(config)
        self._same_lane_evaluator = PredInSameLane(config)
        self._single_lane_evaluator = PredSingleLane(config)

    def evaluate_boolean(self, world: World, time_step, vehicle_ids: List[int]) -> bool:
        cutting_vehicle = world.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_boolean(world, time_step, [vehicle_ids[0]])
        if single_lane:
            return False
        same_lane = self._same_lane_evaluator.evaluate_boolean(world, time_step, vehicle_ids)
        if not same_lane:
            return False
        cutting_lane = cutting_vehicle.get_lane(time_step)
        cutted_lat = cutted_vehicle.get_lat_state(time_step, cutting_lane)
        cutting_lat = cutting_vehicle.get_lat_state(time_step)
        d_p = cutted_lat.d
        d_k = cutting_lat.d
        orient_k = cutting_lat.theta

        result = (d_k < d_p and orient_k > self.eps) or (
            d_k > d_p and orient_k < -self.eps
        )
        return result

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        cutting_vehicle = world.vehicle_by_id(vehicle_ids[0])
        cutted_vehicle = world.vehicle_by_id(vehicle_ids[1])

        single_lane = self._single_lane_evaluator.evaluate_robustness_with_cache(world, time_step,
                                                                                 [vehicle_ids[0], ])
        same_lane = self._same_lane_evaluator.evaluate_robustness_with_cache(world, time_step, vehicle_ids)

        cutting_lane = cutting_vehicle.get_lane(time_step)
        cutted_lat = cutted_vehicle.get_lat_state(time_step, cutting_lane)
        cutting_lat = cutting_vehicle.get_lat_state(time_step)
        r_l_dist = (
                cutted_lat.d
                - cutting_lat.d
        )
        r_l_orient = cutting_lat.theta - self.eps
        l_r_dist = (
            cutting_lat.d
            - cutted_lat.d
        )
        l_r_orient = -self.eps - cutting_lat.theta

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

    @staticmethod
    def _get_color_map():
        return plt.get_cmap("bwr")

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
        latest_value_normalized = (latest_value + 1) / 2
        violation_color = self._get_color_map()(latest_value_normalized)
        violation_color_hex = matplotlib.colors.rgb2hex(violation_color)

        vehicle = vehicle_ids[0]
        draw_params = {
            "dynamic_obstacle": {
                "vehicle_shape": {
                    "occupancy": {
                        "shape": {"rectangle": {"facecolor": violation_color_hex}}
                    }
                }
            }
        }
        add_vehicle_draw_params(vehicle, draw_params)

        draw_functions1 = self._same_lane_evaluator.visualize(
            vehicle_ids,
            add_vehicle_draw_params,
            world,
            time_step,
            predicate_names2vehicle_ids2values,
        )
        draw_functions2 = self._single_lane_evaluator.visualize(
            [vehicle],
            add_vehicle_draw_params,
            world,
            time_step,
            predicate_names2vehicle_ids2values,
        )

        return () + draw_functions1 + draw_functions2

    @staticmethod
    def plot_predicate_visualization_legend(ax):
        points = np.linspace(0, 1, 256)
        points = np.vstack((points, points))
        ax.imshow(points, cmap=PredCutIn._get_color_map(), extent=[-1, 1, 0, 1])
        ax.get_yaxis().set_ticks([])
        ax.set_ylabel('vehicle color')


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
        ax.arrow(x[-2], y[-2], x[-1] - x[-2], y[-1] - y[-2], lw=0, length_includes_head=False, head_width=size, head_length=size, zorder=25, color='r')

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

        # back_again = np.stack(lane_clcs.convert_list_of_points_to_curvilinear_coords([p for p in points_cartesian], 8), axis=0)

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
    predicate_name = "keeps_lane_speed_limit"
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
    predicate_name = "keeps_type_speed_limit"
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle_type = world.vehicle_by_id(vehicle_ids[0]).obstacle_type
        if vehicle_type is ObstacleType.TRUCK:
            return self.config["max_interstate_speed_truck"]
        else:
            return None


class PredFovSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_fov_speed_limit"
    arity = 1

    def get_speed_limit(self, world, time_step, vehicle_ids):
        vehicle = world.vehicle_by_id(vehicle_ids[0])
        return vehicle.vehicle_param.get("fov_speed_limit")


class PredBrSpeedLimit(PredGenericSpeedLimit):
    predicate_name = "keeps_brake_speed_limit"
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


class PredPreceding(BasePredicateEvaluator):
    predicate_name = "precedes"
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


class PredAbruptBreaking(BasePredicateEvaluator):
    predicate_name = "brakes_abruptly"
    arity = 1

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        accel = (
            world.vehicle_by_id(vehicle_ids[0])
            .states_cr[time_step]
            .acceleration
        )
        rob = self.config["a_abrupt"] - accel
        return self._scale_acc(rob)


class PredRelAbruptBreaking(BasePredicateEvaluator):
    predicate_name = "rel_brakes_abruptly"
    arity = 2

    def evaluate_robustness(self, world: World, time_step, vehicle_ids: List[int]) -> float:
        accel_k = (
            world.vehicle_by_id(vehicle_ids[0])
            .states_cr[time_step]
            .acceleration
        )
        accel_p = (
            world.vehicle_by_id(vehicle_ids[1])
            .states_cr[time_step]
            .acceleration
        )
        rob = -accel_k + accel_p + self.config["a_abrupt"]
        return self._scale_acc(rob)
