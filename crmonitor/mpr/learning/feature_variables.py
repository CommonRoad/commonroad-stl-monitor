import enum
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import numpy as np
from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.lanelet import LaneletType

from crmonitor.common import Lane, RoadNetwork, ScenarioType, Vehicle, World
from crmonitor.mpr.state_context import StateContext

from .error import FeatureExtractionError


def _find_right_most_real_lane_for_vehicle_at_time_step(
    vehicle: Vehicle, time_step: int, road_network: RoadNetwork
) -> Lane:
    current_right_most_lane = vehicle.lane_at_time_step(time_step)
    while current_right_most_lane.adj_right is not None:
        # new_right_most_lane = road_network.find_lane_by_id(current_right_most_lane.adj_right)
        new_right_most_lane = current_right_most_lane.adj_right
        if new_right_most_lane is None:
            raise RuntimeError(
                f"Lane {current_right_most_lane.lane_id} references {current_right_most_lane.adj_right} as its adjacent right lane. But this lane does not exist in the road network!"
            )

        current_right_most_lane = new_right_most_lane

    return current_right_most_lane


def _find_left_most_real_lane_for_vehicle_at_time_step(
    vehicle: Vehicle, time_step: int, road_network: RoadNetwork
):
    current_left_most_lane = vehicle.lane_at_time_step(time_step)
    while current_left_most_lane.adj_left is not None:
        # new_left_most_lane = road_network.find_lane_by_id(current_left_most_lane.adj_left)
        new_left_most_lane = current_left_most_lane.adj_left
        if new_left_most_lane is None:
            raise RuntimeError(
                f"Lane {current_left_most_lane.lane_id} references {current_left_most_lane.adj_left} as its adjacent left lane. But this lane does not exist in the road network!"
            )

        current_left_most_lane = new_left_most_lane

    return current_left_most_lane


class AbstractFeatureVariable(ABC):
    arity: int
    name: str

    @classmethod
    @abstractmethod
    def provides(cls, feature_name: str) -> bool: ...

    @classmethod
    @abstractmethod
    def extract(cls, ctx: StateContext) -> dict[str, float]: ...

    def __le__(self, other: object) -> bool:
        if not isinstance(other, AbstractFeatureVariable):
            return False
        return self.arity <= other.arity and self.name <= other.name

    def __eg__(self, other: object) -> bool:
        if not isinstance(other, AbstractFeatureVariable):
            return False
        return self.arity == other.arity and self.name == other.name


class AbstractSingleFeatureVariable(AbstractFeatureVariable, ABC):
    @classmethod
    def provides(cls, feature_name: str) -> bool:
        return feature_name == cls.name

    @classmethod
    @abstractmethod
    def extract_single(cls, ctx: StateContext) -> float: ...

    @classmethod
    def extract(cls, ctx: StateContext) -> dict[str, float]:
        return {cls.name: cls.extract_single(ctx)}


class AbstractLonStateFeatureVariable(AbstractSingleFeatureVariable):
    arity = 1
    _parse_slot: str

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        state = ctx.lon_state(0)
        if not hasattr(state, cls._parse_slot):
            raise FeatureExtractionError(
                cls.name, f"attribute {cls._parse_slot} is missing on state {state}"
            )
        return getattr(state, cls._parse_slot)


class PositionFeatureVariable(AbstractLonStateFeatureVariable):
    name = "position"
    _parse_slot = "s"


class VelocityFeatureVariable(AbstractLonStateFeatureVariable):
    name = "velocity"
    _parse_slot = "v"


class AccelerationFeatureVariable(AbstractLonStateFeatureVariable):
    name = "acceleration"
    _parse_slot = "a"


class JerkFeatureVariable(AbstractLonStateFeatureVariable):
    name = "jerk"
    _parse_slot = "j"


class JerkDotFeatureVariable(AbstractLonStateFeatureVariable):
    name = "jerk_dot"
    _parse_slot = "j_dot"


class AbstractLatStateFeatureVariable(AbstractSingleFeatureVariable):
    arity = 1
    _parse_slot: str

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        state = ctx.lat_state(0)
        if not hasattr(state, cls._parse_slot):
            raise FeatureExtractionError(
                cls.name, f"attribute {cls._parse_slot} is missing on state {state}"
            )
        return getattr(state, cls._parse_slot)


class LateralPositionFeatureVariable(AbstractLatStateFeatureVariable):
    name = "lateral_position"
    _parse_slot = "d"


class OrientationFeatureVariable(AbstractLatStateFeatureVariable):
    name = "orientation"
    _parse_slot = "theta"


class CurvatureFeatureVariable(AbstractLatStateFeatureVariable):
    name = "curvature"
    _parse_slot = "kappa"


class CurvatureDotFeatureVariable(AbstractLatStateFeatureVariable):
    name = "curvature_dot"
    _parse_slot = "kappa_dot"


class CurvatureDdotFeatureVariable(AbstractLatStateFeatureVariable):
    name = "curvature_ddot"
    _parse_slot = "kappa_dot_dot"


class VehicleLengthFeatureVariable(AbstractSingleFeatureVariable):
    name = "length"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        return ctx.vehicle(0).shape.length


class VehicleWidthFeatureVariable(AbstractSingleFeatureVariable):
    name = "width"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        return ctx.vehicle(0).shape.width


class LaneCurvatureFeatureVariable(AbstractSingleFeatureVariable):
    name = "lane_curvature"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        ref_lane = ctx.vehicle(0).lane_at_time_step(ctx.time_step)
        return ref_lane.curvature(ctx.lon_state(0).s)


class LaneCurvatureDotFeatureVariable(AbstractSingleFeatureVariable):
    name = "lane_curvature_dot"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        ref_lane = ctx.vehicle(0).lane_at_time_step(ctx.time_step)
        return ref_lane.curvature_prime(ctx.lon_state(0).s)


class RelativeDistanceFeatureVariable(AbstractSingleFeatureVariable):
    arity = 2
    name = "distance"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        ego_state = ctx.vehicle(0)
        other_state = ctx.vehicle(1)
        return other_state.rear_s(ctx.time_step) - ego_state.front_s(ctx.time_step)


class RelativeLateralDistanceFeatureVariable(AbstractSingleFeatureVariable):
    arity = 2
    name = "lateral_distance"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        ego_state = ctx.lat_state(0)
        other_state = ctx.lat_state(1)
        return other_state.d - ego_state.d


class RelativeVelocityFeatureVariable(AbstractSingleFeatureVariable):
    arity = 2
    name = "velocity"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        ego_state = ctx.lon_state(0)
        other_state = ctx.lon_state(0)
        return ego_state.v - other_state.v


class RelativeLongitudinalVelocityFeatureVariable(AbstractSingleFeatureVariable):
    arity = 2
    name = "relative_longitudinal_velocity"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        lon_ego_state = ctx.lon_state(0)
        lon_other_state = ctx.lon_state(1)
        lat_ego_state = ctx.lat_state(0)
        lat_other_state = ctx.lat_state(1)
        return lon_other_state.v * np.cos(lat_other_state.theta) - lon_ego_state.v * np.cos(
            lat_ego_state.theta
        )


class RelativeLateralVelocityFeatureVariable(AbstractSingleFeatureVariable):
    arity = 2
    name = "relative_lateral_velocity"

    @classmethod
    def extract_single(cls, ctx: StateContext) -> float:
        lon_ego_state = ctx.lon_state(0)
        lon_other_state = ctx.lon_state(1)
        lat_ego_state = ctx.lat_state(0)
        lat_other_state = ctx.lat_state(1)
        return lon_other_state.v * np.sin(lat_other_state.theta) - lon_ego_state.v * np.sin(
            lat_ego_state.theta
        )


class _AbstractDistanceToLaneSide(AbstractFeatureVariable, ABC):
    arity = 1

    @classmethod
    def provides(cls, feature_name: str) -> bool:
        return (
            feature_name == cls.name
            # TODO: Decide how to handle min/max features. Separate feature values? Another layer of abstraction?
            # or feature_name == f"min_{cls.name}"
            # or feature_name == f"max_{cls.name}"
        )

    class Side(enum.Enum):
        LEFT = enum.auto()
        RIGHT = enum.auto()

    @classmethod
    @abstractmethod
    def get_lane(cls, ctx: StateContext) -> Lane: ...

    @classmethod
    @abstractmethod
    def get_side(cls, ctx: StateContext) -> Side: ...

    @classmethod
    def extract(cls, ctx: StateContext) -> dict[str, float]:
        shape = ctx.obstacle(0).obstacle_shape
        lon_state = ctx.lon_state(0)
        lat_state = ctx.lat_state(0)
        x, y = lon_state.s, lat_state.d

        occ_points = list(rotate_translate(shape.vertices[:-1], [x, y], lat_state.theta))

        lane = cls.get_lane(ctx)

        min_dist, max_dist = lane.min_max_distance_to_left(occ_points)
        if cls.get_side(ctx) == _AbstractDistanceToLaneSide.Side.LEFT:
            dist = lane.distance_to_left(x, y)
        else:
            dist = lane.distance_to_right(x, y)

        return {f"min_{cls.name}": min_dist, f"max_{cls.name}": max_dist, cls.name: dist}


class _AbstractDistanceToLane(AbstractFeatureVariable, ABC):
    arity = 1

    @classmethod
    def provides(cls, feature_name: str) -> bool:
        return (
            feature_name == cls.name
            # or feature_name == f"min_{cls.name}"
            # or feature_name == f"max_{cls.name}"
        )

    @classmethod
    @abstractmethod
    def extract_left(cls, ctx: StateContext) -> dict[str, float]: ...

    @classmethod
    @abstractmethod
    def extract_right(cls, ctx: StateContext) -> dict[str, float]: ...

    @classmethod
    def extract(cls, ctx: StateContext) -> dict[str, float]:
        left_values = cls.extract_left(ctx)

        min_key_left = list(left_values.keys())[0]
        min_dist_left = left_values[min_key_left]
        max_key_left = list(left_values.keys())[1]
        max_dist_left = left_values[max_key_left]

        right_values = cls.extract_right(ctx)
        min_key_right = list(right_values.keys())[0]
        min_dist_right = right_values[min_key_right]
        max_key_right = list(right_values.keys())[1]
        max_dist_right = right_values[max_key_right]

        return {
            cls.name: min(max_dist_left, max_dist_right),
            f"{cls.name}_min": min(min_dist_left, min_dist_right),
        }


class DistanceToLeftRoadBoundary(_AbstractDistanceToLaneSide):
    name = "distance_to_road_left"

    @classmethod
    def get_lane(cls, ctx: StateContext) -> Lane:
        vehicle = ctx.vehicle(0)
        return _find_left_most_real_lane_for_vehicle_at_time_step(
            vehicle, ctx.time_step, ctx.world.road_network
        )

    @classmethod
    def get_side(cls, ctx: StateContext) -> _AbstractDistanceToLaneSide.Side:
        return _AbstractDistanceToLaneSide.Side.LEFT


class DistanceToRightRoadBoundary(_AbstractDistanceToLaneSide):
    name = "distance_to_road_right"

    @classmethod
    def get_lane(cls, ctx: StateContext) -> Lane:
        vehicle = ctx.vehicle(0)
        return _find_right_most_real_lane_for_vehicle_at_time_step(
            vehicle, ctx.time_step, ctx.world.road_network
        )

    @classmethod
    def get_side(cls, ctx: StateContext) -> _AbstractDistanceToLaneSide.Side:
        return _AbstractDistanceToLaneSide.Side.RIGHT


class DistanceToRoadBoundary(_AbstractDistanceToLane):
    name = "distance_to_road"

    @classmethod
    def extract_left(cls, ctx: StateContext) -> dict[str, float]:
        return DistanceToLeftRoadBoundary.extract(ctx)

    @classmethod
    def extract_right(cls, ctx: StateContext) -> dict[str, float]:
        return DistanceToRightRoadBoundary.extract(ctx)


class DistanceToRefLaneLeft(_AbstractDistanceToLaneSide):
    name = "distance_to_ref_lane_left"

    @classmethod
    def get_lane(cls, ctx: StateContext) -> Lane:
        return ctx.vehicle(0).lane_at_time_step(ctx.time_step)

    @classmethod
    def get_side(cls, ctx: StateContext) -> _AbstractDistanceToLaneSide.Side:
        return _AbstractDistanceToLaneSide.Side.LEFT


class DistanceToRefLaneRight(_AbstractDistanceToLaneSide):
    name = "distance_to_ref_lane_right"

    @classmethod
    def get_lane(cls, ctx: StateContext) -> Lane:
        return ctx.vehicle(0).lane_at_time_step(ctx.time_step)

    @classmethod
    def get_side(cls, ctx: StateContext) -> _AbstractDistanceToLaneSide.Side:
        return _AbstractDistanceToLaneSide.Side.RIGHT


class DistanceToRefLane(_AbstractDistanceToLane):
    name = "distance_to_ref_lane"

    @classmethod
    def extract_left(cls, ctx: StateContext) -> dict[str, float]:
        return DistanceToRefLaneLeft.extract(ctx)

    @classmethod
    def extract_right(cls, ctx: StateContext) -> dict[str, float]:
        return DistanceToRefLaneRight.extract(ctx)


def _is_main_carriage_way_lane(world: World, lane: Lane) -> bool:
    """Check whether a lane is part of the main carriage way.

    A lane is a main carriage way lane, if all of its lanelet segments are of lanelet type `MAIN_CARRIAGE_WAY`.

    :param world: Reference `World`, where the `lane` is located.
    :param lane: `Lane` inside `world`.

    :returns: Whether `lane` is part of the main carriage way.
    """
    segment_lanelets = [
        world.road_network.lanelet_network.find_lanelet_by_id(segment_id)
        for segment_id in lane.segment_ids
    ]
    return all(
        LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type for lanelet in segment_lanelets
    )


class DistanceToMainCarriageWayBound(_AbstractDistanceToLaneSide):
    name = "distance_to_main_carriage_way_bound"

    @classmethod
    def _get_main_carriage_way_bound_lane(cls, ctx: StateContext) -> Lane:
        main_carriage_way_bound_lane = _find_right_most_real_lane_for_vehicle_at_time_step(
            ctx.vehicle(0), ctx.time_step, ctx.world.road_network
        )
        while not _is_main_carriage_way_lane(ctx.world, main_carriage_way_bound_lane):
            if main_carriage_way_bound_lane.adj_left is None:
                raise RuntimeError(
                    f"Failed to find the main carriage way in scenario {ctx.scenario.scenario_id}!"
                )

            main_carriage_way_bound_lane_id = main_carriage_way_bound_lane.adj_left
            main_carriage_way_bound_lane = ctx.world.road_network.lane_by_id(
                main_carriage_way_bound_lane_id
            )
        return main_carriage_way_bound_lane

    @classmethod
    def get_left_lane(cls, ctx: StateContext) -> Lane:
        return cls._get_main_carriage_way_bound_lane(ctx)

    @classmethod
    def get_right_lane(cls, ctx: StateContext) -> Lane:
        return cls._get_main_carriage_way_bound_lane(ctx)


class DistanceToShoulderFeatureVariable(_AbstractDistanceToLane):
    name = "distance_to_shoulder_lane"

    @classmethod
    def get_lane(cls, ctx: StateContext) -> Lane:
        return ctx.world.road_network.right_most_real_lane

    @classmethod
    def extract_left(cls, ctx: StateContext) -> _AbstractDistanceToLaneSide.Side: ...


# TODO: Implemented intersection feature variables.
# class AbstractDistanceToIntersectionLanes(ABC):
#     @staticmethod
#     def long_distance_to_intersection_lanes(
#         x: float,
#         y: float,
#         occ_points: Iterable[np.ndarray],
#         road_network: RoadNetwork,
#         successors_id: List | Set,
#         lane: Lane,
#     ) -> list[float]:
#         # get longitudinal state of enter and exit of intersection
#         enter_s, exit_s = road_network.get_lanelets_start_end(successors_id, lane)
#         # get longitudinal state of vehicle
#         s_min, s_max = lane.min_max_longitudinal_distance(occ_points)
#         vehicle_s = lane.longitudinal_distance(x, y)
#         return [
#             s_max - enter_s,
#             exit_s - s_min,
#             vehicle_s - enter_s,
#             exit_s - vehicle_s,
#             min(s_max - enter_s, exit_s - s_min),
#             min(vehicle_s - enter_s, exit_s - vehicle_s),
#         ]

#     @staticmethod
#     def lat_distance_to_intersection_lanes(
#         x: float,
#         y: float,
#         occ_points: Iterable[np.ndarray],
#         lane: Lane,
#     ) -> List[float]:
#         # get lateral state of vehicle
#         d_min_right, d_max_right = lane.min_max_distance_to_right(occ_points)
#         d_right = lane.distance_to_right(x, y)

#         d_min_left, d_max_left = lane.min_max_distance_to_left(occ_points)
#         d_left = lane.distance_to_left(x, y)

#         vehicle_s = lane.longitudinal_distance(x, y)
#         lane_width = lane.width(vehicle_s)
#         return [
#             d_right,
#             d_max_right,
#             d_left,
#             d_max_left,
#             min(d_right, d_left),
#             min(d_max_right, d_max_left),
#         ]

#     @staticmethod
#     def distance_stop_line(
#         x: float,
#         y: float,
#         occ_points: Iterable[np.ndarray],
#         road_network: RoadNetwork,
#         lane: Lane,
#         successors_id: Union[List, Set],
#     ) -> List[float]:
#         s_stop_lines = road_network.get_long_distance_stop_lines_from_lane(lane)
#         # TODO: how to define if there is no stop line
#         if len(s_stop_lines) == 0:
#             return [50.0, 50.0]
#         _, s_max = lane.min_max_longitudinal_distance(occ_points)
#         vehicle_s = lane.longitudinal_distance(x, y)
#         closest_distance_index = np.argmin(abs(np.array(s_stop_lines) - s_max))
#         front_distance_stop_line = s_stop_lines[closest_distance_index] - s_max
#         state_distance_stop_line = s_stop_lines[closest_distance_index] - vehicle_s
#         if front_distance_stop_line >= 0:
#             front_distance_stop_line = min(front_distance_stop_line, 50.0)
#         else:
#             front_distance_stop_line = max(front_distance_stop_line, -50.0)
#         if state_distance_stop_line >= 0:
#             state_distance_stop_line = min(state_distance_stop_line, 50.0)
#         else:
#             state_distance_stop_line = max(state_distance_stop_line, -50.0)
#         return [front_distance_stop_line, state_distance_stop_line]


# class AbstractIntersectionTrafficLightSign(ABC):
#     evaluation_index = {
#         TrafficSignIDGermany.ADDITION_LEFT_TURNING_PRIORITY_WITH_OPPOSITE_RIGHT_YIELD: 1.0,
#         TrafficSignIDGermany.ADDITION_LEFT_TURNING_PRIORITY_WITH_OPPOSITE_YIELD: 2.0,
#         TrafficSignIDGermany.ADDITION_LEFT_TURNING_PRIORITY_WITH_RIGHT_YIELD: 3.0,
#         TrafficSignIDGermany.ADDITION_RIGHT_TURNING_PRIORITY_WITH_OPPOSITE_LEFT_YIELD: 4.0,
#         TrafficSignIDGermany.ADDITION_RIGHT_TURNING_PRIORITY_WITH_OPPOSITE_YIELD: 5.0,
#         TrafficSignIDGermany.ADDITION_RIGHT_TURNING_PRIORITY_WITH_LEFT_YIELD: 6.0,
#         TrafficSignIDGermany.ADDITION_LEFT_TRAFFIC_PRIORITY_WITH_STRAIGHT_RIGHT_YIELD: 7.0,
#         TrafficSignIDGermany.ADDITION_LEFT_TRAFFIC_PRIORITY_WITH_STRAIGHT_YIELD: 8.0,
#         TrafficSignIDGermany.ADDITION_RIGHT_TRAFFIC_PRIORITY_WITH_STRAIGHT_LEFT_YIELD: 9.0,
#         TrafficSignIDGermany.ADDITION_RIGHT_TRAFFIC_PRIORITY_WITH_STRAIGHT_YIELD: 10.0,
#         TrafficSignIDGermany.PRIORITY: 11.0,
#         TrafficSignIDGermany.RIGHT_OF_WAY: 12.0,
#         TrafficSignIDGermany.YIELD: 13.0,
#         TrafficSignIDGermany.STOP: 14.0,
#         TrafficSignIDGermany.WARNING_RIGHT_BEFORE_LEFT: 15.0,
#         TrafficSignIDGermany.GREEN_ARROW: 16.0,
#         TrafficSignIDGermany.MAX_SPEED: 17.0,
#     }
#     stop_traffic_sign_deu = TrafficSignIDGermany.STOP

#     @staticmethod
#     def has_traffic_light(
#         lanelets_id: "Union[np.array, List]", road_network: "RoadNetwork"
#     ) -> "List[float]":
#         for l_id in lanelets_id:
#             if road_network.check_traffic_light_relevant(l_id):
#                 return [1.0]
#         return [-1.0]

#     @classmethod
#     def relevant_traffic_sign(
#         cls, lanelets_id: "Union[np.array, List]", road_network: "RoadNetwork"
#     ) -> "List[float]":
#         traffic_signs = road_network.get_traffic_sign_type(lanelets_id)
#         traffic_sign_eval_index = [cls.evaluation_index[ts] for ts in traffic_signs]
#         if len(traffic_sign_eval_index) == 0:
#             return [-1.0]
#         min_eval_index = np.min(traffic_sign_eval_index)
#         return [min_eval_index]

#     @classmethod
#     def relevant_traffic_sign_stop(
#         cls, lanelets_id: "Union[np.array, List]", road_network: "RoadNetwork"
#     ) -> "List[float]":
#         for lanelet_id in lanelets_id:
#             traffic_sign_element_stop = road_network.get_traffic_sign_element(
#                 lanelet_id, cls.stop_traffic_sign_deu
#             )
#             if traffic_sign_element_stop is not None:
#                 return [1.0]
#         return [-1.0]


# class DistanceToLanes(AbstractUnaryFeatureVariable, AbstractDistanceToLanes):
#     names = [
#         "min_distance_to_road_left",
#         "distance_to_road_left",
#         "max_distance_to_road_left",
#         "min_distance_to_road_right",
#         "distance_to_road_right",
#         "max_distance_to_road_right",
#         "distance_to_road",
#         "distance_to_road_min",
#         "min_distance_to_ref_lane_left",
#         "distance_to_ref_lane_left",
#         "max_distance_to_ref_lane_left",
#         "min_distance_to_ref_lane_right",
#         "distance_to_ref_lane_right",
#         "max_distance_to_ref_lane_right",
#         "distance_to_ref_lane",
#         "distance_to_ref_lane_min",
#         "min_distance_to_occ_lanes_left",
#         "distance_to_occ_lanes_left",
#         "max_distance_to_occ_lanes_left",
#         "min_distance_to_occ_lanes_right",
#         "distance_to_occ_lanes_right",
#         "max_distance_to_occ_lanes_right",
#         "distance_to_occ_lanes",
#         "distance_to_occ_lanes_min",
#     ]

#     @classmethod
#     @with_state_context
#     def evaluate(cls, ctx: StateContext) -> Dict[str, float]:
#         vehicle = ctx.vehicle(0)
#         shape = ctx.obstacle(0).obstacle_shape
#         state = ctx.state(0)

#         occ_points = list(rotate_translate(shape.vertices[:-1], [state.s, state.d], state.theta))

#         # road
#         road_left_lane = _find_left_most_real_lane_for_vehicle_at_time_step(
#             vehicle, state.ts, ctx.world.road_network
#         )  # the left most real lane (not phantom!)
#         road_right_lane = _find_right_most_real_lane_for_vehicle_at_time_step(
#             vehicle, state.ts, ctx.world.road_network
#         )  # the right most real lane (not phantom!)
#         # ref
#         ref_lane = vehicle.trajectory_persp.get_reference_lane(
#             state.ts
#         )  # the reference lane (uses the vehicle's center)
#         # occ
#         occ_lanes = vehicle.trajectory_persp.get_occupied_lanes(
#             state.ts
#         )  # list of all occupied lanes
#         occ_left_lane = ctx.world.road_network.left_most_lane(occ_lanes)[
#             0
#         ]  # the left most occupied lane
#         occ_right_lane = ctx.world.road_network.right_most_lane(occ_lanes)[
#             0
#         ]  # the right most occupied lane

#         values = cls.distance_to_lanes(
#             state.s, state.d, occ_points, road_left_lane, road_right_lane
#         )
#         values += cls.distance_to_lanes(state.s, state.d, occ_points, ref_lane, ref_lane)
#         values += cls.distance_to_lanes(state.s, state.d, occ_points, occ_left_lane, occ_right_lane)

#         return dict(zip(cls.names, values))


# class DistanceToIntersection(
#     AbstractUnaryFeatureVariable,
#     AbstractDistanceToIntersectionLanes,
#     AbstractIntersectionTrafficLightSign,
# ):
#     names = [
#         # reference longitudinal
#         "vehicle_front_to_intersection_enter_ref",
#         "vehicle_rear_to_intersection_exit_ref",
#         "state_to_intersection_enter_ref",
#         "state_to_intersection_exit_ref",
#         "shape_inside_intersection_ref",
#         "state_inside_intersection_ref",
#         # reference lateral
#         "state_to_right_bound_ref",
#         "max_shape_to_right_bound_ref",
#         "state_to_left_bound_ref",
#         "max_shape_to_left_bound_ref",
#         "state_to_lane_ref",
#         "shape_to_lane_ref",
#         # road base (stop line, traffic light, traffic sign ...)
#         "relevant_traffic_sign",
#         "vehicle_front_to_stop_line_ref",
#         "state_to_stop_line_ref",
#     ]

#     @classmethod
#     @with_state_context
#     def evaluate(cls, ctx: StateContext) -> "Dict[str, float]":
#         road_network = ctx.world.road_network
#         # state in world frame
#         state = ctx.state(0).get_state_in_world_frame()
#         # ego vehicle
#         vehicle = ctx.vehicle(0)
#         shape = ctx.obstacle(0).obstacle_shape
#         occ_points = list(rotate_translate(shape.vertices[:-1], [state.s, state.d], state.theta))
#         # find turning lanes
#         right_turning_lane = road_network.find_lanes_incoming_by_id(vehicle.incoming.incoming_id)[0]
#         straight_going_lane = road_network.find_lanes_incoming_by_id(vehicle.incoming.incoming_id)[
#             1
#         ]
#         left_turning_lane = road_network.find_lanes_incoming_by_id(vehicle.incoming.incoming_id)[2]
#         ref_path_lane = vehicle.trajectory_persp.get_reference_lane_intersection(state.ts)

#         values = list()
#         # reference lane
#         intersection_lanelets_id = vehicle.incoming.successors_left.union(
#             vehicle.incoming.successors_right
#         ).union(vehicle.incoming.successors_straight)
#         ref_intersection_lanelet_id = list(
#             intersection_lanelets_id.intersection(ref_path_lane.segment_ids)
#         )
#         values += cls.long_distance_to_intersection_lanes(
#             state.s, state.d, occ_points, road_network, ref_intersection_lanelet_id, ref_path_lane
#         )
#         values += cls.lat_distance_to_intersection_lanes(
#             state.s, state.d, occ_points, ref_path_lane
#         )

#         # check traffic light and sign
#         all_lanelets_id = np.unique(
#             right_turning_lane.segment_ids
#             + straight_going_lane.segment_ids
#             + left_turning_lane.segment_ids
#         )

#         values += cls.relevant_traffic_sign(all_lanelets_id, road_network)

#         # distance to stop line
#         values += cls.distance_stop_line(
#             state.s, state.d, occ_points, road_network, ref_path_lane, ref_intersection_lanelet_id
#         )

#         return dict(zip(cls.names, values))


# class RelativeStateVariable(AbstractBinaryFeatureVariable):
#     names = [
#         "distance",
#         "lateral_distance",
#         "relative_longitudinal_velocity",
#         "relative_lateral_velocity",
#     ]

#     @classmethod
#     @with_state_context
#     def evaluate(cls, ctx: StateContext) -> "Dict[str, float]":
#         ego_state = ctx.state(0)
#         other_state = ctx.state(1)
#         values = [
#             other_state.rear_s - ego_state.front_s,
#             ego_state.d - other_state.d,
#             other_state.v * np.cos(other_state.theta) - ego_state.v * np.cos(ego_state.theta),
#             other_state.v * np.sin(other_state.theta) - ego_state.v * np.sin(ego_state.theta),
#         ]
#         return dict(zip(cls.names, values))


# class RelativeIntersectionStateVariable(AbstractBinaryFeatureVariable):
#     names = [
#         "distance",
#         "lateral_distance",
#         "relative_velocity",
#         "relative_longitudinal_velocity",
#         "relative_lateral_velocity",
#     ]

#     @classmethod
#     @with_state_context
#     def evaluate(cls, ctx: StateContext) -> "Dict[str, float]":
#         ego_state = ctx.state(0)
#         other_state = ctx.state(0)
#         values = [
#             other_state.rear_s - ego_state.front_s,
#             # TODO: need min lateral distance?
#             ego_state.d - other_state.d,
#             other_state.v - ego_state.v,
#             other_state.v * np.cos(other_state.theta) - ego_state.v * np.cos(ego_state.theta),
#             other_state.v * np.sin(other_state.theta) - ego_state.v * np.sin(ego_state.theta),
#         ]
#         return dict(zip(cls.names, values))


# class RelativeDistanceToLanes(AbstractBinaryFeatureVariable, AbstractDistanceToLanes):
#     names = [
#         "relative_min_distance_to_ref_lane_left",
#         "relative_distance_to_ref_lane_left",
#         "relative_max_distance_to_ref_lane_left",
#         "relative_min_distance_to_ref_lane_right",
#         "relative_distance_to_ref_lane_right",
#         "relative_max_distance_to_ref_lane_right",
#         "relative_distance_to_ref_lane",
#         "relative_distance_to_ref_lane_min",
#         "relative_min_distance_to_occ_lanes_left",
#         "relative_distance_to_occ_lanes_left",
#         "relative_max_distance_to_occ_lanes_left",
#         "relative_min_distance_to_occ_lanes_right",
#         "relative_distance_to_occ_lanes_right",
#         "relative_max_distance_to_occ_lanes_right",
#         "relative_distance_to_occ_lanes",
#         "relative_distance_to_occ_lanes_min",
#     ]

#     @classmethod
#     @with_state_context
#     def evaluate(cls, ctx: StateContext) -> dict[str, float]:
#         ego_state = ctx.state(0).get_state_in_world_frame()
#         ego_vehicle, other_vehicle = ctx.vehicle(0), ctx.vehicle(1)
#         shape = ctx.obstacle(0).obstacle_shape
#         occ_points = list(
#             rotate_translate(shape.vertices[:-1], [ego_state.s, ego_state.d], ego_state.theta)
#         )

#         # ref
#         ref_lane = other_vehicle.trajectory_persp.get_reference_lane(ego_state.ts)
#         # occ
#         occ_lanes = other_vehicle.trajectory_persp.get_occupied_lanes(ego_state.ts)
#         occ_left_lane = ctx.world.road_network.left_most_lane(occ_lanes)[0]
#         occ_right_lane = ctx.world.road_network.right_most_lane(occ_lanes)[0]

#         values = cls.distance_to_lanes(ego_state.s, ego_state.d, occ_points, ref_lane, ref_lane)
#         values += cls.distance_to_lanes(
#             ego_state.s, ego_state.d, occ_points, occ_left_lane, occ_right_lane
#         )

#         return dict(zip(cls.names, values))


# class RelativeDistanceToIntersection(
#     AbstractBinaryFeatureVariable, AbstractDistanceToIntersectionLanes
# ):
#     names = [
#         # relative reference longitudinal
#         "relative_vehicle_front_to_intersection_enter_ref",
#         "relative_vehicle_rear_to_intersection_exit_ref",
#         "relative_state_to_intersection_enter_ref",
#         "relative_state_to_intersection_exit_ref",
#         "relative_shape_inside_intersection_ref",
#         "relative_state_inside_intersection_ref",
#         # relative reference lateral
#         "relative_state_to_right_bound_ref",
#         "relative_max_shape_to_right_bound_ref",
#         "relative_state_to_left_bound_ref",
#         "relative_max_shape_to_left_bound_ref",
#         "relative_state_to_lane_ref",
#         "relative_shape_to_lane_ref",
#     ]

#     @classmethod
#     @with_state_context
#     def evaluate(cls, ctx: StateContext) -> Dict[str, float]:
#         ego_state = ctx.state(0).get_state_in_world_frame()
#         ego_vehicle = ctx.vehicle(0)
#         other_vehicle = ctx.vehicle(1)
#         shape = ctx.obstacle(0).obstacle_shape
#         occ_points = list(
#             rotate_translate(shape.vertices[:-1], [ego_state.s, ego_state.d], ego_state.theta)
#         )

#         # ref
#         # TODO: check get_reference_path
#         ref_lane = other_vehicle.trajectory_persp.get_reference_lane_intersection(ego_state.ts)
#         incoming_successors = other_vehicle.incoming.successors_right.union(
#             other_vehicle.incoming.successors_straight, other_vehicle.incoming.successors_left
#         )
#         intersection_successor = incoming_successors.intersection(ref_lane.segment_ids)
#         values = cls.long_distance_to_intersection_lanes(
#             ego_state.s,
#             ego_state.d,
#             occ_points,
#             ctx.world.road_network,
#             intersection_successor,
#             ref_lane,
#         )
#         values += cls.lat_distance_to_intersection_lanes(
#             ego_state.s, ego_state.d, occ_points, ref_lane
#         )
#         return dict(zip(cls.names, values))


class FeatureVariableAgentCombination(enum.Enum):
    EGO = "ego"
    OTHER = "other"
    EGO_OTHER = "ego_other"
    OTHER_EGO = "other_ego"

    @property
    def arity(self) -> int:
        match self:
            case FeatureVariableAgentCombination.EGO:
                return 1
            case (
                FeatureVariableAgentCombination.OTHER
                | FeatureVariableAgentCombination.EGO_OTHER
                | FeatureVariableAgentCombination.OTHER_EGO
            ):
                return 2

    def is_reversed(self) -> bool:
        return (
            self == FeatureVariableAgentCombination.OTHER
            or self == FeatureVariableAgentCombination.OTHER_EGO
        )


DesiredFeatureVariables = Mapping[
    FeatureVariableAgentCombination, Iterable[type[AbstractFeatureVariable]]
]

DEFAULT_INTERSTATE_FEATURE_VARIABLES = {
    FeatureVariableAgentCombination.EGO: [
        PositionFeatureVariable,
        VelocityFeatureVariable,
        AccelerationFeatureVariable,
        JerkFeatureVariable,
        JerkDotFeatureVariable,
        LateralPositionFeatureVariable,
        OrientationFeatureVariable,
        CurvatureFeatureVariable,
        CurvatureDotFeatureVariable,
        CurvatureDdotFeatureVariable,
        VehicleLengthFeatureVariable,
        VehicleWidthFeatureVariable,
        DistanceToLeftRoadBoundary,
        DistanceToRightRoadBoundary,
        DistanceToRefLaneLeft,
        DistanceToRefLaneRight,
    ],
    FeatureVariableAgentCombination.OTHER: [
        PositionFeatureVariable,
        VelocityFeatureVariable,
        AccelerationFeatureVariable,
        JerkFeatureVariable,
        LateralPositionFeatureVariable,
        OrientationFeatureVariable,
        CurvatureFeatureVariable,
        CurvatureDotFeatureVariable,
        VehicleLengthFeatureVariable,
        VehicleWidthFeatureVariable,
        DistanceToLeftRoadBoundary,
        DistanceToRightRoadBoundary,
        DistanceToRefLaneLeft,
        DistanceToRefLaneRight,
    ],
    FeatureVariableAgentCombination.EGO_OTHER: [
        RelativeDistanceFeatureVariable,
        RelativeLateralDistanceFeatureVariable,
        RelativeLongitudinalVelocityFeatureVariable,
        RelativeLateralVelocityFeatureVariable,
    ],
}

DEFAULT_INTERSECTION_FEATURE_VARIABLES = {
    FeatureVariableAgentCombination.EGO: {
        PositionFeatureVariable,
        VelocityFeatureVariable,
    },
    FeatureVariableAgentCombination.OTHER: {
        PositionFeatureVariable,
        VelocityFeatureVariable,
    },
}


def default_feature_variable_classes_for_scenario_type(
    scenario_type: ScenarioType,
) -> dict[FeatureVariableAgentCombination, Iterable[type[AbstractFeatureVariable]]]:
    if scenario_type == ScenarioType.INTERSTATE:
        return DEFAULT_INTERSTATE_FEATURE_VARIABLES
    else:
        return DEFAULT_INTERSECTION_FEATURE_VARIABLES


def get_all_available_feature_variables() -> dict[str, AbstractFeatureVariable]:
    classmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    # Simple pre-filter to sort out other types, e.g., type aliases which are also considered
    # classes by `inspect.isclass` but not by `issubclass`...
    classes_with_name = filter(lambda cls: hasattr(cls[1], "name"), classmembers)

    classes_with_feature_variable_base = filter(
        lambda cls: issubclass(cls[1], AbstractFeatureVariable), classes_with_name
    )
    public_classes = filter(
        lambda cls: not cls[0].startswith("_"), classes_with_feature_variable_base
    )
    non_abstract_classes = filter(lambda cls: not inspect.isabstract(cls[1]), public_classes)

    feature_variables = {}
    for feature_variable_tuple in non_abstract_classes:
        feature_variables[feature_variable_tuple[1].name] = feature_variable_tuple[1]

    return feature_variables
