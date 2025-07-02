import copy
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Union

import commonroad_clcs.pycrccosy as pycrccosy
import numpy as np
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState, InitialState, State
from commonroad.scenario.trajectory import Trajectory
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.util import (
    compute_orientation_from_polyline,
    compute_pathlength_from_polyline,
)
from commonroad_dc.feasibility.feasibility_checker import InputState
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, VehicleType
from commonroad_route_planner.route_planner import RoutePlanner
from omegaconf import DictConfig
from shapely import affinity, unary_union
from shapely.geometry import Point, Polygon
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

from crmonitor.common.road_network import Lane, RoadNetwork

rot_mat_factors = np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])

_LOGGER = logging.getLogger(__name__)

# The custom vehicle dynamics are used here, because for low velocity vehicles
# the model-predictive sampling was not producing any feasible states.
# Increasing the steering velocity bounds (by factor of 200...) yields more feasible states.
# However, it is not clear whether this has other unintended consequences.
# TODO: Find out whether the vehicle dynamics are the problem here, or if its a problem with the sampling.
CUSTOM_DEFAULT_VEHICLE_DYNAMICS = VehicleDynamics.KS(VehicleType.BMW_320i)
CUSTOM_DEFAULT_VEHICLE_DYNAMICS.parameters.steering.v_min = -80
CUSTOM_DEFAULT_VEHICLE_DYNAMICS.parameters.steering.v_max = 80


# @numba.njit
# todo: the decorator is removed as it might take a lot of time to initialize
def calc_s(s, width, length, theta):
    s = (
        rot_mat_factors[0] * length / 2.0 * np.cos(theta)
        - rot_mat_factors[1] * width / 2 * np.sin(theta)
        + s
    )
    return s


class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """

    __slots__ = ["s", "v", "a", "j", "j_dot"]

    def __init__(self, **kwargs):
        """Elements of state vector are determined during runtime."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def attributes(self) -> List[str]:
        """Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = "\n"
        for attr in self.attributes:
            state += attr
            state += "= {}\n".format(self.__getattribute__(attr))
        return state


class StateLateral:
    """
    Lateral state in curvilinear coordinate system
    """

    __slots__ = ["d", "theta", "kappa", "kappa_dot", "kappa_dot_dot"]

    def __init__(self, **kwargs):
        """Elements of state vector are determined during runtime."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def attributes(self) -> List[str]:
        """Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = "\n"
        for attr in self.attributes:
            state += attr
            state += "= {}\n".format(self.__getattribute__(attr))
        return state


@dataclass
class CurvilinearVehicleState(CustomState):
    position: np.ndarray = None
    velocity: float = None
    acceleration: float = None
    orientation: float = None
    steering_angle: float = None
    steering_angle_speed: float = None
    s: float = None
    d: float = None
    jerk: float = None
    jerk_dot: float = None
    theta: float = None
    kappa: float = None
    kappa_dot: float = None
    kappa_ddot: float = None


class CurvilinearVehicleTrajectory:
    def __init__(
        self,
        initial_time_step: int,
        final_time_step: int,
        dt: float,
        s: np.ndarray,
        d: np.ndarray,
        v: np.ndarray | None,
        a: np.ndarray | None,
        kappa: np.ndarray | None,
        vehicle_dynamics: VehicleDynamics = CUSTOM_DEFAULT_VEHICLE_DYNAMICS,
    ) -> None:
        self.initial_time_step = initial_time_step
        self.final_time_step = final_time_step

        self._dt = dt
        self._l_wb = vehicle_dynamics.parameters.a + vehicle_dynamics.parameters.b

        self._s = s
        self._d = d
        self._polyline = np.array([self._s, self._d]).T
        self._pathlength = compute_pathlength_from_polyline(self._polyline)

        self._v = self._compute_v_from_pathlength(self._pathlength) if v is None else v
        self._a = self._compute_a_from_v(self._v) if a is None else a
        self._jerk = self._compute_jerk_from_a(self._a)
        self._jerk_dot = self._compute_j_dot_from_j(self._jerk)
        self._theta = compute_orientation_from_polyline(self._polyline)
        self._kappa = pycrccosy.Util.compute_curvature(self._polyline) if kappa is None else kappa
        self._kappa_dot = self._compute_kappa_dot_from_kappa(self._kappa)
        self._kappa_ddot = self._compute_kappa_ddot_from_kappa_dot(self._kappa_dot)
        self._steering_angle = self._compute_steering_angle_from_kappa(self._kappa)
        self._steering_angle_speed = self._compute_steering_angle_speed_from_kappa_and_kappa_dot(
            self._kappa, self._kappa_dot
        )

    def _time_step_to_index(self, time_step: int) -> int:
        if time_step > self.final_time_step:
            raise ValueError()
        if time_step < self.initial_time_step:
            raise ValueError()

        return time_step - self.initial_time_step

    def s(self, time_step: int) -> float:
        return self._s[self._time_step_to_index(time_step)]

    def d(self, time_step: int) -> float:
        return self._d[self._time_step_to_index(time_step)]

    def position(self, time_step: int, clcs: CurvilinearCoordinateSystem) -> np.ndarray:
        position = clcs.convert_to_cartesian_coords(self.s(time_step), self.d(time_step))
        return position

    def velocity(self, time_step: int) -> float:
        return self._v[self._time_step_to_index(time_step)]

    def acceleration(self, time_step: int) -> float:
        return self._a[self._time_step_to_index(time_step)]

    def orientation(self, time_step: int) -> float:
        return self._theta[self._time_step_to_index(time_step)]

    def steering_angle(self, time_step: int) -> float:
        return self._steering_angle[self._time_step_to_index(time_step)]

    def steering_angle_speed(self, time_step: int) -> float:
        return self._steering_angle_speed[self._time_step_to_index(time_step)]

    def state_at_time_step(
        self, time_step: int, clcs: CurvilinearCoordinateSystem
    ) -> CurvilinearVehicleState:
        return CurvilinearVehicleState(
            time_step=time_step,
            position=self.position(time_step, clcs),
            velocity=self.velocity(time_step),
            acceleration=self.acceleration(time_step),
            orientation=self.orientation(time_step),
        )

    def initial_state_at_time_step(
        self, time_step: int, clcs: CurvilinearCoordinateSystem
    ) -> InitialState:
        return InitialState(
            time_step=int(time_step),
            position=self.position(time_step, clcs),
            velocity=self.velocity(time_step),
            orientation=self.orientation(time_step),
            acceleration=self.acceleration(time_step),
        )

    def input_state_at_time_step(
        self, time_step: int, clcs: CurvilinearCoordinateSystem
    ) -> InputState:
        return InputState(
            time_step=int(time_step),
            steering_angle_speed=self.steering_angle_speed(time_step),
            acceleration=self.acceleration(time_step),
        )

    def _convert_to_commonroad_trajectory(
        self,
        state_converter,
        clcs: CurvilinearCoordinateSystem,
        initial_time_step: int | None = None,
        final_time_step: int | None = None,
    ) -> Trajectory:
        if initial_time_step is None:
            initial_time_step = self.initial_time_step

        if final_time_step is None:
            final_time_step = self.final_time_step

        state_list = [
            state_converter(time_step, clcs)
            for time_step in range(initial_time_step, final_time_step + 1)
        ]

        return Trajectory(int(initial_time_step), state_list)

    def convert_to_commonroad_input_trajectory(
        self,
        clcs: CurvilinearCoordinateSystem,
        initial_time_step: int | None = None,
        final_time_step: int | None = None,
    ) -> Trajectory:
        return self._convert_to_commonroad_trajectory(
            self.input_state_at_time_step, clcs, initial_time_step, final_time_step
        )

    def convert_to_commonroad_trajectory(self, clcs: CurvilinearCoordinateSystem) -> Trajectory:
        return self._convert_to_commonroad_trajectory(self.state_at_time_step, clcs)

    def _compute_v_from_pathlength(self, pathlength: np.ndarray) -> np.ndarray:
        return np.gradient(pathlength, self._dt)

    def _compute_a_from_v(self, v: np.ndarray) -> np.ndarray:
        return np.gradient(v, self._dt)

    def _compute_jerk_from_a(self, a: np.ndarray) -> np.ndarray:
        return np.gradient(a, self._dt)

    def _compute_j_dot_from_j(self, j: np.ndarray) -> np.ndarray:
        return np.gradient(j, self._dt)

    def _compute_kappa_dot_from_kappa(self, kappa: np.ndarray) -> np.ndarray:
        return np.gradient(kappa, self._pathlength)

    def _compute_kappa_ddot_from_kappa_dot(self, kappa_dot: np.ndarray) -> np.ndarray:
        return np.gradient(kappa_dot, self._pathlength)

    def _compute_steering_angle_from_kappa(self, kappa: np.ndarray) -> np.ndarray:
        return np.arctan(kappa * self._l_wb)

    def _compute_steering_angle_speed_from_kappa_and_kappa_dot(
        self, kappa: np.ndarray, kappa_dot: np.ndarray
    ) -> np.ndarray:
        return self._l_wb * kappa_dot / (1 + self._l_wb**2 * kappa**2)


class Input:
    """
    Lateral and longitudinal vehicle input
    """

    __slots__ = ["a", "kappa_dot_dot"]

    @property
    def attributes(self) -> List[str]:
        """Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = "\n"
        for attr in self.attributes:
            state += attr
            state += "= {}\n".format(self.__getattribute__(attr))
        return state


@dataclass
class CurvilinearStateManager:
    """
    Manage, cache curvilinear states
    """

    road_network: RoadNetwork
    cache: Dict[Tuple[State, Lane], Tuple[StateLongitudinal, StateLateral]] = field(
        default_factory=dict
    )

    @staticmethod
    def _compute_curvilinear_state(
        state: State, lane: Lane
    ) -> Optional[Tuple[StateLongitudinal, StateLateral]]:
        try:
            s, d = lane.clcs.convert_to_curvilinear_coords(*state.position)
        except ValueError:
            _LOGGER.debug("Vehicle out of projection domain: consider large clcs")
            try:
                s, d = lane.clcs_large_step.convert_to_curvilinear_coords(*state.position)
            except ValueError:
                _LOGGER.debug("Vehicle out of projection domain: State will not be considered")
                return None
        # Originally, the speed was calcuclated as the magnitude of the combined directed vector of velocity and velocity_y.
        # However, this resulted in issues because the orientation of the vehicle was not considered relative to the lane orientation.
        # This could result in negative velocities (=reversing) even though the vehicle was driving forward (https://gitlab.lrz.de/cps/commonroad/commonroad-stl-monitor/-/issues/59).
        # As velocity_y is usually very small, and velocity is good enough, we can also simply use the velocity.
        speed = state.velocity
        if hasattr(state, "acceleration"):
            # Similarly to speed, this was originally calculated from acceleration and acceleration_y.
            accel = state.acceleration
            if hasattr(state, "jerk"):
                if hasattr(state, "jerk_dot"):
                    x_lon = StateLongitudinal(
                        s=s,
                        v=speed,
                        a=accel,
                        j=state.jerk,
                        j_dot=state.jerk_dot,
                    )
                else:
                    x_lon = StateLongitudinal(s=s, v=speed, a=accel, j=state.jerk)
            else:
                x_lon = StateLongitudinal(s=s, v=speed, a=accel)
        else:
            x_lon = StateLongitudinal(s=s, v=speed)

        # Make sure the resulting theta lies in [-pi, +pi].
        theta_cl = lane.orientation(s) % (2 * math.pi)
        orientation = state.orientation % (2 * math.pi)
        theta = (orientation - theta_cl + math.pi) % (2 * math.pi) - math.pi
        if (
            hasattr(state, "kappa")
            and hasattr(state, "kappa_dot")
            and hasattr(state, "kappa_dot_dot")
        ):
            x_lat = StateLateral(
                d=d,
                theta=theta,
                kappa=state.kappa,
                kappa_dot=state.kappa_dot,
                kappa_dot_dot=state.kappa_dot_dot,
            )
        elif hasattr(state, "kappa") and hasattr(state, "kappa_dot"):
            x_lat = StateLateral(
                d=d,
                theta=theta,
                kappa=state.kappa,
                kappa_dot=state.kappa_dot,
            )
        elif hasattr(state, "kappa"):
            x_lat = StateLateral(d=d, theta=theta, kappa=state.kappa)
        else:
            x_lat = StateLateral(d=d, theta=theta)
        return x_lon, x_lat

    def get_curvilinear_state(
        self, state: State, lane: Lane
    ) -> Tuple[StateLongitudinal, StateLateral]:
        """

        :param state:
        :param lane: Reference lane
        :return:
        """
        key = (state.time_step, lane.lane_id)
        ccosy_state = self.cache.get(key)
        if ccosy_state is None:
            ccosy_state = self._compute_curvilinear_state(state, lane)
            self.cache[key] = ccosy_state
        return ccosy_state


@dataclass
class PredicateCache:
    # Levels: time step, predicate name, agent_ids
    cache: Dict[int, Dict[str, Dict[int, float]]] = field(
        default_factory=partial(defaultdict, partial(defaultdict, dict))
    )

    def get_robustness(
        self, time_step: int, predicate_name: str, other_ids: Union[Tuple[int], int]
    ) -> Optional[float]:
        return self.cache[time_step][predicate_name].get(other_ids)

    def set_robustness(
        self,
        time_step: int,
        predicate_name: str,
        other_ids: Union[Tuple[int], int],
        robustness: float,
    ):
        self.cache[time_step][predicate_name][other_ids] = robustness

    def __contains__(self, item):
        assert isinstance(item, tuple) and len(item) == 3
        time_step, predicate_name, other_ids = item
        rob = self.cache[time_step][predicate_name].get(other_ids)
        return rob is not None

    def __getitem__(self, item):
        assert isinstance(item, tuple) and len(item) == 3
        time_step = item[0]
        predicate_name = item[1]
        ids = item[2]
        if isinstance(predicate_name, slice):
            # Only accept slice over all predicates
            assert (
                predicate_name.start is None
                and predicate_name.stop is None
                and predicate_name.step is None
            )
            return {
                n: pred_vals[ids]
                for n, pred_vals in self.cache[time_step].items()
                if len(pred_vals) > 0 and len(list(pred_vals.keys())[0]) == 1
            }
        else:
            return self.get_robustness(*item)

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and len(key) == 3
        self.set_robustness(*key, value)


# The vehicle parameters somewhat duplicate the existing paramters from commonroad-vehicle-models.
# TODO: Are they still required, or could they be merged with the paramters from commonroad-vehicle-models?
@dataclass
class VehicleParameters:
    # TODO: most of the following parameters are not used anywhere. Can we get rid of them, or only require them in the constructor to compute the speed limits?
    a_max: float = 5.0
    a_min: float = -10.5
    a_corr: float = 0.0
    v_max: float = 60.0
    v_min: float = 0.0
    j_max: float = 10.0
    j_min: float = -10.0
    t_react: float = 0.4
    fov: float = 20
    v_des: float = 30.0
    const_dist_offset: float = 0.0

    fov_speed_limit: float = 50.0
    braking_speed_limit: float = 43.0
    road_condition_speed_limit: float = 50.0

    emergency_profile: list[float] = field(default_factory=list)
    emergency_profile_num_steps_fb: float = 200.0

    dynamics_param: DictConfig = field(default_factory=parameters_vehicle2)

    def __post_init__(self) -> None:
        self.emergency_profile += [self.j_min * self.emergency_profile_num_steps_fb]

    @classmethod
    def create(cls, dt: float, vehicle_number: int = 2, **kwargs) -> "VehicleParameters":
        if vehicle_number == 1:
            dynamics_param = parameters_vehicle1()
        elif vehicle_number == 2:
            dynamics_param = parameters_vehicle2()
        elif vehicle_number == 3:
            dynamics_param = parameters_vehicle3()
        else:
            raise ValueError(f"Vehicle number {vehicle_number} is not supported.")

        # TODO: The construction with the dynamics parmaters is not clean, because the caller could also provide custom dynamic params which would conflict.
        ego_vehicle_param = cls(dynamics_param=dynamics_param, **kwargs)
        if not -1e-12 <= (Decimal(str(ego_vehicle_param.t_react)) % Decimal(str(dt))) <= 1e-12:
            raise ValueError("Reaction time must be multiple of time step size.")

        return ego_vehicle_param

    # TODO: This pattern is also not very clean, because a_max and a_min are treated special and are kind of obscure defaults.
    @classmethod
    def create_for_ego_vehicle(
        cls, dt: float, vehicle_number: int = 2, a_max=3.0, a_min=-10.0, **kwargs
    ) -> "VehicleParameters":
        return cls.create(dt, vehicle_number, a_max=a_max, a_min=a_min, **kwargs)


class Vehicle:
    def __init__(
        self,
        id,
        obstacle_type,
        vehicle_param: VehicleParameters,
        shape,
        states_cr,
        signal_series,
        ccosy_cache,
        lanelet_assignment: Dict[int, Set[int]],
        predicate_cache=None,
        road_network: Optional[RoadNetwork] = None,
        goal=None,
    ):
        self.id = id
        self.obstacle_type = obstacle_type
        self.vehicle_param = vehicle_param
        self.shape = shape
        self.states_cr = states_cr
        self.signal_series = signal_series
        self.ccosy_cache = ccosy_cache
        self.lanelet_assignment = lanelet_assignment
        self.predicate_cache = predicate_cache or PredicateCache()
        self.road_network = road_network
        if self.road_network is None:
            self.lanelets_dir = None
            self.ref_path_lane = None
            self.lanelets_dir_center_vertices = None
            self.lanelets_dir_left_vertices = None
            self.lanelets_dir_right_vertices = None
            self.incoming_intersection = None
        else:
            # intersection scenario
            (
                self.lanelets_dir,
                self.ref_path_lane,
                self.lanelets_dir_center_vertices,
                self.lanelets_dir_left_vertices,
                self.lanelets_dir_right_vertices,
            ) = self._initial_lanelets_dir(self.road_network, goal)
            self.incoming_intersection = self.road_network.find_incoming_intersection(
                self.lanelets_dir
            )
            # three circle approximation
            (
                self.circle_appr_geo,
                self.circle_radius,
            ) = self._initial_circle_approximation()

    def rear_s(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates rear s-coordinate of vehicle

        :param time_step: time step to consider
        :returns rear s-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        if lane is None:
            return None
        curvi_state = self.ccosy_cache.get_curvilinear_state(self.states_cr[time_step], lane)
        if curvi_state is None:
            return None
        state_lon, state_lat = curvi_state
        center_s = state_lon.s
        width = self.shape.width
        length = self.shape.length
        theta = state_lat.theta
        rear_s = np.min(calc_s(center_s, width, length, theta))
        return rear_s

    def front_s(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates front s-coordinate of vehicle

        :param time_step: time step to consider
        :returns front s-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        curvi_state = self.ccosy_cache.get_curvilinear_state(self.states_cr[time_step], lane)
        if curvi_state is None:
            return None
        state_lon, state_lat = curvi_state
        center_s = state_lon.s
        width = self.shape.width
        length = self.shape.length
        theta = state_lat.theta
        front_s = np.max(calc_s(center_s, width, length, theta))
        return front_s

    def left_d(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates left d-coordinate of vehicle

        :param time_step: time step to consider
        :returns left d-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        state_lon, state_lat = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane
        )
        d = state_lat.d
        width = self.shape.width
        length = self.shape.length
        theta = state_lat.theta
        return max(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def right_d(self, time_step: int, lane: Lane = None) -> float:
        """
        Calculates right d-coordinate of vehicle

        :param time_step: time step to consider
        :returns right d-coordinate [m]
        """
        lane = lane or self.get_lane(time_step)
        state_lon, state_lat = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane
        )
        d = state_lat.d
        width = self.shape.width
        length = self.shape.length
        theta = state_lat.theta
        return min(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def get_lat_state(self, time_step: int, lane: Lane = None) -> StateLateral:
        lane = lane or self.get_lane(time_step)
        state_lon, state_lat = self.ccosy_cache.get_curvilinear_state(
            self.states_cr[time_step], lane
        )
        return state_lat

    def get_lon_state(self, time_step: int, lane: Lane = None) -> StateLongitudinal:
        lane = lane or self.get_lane(time_step)
        states = self.ccosy_cache.get_curvilinear_state(self.states_cr[time_step], lane)
        return states[0] if states is not None else None

    def occupancy_at_time_step(self, time_step) -> Rectangle:
        state = self.states_cr[time_step]
        orientation = state.orientation
        shape = self.shape.rotate_translate_local(state.position, orientation)
        return shape

    def shapely_occupancy_at_time_step(self, time_step):
        state = self.states_cr[time_step]
        orientation = state.orientation
        shape = self.shape.shapely_object
        cos = np.cos(orientation)
        sin = np.sin(orientation)
        mat = [cos, -sin, sin, cos, state.position[0], state.position[1]]
        new_shape = affinity.affine_transform(shape, mat)
        return new_shape

    def circle_appr_occupancy_at_time_step(self, time_step) -> Polygon:
        state = self.states_cr[time_step]
        orientation = state.orientation
        shape = self.circle_appr_geo
        cos = np.cos(orientation)
        sin = np.sin(orientation)
        mat = [cos, -sin, sin, cos, state.position[0], state.position[1]]
        new_shape = affinity.affine_transform(shape, mat)
        return new_shape

    def is_valid(self, time_step):
        state = self.states_cr.get(time_step)
        return state is not None

    def lanes_at_state(self, time_step) -> Set[Lane]:
        lanelets = self.lanelet_assignment[time_step]
        return self.ccosy_cache.road_network.find_lanes_by_lanelets(lanelets)

    def get_lane(self, time_step):
        # Todo: How to decide lane assignment generally?
        lanes = self.lanes_at_state(time_step)
        return lanes.pop() if len(lanes) > 0 else None

    @property
    def end_time(self) -> int:
        return max(map(lambda state: state.time_step, self.states_cr.values()))

    @property
    def start_time(self) -> int:
        return min(map(lambda state: state.time_step, self.states_cr.values()))

    @property
    def state_list_cr(self) -> List[State]:
        return list(self.states_cr.values())

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def _initial_circle_approximation(self):
        """
        generate three circle approximation
        """
        circle_radius = np.sqrt(self.shape.width**2 + (self.shape.length / 3) ** 2) / 2
        center_of_vehicle = Point(0, 0)
        front_point = Point(self.shape.length / 3, 0)
        rear_point = Point(-self.shape.length / 3, 0)

        circle_center = center_of_vehicle.buffer(circle_radius)
        circle_front = front_point.buffer(circle_radius)
        circle_rear = rear_point.buffer(circle_radius)

        combined_geometry = unary_union([circle_center, circle_front, circle_rear])
        return combined_geometry, circle_radius

    def _initial_lanelets_dir(
        self, road_network: RoadNetwork, goal=None
    ) -> (List[int], Lane, np.ndarray, np.ndarray, np.ndarray):
        """
        initialize lanelets_dir
        """
        if goal is None:
            initial_state = self.states_cr[self.start_time]
            end_time = self.end_time
            end_position = self.states_cr[end_time].position
            end_orientation = self.states_cr[end_time].orientation
            end_velocity = self.states_cr[end_time].velocity
            attributes = {
                "time_step": Interval(start=end_time - 1, end=end_time + 1),
                "position": Rectangle(length=1.0, width=1.0, center=end_position),
                # + np.array([np.cos(end_orientation), np.sin(end_orientation)])),
                "velocity": Interval(start=end_velocity, end=end_velocity + 1),
                "orientation": AngleInterval(
                    start=end_orientation - 0.1, end=end_orientation + 0.1
                ),
            }
        else:
            initial_state = goal["initial_state"]
            attributes = goal["attributes"]
            end_position = goal["end_position"]
            end_orientation = goal["end_orientation"]
        try:
            route = self._route_planner(initial_state, attributes, road_network)
        except Exception:
            route = None
        # replan route to fix no solution from route planner
        replanned_route = self._replan_route(
            initial_state, end_position, end_orientation, attributes, road_network
        )
        if route is None:
            route = next(replanned_route)
        # extend lanelets from route
        lanelets_leading_to_goal = self._extend_route_plan(route.lanelet_ids, road_network)
        # get reference lane from lanelets_leading_to_goal
        ref_path_lanes = self._initial_ref_path_lane(road_network, lanelets_leading_to_goal)
        # if no reference lane is found, replan the route
        while len(ref_path_lanes) == 0 and route is not None:
            route = next(replanned_route)
            lanelets_leading_to_goal = self._extend_route_plan(route.lanelet_ids, road_network)
            ref_path_lanes = self._initial_ref_path_lane(road_network, lanelets_leading_to_goal)
        # get properties from reference lane
        ref_path_lane = ref_path_lanes[0]
        center_vertices = road_network.lanelet_network.find_lanelet_by_id(
            lanelets_leading_to_goal[0]
        ).center_vertices
        left_vertices = road_network.lanelet_network.find_lanelet_by_id(
            lanelets_leading_to_goal[0]
        ).left_vertices
        right_vertices = road_network.lanelet_network.find_lanelet_by_id(
            lanelets_leading_to_goal[0]
        ).right_vertices
        for lanelet_id in lanelets_leading_to_goal[1:]:
            lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            center_vertices = np.append(center_vertices, lanelet.center_vertices, axis=0)
            left_vertices = np.append(left_vertices, lanelet.left_vertices, axis=0)
            right_vertices = np.append(right_vertices, lanelet.right_vertices, axis=0)
        return (
            lanelets_leading_to_goal,
            ref_path_lane,
            center_vertices,
            left_vertices,
            right_vertices,
        )

    @staticmethod
    def _route_planner(initial_state, attributes, road_network: RoadNetwork):
        """
        route planner by given intial state and attributes
        """
        end_state = CustomState(**attributes)
        goal_region = GoalRegion(state_list=[end_state])
        planning_problem = PlanningProblem(0, initial_state, goal_region)
        route_planner = RoutePlanner(
            lanelet_network=road_network.lanelet_network,
            planning_problem=planning_problem,
        )
        candidate_holder = route_planner.plan_routes()
        route = candidate_holder.retrieve_first_route()
        return route

    def _replan_route(
        self,
        initial_state: InitialState,
        end_position,
        end_orientation,
        attributes,
        road_network,
    ):
        """
        replan route to fix no solution in route planner
        """
        initial_state_candidates = [initial_state]
        end_position_candidates = [end_position]
        extend_length = [1.0, 1.5]
        for length in extend_length:
            right_start_position = initial_state.position + np.array(
                [
                    length * np.cos(initial_state.orientation - np.pi / 2),
                    length * np.sin(initial_state.orientation - np.pi / 2),
                ]
            )
            right_initial_state = copy.copy(initial_state)
            right_initial_state.position = right_start_position
            initial_state_candidates.append(right_initial_state)

            left_start_position = initial_state.position - np.array(
                [
                    length * np.cos(initial_state.orientation - np.pi / 2),
                    length * np.sin(initial_state.orientation - np.pi / 2),
                ]
            )
            left_initial_state = copy.copy(initial_state)
            left_initial_state.position = left_start_position
            initial_state_candidates.append(left_initial_state)

            right_end_position = end_position + np.array(
                [
                    length * np.cos(end_orientation - np.pi / 2),
                    1.0 * np.sin(end_orientation - np.pi / 2),
                ]
            )
            end_position_candidates.append(right_end_position)

            left_end_position = end_position - np.array(
                [
                    length * np.cos(end_orientation - np.pi / 2),
                    length * np.sin(end_orientation - np.pi / 2),
                ]
            )
            end_position_candidates.append(left_end_position)
        for i in range(len(initial_state_candidates)):
            for j in range(len(end_position_candidates)):
                if i == 0 and j == 0:
                    continue
                attributes["position"] = Rectangle(
                    length=1.0,
                    width=1.0,
                    center=end_position_candidates[j],
                    orientation=end_orientation,
                )
                try:
                    route = self._route_planner(
                        initial_state=initial_state_candidates[i],
                        attributes=attributes,
                        road_network=road_network,
                    )
                except Exception:
                    route = None
                if route is not None:
                    yield route
        yield None

    @staticmethod
    def _extend_route_plan(lanelets_leading_to_goal, road_network: RoadNetwork) -> List[int]:
        """
        extend lanelets from route
        """
        # extend the route path:
        first_lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelets_leading_to_goal[0])
        if first_lanelet.predecessor:
            selected_predecessor = first_lanelet.predecessor[0]
            # if there are more than one predecessor, find the one with the minimum orientation change
            # compared with first lanelet
            if len(first_lanelet.predecessor) > 1:
                min_offset = np.inf
                for predecessor_lanelet_id in first_lanelet.predecessor:
                    predecessor_lanelet = road_network.lanelet_network.find_lanelet_by_id(
                        predecessor_lanelet_id
                    )
                    orientation_pre = np.arctan2(
                        predecessor_lanelet.center_vertices[0, 1]
                        - predecessor_lanelet.center_vertices[1, 1],
                        predecessor_lanelet.center_vertices[0, 0]
                        - predecessor_lanelet.center_vertices[1, 0],
                    )
                    orientation_first = np.arctan2(
                        first_lanelet.center_vertices[0, 1] - first_lanelet.center_vertices[1, 1],
                        first_lanelet.center_vertices[0, 0] - first_lanelet.center_vertices[1, 0],
                    )
                    offset = abs(orientation_pre - orientation_first)
                    if np.min(offset) < min_offset:
                        min_offset = np.min(offset)
                        selected_predecessor = predecessor_lanelet_id
            lanelets_leading_to_goal.insert(0, selected_predecessor)
        last_lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelets_leading_to_goal[-1])
        if last_lanelet.successor:
            selected_successor = last_lanelet.successor[0]
            # if there are more than one successor, find the one with the minimum orientation change
            # compared with last lanelet
            if len(last_lanelet.successor) > 1:
                min_offset = np.inf
                for successor_lanelet_id in last_lanelet.successor:
                    successor_lanelet = road_network.lanelet_network.find_lanelet_by_id(
                        successor_lanelet_id
                    )
                    orientation_suc = np.arctan2(
                        successor_lanelet.center_vertices[-1, 1]
                        - successor_lanelet.center_vertices[-2, 1],
                        successor_lanelet.center_vertices[-1, 0]
                        - successor_lanelet.center_vertices[-2, 0],
                    )
                    orientation_last = np.arctan2(
                        last_lanelet.center_vertices[-1, 1] - last_lanelet.center_vertices[-2, 1],
                        last_lanelet.center_vertices[-1, 0] - last_lanelet.center_vertices[-2, 0],
                    )
                    offset = abs(orientation_suc - orientation_last)
                    if np.min(offset) < min_offset:
                        min_offset = np.min(offset)
                        selected_successor = successor_lanelet_id
            lanelets_leading_to_goal.append(selected_successor)
        return lanelets_leading_to_goal

    @staticmethod
    def _initial_ref_path_lane(road_network: RoadNetwork, lanelets: List[int]):
        """
        finds reference lane based on lanelets dir
        """
        lanes = list()
        lanelets = lanelets
        if len(lanelets) == 1:
            return list(road_network.find_lanes_by_lanelets(set(lanelets)))
        for lanelet_id in lanelets:
            lanes.append(
                road_network.find_lanes_by_lanelets(
                    {
                        lanelet_id,
                    }
                )
            )
        ref_path = lanes[0]
        for i in range(len(lanes) - 1):
            ref_path = ref_path.intersection(lanes[i + 1])
        reference_path = list(ref_path)
        return reference_path

    # ---------------------------------------------------------------------#
    # def ref_path_lanes(self, timestep: int) -> Tuple[Lane]:
    #     """
    #     Determine all possible lanes for a vehicle from the given moment.
    #
    #     Idea: A vehicle should drive on a connected sequence of lanelets to get to
    #     the current
    #     position. Hence, the intersection of the initially occupied lanes (all paths
    #     from the first state)
    #     and the currently occupied lanes should not be empty and only contain the
    #     lanes that have been driven on.
    #
    #     :param timestep:
    #     :return:
    #     """
    #
    #     initial_lanes = self.lanes_at_state(self.start_time)
    #     current_lanes = self.lanes_at_state(timestep)
    #
    #     return tuple(initial_lanes.intersection(current_lanes))


#
#     def lanelets_dir(self, timestep: int) -> Tuple[int]:
#         """
#         Get the lanelets in driving direction occupied at the current time step.
#
#         Implementation: Intersect the current lanelets with the reference path.
#
#         :param self:
#         :param timestep:
#         :return:
#         """
#         ref_lanes = self.ref_path_lanes(timestep)
#         current_lanelets = self.lanelet_assignment[timestep]
#         ref_lanelets = set()
#         for lane in ref_lanes:
#             ref_lanelets.update(lane.contained_lanelets)
#         return tuple(ref_lanelets.intersection(current_lanelets))
# ----------------------------------------------------------------#


class ControlledVehicle(Vehicle):
    def __init__(
        self,
        obstacle_id,
        vehicle_param,
        shape,
        road_network: RoadNetwork,
        inital_state,
        obstacle_type=ObstacleType.CAR,
        initial_signal=None,
    ):
        states_cr = {inital_state.time_step: inital_state}
        signal_series = {inital_state.time_step: initial_signal}
        ccosy_cache = CurvilinearStateManager(road_network)
        self.lanelet_network = road_network.lanelet_network
        initial_lanelets = road_network.lanelet_network.find_lanelet_by_shape(
            shape.rotate_translate_local(inital_state.position, inital_state.orientation)
        )
        lanelet_assignment = {inital_state.time_step: initial_lanelets}
        super().__init__(
            obstacle_id,
            obstacle_type,
            vehicle_param,
            shape,
            states_cr,
            signal_series,
            ccosy_cache,
            lanelet_assignment,
        )

    def add_state(self, state: State, signal_state=None):
        self.states_cr[state.time_step] = state
        loc_shape = self.shape.rotate_translate_local(state.position, state.orientation)
        self.lanelet_assignment[state.time_step] = self.lanelet_network.find_lanelet_by_shape(
            loc_shape
        )
        self.signal_series[state.time_step] = signal_state


class DynamicObstacleVehicle(Vehicle):
    """
    Vehicle with state and input profiles and other information for complete
    simulation horizon.
    """

    def __init__(
        self,
        obstacle: DynamicObstacle,
        ccosy_cache: CurvilinearStateManager,
        vehicle_param: Optional[VehicleParameters] = None,
        predicate_cache=None,
        road_network=None,
        goal=None,
    ):
        lanelet_assignment = obstacle.prediction.shape_lanelet_assignment.copy()
        id = obstacle.obstacle_id
        obstacle_type = obstacle.obstacle_type
        states_cr = {
            state.time_step: state
            for state in [obstacle.initial_state] + obstacle.prediction.trajectory.state_list
        }
        shape = obstacle.obstacle_shape
        if obstacle.signal_series is not None:
            signal_series = {state.time_step: state for state in obstacle.signal_series}
        else:
            signal_series = None
        ccosy_cache = ccosy_cache
        lanelet_assignment[obstacle.initial_state.time_step] = obstacle.initial_shape_lanelet_ids

        if vehicle_param is None:
            vehicle_param = VehicleParameters()
        super().__init__(
            id,
            obstacle_type,
            vehicle_param,
            shape,
            states_cr,
            signal_series,
            ccosy_cache,
            lanelet_assignment,
            predicate_cache,
            road_network,
            goal,
        )

    # @property
    # def states_lon(self) -> Dict[int, StateLongitudinal]:
    #     return self._states_lon
    #
    # @property
    # def states_lat(self) -> Dict[int, StateLateral]:
    #     return self._states_lat
