import copy
import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List

import commonroad_clcs.pycrccosy as pycrccosy
import numpy as np
from commonroad.common.util import AngleInterval, Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, TrajectoryPrediction
from commonroad.scenario.state import CustomState, InitialState, InputState, State, TraceState
from commonroad.scenario.trajectory import Trajectory
from commonroad_clcs.clcs import CurvilinearCoordinateSystem
from commonroad_clcs.util import (
    compute_orientation_from_polyline,
    compute_pathlength_from_polyline,
)
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, VehicleType
from commonroad_route_planner.route_planner import RoutePlanner
from omegaconf import DictConfig
from shapely import affinity, unary_union
from shapely.geometry import Point, Polygon
from typing_extensions import Self
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

from crmonitor.common.road_network import Lane, RoadNetwork
from crmonitor.common.scenario_type import ScenarioType

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


@dataclass(slots=True, frozen=True)
class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """

    time_step: int
    s: float
    v: float
    a: float
    j: float
    j_dot: float

    @property
    def attributes(self) -> list[str]:
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


@dataclass(slots=True, frozen=True)
class StateLateral:
    """
    Lateral state in curvilinear coordinate system
    """

    time_step: int
    d: float
    theta: float
    kappa: float
    kappa_dot: float
    kappa_dot_dot: float

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
        v: np.ndarray | None = None,
        a: np.ndarray | None = None,
        kappa: np.ndarray | None = None,
        theta: np.ndarray | None = None,
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
        self._theta = compute_orientation_from_polyline(self._polyline) if theta is None else theta
        self._kappa = pycrccosy.Util.compute_curvature(self._polyline) if kappa is None else kappa
        self._kappa_dot = self._compute_kappa_dot_from_kappa(self._kappa)
        self._kappa_ddot = self._compute_kappa_ddot_from_kappa_dot(self._kappa_dot)
        self._steering_angle = self._compute_steering_angle_from_kappa(self._kappa)
        self._steering_angle_speed = self._compute_steering_angle_speed_from_kappa_and_kappa_dot(
            self._kappa, self._kappa_dot
        )

    @classmethod
    def from_cartesian_state_list(
        cls,
        initial_time_step: int,
        final_time_step: int,
        state_list: list[TraceState],
        dt: float,
        lane: Lane,
    ) -> Self:
        curvilinear_coords = np.array(
            [lane.convert_to_curvilinear_coords(*state.position) for state in state_list]
        )

        s = curvilinear_coords.T[0]
        d = curvilinear_coords.T[1]

        v = np.array([state.velocity for state in state_list])

        a = None
        if all(state.has_value("acceleration") for state in state_list):
            a = np.array([state.acceleration for state in state_list])

        thetas = []
        for i, state in enumerate(state_list):
            theta_cl = lane.orientation(s[i]) % (2 * math.pi)
            orientation = state.orientation % (2 * math.pi)
            theta = (orientation - theta_cl + math.pi) % (2 * math.pi) - math.pi
            thetas.append(theta)

        return cls(
            initial_time_step, final_time_step, dt=dt, s=s, d=d, v=v, a=a, theta=np.array(thetas)
        )

    def _time_step_to_index(self, time_step: int) -> int:
        if time_step > self.final_time_step:
            raise ValueError(
                f"Time step {time_step} is larger than final time step of trajectory {self.final_time_step}."
            )
        if time_step < self.initial_time_step:
            raise ValueError(
                f"Time step {time_step} is smaller than final time step of trajectory {self.initial_time_step}."
            )

        return time_step - self.initial_time_step

    @property
    def length(self) -> int:
        return self.final_time_step - self.initial_time_step + 1

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

    def theta(self, time_step: int) -> float:
        return self._theta[self._time_step_to_index(time_step)]

    def jerk(self, time_step: int) -> float:
        return self._jerk[self._time_step_to_index(time_step)]

    def jerk_dot(self, time_step: int) -> float:
        return self._jerk_dot[self._time_step_to_index(time_step)]

    def kappa(self, time_step: int) -> float:
        return self._kappa[self._time_step_to_index(time_step)]

    def kappa_dot(self, time_step: int) -> float:
        return self._kappa_dot[self._time_step_to_index(time_step)]

    def kappa_dot_dot(self, time_step: int) -> float:
        return self._kappa_ddot[self._time_step_to_index(time_step)]

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

    def lon_state_at_time_step(self, time_step: int) -> StateLongitudinal:
        return StateLongitudinal(
            time_step=time_step,
            s=self.s(time_step),
            v=self.velocity(time_step),
            a=self.acceleration(time_step),
            j=self.jerk(time_step),
            j_dot=self.jerk_dot(time_step),
        )

    def lat_state_at_time_step(self, time_step: int) -> StateLateral:
        return StateLateral(
            time_step,
            d=self.d(time_step),
            theta=self.theta(time_step),
            kappa=self.kappa(time_step),
            kappa_dot=self.kappa_dot(time_step),
            kappa_dot_dot=self.kappa_dot_dot(time_step),
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
        if self.length <= 1:
            return np.array([0.0])
        return np.gradient(pathlength, self._dt)

    def _compute_a_from_v(self, v: np.ndarray) -> np.ndarray:
        if self.length <= 1:
            return np.array([0.0])
        return np.gradient(v, self._dt)

    def _compute_jerk_from_a(self, a: np.ndarray) -> np.ndarray:
        if self.length <= 1:
            return np.array([0.0])
        return np.gradient(a, self._dt)

    def _compute_j_dot_from_j(self, j: np.ndarray) -> np.ndarray:
        if self.length <= 1:
            return np.array([0.0])
        return np.gradient(j, self._dt)

    def _compute_kappa_dot_from_kappa(self, kappa: np.ndarray) -> np.ndarray:
        if self.length <= 1:
            return np.array([0.0])
        return np.gradient(kappa, self._pathlength)

    def _compute_kappa_ddot_from_kappa_dot(self, kappa_dot: np.ndarray) -> np.ndarray:
        if self.length <= 1:
            return np.array([0.0])
        return np.gradient(kappa_dot, self._pathlength)

    def _compute_steering_angle_from_kappa(self, kappa: np.ndarray) -> np.ndarray:
        return np.arctan(kappa * self._l_wb)

    def _compute_steering_angle_speed_from_kappa_and_kappa_dot(
        self, kappa: np.ndarray, kappa_dot: np.ndarray
    ) -> np.ndarray:
        return self._l_wb * kappa_dot / (1 + self._l_wb**2 * kappa**2)


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
        id: int,
        obstacle_type: ObstacleType,
        shape,
        states_cr,
        lanelet_assignment: dict[int, set[int]] | None,
        road_network: RoadNetwork,
        dt: float,
        signal_series=None,
        goal=None,
        vehicle_param: VehicleParameters | None = None,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
    ):
        self.id = id
        self.obstacle_type = obstacle_type
        self.vehicle_param = vehicle_param
        self.shape = shape
        self.states_cr = states_cr
        self.signal_series = signal_series
        self._road_network = road_network
        self._dt = dt

        # Mappings to improve lookup speed for lanelet and lane queries.
        # Currently, the two assignments are tracked separately to ensure compatability with old implementations,
        # since those might directly access the lanelet assignments dict, and do not use the query
        # methods from below.
        if lanelet_assignment is not None:
            self._lanelet_assignment = lanelet_assignment
        else:
            self._lanelet_assignment = self._initialize_lanelet_assignment()
        self._lane_assignment: dict[int, set[int]] = dict()

        if vehicle_param is None:
            vehicle_param = VehicleParameters()
        self.vehicle_param = vehicle_param

        self._curvilinear_trajectories = {}
        self._start_time = min(map(lambda state: state.time_step, self.states_cr.values()))
        self._end_time = max(map(lambda state: state.time_step, self.states_cr.values()))

        if scenario_type == ScenarioType.INTERSTATE:
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
            ) = self._initial_lanelets_dir(self._road_network, goal)
            self.incoming_intersection = self._road_network.find_incoming_intersection(
                self.lanelets_dir
            )
            # three circle approximation
            (
                self.circle_appr_geo,
                self.circle_radius,
            ) = self._initial_circle_approximation()

    @property
    def lanelet_assignment(self) -> dict[int, set[int]]:
        return self._lanelet_assignment

    @classmethod
    def from_dynamic_obstacle(
        cls,
        obstacle: DynamicObstacle,
        road_network: RoadNetwork,
        dt: float,
        scenario_type: ScenarioType = ScenarioType.INTERSTATE,
        vehicle_param: VehicleParameters | None = None,
    ) -> Self:
        if obstacle.signal_series is not None:
            signal_series = {state.time_step: state for state in obstacle.signal_series}
        else:
            signal_series = None

        # A dynamic obstacle might not have a trajectory prediction.
        # Then the resulting states only consist of the initial state.
        state_list = []
        if isinstance(obstacle.prediction, TrajectoryPrediction):
            state_list = obstacle.prediction.trajectory.state_list

        states_cr = {state.time_step: state for state in [obstacle.initial_state] + state_list}

        # If the obstacle already has lanelet assignments (e.g., because `CommonRoadFileReader`
        #  was invoked with `lanelet_assignment=True`), those can be used to speed up processing.
        lanelet_assignment = None
        if (
            isinstance(obstacle.prediction, TrajectoryPrediction)
            and obstacle.prediction.shape_lanelet_assignment is not None
        ):
            lanelet_assignment = copy.deepcopy(obstacle.prediction.shape_lanelet_assignment)

        return cls(
            id=obstacle.obstacle_id,
            obstacle_type=obstacle.obstacle_type,
            shape=obstacle.obstacle_shape,
            states_cr=states_cr,
            road_network=road_network,
            dt=dt,
            lanelet_assignment=lanelet_assignment,
            signal_series=signal_series,
            vehicle_param=vehicle_param,
            scenario_type=scenario_type,
        )

    def get_curvilinear_trajectory(self, lane: Lane) -> CurvilinearVehicleTrajectory:
        if lane.lane_id in self._curvilinear_trajectories:
            return self._curvilinear_trajectories[lane.lane_id]

        state_list_cr = list(self.states_cr.values())

        if len(state_list_cr) != (self.end_time - self.start_time + 1):
            raise ValueError(
                f"Cannot create curvilinear trajectory for vehicle {self.id}: The state list of length {len(state_list_cr)} is not continous over the time frame from {self.start_time} to {self.end_time}"
            )

        curvilinear_trajectory = CurvilinearVehicleTrajectory.from_cartesian_state_list(
            self.start_time, self.end_time, list(self.states_cr.values()), self._dt, lane
        )

        self._curvilinear_trajectories[lane.lane_id] = curvilinear_trajectory
        return curvilinear_trajectory

    def rear_s(self, time_step: int, lane: Lane | None = None) -> float:
        """
        Calculates rear s-coordinate of vehicle

        :param time_step: time step to consider
        :returns: rear s-coordinate [m]
        """
        lane = lane or self.lane_at_time_step(time_step)
        if lane is None:
            return None

        curvi_trajectory = self.get_curvilinear_trajectory(lane)
        center_s = curvi_trajectory.s(time_step)
        width = self.shape.width
        length = self.shape.length
        theta = curvi_trajectory.theta(time_step)
        rear_s = np.min(calc_s(center_s, width, length, theta))
        return rear_s

    def front_s(self, time_step: int, lane: Lane | None = None) -> float:
        """
        Calculates front s-coordinate of vehicle

        :param time_step: time step to consider
        :returns: front s-coordinate [m]
        """
        lane = lane or self.lane_at_time_step(time_step)
        if lane is None:
            return None

        curvi_trajectory = self.get_curvilinear_trajectory(lane)

        center_s = curvi_trajectory.s(time_step)
        width = self.shape.width
        length = self.shape.length
        theta = curvi_trajectory.theta(time_step)
        front_s = np.max(calc_s(center_s, width, length, theta))
        return front_s

    def left_d(self, time_step: int, lane: Lane | None = None) -> float:
        """
        Calculates left d-coordinate of vehicle

        :param time_step: time step to consider
        :returns: left d-coordinate [m]
        """
        lane = lane or self.lane_at_time_step(time_step)
        if lane is None:
            return None

        curvi_trajectory = self.get_curvilinear_trajectory(lane)
        d = curvi_trajectory.d(time_step)
        width = self.shape.width
        length = self.shape.length
        theta = curvi_trajectory.theta(time_step)
        return max(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def right_d(self, time_step: int, lane: Lane | None = None) -> float:
        """
        Calculates right d-coordinate of vehicle

        :param time_step: time step to consider
        :returns: right d-coordinate [m]
        """
        lane = lane or self.lane_at_time_step(time_step)
        if lane is None:
            return None

        curvi_trajectory = self.get_curvilinear_trajectory(lane)
        d = curvi_trajectory.d(time_step)
        width = self.shape.width
        length = self.shape.length
        theta = curvi_trajectory.theta(time_step)
        return min(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def get_lat_state(self, time_step: int, lane: Lane = None) -> StateLateral:
        lane = lane or self.lane_at_time_step(time_step)
        return self.get_curvilinear_trajectory(lane).lat_state_at_time_step(time_step)

    def get_lon_state(self, time_step: int, lane: Lane = None) -> StateLongitudinal:
        lane = lane or self.lane_at_time_step(time_step)
        return self.get_curvilinear_trajectory(lane).lon_state_at_time_step(time_step)

    def get_cr_state(self, time_step: int) -> State:
        return self.states_cr[time_step]

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

    def is_valid(self, time_step: int) -> bool:
        state = self.states_cr.get(time_step)
        return state is not None

    def _find_lane_ids_at_time_step(self, time_step: int) -> set[int]:
        """
        Determine the IDs of lanes on which the vehicle is at the time step.

        Uses the internal lanelet assignment to determine the lane at a time step.
        """
        # By using the internal lanelet assignment, backwards compatability is ensured.
        # This makes it possible to override the lane and lanelet assignment, which is currently used in some tests.
        lanelet_ids = self.lanelet_ids_at_time_step(time_step)
        lane_ids = self._road_network.find_lane_ids_by_lanelets(lanelet_ids)
        return lane_ids

    def lane_ids_at_time_step(self, time_step: int) -> set[int]:
        """
        Determine the lane IDs which are occupied by the vehicle at the time step.

        :param time_step: Time step for which the lane IDs for this vehicle should be determined. Must be in the interval [start_time, end_time].

        :returns: The set of occupied lane IDs. Might be empty, if the vehicle does not occupy any lanes at the time step.
        """
        if time_step in self._lane_assignment:
            return self._lane_assignment[time_step]

        lane_ids = self._find_lane_ids_at_time_step(time_step)
        self._lane_assignment[time_step] = lane_ids
        return lane_ids

    def lanes_at_time_step(self, time_step: int) -> set[Lane]:
        """
        Determine the lanes which are occupied by the vehicle at the time step.

        :param time_step: Time step for which the lanes for this vehicle should be determined. Must be in the interval [start_time, end_time].

        :returns: The set of occupied lanes. Might be empty, if the vehicle does not occupy any lanes at the time step.
        """
        lane_ids = self.lane_ids_at_time_step(time_step)

        lanes = set()
        for lane_id in lane_ids:
            lane = self._road_network.find_lane_by_id(lane_id)
            if lane is None:
                raise RuntimeError(
                    f"Invalid lane assignment for vehicle {self.id} at time step {time_step}: lane {lane_id} is not part of the road network"
                )

            lanes.add(lane)

        return lanes

    def _find_lanelet_ids_at_time_step(self, time_step: int) -> set[int]:
        state = self.get_state_at_time_step(time_step)
        loc_shape = self.shape.rotate_translate_local(state.position, state.orientation)
        lanelet_ids = self._road_network.lanelet_network.find_lanelet_by_shape(loc_shape)
        return set(lanelet_ids)

    def lanelet_ids_at_time_step(self, time_step: int) -> set[int]:
        """
        Determine the lanelet IDs which are occupied by the vehicle at the time step.

        :param time_step: Time step for which the lanelet IDs for this vehicle should be determined. Must be in the interval [start_time, end_time].

        :returns: The set of occupied lanelet IDs. Might be empty, if the vehicle does not occupy any lanelets at the time step.
        """
        if time_step in self._lanelet_assignment:
            return self._lanelet_assignment[time_step]

        lanelet_ids = self._find_lanelet_ids_at_time_step(time_step)
        self._lanelet_assignment[time_step] = lanelet_ids
        return lanelet_ids

    def _initialize_lanelet_assignment(self) -> dict[int, set[int]]:
        """
        Compute the internal lanelet assignment.

        Queries the occupied lanelets for every state in the state list of the vehicles.
        """
        lanelet_assignment = {}
        for time_step in self.states_cr.keys():
            lanelet_assignment[time_step] = self._find_lanelet_ids_at_time_step(time_step)

        return lanelet_assignment

    def lane_id_at_time_step(self, time_step: int) -> int | None:
        """
        Determine the ID of the most likely lane at the state of the time step.

        To improve efficiency, this method will use its internal lane assignment for the lookup, if possible.

        :param time_step: Time step for which the lane ID for this vehicle should be determined. Must be in the interval [start_time, end_time].

        :returns: The smallest lane ID which the vehicle occupies at the time step, or None, if it occupies no lanes at the time step.
        """
        lane_ids = self.lane_ids_at_time_step(time_step)
        if len(lane_ids) == 0:
            return None

        # Sort the lanes by ID, and selected the one with the smallest ID.
        stored_lane_ids = sorted(lane_ids)
        return stored_lane_ids[0]

    def lane_at_time_step(self, time_step: int) -> Lane | None:
        """
        Determine the most likely lane at the state of the time step.

        To improve efficiency, this method will use its internal lane assignment for the lookup, if possible.

        :param time_step: Time step for which the lane for this vehicle should be determined. Must be in the interval [start_time, end_time].

        :returns: The lane with the smallest ID which the vehicle occupies at the time step, or None, if it occupies no lanes at the time step.
        """
        lane_id = self.lane_id_at_time_step(time_step)
        if lane_id is None:
            return None

        lane = self._road_network.find_lane_by_id(lane_id)
        return lane

    def get_state_at_time_step(self, time_step: int) -> TraceState:
        return self.states_cr[time_step]

    def set_state_at_time_step(self, time_step: int, state: TraceState) -> None:
        self.states_cr[time_step] = state

        # The curvilinear trajectories must be invalidated.
        # TODO: Since this only affects one time step, we could also just recompute for this time step.
        # Currently this is however not supported by the `CurvilinearVehicleTrajectory`.
        self._curvilinear_trajectories = {}
        del self._lanelet_assignment[time_step]
        del self._lane_assignment[time_step]

    @property
    def end_time(self) -> int:
        return self._end_time

    @property
    def start_time(self) -> int:
        return self._start_time

    @property
    def state_list_cr(self) -> List[State]:
        return list(self.states_cr.values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vehicle):
            return False
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
            routes = self._route_planner(initial_state, attributes, road_network)
            route = routes[0]
        except ValueError as e:
            _LOGGER.debug("Route planner failed for vehicle %s: %s", self.id, e)
            replanned_route = self._replan_route(
                initial_state, end_position, end_orientation, attributes, road_network
            )
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
        routes = route_planner.plan_routes()
        return routes

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
        dt: float,
        obstacle_type=ObstacleType.CAR,
        initial_signal=None,
    ):
        states_cr = {inital_state.time_step: inital_state}
        signal_series = {inital_state.time_step: initial_signal}
        self.lanelet_network = road_network.lanelet_network
        initial_lanelets = road_network.lanelet_network.find_lanelet_by_shape(
            shape.rotate_translate_local(inital_state.position, inital_state.orientation)
        )
        lanelet_assignment = {inital_state.time_step: initial_lanelets}
        super().__init__(
            id=obstacle_id,
            obstacle_type=obstacle_type,
            shape=shape,
            states_cr=states_cr,
            lanelet_assignment=lanelet_assignment,
            road_network=road_network,
            dt=dt,
            signal_series=signal_series,
            vehicle_param=vehicle_param,
        )

    def add_state(self, state: State, signal_state=None):
        self.set_state_at_time_step(state.time_step, state)
        self.signal_series[state.time_step] = signal_state
