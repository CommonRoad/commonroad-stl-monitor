import enum
from typing import Union, Set, Dict, List

import numba
import numpy as np
from commonroad.geometry.shape import Shape, Rectangle
from commonroad.scenario.obstacle import ObstacleType, SignalState
from commonroad.scenario.trajectory import State
from crmonitor.common.road_network import Lane
from shapely import affinity

rot_mat_factors = np.array([[1., 1., -1., -1.], [1., -1., 1., -1.]])


@numba.njit
def calc_s(s, w, l, theta):
    s = rot_mat_factors[0] * l / 2. * np.cos(theta) - rot_mat_factors[
        1] * w / 2 * np.sin(theta) + s
    return s


class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """

    __slots__ = ["s", "v", "a", "j"]

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

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

    __slots__ = ["d", "theta", "kappa", "kappa_dot"]

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

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


@enum.unique
class VehicleClassification(enum.Enum):
    EGO_VEHICLE = 0
    CROSSING_VEHICLE = 1
    ADJACENT_VEHICLE = 2


class Vehicle:
    """
    Representation of a vehicle with state and input profiles and other information for complete simulation horizon
    """

    def __init__(
        self,
        states_lon: Dict[int, StateLongitudinal],
        states_lat: Dict[int, StateLateral],
        shape: Union[Shape, Rectangle],
        cr_states: Dict[int, State],
        vehicle_id: int,
        obstacle_type: ObstacleType,
        vehicle_param: Dict,
        lanelet_assignments: Dict[int, Set[int]],
        signal_states: Dict[int, SignalState] = None,
        vehicle_classification: Dict[int, VehicleClassification] = None,
        lane: Union[Lane, List[Lane]] = None,
        robust_lanelet_assignment: Dict[int, Set[int]] = None
    ):
        """
        :param states_lon: list of longitudinal states for initialization
        :param states_lat: list of lateral states for initialization
        :param shape: CommonRoad shape of vehicle
        :param cr_states: initial CommonRoad state of vehicle
        :param vehicle_id: id of vehicle
        :param obstacle_type: type of the vehicle, e.g. parked car, car, bus, ...
        :param lanelet_assignments: initial lanelet assignment
        :param signal_states: initial signal state of vehicle
        """
        self._states_lon = states_lon
        self._states_lat = states_lat
        self._states_cr = cr_states
        self._lanelet_assignment = lanelet_assignments
        self._signal_series = signal_states
        self._shape = shape
        self._id = vehicle_id
        self._obstacle_type = obstacle_type
        self._vehicle_classification = vehicle_classification
        self._lane = lane
        self._vehicle_param = vehicle_param
        self._robust_lanelet_assignment = robust_lanelet_assignment

    @property
    def robust_lanelet_assignment(self):
        return self._robust_lanelet_assignment

    @property
    def start_time(self):
        return min(self._states_lon.keys())

    @property
    def end_time(self):
        return max(self._states_lon.keys())

    @property
    def shape(self) -> Rectangle:
        return self._shape

    @property
    def id(self) -> int:
        return self._id

    @property
    def states_lon(self) -> Dict[int, StateLongitudinal]:
        return self._states_lon

    @property
    def states_lat(self) -> Dict[int, StateLateral]:
        return self._states_lat

    @property
    def states_cr(self) -> Dict[int, State]:
        return self._states_cr

    @property
    def state_list_cr(self) -> List[State]:
        state_list = []
        for state in self._states_cr.values():
            state_list.append(state)
        return state_list

    @property
    def obstacle_type(self) -> ObstacleType:
        return self._obstacle_type

    @property
    def lanelet_assignment(self) -> Dict[int, Set[int]]:
        return self._lanelet_assignment

    @property
    def signal_series(self) -> Dict[int, SignalState]:
        return self._signal_series

    @property
    def lane(self) -> Lane:
        return self._lane

    @lane.setter
    def lane(self, lane: Lane):
        self._lane = lane

    @property
    def vehicle_param(self) -> Dict:
        return self._vehicle_param

    def rear_s(self, time_step: int) -> float:
        """
        Calculates rear s-coordinate of vehicle

        :param time_step: time step to consider
        :returns rear s-coordinate [m]
        """
        s = self._states_lon[time_step].s
        w = self.shape.width
        l = self.shape.length
        theta = self.states_lat[time_step].theta
        rear_s = np.min(calc_s(s, w, l, theta))
        return rear_s

    def front_s(self, time_step: int) -> float:
        """
        Calculates front s-coordinate of vehicle

        :param time_step: time step to consider
        :returns front s-coordinate [m]
        """
        s = self._states_lon[time_step].s
        w = self.shape.width
        l = self.shape.length
        theta = self.states_lat[time_step].theta
        front_s = np.max(calc_s(s, w, l, theta))
        return front_s

    def right_d(self, time_step: int) -> float:
        """
        Calculates right d-coordinate of vehicle

        :param time_step: time step to consider
        :returns right d-coordinate [m]
        """
        s = self._states_lon[time_step].s
        d = self.states_lat[time_step].d
        width = self.shape.width
        length = self.shape.length
        theta = self.states_lat[time_step].theta
        return min(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def left_d(self, time_step: int) -> float:
        """
        Calculates left d-coordinate of vehicle

        :param time_step: time step to consider
        :returns left d-coordinate [m]
        """
        s = self._states_lon[time_step].s
        d = self.states_lat[time_step].d
        width = self.shape.width
        length = self.shape.length
        theta = self.states_lat[time_step].theta
        return max(
            (width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (length / 2) * np.sin(theta) + d,
            (-width / 2) * np.cos(theta) - (-length / 2) * np.sin(theta) + d,
        )

    def append_time_step(
        self,
        time_step: int,
        state_lon: StateLongitudinal,
        state_lat: StateLateral,
        state_cr: State,
        lanelet_assignment: Set[int],
        signal_state: State = None,
    ):
        """
        Adds information for a specific time step to vehicle

        :param time_step: time step of new data
        :param state_lon: longitudinal state to append
        :param state_lat: lateral state to append
        :param state_cr: CommonRoad state to append
        :param lanelet_assignment: lanelet assignment to append
        :param signal_state: signal state to append
        """
        self._states_lon[time_step] = state_lon
        self._states_lat[time_step] = state_lat
        self._states_cr[time_step] = state_cr
        self._lanelet_assignment[time_step] = lanelet_assignment
        self._signal_series[time_step] = signal_state

    def occupancy_at_time_step(self, time_step):
        state = self.states_cr[time_step]
        orientation = self.states_lat[time_step].theta
        shape = self.shape.rotate_translate_local(state.position,
                                               orientation)
        return shape

    def shapely_occupancy_at_time_step(self, time_step):
        state = self.states_cr[time_step]
        orientation = self.states_lat[time_step].theta
        shape = self.shape.shapely_object
        cos = np.cos(orientation)
        sin = np.sin(orientation)
        mat = [cos, -sin, sin, cos, state.position[0], state.position[1]]
        new_shape = affinity.affine_transform(shape, mat)
        return new_shape

    def is_valid(self, time_step):
        state = self.states_cr.get(time_step)
        return state is not None

