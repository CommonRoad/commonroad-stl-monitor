from commonroad.geometry.shape import Shape, Rectangle
from typing import Union, Set, Dict, List
from commonroad.scenario.trajectory import State
from commonroad.scenario.obstacle import ObstacleType, SignalState


class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """
    def __init__(self, s: float, v: float, a: float):
        """
        :param s: longitudinal position in curvilinear coordinates
        :param v: longitudinal velocity in curvilinear coordinates
        :param a: longitudinal acceleration in curvilinear coordinates
        """
        self._s = s
        self._v = v
        self._a = a

    @property
    def s(self) -> float:
        return self._s

    @s.setter
    def s(self, value: float):
        self._s = value

    @property
    def v(self) -> float:
        return self._v

    @v.setter
    def v(self, value: float):
        self._v = value

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, value: float):
        self._a = value


class StateLateral:
    """
    Lateral state in curvilinear coordinate system
    """
    def __init__(self, d: float, theta: float, kappa: float, kappa_dot: float):
        """
        :param d: lateral position in curvilinear coordinates
        :param theta: orientation of vehicle
        :param kappa: curvature of vehicle
        :param kappa_dot: change of curvature of vehicle
        """
        self._d = d
        self._theta = theta
        self._kappa = kappa
        self._kappa_dot = kappa_dot

    @property
    def d(self) -> float:
        return self._d

    @d.setter
    def d(self, value: float):
        self._d = value

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value: float):
        self._theta = value

    @property
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self, value: float):
        self._kappa = value

    @property
    def kappa_dot(self) -> float:
        return self._kappa_dot

    @kappa_dot.setter
    def kappa_dot(self, value: float):
        self._kappa_dot = value


class Vehicle:
    """
    Representation of a vehicle with state and input profiles and other information for complete simulation horizon
    """
    def __init__(self, state_lon: StateLongitudinal, state_lat: StateLateral, shape: Union[Shape, Rectangle],
                 cr_state: State, vehicle_id: int, obstacle_type: ObstacleType, lanelet_assignment: Set[int],
                 signal_state: SignalState):
        """
        :param state_lon: initial longitudinal state of vehicle
        :param state_lat: initial lateral state of vehicle
        :param shape: CommonRoad shape of vehicle
        :param cr_state: initial CommonRoad state of vehicle
        :param vehicle_id: id of vehicle
        :param obstacle_type: type of the vehicle, e.g. parked car, car, bus, ...
        :param lanelet_assignment: initial lanelet assignment
        :param signal_state: initial signal state of vehicle
        """
        self._states_lon = {cr_state.time_step: state_lon}
        self._states_lat = {cr_state.time_step: state_lat}
        self._states_cr = {cr_state.time_step: cr_state}
        self._jerk_profile = {}
        self._kappa_dot_dot_profile = {}
        self._shape = shape
        self._id = vehicle_id
        self._obstacle_type = obstacle_type
        self._lanelet_assignment = {cr_state.time_step: lanelet_assignment}
        self._signal_series = {cr_state.time_step: signal_state}

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
    def jerk_profile(self) -> Dict[int, float]:
        return self._jerk_profile

    @property
    def obstacle_type(self) -> ObstacleType:
        return self._obstacle_type

    @property
    def lanelet_assignment(self) -> Dict[int, Set[int]]:
        return self._lanelet_assignment

    @property
    def signal_series(self) -> Dict[int, SignalState]:
        return self._signal_series

    def rear_position(self, time_step: int) -> float:
        """
        Calculates rear position of vehicle based on longitudinal curvilinear state

        :param time_step: time step to consider
        :returns rear position [m]
        """
        return self._states_lon[time_step].s - self.shape.length/2

    def front_position(self, time_step: int) -> float:
        """
        Calculates front position of vehicle based on longitudinal curvilinear state

        :param time_step: time step to consider
        :returns front position [m]
        """
        return self._states_lon[time_step].s + self.shape.length/2

    def append_state_lon(self, state: StateLongitudinal, time_step: int):
        """
        Appends a state to the longitudinal curvilinear state list

        :param state: state to append
        :param time_step: time step of new data
        """
        self._states_lon[time_step] = state

    def append_state_lat(self, state: StateLateral, time_step: int):
        """
        Appends a state to the lateral curvilinear state list

        :param state: state to append
        :param time_step: time step of new data
        """
        self._states_lat[time_step] = state

    def append_state_cr(self, state: State, time_step: int):
        """
        Appends a state to the CommonRoad state list

        :param state: state to append
        :param time_step: time step of new data
        """
        self._states_cr[time_step] = state

    def append_lanelet_assignment(self, lanelets: Set[int], time_step: int):
        """
        Sets lanelets at a specific time step

        :param lanelets: lanelet IDs to append
        :param time_step: time step of new data
        """
        self._lanelet_assignment[time_step] = lanelets

    def append_signal_state(self, signal_state: SignalState, time_step: int):
        """
        Sets signal state at a specific time step

        :param signal_state: the CommonRaod signal state to append
        :param time_step: time step of new data
        """
        self._signal_series[time_step] = signal_state

    def append_jerk(self, jerk: float, time_step: int):
        """
        Sets jerk at a specific time step

        :param jerk: jerk of vehicle
        :param time_step: time step of new data
        """
        self._jerk_profile[time_step] = jerk
