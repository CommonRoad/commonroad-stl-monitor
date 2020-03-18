from commonroad.geometry.shape import Shape, Rectangle
from typing import Union, Set, Dict, List
from commonroad.scenario.trajectory import State
from commonroad.scenario.obstacle import ObstacleType, SignalState


class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """
    __slots__ = ['s', 'v', 'a', 'j']

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

    @property
    def attributes(self) -> List[str]:
        """ Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = '\n'
        for attr in self.attributes:
            state += attr
            state += '= {}\n'.format(self.__getattribute__(attr))
        return state


class StateLateral:
    """
    Lateral state in curvilinear coordinate system
    """
    __slots__ = ['d', 'theta', 'kappa', 'kappa_dot']

    def __init__(self, **kwargs):
        """ Elements of state vector are determined during runtime."""
        for (field, value) in kwargs.items():
            setattr(self, field, value)

    @property
    def attributes(self) -> List[str]:
        """ Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = '\n'
        for attr in self.attributes:
            state += attr
            state += '= {}\n'.format(self.__getattribute__(attr))
        return state


class Input:
    """
    Lateral and longitudinal vehicle input
    """
    __slots__ = ['a', 'kappa_dot_dot']

    @property
    def attributes(self) -> List[str]:
        """ Returns all dynamically set attributes of an instance of State.

        :return: subset of slots which are dynamically assigned to the object.
        """
        attributes = list()
        for slot in self.__slots__:
            if hasattr(self, slot):
                attributes.append(slot)
        return attributes

    def __str__(self):
        state = '\n'
        for attr in self.attributes:
            state += attr
            state += '= {}\n'.format(self.__getattribute__(attr))
        return state


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
        self._lanelet_assignment = {cr_state.time_step: lanelet_assignment}
        self._signal_series = {cr_state.time_step: signal_state}
        self._shape = shape
        self._id = vehicle_id
        self._obstacle_type = obstacle_type
        #self._coordinate_system

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

    def rear_s(self, time_step: int) -> float:
        """
        Calculates rear s-coordinate of vehicle

        :param time_step: time step to consider
        :returns rear s-coordinate [m]
        """
        return self._states_lon[time_step].s - self.shape.length/2

    def front_s(self, time_step: int) -> float:
        """
        Calculates front s-coordinate of vehicle

        :param time_step: time step to consider
        :returns front s-coordinate [m]
        """
        return self._states_lon[time_step].s + self.shape.length/2

    def right_position(self, time_step: int) -> float:
        """
        Calculates right d-coordinate of vehicle

        :param time_step: time step to consider
        :returns front s-coordinate [m]
        """
        return self._states_lat[time_step].d - self.shape.width/2

    def left_position(self, time_step: int) -> float:
        """
        Calculates left d-coordinate of vehicle

        :param time_step: time step to consider
        :returns front d-coordinate [m]
        """
        return self._states_lat[time_step].d + self.shape.width/2

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
