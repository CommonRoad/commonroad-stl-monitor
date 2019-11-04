from commonroad.geometry.shape import Shape, Rectangle
from typing import Union, List
from commonroad.scenario.trajectory import State


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
    def __init__(self, d: float, v_d: float, a_d: float, theta: float):
        """
        :param d: lateral position in curvilinear coordinates
        :param v_d: lateral velocity in curvilinear coordinates
        :param a_d: lateral acceleration in curvilinear coordinates
        :param a_d: lateral orientation in curvilinear coordinates
        """
        self._d = d
        self._v_d = v_d
        self._a_d = a_d
        self._theta = theta

    @property
    def d(self) -> float:
        return self._d

    @d.setter
    def d(self, value: float):
        self._d = value

    @property
    def v_d(self) -> float:
        return self._v_d

    @v_d.setter
    def v_d(self, value: float):
        self._v_d = value

    @property
    def a_d(self) -> float:
        return self._a_d

    @a_d.setter
    def a_d(self, value: float):
        self._a_d = value

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value: float):
        self._theta = value


class Vehicle:
    """
    Representation of a vehicle with state lists over complete simulation horizon
    """
    def __init__(self, state_lon: StateLongitudinal, state_lat: StateLateral, shape: Union[Shape, Rectangle],
                 lane_number: int, cr_state: State, vehicle_id: int):
        """
        :param state_lon: initial longitudinal state of vehicle
        :param state_lat: initial lateral state of vehicle
        :param shape: CommonRoad rectangle representing shape of vehicle
        :param lane_number: -1, 0, 1 representing left, same lane, or right lane of vehicle at initial time step,
        respectively
        :param cr_state: initial CommonRoad state of vehicle
        :param vehicle_id: id of vehicle
        """
        self._state_list_lon = [state_lon]  # (index corresponds to time step)
        self._state_list_lat = [state_lat]  # (index corresponds to time step)
        self._state_list_cr = [cr_state]  # (index corresponds to time step)
        self._safe_distance_list = []  # (index corresponds to time step)
        self._shape = shape
        self._lane_number_list = [lane_number]  # (index corresponds to time step)
        self._id = vehicle_id

    @property
    def shape(self) -> Rectangle:
        return self._shape

    @property
    def lane_number_list(self) -> List[int]:
        return self._lane_number_list

    @property
    def id(self) -> int:
        return self._id

    @property
    def state_list_lon(self) -> List[StateLongitudinal]:
        return self._state_list_lon

    @property
    def state_list_lat(self) -> List[StateLateral]:
        return self._state_list_lat

    @property
    def state_list_cr(self) -> List[State]:
        return self._state_list_cr

    @property
    def safe_distance_list(self) -> List[float]:
        return self._safe_distance_list

    def rear_position(self, time_step: int) -> float:
        """
        Calculates rear position of vehicle based on longitudinal curvilinear state

        :param time_step: time step to consider
        :returns rear position [m]
        """
        return self._state_list_lon[time_step].s - self.shape.length/2

    def front_position(self, time_step: int) -> float:
        """
        Calculates front position of vehicle based on longitudinal curvilinear state

        :param time_step: time step to consider
        :returns front position [m]
        """
        return self._state_list_lon[time_step].s + self.shape.length/2

    def append_state_lon(self, state: StateLongitudinal, time_step: int):
        """
        Appends a state to the longitudinal curvilinear state list

        :param state: state to append
        :param time_step: time step of new data
        """
        if len(self._state_list_lon) == time_step:
            self._state_list_lon.append(state)
        else:
            for idx in range(len(self._state_list_lon), time_step):
                self._state_list_lon.append(None)
            self._state_list_lon.append(state)

    def append_state_lat(self, state: StateLateral, time_step: int):
        """
        Appends a state to the lateral curvilinear state list

        :param state: state to append
        :param time_step: time step of new data
        """
        if len(self._state_list_lat) == time_step:
            self._state_list_lat.append(state)
        else:
            for idx in range(len(self._state_list_lat), time_step):
                self._state_list_lat.append(None)
            self._state_list_lat.append(state)

    def append_state_cr(self, state: State, time_step: int):
        """
        Appends a state to the CommonRoad state list

        :param state: state to append
        :param time_step: time step of new data
        """
        if len(self._state_list_cr) == time_step:
            self._state_list_cr.append(state)
        else:
            for idx in range(len(self._state_list_cr), time_step):
                self._state_list_cr.append(None)
            self._state_list_cr.append(state)

    def append_lane_number(self, lane_number: int, time_step: int):
        """
        Appends a lane number to the lane number list (index corresponds to time step)

        :param lane_number: value to append
        :param time_step: time step of new data
        """
        assert -1 <= lane_number <= 1, '<Vehicle/lane>: Lane number not in valid range! ' \
                                       'Provided lane number: {}'.format(lane_number)
        if len(self._lane_number_list) == time_step:
            self._lane_number_list.append(lane_number)
        else:
            for idx in range(len(self._lane_number_list), time_step):
                self._lane_number_list.append(None)
            self._lane_number_list.append(lane_number)

    def append_safe_distance(self, distance: float, time_step: int):
        """
        Sets safe distance at a specific time step

        :param distance: safe distance of vehicle
        :param time_step: time step of new data
        """
        if len(self._safe_distance_list) == time_step:
            self._safe_distance_list.append(distance)
        else:
            for idx in range(len(self._safe_distance_list), time_step):
                self._safe_distance_list.append(None)
            self._safe_distance_list.append(distance)
