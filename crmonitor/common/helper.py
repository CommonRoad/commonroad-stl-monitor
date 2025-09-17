import math
from typing import Dict, List, Tuple

from commonroad.common.util import Interval as CommonRoadInterval
from commonroad.scenario.scenario import Scenario
from rtamt.semantics.interval.interval import Interval as RtamtInterval


def calc_v_max_fov(ego_vehicle_param: Dict, simulation_param: Dict) -> int:
    """
    Calculates safety (field of view) based maximum allowed velocity rounded to next lower integer value

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :returns maximum allowed velocity
    """
    v_ego = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
    a_min = ego_vehicle_param.get("a_min") + ego_vehicle_param.get("a_corr")
    emergency_profile = ego_vehicle_param.get("emergency_profile")
    (
        a_corr,
        a_ego,
        a_max,
        dist_offset,
        dt,
        j_max,
        s_ego,
        stopping_distance,
        t_react,
        v_max,
        v_min,
    ) = init_v_max_calculation(a_min, ego_vehicle_param, emergency_profile, simulation_param, v_ego)
    while (
        dist_offset <= 0
        or dist_offset >= 0.5
        and not (
            v_max == ego_vehicle_param.get("dynamics_param").longitudinal.v_max
            and dist_offset > 0.5
        )
    ):
        if (
            ego_vehicle_param.get("fov")
            - stopping_distance
            - ego_vehicle_param.get("const_dist_offset")
            < 0
        ):
            v_max -= 0.001
        else:
            v_max += 0.001
        if v_max > ego_vehicle_param.get("dynamics_param").longitudinal.v_max:
            v_max = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
        if v_max < v_min:
            v_max = v_min
        stopping_distance = emg_stopping_distance(
            s_ego,
            v_ego,
            a_ego,
            dt,
            t_react,
            a_min,
            a_max,
            j_max,
            v_min,
            v_max,
            a_corr,
            emergency_profile,
        )
        dist_offset = (
            ego_vehicle_param.get("fov")
            - stopping_distance
            - ego_vehicle_param.get("const_dist_offset")
        )

    return math.floor(v_max)


def calc_v_max_braking(
    ego_vehicle_param: Dict, simulation_param: Dict, traffic_rule_param: Dict
) -> int:
    """
    Calculates braking based maximum allowed velocity rounded to next lower integer value

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param traffic_rule_param: dictionary with parameters related to traffic rules
    :returns maximum allowed velocity
    """
    v_max_delta = ego_vehicle_param.get(
        "dynamics_param"
    ).longitudinal.v_max - traffic_rule_param.get("max_velocity_limit_free_driving")
    v_ego = v_max_delta
    a_min = traffic_rule_param.get("a_abrupt")
    emergency_profile = 2500 * [traffic_rule_param.get("j_abrupt")]
    (
        a_corr,
        a_ego,
        a_max,
        dist_offset,
        dt,
        j_max,
        s_ego,
        stopping_distance,
        t_react,
        v_max,
        v_min,
    ) = init_v_max_calculation(a_min, ego_vehicle_param, emergency_profile, simulation_param, v_ego)
    while (
        dist_offset <= 0 or dist_offset >= 0.5 and not (v_max == v_max_delta and dist_offset > 0.5)
    ):
        if ego_vehicle_param.get("fov") - stopping_distance < 0:
            v_max -= 0.001
        else:
            v_max += 0.001
        if v_max > v_max_delta:
            v_max = v_max_delta
        if v_max < v_min:
            v_max = v_min
        stopping_distance = emg_stopping_distance(
            s_ego,
            v_ego,
            a_ego,
            dt,
            t_react,
            a_min,
            a_max,
            j_max,
            v_min,
            v_max,
            a_corr,
            emergency_profile,
        )
        dist_offset = (
            ego_vehicle_param.get("fov")
            - stopping_distance
            - ego_vehicle_param.get("const_dist_offset")
        )

    return math.floor(v_max + traffic_rule_param.get("max_velocity_limit_free_driving"))


def init_v_max_calculation(a_min, ego_vehicle_param, emergency_profile, simulation_param, v_ego):
    """
    Helper function to initialize values for calculation of maximum velocity based on field of view and braking

    :param a_min: minimum acceleration of the ego vehicle
    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param emergency_profile: emergency jerk profile which is executed in case of a fail-safe braking maneuver
    :param simulation_param: dictionary with parameters of the simulation environment
    :param v_ego: ego vehicle velocity
    :returns different parameters for the calculation of the maximum allowed velocity
    """
    s_ego = 0
    a_ego = 0  # ego vehicle is already at v_max
    dt = simulation_param.get("dt")
    t_react = ego_vehicle_param.get("t_react")
    a_max = ego_vehicle_param.get("a_max")
    a_corr = ego_vehicle_param.get("a_corr")
    j_max = ego_vehicle_param.get("j_max")
    v_min = ego_vehicle_param.get("v_min")
    v_max = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
    stopping_distance = emg_stopping_distance(
        s_ego,
        v_ego,
        a_ego,
        dt,
        t_react,
        a_min,
        a_max,
        j_max,
        v_min,
        v_max,
        a_corr,
        emergency_profile,
    )
    dist_offset = (
        ego_vehicle_param.get("fov")
        - stopping_distance
        - ego_vehicle_param.get("const_dist_offset")
    )

    return (
        a_corr,
        a_ego,
        a_max,
        dist_offset,
        dt,
        j_max,
        s_ego,
        stopping_distance,
        t_react,
        v_max,
        v_min,
    )


def emg_stopping_distance(
    s: float,
    v: float,
    a: float,
    dt: float,
    t_react: float,
    a_min: float,
    a_max: float,
    j_max: float,
    v_min: float,
    v_max: float,
    a_corr: float,
    emergency_profile: List[float],
) -> float:
    """
    Calculates stopping distance of a vehicle which applies predefined emergency jerk profile
     and considering reaction time

    :param s: current longitudinal front position of vehicle
    :param v: current velocity of vehicle
    :param a: current acceleration of vehicle
    :param dt: time step size
    :param t_react: reaction time of vehicle
    :param a_min: minimum acceleration of vehicle
    :param a_max: maximum acceleration of vehicle
    :param j_max: maximum jerk of vehicle
    :param v_max: maximum velocity of vehicle
    :param v_min: minimum velocity of vehicle
    :param a_corr: maximum deviation of vehicle from real acceleration
    :param emergency_profile: jerk emergency profile
    :returns: stopping distance
    """
    # application of reaction time (maximum jerk of vehicle):
    a = min(a + a_corr, a_max)
    if v == v_max:
        a = 0
    steps_reaction_time = round(t_react / dt)
    for i in range(steps_reaction_time):
        s, v, a = vehicle_dynamics_jerk(s, v, a, j_max, v_min, v_max, a_min, a_max, dt)

    # application of the emergency profile:
    index = 0
    while v > 0:
        a = min(a + a_corr, a_max)
        if v == v_max:
            a = 0
        s, v, a = vehicle_dynamics_jerk(
            s, v, a, emergency_profile[index], v_min, v_max, a_min, a_max, dt
        )
        index = index + 1

    return s


def vehicle_dynamics_jerk(
    s_0: float,
    v_0: float,
    a_0: float,
    j_input: float,
    v_min: float,
    v_max: float,
    a_min: float,
    a_max: float,
    dt: float,
) -> Tuple[float, float, float]:
    """
    Applying vehicle dynamics for one times step with jerk as input

    :param s_0: current longitudinal position at vehicle's front
    :param v_0: current velocity of vehicle
    :param a_0: current acceleration of vehicle
    :param j_input: jerk input for vehicle
    :param v_min: minimum velocity of vehicle
    :param v_max: maximum velocity of vehicle
    :param a_min: minimum acceleration of vehicle
    :param a_max: maximum acceleration of vehicle
    :param dt: time step size
    :return: new position, velocity, acceleration
    """
    a_new = a_0 + j_input * dt
    if a_new > a_max:
        t_a = abs((a_max - a_0) / j_input)  # time until a_max is reached
        a_new = a_max
    elif a_new < a_min:
        t_a = abs((a_0 - a_min) / j_input)  # time until a_min is reached
        a_new = a_min
    else:
        t_a = dt

    v_new = v_0 + a_0 * dt + 0.5 * j_input * t_a**2
    if v_new > v_max and j_input != 0.0:
        t_v = calculate_tv(a_0, j_input, v_0, v_max)  # time until v_max is reached
        t_a = t_v
        v_new = v_max
    elif v_new > v_max and j_input == 0.0:
        t_v = abs((v_max - v_0) / a_0)
        t_a = t_v
        v_new = v_max
    if v_new < v_min and j_input != 0.0:
        t_v = calculate_tv(a_0, j_input, v_0, v_min)  # time until v_min is reached
        t_a = t_v
        v_new = v_min
    elif v_new < v_min and j_input == 0.0:
        t_v = abs((v_0 - v_min) / a_0)
        t_a = t_v
        v_new = v_min
    else:
        t_v = dt

    if v_new == v_max or v_new == v_min:
        a_new = 0

    s_new = s_0 + v_0 * t_v + 0.5 * a_0 * t_a**2 + (1 / 6) * j_input * t_a**3

    return s_new, v_new, a_new


def calculate_tv(a_0, j_input, v_0, v_max):
    """
    Calculates time how long input can be applied until minimum/maximum velocity is reached

    :param a_0: current acceleration of vehicle
    :param j_input: jerk input for vehicle
    :param v_0: current velocity of vehicle
    :param v_max: maximum velocity of vehicle
    :returns time until v_max is reached
    """
    d = abs(a_0) ** 2 - 4 * 0.5 * abs(j_input) * (v_max - v_0)
    t_1 = (-abs(a_0) + math.sqrt(d)) / (2 * 0.5 * abs(j_input))
    t_2 = (-abs(a_0) - math.sqrt(d)) / (2 * 0.5 * abs(j_input))
    t_v = min(abs(t_1), abs(t_2))

    return t_v


def rtamt_interval_to_commonroad_interval(
    interval: RtamtInterval, scenario_context: Scenario
) -> CommonRoadInterval:
    """
    Convert a rtamt interval with units to a time step based interval in the context of the scenario.

    :param interval: A rtamt interval, with optional units.
    :param scenario_context: The scenario in which this interval should be valid.

    :returns: A CommonRoad interval in time steps, which is valid in regards to the scenario context.

    :raises RuntimeError: If an invalid combination of units is used.
    """
    if len(interval.begin_unit) == 0 and len(interval.end_unit) == 0:
        normalized_begin = int(interval.begin)
        normalized_end = int(interval.end)
    elif interval.begin_unit == "s" or interval.end_unit == "s":
        normalized_begin = interval.begin / scenario_context.dt
        normalized_end = interval.end / scenario_context.dt
    else:
        raise RuntimeError(
            f"Cannot convert rtamt interval: combination of time units '{interval.begin_unit}' and '{interval.end_unit}' is not supported! Use 's' for seconds, or omit for time steps."
        )

    begin = max(0, normalized_begin)

    return CommonRoadInterval(begin, normalized_end)
