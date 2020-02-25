from parameters_vehicle1 import parameters_vehicle1
from parameters_vehicle2 import parameters_vehicle2
from parameters_vehicle3 import parameters_vehicle3
from typing import Dict, Union, List, Tuple
import ruamel.yaml
import math
from decimal import Decimal

from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry


def create_ego_vehicle_param(ego_vehicle_param: Dict, simulation_param: Dict, traffic_rule_param: Dict) -> Dict:
    """
    Update ACC vehicle parameters

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param traffic_rule_param: dictionary with parameters related to traffic rules
    :returns updated dictionary with parameters of ACC vehicle
    """
    if ego_vehicle_param.get("vehicle_number") == 1:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle1()
    elif ego_vehicle_param.get("vehicle_number") == 2:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle2()
    elif ego_vehicle_param.get("vehicle_number") == 3:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for ACC vehicle in config file defined.')

    emergency_profile = ego_vehicle_param.get("emergency_profile")
    emergency_profile += [ego_vehicle_param.get("j_min")] * ego_vehicle_param.get("emergency_profile_num_steps_fb")
    ego_vehicle_param["emergency_profile"] = emergency_profile

    ego_vehicle_param["fov_speed_limit"] = calc_v_max_fov(ego_vehicle_param, simulation_param)
    ego_vehicle_param["braking_speed_limit"] = calc_v_max_braking(ego_vehicle_param, simulation_param,
                                                                  traffic_rule_param)

    if not -1e-12 <= (Decimal(str(ego_vehicle_param.get("t_react"))) %
                      Decimal(str(simulation_param.get("dt")))) <= 1e-12:
        raise ValueError('Reaction time must be multiple of time step size.')

    return ego_vehicle_param


def create_other_vehicles_param(other_vehicles_param: Dict) -> Dict:
    """
    Update other vehicle's parameters

    :param other_vehicles_param: dictionary with physical parameters of other vehicles
    :returns updated dictionary with parameters of other vehicles
    """
    if other_vehicles_param.get("vehicle_number") == 1:
        other_vehicles_param["dynamics_param"] = parameters_vehicle1()
    elif other_vehicles_param.get("vehicle_number") == 2:
        other_vehicles_param["dynamics_param"] = parameters_vehicle2()
    elif other_vehicles_param.get("vehicle_number") == 3:
        other_vehicles_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for leading vehicle in config file defined.')

    return other_vehicles_param


def create_simulation_param(simulation_param: Dict, dt: float, country: str) -> Dict:
    """
    Update simulation parameters

    :param simulation_param: dictionary with parameters of the simulation environment
    :param country: country of CommonRoad scenario
    :param dt: time step size of CommonRoad scenario
    :returns updated dictionary with parameters of CommonRoad scenario
    """
    simulation_param["dt"] = dt
    simulation_param["country"] = SupportedTrafficSignCountry(country)

    return simulation_param


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
    a_corr, a_ego, a_max, dist_offset, dt, j_max, s_ego, stopping_distance, t_react, v_max, v_min = \
        init_v_max_calculation(a_min, ego_vehicle_param, emergency_profile, simulation_param, v_ego)
    while dist_offset <= 0 or dist_offset >= 0.5 \
            and not (v_max == ego_vehicle_param.get("dynamics_param").longitudinal.v_max and dist_offset > 0.5):
        if ego_vehicle_param.get("fov") - stopping_distance - ego_vehicle_param.get("const_dist_offset") < 0:
            v_max -= 0.001
        else:
            v_max += 0.001
        if v_max > ego_vehicle_param.get("dynamics_param").longitudinal.v_max:
            v_max = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
        if v_max < v_min:
            v_max = v_min
        stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt, t_react, a_min, a_max, j_max, v_min, v_max,
                                                  a_corr, emergency_profile)
        dist_offset = ego_vehicle_param.get("fov") - stopping_distance - ego_vehicle_param.get("const_dist_offset")

    return math.floor(v_max)


def calc_v_max_braking(ego_vehicle_param: Dict, simulation_param: Dict, traffic_rule_param: Dict) -> int:
    """
    Calculates braking based maximum allowed velocity rounded to next lower integer value

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param traffic_rule_param: dictionary with parameters related to traffic rules
    :returns maximum allowed velocity
    """
    v_max_delta = \
        ego_vehicle_param.get("dynamics_param").longitudinal.v_max - \
        traffic_rule_param.get("max_velocity_limit_free_driving")
    v_ego = v_max_delta
    a_min = traffic_rule_param.get("a_abrupt")
    emergency_profile = 2500 * [traffic_rule_param.get("j_abrupt")]
    a_corr, a_ego, a_max, dist_offset, dt, j_max, s_ego, stopping_distance, t_react, v_max, v_min = \
        init_v_max_calculation(a_min, ego_vehicle_param, emergency_profile, simulation_param, v_ego)
    while dist_offset <= 0 or dist_offset >= 0.5 \
            and not (v_max == v_max_delta and dist_offset > 0.5):
        if ego_vehicle_param.get("fov") - stopping_distance < 0:
            v_max -= 0.001
        else:
            v_max += 0.001
        if v_max > v_max_delta:
            v_max = v_max_delta
        if v_max < v_min:
            v_max = v_min
        stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt, t_react, a_min, a_max, j_max, v_min, v_max,
                                                  a_corr, emergency_profile)
        dist_offset = ego_vehicle_param.get("fov") - stopping_distance - ego_vehicle_param.get("const_dist_offset")

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
    stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt, t_react, a_min, a_max, j_max, v_min, v_max,
                                              a_corr, emergency_profile)
    dist_offset = ego_vehicle_param.get("fov") - stopping_distance - ego_vehicle_param.get("const_dist_offset")

    return a_corr, a_ego, a_max, dist_offset, dt, j_max, s_ego, stopping_distance, t_react, v_max, v_min


def emg_stopping_distance(s: float, v: float, a: float, dt: float, t_react: float, a_min: float, a_max: float,
                          j_max: float, v_min: float, v_max: float, a_corr: float,
                          emergency_profile: List[float]) -> float:
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
        s, v, a = vehicle_dynamics_jerk(s, v, a, emergency_profile[index], v_min, v_max, a_min, a_max, dt)
        index = index + 1

    return s


def vehicle_dynamics_jerk(s_0: float, v_0: float, a_0: float, j_input: float, v_min: float, v_max: float, a_min: float,
                          a_max: float, dt: float) -> Tuple[float, float, float]:
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
        t_v = calculate_tv(a_0, j_input, v_0, v_min)    # time until v_min is reached
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

    s_new = s_0 + v_0 * t_v + 0.5 * a_0 * t_a ** 2 + (1/6) * j_input * t_a ** 3

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


def vehicle_dynamics_acc(s_0: float, v_0: float, a_input: float, v_min: float, v_max: float,
                         dt: float) -> Tuple[float, float]:
    """
    Applying vehicle dynamics for one times step with acceleration as input

    :param s_0: current longitudinal position at vehicle's front
    :param v_0: current velocity of vehicle
    :param a_input: acceleration input for vehicle
    :param v_min: minimum velocity of vehicle
    :param v_max: maximum velocity of vehicle
    :param dt: time step size
    :return: new position and velocity
    """
    v_new = v_0 + a_input * dt
    if v_new > v_max:
        t_v = (v_max - v_0) / a_input   # time until v_max is reached
        v_new = v_max
    elif v_new < v_min:
        t_v = (v_0 - v_min) / a_input
        v_new = v_min
    else:
        t_v = dt

    s_new = s_0 + v_0 * t_v + 0.5 * a_input * t_v ** 2

    return s_new, v_new


def load_yaml(file_name: str) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    with open(file_name, 'r') as stream:
        try:
            config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            return None
