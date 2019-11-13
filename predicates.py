from commonroad.scenario.trajectory import State
from typing import List, Tuple
import numpy as np


def keeps_speed_limit(state: State, speed_limit: float):
    """
    Predicate for speed limit evaluation

    :param state: CommonRoad state to evaluate
    :param speed_limit: Speed limit at state position
    :returns Boolean indicating predicate satisfaction
    """
    if speed_limit < state.velocity:
        return False
    else:
        return True


def keeps_safe_distance(self, state: State, s_ego, obstacle_states_same_lane_clc,
                        obstacle_states_same_lane_cr: List[State]) -> bool:
    safe_distance_valid = True
    for idx, obs in enumerate(obstacle_states_same_lane_clc):
        if 0 < obs[0] - s_ego < self.safe_distance(state.velocity, obstacle_states_same_lane_cr[idx].velocity,
                                                   self._ego_vehicle_param.get("a_min"),
                                                   self._other_vehicles_param.get("a_min"),
                                                   self._ego_vehicle_param.get("t_react"), 0, 0,  # TODO change
                                                   self._ego_vehicle_param.get("j_min"),
                                                   self._other_vehicles_param.get("j_min"),
                                                   self._simulation_param.get("dt"),
                                                   s_ego, obs[0], self._ego_vehicle_param.get("a_max"),
                                                   self._ego_vehicle_param.get("j_max")):
            safe_distance_valid = False

    return safe_distance_valid


def safe_distance(v_follow: float, v_lead: float, a_min_follow: float, a_min_lead: float, t_react: float,
                  a_follow: float, a_lead: float, j_min_follow: float, j_min_lead: float, dt: float,
                  s_x_follow: float, s_x_lead: float, a_max_follow: float, j_max_follow: float) -> float:
    """
    Calculates safe distance between two vehicles, if both have the same negative acceleration

    :param v_follow: current velocity of ACC vehicle [m/s]
    :param v_lead: current velocity of leading vehicle [m/s]
    :param a_min_lead: minimum negative acceleration of leading vehicle [m/s^2]
    :param a_min_follow: minimum negative acceleration of ACC vehicle [m/s^2]
    :param a_max_follow: maximum positive acceleration of ACC vehicle [m/s^2]
    :param t_react: reaction time of ACC vehicle [s]
    :param a_follow: current acceleration of ACC vehicle [m/s^2]
    :param a_lead: current acceleration of leading vehicle [m/s^2]
    :param s_x_follow: current position of ACC vehicle [m/s^2]
    :param s_x_lead: current position of leading vehicle [m/s^2]
    :param j_min_follow: minimum negative jerk of ACC vehicle [m/s^3]
    :param j_min_lead: minimum negative acceleration of leading vehicle [m/s^3]
    :param j_max_follow: maximum positive jerk of ACC vehicle [m/s^3]
    :param dt: time step size [s]
    :return: safe distance between ACC vehicle and leading vehicle [m]
    """
    t = 0  # time step
    s_acc_list = [s_x_follow]
    v_acc_list = [v_follow]
    s_lead_list = [s_x_lead]
    v_lead_list = [v_lead]
    initial_distance = s_x_lead - s_x_follow

    if v_follow == 0:
        return 0.0

    # Leading vehicle braking:
    s_lead_list, v_lead_list = simulate_vehicle_braking(v_lead_list, s_lead_list, a_lead, a_min_lead, dt, j_min_lead)

    # ACC vehicle motion during reaction time:
    while t < t_react and v_follow > 0:
        if v_follow + min(a_max_follow, a_follow + j_max_follow * dt) * dt >= 0:
            a_acc = min(a_max_follow, a_follow + j_max_follow * dt)
            s_acc_curr = s_acc_list[-1] + v_follow * dt + 0.5 * a_acc * dt ** 2
            v_follow = v_follow + a_acc * dt
        else:
            a_acc = min(a_max_follow, a_follow + j_max_follow * dt)
            delta_t_short = v_follow / abs(a_acc)
            s_acc_curr = s_acc_list[-1] + v_follow * delta_t_short + 0.5 * a_acc * delta_t_short ** 2
            v_acc = 0
        s_acc_list.append(s_acc_curr)
        v_acc_list.append(v_follow)

        s_acc_curr = s_acc_list[-1] + v_follow * dt
        s_acc_list.append(s_acc_curr)
        t += dt

    # ACC vehicle braking:
    s_acc_list, v_acc_list = simulate_vehicle_braking(v_acc_list, s_acc_list, a_follow, a_min_follow, dt, j_min_follow)

    # Equalize list size:
    if len(s_acc_list) > len(s_lead_list):
        s_lead_list = s_lead_list + [s_lead_list[-1]] * (len(s_acc_list) - len(s_lead_list))
    elif len(s_lead_list) > len(s_acc_list):
        s_acc_list = s_acc_list + [s_acc_list[-1]] * (len(s_lead_list) - len(s_acc_list))

    # Safe distance calculation:
    delta_s = np.subtract(s_lead_list[1::], s_acc_list[1::])

    min_delta_s = np.min(delta_s)

    safe_dist = max(0, initial_distance - min_delta_s)

    return safe_dist


def simulate_vehicle_braking(velocity_list: List[float], position_list: List[float], acceleration: float,
                             a_min: float, dt: float, j_min: float) -> Tuple[List[float], List[float]]:
    """
    Forward simulation with kinematic single-track model

    :param velocity_list: list of previous velocity values of vehicle [m/s]
    :param position_list: list of previous x-position values of vehicle [m]
    :param acceleration: current acceleration of vehicle [rad]
    :param a_min: minimum acceleration of vehicle [m/s]
    :param j_min: minimum jerk of vehicle [rad]
    :param dt: time step size [s]
    :return: velocity and position list of braking vehicle
    """
    current_velocity = velocity_list[-1]
    current_position = position_list[-1]
    while current_velocity > 0:
        acceleration = max(a_min, acceleration + j_min * dt)
        if current_velocity + acceleration * dt >= 0:  # acceleration does
            # not lead to velocity < 0
            current_position = current_position + current_velocity * dt + 0.5 * acceleration * dt ** 2
            current_velocity = current_velocity + acceleration * dt
        else:  # acceleration leads to velocity < 0 -> prevent this
            delta_t_short = current_velocity / abs(acceleration)
            current_position = \
                current_position + current_velocity * delta_t_short + 0.5 * acceleration * delta_t_short ** 2

            current_velocity = 0
        position_list.append(current_position)
        velocity_list.append(current_velocity)

    return position_list, velocity_list
