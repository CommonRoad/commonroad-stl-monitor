from commonroad.scenario.trajectory import State
from commonroad_ccosy.geometry.util import chaikins_corner_cutting, resample_polyline
from commonroad.scenario.scenario import Scenario
import numpy as np
from commonroad.scenario.lanelet import Lanelet
from vehicle import StateLongitudinal, StateLateral, Vehicle
import math
from commonroad.planning.planning_problem import PlanningProblem
from commonroad_ccosy.geometry.trapezoid_coordinate_system import create_coordinate_system_from_polyline
from typing import Dict, Tuple, List


def create_curvilinear_coordinate_system_from_state(lanelet_network, state, fov):
    point_list = list(np.array([state.position]))
    lanelet_id = lanelet_network.find_lanelet_by_position(point_list)
    lanelet = lanelet_network.find_lanelet_by_id(lanelet_id[0][0])

    if lanelet.successor is not None and len(lanelet.successor) > 0:
        lanes = Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, lanelet_network, fov+100)[0][0]
    else:
        lanes = lanelet
    reference_path = lanes.center_vertices
    curvilinear_cosy = create_curvilinear_coordinate_system_from_lanelet(reference_path)

    return curvilinear_cosy, lanes


def create_curvilinear_coordinate_system_from_lanelet(reference_path):
    theta_ref = compute_orientation_from_polyline(reference_path)
    new_reference_path = np.array([])
    for i in range(0, 100):
        new_reference_path = chaikins_corner_cutting(reference_path)
    new_reference_path = resample_polyline(new_reference_path, 0.1)

    curvilinear_cosy = create_coordinate_system_from_polyline(new_reference_path)

    return curvilinear_cosy


def simulate_ego_vehicle(ego_vehicle: Vehicle, dt: float, a_lat, a_lon, curvilinear_cosy, reference_path, a_min, a_max, time_step):
    x_lon = ego_vehicle.state_list_lon[time_step]
    x_lat = ego_vehicle.state_list_lat[time_step]

    if math.sqrt(a_lat**2 + a_lon**2) > max(abs(a_min), a_max):
        a_lon = (abs(a_min) - abs(a_lat)) * np.sign(a_lon)

    if x_lon.v + a_lon * dt < 0:
        dt = x_lon.v/a_lon
        v_new = 0.0
    else:
        v_new = a_lon * dt + x_lon.v

    s_new = 0.5 * a_lon * dt ** 2 + x_lon.v * dt + x_lon.s
    v_d_new = a_lat * dt + x_lat.v_d
    d_new = 0.5 * a_lat * dt ** 2 + x_lat.v_d * dt + x_lat.d

    theta_ref = compute_orientation_from_polyline(reference_path)
    ref_pos = compute_pathlength_from_polyline(reference_path)
    theta_cl = np.interp(s_new, ref_pos, theta_ref)

    x_lon_new = StateLongitudinal(s_new, v_new, a_lon)
    x_lat_new = StateLateral(d_new, v_d_new, a_lat, theta_cl)
    cartesian_coord = curvilinear_cosy.convert_to_cartesian_coords(s_new, d_new)

    # TODO ask Christian for conversion to kinematic single track model
    x_cr_new = State(position=cartesian_coord, velocity=v_new, orientation=0, time_step=time_step+1)

    ego_vehicle.append_state_lon(x_lon_new, time_step)
    ego_vehicle.append_state_lat(x_lat_new, time_step)
    ego_vehicle.append_state_cr(x_cr_new, time_step)
    ego_vehicle.append_lane_number(0, time_step)
    ego_vehicle.append_safe_distance(0, time_step)

    return ego_vehicle


def compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the orientations along a given polyline
    :param polyline: The polyline to check
    :return: The orientations along the polyline
    """
    assert isinstance(polyline, np.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(
        polyline[0, :]) == 2, '<Math>: not a valid polyline. polyline = {}'.format(polyline)

    if (len(polyline) < 2):
        raise NameError('Cannot create orientation from polyline of length < 2')

    orientation = [0]
    for i in range(1, len(polyline)):
        pt1 = polyline[i - 1]
        pt2 = polyline[i]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    return orientation


def compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the curvature of a given polyline
    :param polyline: The polyline for the curvature computation
    :return: The curvature of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for curvature computation p={}'.format(polyline)
    x_d = np.gradient(polyline[:, 0])
    x_dd = np.gradient(x_d)
    y_d = np.gradient(polyline[:, 1])
    y_dd = np.gradient(y_d)

    return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))


def compute_pathlength_from_polyline(polyline: np.ndarray) -> np.ndarray:
    """
    Computes the pathlength of a given polyline

    :param polyline: The polyline
    :return: The pathlength of the polyline
    """
    assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
        polyline[:, 0]) > 2, 'Polyline malformed for pathlenth computation p={}'.format(polyline)
    distance = np.zeros((len(polyline),))
    for i in range(1, len(polyline)):
        distance[i] = distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1])

    return np.array(distance)


def create_curvilinear_states(cr_state: State, curvilinear_cosy, theta_ref, ref_pos) -> Tuple[StateLongitudinal, StateLateral]:
    """
    Computes initial state of ego vehicle

    :param cr_state: The current CommonRoad state of the vehicle
    :param curvilinear_cosy: The curvilinear coordinate system based on the reference path of the ego vehicle lane
    :return: The lateral and longitudinal state of the ego vehicle
    """
    s, d = curvilinear_cosy.convert_to_curvilinear_coords(cr_state.position[0],
                                                          cr_state.position[1])

    theta_cl = np.interp(s, ref_pos, theta_ref)
    x_lon = StateLongitudinal(s, cr_state.velocity, 0)
    x_lat = StateLateral(d, 0, 0, theta_cl - cr_state.orientation)
    return x_lon, x_lat


def update_ego_lane_info(scenario: Scenario, ego_state: State, ego_vehicle_param: Dict):
    """
    Update information for ego vehicle planning

    :param scenario: CommonRoad scenario
    :param ego_state: current state of ego vehicle
    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :returns lanelet id of ego vehicle center position, ego lanelet, ego lane, curvilinear coordinate system,
    left adjacent lane of ego vehicle, right adjacent lane of ego vehicle
    """
    ego_lanelet_id = scenario.lanelet_network.find_lanelet_by_position([ego_state.position])
    ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id[0][0])
    curvilinear_cosy_ego_lane, ego_lane = create_curvilinear_coordinate_system_from_state(scenario.lanelet_network,
                                                                                          ego_state,
                                                                                          ego_vehicle_param.get("fov"))
    left_lane = None
    right_lane = None

    if ego_lanelet.adj_left_same_direction is True:
        if len(scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_left).successor) == 0:
            left_lane = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_left)
        else:
            left_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_left),
                scenario.lanelet_network, ego_vehicle_param.get("fov"))[0][0]  # TODO better solution instead of selecting single path
    if ego_lanelet.adj_right_same_direction is True:
        if len(scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_right).successor) == 0:
            right_lane = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_right)
        else:
            right_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.adj_right),
                scenario.lanelet_network, ego_vehicle_param.get("fov"))[0][0]  # TODO better solution instead of selecting single path

    return ego_lanelet_id, ego_lanelet, ego_lane, curvilinear_cosy_ego_lane, left_lane, right_lane


def safe_distance_full_brake(self, x_ego: float, v_ego: float, a_ego: float,
                                x_lead: float, v_lead: float, a_lead: float) -> float:
    """
    Safe distance between two vehicles, where the following vehicle (ego vehicle) executes a predefined (random)
    acceleration profile and the leading vehicle applies full brake. The safe distance considers jerk limitations
    and the reaction time of the following vehicle.
    :param x_ego: current front x-position of ego vehicle
    :param v_ego: current velocity of ego vehicle
    :param a_ego: current acceleration of ego vehicle
    :param x_lead: current rear x-position of leading vehicle
    :param v_lead: current velocity of leading vehicle
    :param a_lead: current acceleration of leading vehicle
    :return: d_safe: safe distance
    """
    v_lead_profile = [v_lead]
    x_lead_profile = [x_lead]
    a_lead_profile = [a_lead]
    v_ego_profile = [v_ego]
    x_ego_profile = [x_ego]
    a_ego_profile = [a_ego]

    # braking of leading vehicle with considered jerk:
    while v_lead_profile[-1] > 0:
        a_lead_tmp = max(conf.a_min_lead, a_lead_profile[-1] + conf.j_min_lead * conf.dt)
        x_lead_tmp = x_lead_profile[-1] + v_lead_profile[-1] * conf.dt + 0.5 * a_lead_tmp * conf.dt ** 2
        v_lead_tmp = v_lead_profile[-1] + a_lead_tmp * conf.dt

        if v_lead_tmp <= 0:
            v_lead_tmp = 0
            x_lead_tmp = x_lead_profile[-1] + v_lead_profile[-1] ** 2 / 2 * abs(a_lead_tmp)

        x_lead_profile.append(x_lead_tmp)
        v_lead_profile.append(v_lead_tmp)
        a_lead_profile.append(a_lead_tmp)

    # ego vehicle motion during reaction time (max acceleration of ego vehicle):
    steps_reaction_time = int(conf.t_react / conf.dt)
    for i in range(steps_reaction_time):
        a_ego_tmp = max(conf.a_min_ego, a_ego_profile[-1] + conf.j_max_lead * conf.dt)
        x_ego_tmp = x_ego_profile[-1] + v_ego_profile[-1] * conf.dt + 0.5 * a_ego_tmp * conf.dt ** 2
        v_ego_tmp = v_ego_profile[-1] + a_ego_tmp * conf.dt
        if v_ego_tmp > conf.v_max_ego:
            v_ego_tmp = conf.v_max_ego
            x_ego_tmp = x_ego_profile[-1] + v_ego_profile[-1] * conf.dt + a_ego_tmp * (
                    v_ego_profile[-1] / a_ego_tmp) ** 2 + (
                                conf.dt - (v_ego_profile[-1] / a_ego_tmp) * (conf.v_max_ego - v_ego_profile[-1]))
        x_ego_profile.append(x_ego_tmp)
        v_ego_profile.append(v_ego_tmp)
        a_ego_profile.append(a_ego_tmp)

    # application of full braking by the ego vehicle:
    while v_ego_profile[-1] > 0:
        a_ego_tmp = max(conf.a_min_ego, a_ego_profile[-1] + conf.j_min_ego * conf.dt)
        x_ego_tmp = x_ego_profile[-1] + v_ego_profile[-1] * conf.dt + 0.5 * a_ego_tmp * conf.dt ** 2
        v_ego_tmp = v_ego_profile[-1] + a_ego_tmp * conf.dt

        if v_ego_tmp < 0:
            v_ego_tmp = 0
            x_ego_tmp = x_ego_profile[-1] + v_ego_profile[-1] ** 2 / 2 * abs(a_ego_tmp)

        x_ego_profile.append(x_ego_tmp)
        v_ego_profile.append(v_ego_tmp)
        a_ego_profile.append(a_ego_tmp)

    #  safe distance based on the position profile during braking of the ego vehicle and the leading vehicle:
    len_ego = len(x_ego_profile)
    len_lead = len(x_lead_profile)
    if len_ego < len_lead:
        x_ego_profile = x_ego_profile + [x_ego_profile[-1]] * (len_lead - len_ego)
    else:
        x_lead_profile = x_lead_profile + [x_lead_profile[-1]] * (len_ego - len_lead)

    diff_max = np.min(np.array(x_lead_profile) - np.array(x_ego_profile))
    d_safe = x_lead - x_ego - diff_max
    d_safe = d_safe + conf.const_dist_offset

    return d_safe


def get_max_scenario_duration(planning_problem: PlanningProblem) -> int:
    max_time = 0
    for state in planning_problem.goal.state_list:
        if state.time_step.end > max_time:
            max_time = state.time_step.end

    return max_time


def calculate_orientation_from_polyline(polyline):
    if (len(polyline) < 2):
        raise NameError('Cannot create orientation from polyline of length < 2')

    orientation = []
    for i in range(0, len(polyline) - 1):
        pt1 = polyline[i]
        pt2 = polyline[i + 1]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    for i in range(len(polyline) - 1, len(polyline)):
        pt1 = polyline[i - 1]
        pt2 = polyline[i]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    orientation = [0]
    for i in range(1, len(polyline)):
        pt1 = polyline[i - 1]
        pt2 = polyline[i]
        tmp = pt2 - pt1
        orientation.append(np.arctan2(tmp[1], tmp[0]))

    return orientation


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
    s_lead_list, v_lead_list = simulate_vehicle_braking(v_lead_list, s_lead_list, a_lead, a_min_lead, dt,
                                                        j_min_lead)

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
    s_acc_list, v_acc_list = simulate_vehicle_braking(v_acc_list, s_acc_list, a_follow, a_min_follow, dt,
                                                      j_min_follow)

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

