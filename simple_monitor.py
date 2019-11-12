import mtl
from typing import Dict, List, Tuple
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
import numpy as np
from util import update_ego_lane_info, create_curvilinear_states, compute_orientation_from_polyline, \
    compute_pathlength_from_polyline
from commonroad.geometry.shape import Rectangle
from parameters_vehicle2 import parameters_vehicle2
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle


class SimpleMonitor:
    def __init__(self, traffic_rules: Dict[str, str], predicates: Dict[str, List[str]],
                 simulation_param: Dict, ego_vehicle_param: Dict, other_vehicles_param: Dict):
        """
        :param traffic_rules: dictionary with MTL formulas of traffic rules
        :param predicates: dictionary predicates for each MTL formula
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        self._rules = self.create_monitors(traffic_rules)
        self._formula_predicates = predicates
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param

    @staticmethod
    def create_monitors(mtl_rules: Dict[str, str]) -> List[str]:
        """
        Initialization of monitor for each MTL rule

        :param mtl_rules: dictionary with MTL formulas
        :returns list of monitors
        """
        monitors = []
        for key, value in mtl_rules.items():
            monitor = mtl.parse(value)
            monitors.append(monitor)
        return monitors

    def evaluate_predicates(self, predicate: str, state: State, scenario: Scenario) -> bool:
        # ego_lanelet_id, ego_lanelet, ego_lane, curvilinear_cosy_ego_lane, left_lane, \
        # right_lane = update_ego_lane_info(scenario, state, self._ego_vehicle_param)
        # s_ego, d_ego = curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(state.position[0], state.position[1])
        # obstacle_states_same_lane_cr, obstacle_states_right_lane_cr, obstacle_states_left_lane_cr = \
        #     self.obstacles_at_time_step(state.time_step, scenario.dynamic_obstacles, right_lane, left_lane, ego_lane)
        # obstacle_states_same_lane_clc = self.convert_to_curvilinear(obstacle_states_same_lane_cr,
        #                                                             curvilinear_cosy_ego_lane)
        #obstacle_states_same_lane_cr_fov = self.obstacles_fov(state.time_step, s_ego, obstacle_states_same_lane_clc)

        #if predicate == "keeps_safe_distance":
        #    return self.keeps_safe_distance(state, s_ego, obstacle_states_same_lane_clc, obstacle_states_same_lane_cr)
        if predicate == "keeps_speed_limit":
            return self.keeps_speed_limit(state, scenario)

    def keeps_speed_limit(self, state: State, scenario: Scenario):
        lanelet_id = scenario.lanelet_network.find_lanelet_by_position([state.position])
        lanlet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0][0])
        if lanlet.speed_limit < state.velocity:
            return False
        else:
            return True

    def convert_to_curvilinear(self, obstacle_states_cr: List[State], curvilinear_coord_system):
        clc_state_list = []
        for state in obstacle_states_cr:
            s, d = curvilinear_coord_system.convert_to_curvilinear_coords(state.position[0], state.position[1])
            clc_state_list.append([s, d])

        return clc_state_list


    def obstacles_at_time_step(self, time_step: int, dynamic_obstacles: List[DynamicObstacle], right_lane: Lanelet,
                               left_lane: Lanelet, same_lane: Lanelet) -> Tuple[List[State], List[State], List[State]]:
        """
        Extract obstacles existing at current time step from CommonRoad scenario
        :param time_step: current time step
        :returns: list with dictionaries containing obstacle properties and states
        """
        obstacle_states_same_lane = []
        obstacle_states_left_lane = []
        obstacle_states_right_lane = []

        for obs in dynamic_obstacles:
            state_cr = None
            if time_step == obs.initial_state.time_step:
                state_cr = obs.initial_state
            else:
                for state in obs.prediction.trajectory.state_list:
                    if state.time_step == time_step:
                        state_cr = state
                        break
            if state_cr is not None:
                if right_lane is not None and right_lane.contains_points(np.array([np.array([state_cr.position[0],
                                                                                             state_cr.position[1]])]))[0]:
                    obstacle_states_right_lane.append(state_cr)
                elif left_lane is not None and left_lane.contains_points(np.array([np.array([state_cr.position[0],
                                                                                             state_cr.position[1]])]))[0]:
                    obstacle_states_left_lane.append(state_cr)
                elif same_lane.contains_points(np.array([np.array([state_cr.position[0], state_cr.position[1]])]))[0]:
                    obstacle_states_same_lane.append(state_cr)
        return obstacle_states_same_lane, obstacle_states_right_lane, obstacle_states_left_lane

    def keeps_safe_distance(self, state: State, s_ego, obstacle_states_same_lane_clc,
                            obstacle_states_same_lane_cr: List[State]) -> bool:
        for idx, obs in enumerate(obstacle_states_same_lane_clc):
            if obs[0] - s_ego > 0 \
                    and obs[0] - s_ego < self.safe_distance(state.velocity,
                                                            obstacle_states_same_lane_cr[idx].velocity,
                                                            self._ego_vehicle_param.get("a_min"),
                                                            self._other_vehicles_param.get("a_min"),
                                                            self._ego_vehicle_param.get("t_react"),
                                                            0, 0, # TODO change
                                                            self._ego_vehicle_param.get("j_min"),
                                                            self._other_vehicles_param.get("j_min"),
                                                            self._simulation_param.get("dt"),
                                                            s_ego, obs[0], self._ego_vehicle_param.get("a_max"),
                                                            self._ego_vehicle_param.get("j_max")):
                return False
            else:
                return True

    @staticmethod
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

    @staticmethod
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
        s_lead_list, v_lead_list = SimpleMonitor.simulate_vehicle_braking(v_lead_list, s_lead_list, a_lead,
                                                                          a_min_lead, dt, j_min_lead)

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
        s_acc_list, v_acc_list = SimpleMonitor.simulate_vehicle_braking(v_acc_list, s_acc_list, a_follow, a_min_follow,
                                                                        dt, j_min_follow)

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

    def evaluate_trajectory(self, scenario: Scenario, trajectory: List[State]):
        """
        :param scenario: CommonRoad scenario
        :param trajectory: ego vehicle trajectory
        """

        data = {}
        for state in trajectory:
            for key, value in self._formula_predicates.items():
                for predicate in value:
                    if data.get(predicate) is None:
                        data[predicate] = []
                    data[predicate].append((state.time_step, self.evaluate_predicates(predicate, state, scenario)))

        for rule in self._rules:
            print(rule(data, quantitative=False))
