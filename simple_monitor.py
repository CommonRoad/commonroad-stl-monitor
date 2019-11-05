import mtl
from typing import Dict, List, Tuple
from commonroad.scenario.scenario import Scenario
from vehicle import Vehicle
from commonroad.scenario.trajectory import State
import numpy as np


class SimpleMonitor:
    def __init__(self, traffic_rules: Dict[str, str], predicates: Dict[str, List[str]],
                 simulation_param: Dict, ego_vehicle_param: Dict, other_vehicles_param: Dict):
        """
        :param traffic_rules: dictionary with MTl traffic rule formalizations
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        self._rules = self.create_monitors(traffic_rules)
        self._formula_predicates = predicates

    @staticmethod
    def create_monitors(mtl_rules: Dict[str, str]) -> List[str]:
        monitors = []
        for key, value in mtl_rules.items():
            monitor = mtl.parse(value)
            monitors.append(monitor)
        return monitors

    def keeps_speed_limit(self, state: State, scenario: Scenario):
        lanelet_id = scenario.lanelet_network.find_lanelet_by_position([state.position])
        lanlet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0][0])
        if lanlet.speed_limit < state.velocity:
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
    def safe_distance(v_acc: float, v_lead: float, a_min_acc: float, a_min_lead: float, t_react: float,
                      a_acc: float, a_lead: float, j_min_acc: float, j_min_lead: float, dt: float,
                      s_x_acc: float, s_x_lead: float, a_max_acc: float, j_max_acc: float) -> float:
        """
        Calculates safe distance between two vehicles, if both have the same negative acceleration

        :param v_acc: current velocity of ACC vehicle [m/s]
        :param v_lead: current velocity of leading vehicle [m/s]
        :param a_min_lead: minimum negative acceleration of leading vehicle [m/s^2]
        :param a_min_acc: minimum negative acceleration of ACC vehicle [m/s^2]
        :param a_max_acc: maximum positive acceleration of ACC vehicle [m/s^2]
        :param t_react: reaction time of ACC vehicle [s]
        :param a_acc: current acceleration of ACC vehicle [m/s^2]
        :param a_lead: current acceleration of leading vehicle [m/s^2]
        :param s_x_acc: current position of ACC vehicle [m/s^2]
        :param s_x_lead: current position of leading vehicle [m/s^2]
        :param j_min_acc: minimum negative jerk of ACC vehicle [m/s^3]
        :param j_min_lead: minimum negative acceleration of leading vehicle [m/s^3]
        :param j_max_acc: maximum positive jerk of ACC vehicle [m/s^3]
        :param dt: time step size [s]
        :return: safe distance between ACC vehicle and leading vehicle [m]
        """
        t = 0  # time step
        s_acc_list = [s_x_acc]
        v_acc_list = [v_acc]
        s_lead_list = [s_x_lead]
        v_lead_list = [v_lead]
        initial_distance = s_x_lead - s_x_acc

        if v_acc == 0:
            return 0.0

        # Leading vehicle braking:
        s_lead_list, v_lead_list = SimpleMonitor.simulate_vehicle_braking(v_lead_list, s_lead_list, a_lead,
                                                                          a_min_lead, dt, j_min_lead)

        # ACC vehicle motion during reaction time:
        while t < t_react and v_acc > 0:
            if v_acc + min(a_max_acc, a_acc + j_max_acc * dt) * dt >= 0:
                a_acc = min(a_max_acc, a_acc + j_max_acc * dt)
                s_acc_curr = s_acc_list[-1] + v_acc * dt + 0.5 * a_acc * dt ** 2
                v_acc = v_acc + a_acc * dt
            else:
                a_acc = min(a_max_acc, a_acc + j_max_acc * dt)
                delta_t_short = v_acc / abs(a_acc)
                s_acc_curr = s_acc_list[-1] + v_acc * delta_t_short + 0.5 * a_acc * delta_t_short ** 2
                v_acc = 0
            s_acc_list.append(s_acc_curr)
            v_acc_list.append(v_acc)

            s_acc_curr = s_acc_list[-1] + v_acc * dt
            s_acc_list.append(s_acc_curr)
            t += dt

        # ACC vehicle braking:
        s_acc_list, v_acc_list = SimpleMonitor.simulate_vehicle_braking(v_acc_list, s_acc_list, a_acc, a_min_acc,
                                                                        dt, j_min_acc)

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
                    data[predicate].append((state.time_step, self.keeps_speed_limit(state, scenario)))

        for rule in self._rules:
            print(rule(data, quantitative=False))
