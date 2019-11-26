from commonroad.scenario.trajectory import State
from typing import List, Dict
from util import update_ego_lane_info
from commonroad.scenario.scenario import Scenario
from util import safe_distance
from commonroad.scenario.lanelet import Lanelet


class Predicate:
    def __init__(self, simulation_param: Dict,
                 ego_vehicle_param: Dict, other_vehicles_param: Dict, scenario: Scenario, initial_state: State):
        """
        :param initial_state: initial state of ego vehicle
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param scenario: CommonRoad scenario
        """
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._scenario = scenario
        self._ego_lanelet_id, self._ego_lanelet, self._ego_lane, self._curvilinear_cosy_ego_lane, self._left_lane, \
        self._right_lane = update_ego_lane_info(scenario, initial_state, self._ego_vehicle_param)

    def brakes_abruptly(self, state: State, obstacle_ids: List[int], j_min_abrupt: float,
                        delta_a_abrupt: float) -> bool:
        """ Predicate to check whether an obstacle brakes abruptly

        :param state: CommonRoad state
        :param obstacle_ids: IDs of obstacles on lanelet at specific time step
        :param j_min_abrupt: the maximum allowed jerk for a braking maneuver to be considered not abrupt
        :param delta_a_abrupt: the maximum allowed acceleration difference between two vehicles for a braking maneuver
        to be considered not abrupt
        :return: boolean indicating if vehicle brakes abruptly at current state
        """
        if len(obstacle_ids) == 0 and state.jerk < j_min_abrupt:
            return True
        s_ego, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(state.position[0], state.position[1])
        for idx, obs_id in enumerate(obstacle_ids):
            obs_state = self._scenario.obstacle_by_id(obs_id).prediction.trajectory.state_list[state.time_step-1]
            s_obs, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(obs_state.position[0],
                                                                                     obs_state.position[1])
            delta_s = s_obs - s_ego
            if delta_s > self._ego_vehicle_param.get("fov"):
                continue

            if state.jerk < j_min_abrupt and obs_state.acceleration < 0 \
                    and state.acceleration - obs_state.acceleration < delta_a_abrupt:
                return True
        return False

    @staticmethod
    def keeps_speed_limit(velocity: float, speed_limit: float):
        """
        Predicate for speed limit evaluation

        :param velocity: Velocity of CommonRoad state
        :param speed_limit: Speed limit at state position
        :returns Boolean indicating predicate satisfaction
        """
        if speed_limit < velocity:
            return False
        else:
            return True

    def keeps_safe_distance(self, state: State, obstacle_ids: List[int]) -> bool:
        """
        Predicate for safe distance evaluation

        :param state: CommonRoad state
        :param obstacle_ids: IDs of obstacles on lanelet at specific time step
        :returns Boolean indicating predicate satisfaction
        """
        s_ego, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(state.position[0], state.position[1])
        safe_distance_satisfied = True
        for idx, obs_id in enumerate(obstacle_ids):
            obs_state = self._scenario.obstacle_by_id(obs_id).prediction.trajectory.state_list[state.time_step-1]
            s_obs, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(obs_state.position[0],
                                                                                     obs_state.position[1])
            if s_obs - s_ego > self._ego_vehicle_param.get("fov"):
                continue
            if 0 < s_obs - s_ego < safe_distance(state.velocity, obs_state.velocity,
                                                 self._ego_vehicle_param.get("a_min"),
                                                 self._other_vehicles_param.get("a_min"),
                                                 self._ego_vehicle_param.get("t_react"), state.acceleration,
                                                 obs_state.acceleration, self._ego_vehicle_param.get("j_min"),
                                                 self._other_vehicles_param.get("j_min"),
                                                 self._simulation_param.get("dt"), s_ego, s_obs,
                                                 self._ego_vehicle_param.get("a_max"),
                                                 self._ego_vehicle_param.get("j_max")):
                safe_distance_satisfied = False

        return safe_distance_satisfied

    def congestion_on_lanelet(self, lanelet: Lanelet, time_step: int, min_num_vehicles: int,
                              max_dist_between_cars: float, v_max: float):
        """ Predicate to check whether there is congestion on a lanelet at a specific time step.

        :param lanelet: CommonRoad lanelet to be evaluated
        :param time_step: time step at which congestion is evaluated
        :param min_num_vehicles: minimum number of vehicles that have to be part of the congestion
        :param max_dist_between_cars: maximum distance between two cars that are part of the congestion
        :param v_max: maximum velocity obstacles are allowed to drive in the congestion
        :return: boolean indicating if there is a congestion on a lanelet
        """
        obstacles_on_lane = lanelet.dynamic_obstacles_on_lanelet[time_step]
        if len(obstacles_on_lane) < min_num_vehicles:
            return False
        for obs_id in obstacles_on_lane:
            obstacle = self._scenario.obstacle_by_id(obs_id)
        # TODO
            # (front, front_distance) = front_vehicles(obstacle, obstacles_on_lane)
            # if obstacle.get_velocity(time_step) > maximum_velocity \
            #         or (front and front_distance > maximum_distance_between_cars) \
            #         or (back and -back_distance > maximum_distance_between_cars):
            #     return False
        return True

