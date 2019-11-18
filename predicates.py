from commonroad.scenario.trajectory import State
from typing import List, Tuple, Dict
import numpy as np
from util import update_ego_lane_info
from commonroad.scenario.scenario import Scenario
from util import safe_distance
from commonroad_ccosy.geometry.util import chaikins_corner_cutting, resample_polyline
import sys

class Predicate:
    def __init__(self, simulation_param: Dict,
                 ego_vehicle_param: Dict, other_vehicles_param: Dict, scenario: Scenario, initial_state: State):
        """
        :param traffic_rules: dictionary with MTL formulas of traffic rules
        :param predicates_per_mtl: dictionary with predicates for each MTL formula
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

    def brakes_abruptly(self, state: State, obstacle_ids: List[int], j_min_abrupt: float, delta_a_abrupt: float) -> bool:
        """ Predicate to check whether an obstacle brakes abruptly.

        :param j_ego: jerk of the considered vehicle
        :param a_ego: acceleration of the considered vehicle
        :param a_lead: acceleration of the leading vehicle
        :param j_min_abrupt: the maximum allowed jerk for a braking to be considered abrupt if also other
        conditions hold
        :param delta_a_abrupt: the maximum difference between the acceleration of the considered and
        a possible front vehicle that is allowed such that the braking is not classified as abrupt if the considered vehicle
        brakes with a jerk that violates the maximum_allowed_jerk limit
        :return: true if braking occurs with a higher jerk than allowed and without a front vehicle, or with a higher jerk
        than allowed ands a higher acceleration difference to a front vehicle than the allowed maximum difference, false
        otherwise.
        """
        if len(obstacle_ids) == 0 and state.jerk < j_min_abrupt:
            # If there is no front vehicle any braking that violates the maximum allowed jerk limit is considered abrupt
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
                # If there is a front vehicle, the braking is considered abrupt if the acceleration difference between
                # the two cars is greater than the user-defined maximum acceleration difference to the front vehicle.
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

        :param state: Velocity of CommonRoad state
        :param s_ego: Speed limit at state position
        :param obstacle_states_same_lane_clc: List of obstacle positions in curvilinear coordinate system
        :param obstacle_states_same_lane_cr: List of CommonRoad states of obstacles
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
                                                 obs_state.acceleration,   self._ego_vehicle_param.get("j_min"),
                                                 self._other_vehicles_param.get("j_min"),
                                                 self._simulation_param.get("dt"), s_ego, s_obs,
                                                 self._ego_vehicle_param.get("a_max"),
                                                 self._ego_vehicle_param.get("j_max")):
                safe_distance_satisfied = False

        return safe_distance_satisfied
