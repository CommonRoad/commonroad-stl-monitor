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
import cProfile
from predicates import *


class SimpleMonitor:
    def __init__(self, traffic_rules: Dict[str, str], predicates_per_mtl: Dict[str, List[str]], simulation_param: Dict,
                 ego_vehicle_param: Dict, other_vehicles_param: Dict, scenario: Scenario, initial_state: State):
        """
        :param traffic_rules: dictionary with MTL formulas of traffic rules
        :param predicates_per_mtl: dictionary with predicates for each MTL formula
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param scenario: CommonRoad scenario
        """
        self._rules = self.create_monitors(traffic_rules)
        self._predicates_per_mtl = predicates_per_mtl
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._scenario = scenario
        self._ego_lanelet_id, self._ego_lanelet, self._ego_lane, self._curvilinear_cosy_ego_lane, self._left_lane, \
        self._right_lane = update_ego_lane_info(scenario, initial_state, self._ego_vehicle_param)

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
        s_ego, d_ego = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(state.position[0],
                                                                                     state.position[1])
        obstacle_states_same_lane_cr, obstacle_states_right_lane_cr, obstacle_states_left_lane_cr = \
            self.obstacles_at_time_step(state.time_step, scenario.dynamic_obstacles, self._right_lane,
                                        self._left_lane, self._ego_lane)
        obstacle_states_same_lane_clc = self.convert_to_curvilinear(obstacle_states_same_lane_cr,
                                                                    self._curvilinear_cosy_ego_lane)
        #obstacle_states_same_lane_cr_fov = self.obstacles_fov(state.time_step, s_ego, obstacle_states_same_lane_clc)

        if predicate == "keeps_speed_limit":
            return keeps_speed_limit(state, self._ego_lane)
        if predicate == "keeps_safe_distance":
            return keeps_safe_distance(state, s_ego, obstacle_states_same_lane_clc, obstacle_states_same_lane_cr)


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




    @staticmethod


    def evaluate_trajectory(self, scenario: Scenario, trajectory: List[State]):
        """
        :param scenario: CommonRoad scenario
        :param trajectory: ego vehicle trajectory
        """

        data = {}
        for state in trajectory:
            for key, value in self._predicates_per_mtl.items():
                for predicate in value:
                    if data.get(predicate) is None:
                        data[predicate] = []
                    #pr = cProfile.Profile()
                    #pr.enable()
                    data[predicate].append((state.time_step, self.evaluate_predicates(predicate, state, scenario)))
                    #pr.disable()
                    #pr.print_stats(sort='time')

        for idx, rule in enumerate(self._rules):
            predicate_list = self._predicates_per_mtl.get(idx + 1)
            formula_predicates = {}
            for predicate in predicate_list:
                formula_predicates[predicate] = data[predicate]
            print(rule(formula_predicates, quantitative=False))
