from typing import List, Dict, Set
from predicate_collection import PredicateCollection
from commonroad.scenario.lanelet import LaneletNetwork
from common.vehicle import Vehicle


class SafetyPredicateCollection(PredicateCollection):
    def __init__(self, lanelet_network: LaneletNetwork, simulation_param: Dict, ego_vehicle_param: Dict, other_vehicles_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        super().__init__(lanelet_network, simulation_param, ego_vehicle_param, other_vehicles_param)
        #self._ego_lanelet_id, self._ego_lanelet, self._ego_lane, self._curvilinear_cosy_ego_lane, self._left_lane, \
        #self._right_lane = update_ego_lane_info(scenario, initial_state, self._ego_vehicle_param)

    # def brakes_abruptly(self, state: State, obstacle_ids: List[int], j_min_abrupt: float,
    #                     delta_a_abrupt: float) -> bool:
    #     """ Predicate to check whether an obstacle brakes abruptly
    #
    #     :param state: CommonRoad state
    #     :param obstacle_ids: IDs of obstacles on lanelet at specific time step
    #     :param j_min_abrupt: the maximum allowed jerk for a braking maneuver to be considered not abrupt
    #     :param delta_a_abrupt: the maximum allowed acceleration difference between two vehicles for a braking maneuver
    #     to be considered not abrupt
    #     :return: boolean indicating if vehicle brakes abruptly at current state
    #     """
    #     if len(obstacle_ids) == 0 and state.jerk < j_min_abrupt:
    #         return True
    #     s_ego, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(state.position[0], state.position[1])
    #     for idx, obs_id in enumerate(obstacle_ids):
    #         obs_state = self._scenario.obstacle_by_id(obs_id).prediction.trajectory.state_list[state.time_step-1]
    #         s_obs, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(obs_state.position[0],
    #                                                                                  obs_state.position[1])
    #         delta_s = s_obs - s_ego
    #         if delta_s > self._ego_vehicle_param.get("fov"):
    #             continue
    #
    #         if state.jerk < j_min_abrupt and obs_state.acceleration < 0 \
    #                 and state.acceleration - obs_state.acceleration < delta_a_abrupt:
    #             return True
    #     return False

    def _keeps_speed_limit(self, velocity: float, lanelet_ids: Set[int]) -> bool:
        """
        Predicate for speed limit evaluation

        :param velocity: Velocity of vehicle
        :param lanelet_ids: IDs of lanelets the vehicle is on
        :returns Boolean indicating speed limit satisfaction
        """
        speed_limits = []
        for lanelet_id in lanelet_ids:
            lanelet = self._lanelet_network.find_lanelet_by_id(lanelet_id)
            for traffic_sign_id in lanelet.traffic_signs:
                lanelet_speed_limits = \
                    self._lanelet_network.find_traffic_sign_by_id(traffic_sign_id).speed_limit(self._country)
                speed_limits.append(lanelet_speed_limits)
        if min(speed_limits) < velocity:
            return False
        else:
            return True

    @staticmethod
    def safe_distance(v_follow: float, v_lead: float, a_min_follow: float,
                      a_min_lead: float, t_react_follow: float) -> float:

        u_max_follow = (v_follow**2) / (2 * abs(a_min_follow)) + v_follow * t_react_follow
        v_lead_t_react = v_lead + a_min_lead * t_react_follow
        t_stop_lead_remain = v_lead_t_react / abs(a_min_lead)
        delta_s_lead = v_lead * t_react_follow + 0.5 * a_min_lead * t_react_follow**2
        t_stop_ego = v_follow / abs(a_min_follow)

        precondition = (delta_s_lead <= u_max_follow and abs(a_min_lead) < abs(a_min_follow)
                        and v_lead_t_react < v_follow and t_stop_ego < t_stop_lead_remain)

        d_safe_1 = (v_lead - abs(a_min_lead) * t_react_follow - v_follow) / \
                   (-2 * (abs(a_min_lead) - abs(a_min_follow))) - \
                   v_lead * t_react_follow + 0.5 * abs(a_min_lead) * t_react_follow**2

        d_safe_2 = v_lead**2 / (-2 * abs(a_min_lead)) - v_follow**2 / (-2 * abs(a_min_follow)) + \
                   v_follow * t_react_follow

        if precondition:
            return d_safe_1
        else:
            return d_safe_2

    # def _keeps_safe_distance(self, state: State, obstacle_ids: List[int]) -> bool:
    #     """
    #     Predicate for safe distance evaluation
    #
    #     :param state: CommonRoad state
    #     :param obstacle_ids: IDs of obstacles on lanelet at specific time step
    #     :returns Boolean indicating predicate satisfaction
    #     """
    #     s_ego, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(state.position[0], state.position[1])
    #     safe_distance_satisfied = True
    #     for idx, obs_id in enumerate(obstacle_ids):
    #         obs_state = self._scenario.obstacle_by_id(obs_id).prediction.trajectory.state_list[state.time_step-1]
    #         s_obs, _ = self._curvilinear_cosy_ego_lane.convert_to_curvilinear_coords(obs_state.position[0],
    #                                                                                  obs_state.position[1])
    #         if s_obs - s_ego > self._ego_vehicle_param.get("fov"):
    #             continue
    #         if 0 < s_obs - s_ego < safe_distance(state.velocity, obs_state.velocity,
    #                                              self._ego_vehicle_param.get("a_min"),
    #                                              self._other_vehicles_param.get("a_min"),
    #                                              self._ego_vehicle_param.get("t_react"), state.acceleration,
    #                                              obs_state.acceleration, self._ego_vehicle_param.get("j_min"),
    #                                              self._other_vehicles_param.get("j_min"),
    #                                              self._simulation_param.get("dt"), s_ego, s_obs,
    #                                              self._ego_vehicle_param.get("a_max"),
    #                                              self._ego_vehicle_param.get("j_max")):
    #             safe_distance_satisfied = False
    #
    #     return safe_distance_satisfied

    # def congestion_on_lanelet(self, lanelet: Lanelet, time_step: int, min_num_vehicles: int,
    #                           max_dist_between_cars: float, v_max: float):
    #     """ Predicate to check whether there is congestion on a lanelet at a specific time step.
    #
    #     :param lanelet: CommonRoad lanelet to be evaluated
    #     :param time_step: time step at which congestion is evaluated
    #     :param min_num_vehicles: minimum number of vehicles that have to be part of the congestion
    #     :param max_dist_between_cars: maximum distance between two cars that are part of the congestion
    #     :param v_max: maximum velocity obstacles are allowed to drive in the congestion
    #     :return: boolean indicating if there is a congestion on a lanelet
    #     """
    #     obstacles_on_lane = lanelet.dynamic_obstacles_on_lanelet[time_step]
    #     if len(obstacles_on_lane) < min_num_vehicles:
    #         return False
    #     for obs_id in obstacles_on_lane:
    #         obstacle = self._scenario.obstacle_by_id(obs_id)
    #     # TODO
    #         # (front, front_distance) = front_vehicles(obstacle, obstacles_on_lane)
    #         # if obstacle.get_velocity(time_step) > maximum_velocity \
    #         #         or (front and front_distance > maximum_distance_between_cars) \
    #         #         or (back and -back_distance > maximum_distance_between_cars):
    #         #     return False
    #     return True

    def evaluate_predicates(self, vehicle: Vehicle) -> Dict[str, List[bool]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param vehicle: vehicle object
        """
        predicate_trace = {"keeps_speed_limit": []}
                           #"keeps_safe_distance": [],
                          # "brakes_abruptly": []}
        for idx in range(len(vehicle.state_list_cr)):
            predicate_trace["keeps_speed_limit"].append(self._keeps_speed_limit(vehicle.states_lon[idx].v,
                                                                                vehicle.lanelet_assignment[idx]))
            #predicate_trace["keeps_safe_distance"][idx] = self.keeps_safe_distance()
            #predicate_trace["brakes_abruptly"][idx] = self.brakes_abruptly()

        return predicate_trace
