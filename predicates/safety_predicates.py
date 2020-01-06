from typing import List, Dict, Set
from predicates.predicate_collection import PredicateCollection
from commonroad.scenario.lanelet import LaneletNetwork
from common.vehicle import Vehicle, VehicleLocalization


class SafetyPredicateCollection(PredicateCollection):
    def __init__(self, lanelet_network: LaneletNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, traffic_rules_param: Dict):
        """
        :param lanelet_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        """
        super().__init__(lanelet_network, simulation_param, ego_vehicle_param, other_vehicles_param,
                         traffic_rules_param)

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

    def _keeps_lane_speed_limit(self, velocity: float, lanelet_ids: Set[int]) -> bool:
        """
        Predicate for lanelet speed limit evaluation

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

    def _keeps_min_speed_limit(self, velocity: float, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Predicate for minimum speed limit evaluation

        :param velocity: Velocity of vehicle
        :param time_step: current time step
        :returns Boolean indicating speed limit satisfaction
        """
        for veh in other_vehicles:
            if veh.classification[time_step] == VehicleLocalization.EGO_LANE_FRONT:
                if veh.states_lon[time_step].v - velocity < self._traffic_rule_param.get("min_velocity_dif"):
                    return False
        return True

    def _keeps_fov_speed_limit(self, velocity: float) -> bool:
        """
        Predicate for field of view speed limit evaluation

        :param velocity: Velocity of vehicle
        :returns boolean indicating speed limit satisfaction
        """
        if self._ego_vehicle_param.get("fov_speed_limit") < velocity:
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

        d_safe_2 = (v_lead**2) / (-2 * abs(a_min_lead)) - (v_follow**2) / (-2 * abs(a_min_follow)) + \
                   v_follow * t_react_follow

        if precondition:
            return d_safe_1
        else:
            return d_safe_2

    def _keeps_safe_distance(self, s_follow: float, s_lead:float, v_follow: float, v_lead: float,
                             a_min_follow: float, a_min_lead: float, t_react_follow: float) -> bool:
        if s_lead - s_follow < self.safe_distance(v_follow, v_lead, a_min_follow, a_min_lead, t_react_follow):
            return False
        else:
            return True

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"keeps_lane_speed_limit": {ego_vehicle.id: {}},
                           "keeps_fov_speed_limit": {ego_vehicle.id: {}},
                           "keeps_min_speed_limit": {ego_vehicle.id: {}},
                           "keeps_safe_distance": {}}

        for idx in range(len(ego_vehicle.state_list_cr)):
            time_step = ego_vehicle.state_list_cr[idx].time_step
            predicate_trace["keeps_lane_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_lane_speed_limit(ego_vehicle.states_lon[idx].v, ego_vehicle.lanelet_assignment[idx])
            predicate_trace["keeps_fov_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_fov_speed_limit(ego_vehicle.states_lon[idx].v)
            predicate_trace["keeps_min_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_min_speed_limit(ego_vehicle.states_lon[idx].v, other_vehicles, idx)
            #predicate_trace["brakes_abruptly"][idx] = self.brakes_abruptly()

        for other_vehicle in other_vehicles:
            for idx in range(len(other_vehicle.state_list_cr)):
                time_step = other_vehicle.state_list_cr[idx].time_step
                if VehicleLocalization.EGO_LANE_FRONT in other_vehicle.classification[idx]:
                    if predicate_trace["keeps_safe_distance"].get(other_vehicle.id) is None:
                        predicate_trace["keeps_safe_distance"][other_vehicle.id] = {}
                    predicate_trace["keeps_safe_distance"][other_vehicle.id][time_step] = \
                        self._keeps_safe_distance(ego_vehicle.states_lon[idx].s, other_vehicle.states_lon[idx].s,
                                                  ego_vehicle.states_lon[idx].v, other_vehicle.states_lon[idx].v,
                                                  self._ego_vehicle_param.get("a_min"),
                                                  self._other_vehicles_param.get("a_min"),
                                                  self._ego_vehicle_param.get("t_react"))
        return predicate_trace
