from typing import List, Dict, Set, Union
from predicates.predicate_collection import PredicateCollection
from common.road_network import RoadNetwork
from common.vehicle import Vehicle, VehicleLocalization
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry, TrafficSignIDGermany
from predicates.position_predicates import PositionPredicateCollection


class VehicleStatePredicateCollection(PredicateCollection):
    def __init__(self, road_network: RoadNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, traffic_rules_param: Dict):
        """
        :param road_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        """
        super().__init__(road_network, simulation_param, ego_vehicle_param, other_vehicles_param,
                         traffic_rules_param)

    def _brakes_abruptly(self, a_ego: float, j_ego: float, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """ Predicate to check whether an obstacle brakes abruptly

        :param state: CommonRoad state
        :param obstacle_ids: IDs of obstacles on lanelet at specific time step
        :param j_min_abrupt: the maximum allowed jerk for a braking maneuver to be considered not abrupt
        :param delta_a_abrupt: the maximum allowed acceleration difference between two vehicles for a braking maneuver
        to be considered not abrupt
        :return: boolean indicating if vehicle brakes abruptly at current state
        """
        if a_ego >= 0:
            return False

        a_min_other = 100
        for veh in other_vehicles:
            if veh.classification[time_step] == VehicleLocalization.EGO_LANE_FRONT:
                if veh.states_lon[time_step].a < a_min_other:
                    a_min_other = veh.states_lon[time_step].a

        if a_min_other == 100 and j_ego < self._traffic_rule_param.get("j_min_abrupt"):  # no leading vehicle
            return True

        if j_ego < self._traffic_rule_param.get("j_min_abrupt") and \
                a_ego - a_min_other < self._traffic_rule_param.get("delta_a_abrupt"):
            return True

        return False

    def _keeps_lane_speed_limit(self, velocity: float, lanelet_ids: Set[int]) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param velocity: Velocity of vehicle
        :param lanelet_ids: IDs of lanelets the vehicle is on
        :returns Boolean indicating speed limit satisfaction
        """
        speed_limit = self.active_speed_limit(lanelet_ids)
        if speed_limit is None:
            return True
        elif speed_limit < velocity:
            return False
        else:
            return True

    def active_speed_limit(self, lanelet_ids: Set[int]) -> Union[float, None]:
        speed_limits = []
        for lanelet_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            for traffic_sign_id in lanelet.traffic_signs:
                traffic_sign = self._road_network.lanelet_network.find_traffic_sign_by_id(traffic_sign_id)
                if self._country == SupportedTrafficSignCountry.GERMANY:
                    for elem in traffic_sign.traffic_sign_elements:
                        if elem.traffic_sign_element_id == TrafficSignIDGermany.MAXSPEED.value:
                            speed_limits.append(float(elem.additional_values[0]))
                        if elem.traffic_sign_element_id == TrafficSignIDGermany.MAXSPEED.value:
                            speed_limits.append(50.0)
        if len(speed_limits) == 0:
            return None
        else:
            return min(speed_limits)

    def _keeps_braking_speed_limit(self, velocity: float) -> bool:
        """
        Predicate for velocity limit to ensure comfortable braking for speed limits

        :param velocity: Velocity of vehicle
        :returns Boolean indicating speed limit satisfaction
        """
        if self._ego_vehicle_param.get("braking_speed_limit") < velocity:
            return False
        else:
            return True

    def leading_vehicle(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int):
        front_vehicles = PositionPredicateCollection.front_vehicle_same_lane(ego_vehicle, other_vehicles, time_step)

        return front_vehicles

    def _keeps_flow_min_speed_limit(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Predicate for minimum speed limit evaluation

        :param velocity: Velocity of vehicle
        :param time_step: current time step
        :returns Boolean indicating speed limit satisfaction
        """
        front_vehicle = self.leading_vehicle(ego_vehicle, other_vehicles, time_step)
        for veh in other_vehicles:
            if VehicleLocalization.EGO_LANE_FRONT in veh.classification[time_step]:
                if veh.states_lon[time_step].v - ego_vehicle.states_lon[time_step].v > \
                        self._traffic_rule_param.get("min_velocity_dif"):
                    return False
        return True

    def _close_to_max_velocity(self, velocity: float, v_max_ego: float, lanelet_ids: Set[int]) -> bool:
        """
        Predicate for evaluation if vehicle drives close to maximum velocity

        :param velocity: Velocity of vehicle
        :param time_step: current time step
        :returns Boolean indicating speed limit satisfaction
        """
        speed_limit = self.active_speed_limit(lanelet_ids)
        if speed_limit is None:
            speed_limit = self._traffic_rule_param.get("desired_highway_velocity")
        if velocity - self._traffic_rule_param.get("min_velocity_dif") < min(speed_limit, v_max_ego):
            return False
        else:
            return True

    def _keeps_sign_min_speed_limit(self, velocity: float, lanelet_ids: Set[int]) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param velocity: Velocity of vehicle
        :param lanelet_ids: IDs of lanelets the vehicle is on
        :returns Boolean indicating speed limit satisfaction
        """
        speed_limits = [0]
        for lanelet_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
            for traffic_sign_id in lanelet.traffic_signs:
                traffic_sign = self._road_network.lanelet_network.find_traffic_sign_by_id(traffic_sign_id)
                if self._country == SupportedTrafficSignCountry.GERMANY:
                    for elem in traffic_sign.traffic_sign_elements:
                        if elem.traffic_sign_element_id == TrafficSignIDGermany.MINSPEED.value:
                            speed_limits.append(float(elem.additional_values[0]))
        if min(speed_limits) > velocity:
            return False
        else:
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

    def _in_standstill(self, velocity: float):
        if self._traffic_rule_param.get("standstill_error") < velocity < \
                self._traffic_rule_param.get("standstill_error"):
            return True
        else:
            return False

    @staticmethod
    def _safe_distance(v_follow: float, v_lead: float, a_min_follow: float,
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
        if s_lead - s_follow < self._safe_distance(v_follow, v_lead, a_min_follow, a_min_lead, t_react_follow):
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
                           "keeps_flow_min_speed_limit": {ego_vehicle.id: {}},
                           "keeps_sign_min_speed_limit": {ego_vehicle.id: {}},
                           "keeps_braking_speed_limit": {ego_vehicle.id: {}},
                           "brakes_abruptly": {ego_vehicle.id: {}},
                           "keeps_safe_distance": {}}

        for idx in range(len(ego_vehicle.state_list_cr)):
            time_step = ego_vehicle.state_list_cr[idx].time_step
            predicate_trace["keeps_lane_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_lane_speed_limit(ego_vehicle.states_lon[idx].v, ego_vehicle.lanelet_assignment[idx])
            predicate_trace["keeps_fov_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_fov_speed_limit(ego_vehicle.states_lon[idx].v)
            predicate_trace["keeps_braking_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_braking_speed_limit(ego_vehicle.states_lon[idx].v)
            predicate_trace["keeps_flow_min_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_flow_min_speed_limit(ego_vehicle, other_vehicles, idx)
            predicate_trace["keeps_sign_min_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_sign_min_speed_limit(ego_vehicle.states_lon[idx].v, ego_vehicle.lanelet_assignment[idx])
            predicate_trace["brakes_abruptly"][ego_vehicle.id][time_step] = \
                self._brakes_abruptly(ego_vehicle.states_lon[idx].a, ego_vehicle.jerk_profile[idx], other_vehicles, idx)

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
