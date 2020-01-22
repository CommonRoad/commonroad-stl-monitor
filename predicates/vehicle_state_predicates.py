from typing import List, Dict, Set, Union
from predicates.predicate_collection import PredicateCollection
from common.road_network import RoadNetwork
from common.vehicle import Vehicle
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

    def _unnecessary_braking(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """ Predicate to check whether an obstacle brakes abruptly

        :param ego_vehicle: ego vehicle
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :return: boolean indicating if vehicle brakes abruptly at current state
        """
        v_ego = ego_vehicle.states_lon[time_step].v
        a_ego = ego_vehicle.states_lon[time_step].a
        j_ego = ego_vehicle.states_lon[time_step].j
        if a_ego >= 0:
            return False
        if self.velocity_reduction_necessary(v_ego):
            return False
        ego_vehicle_lanelets = ego_vehicle.lanelet_assignment[time_step]

        a_min_other = None
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if PositionPredicateCollection.in_front_of(ego_vehicle.states_lon[time_step].s,
                                                       veh_o.states_lon[time_step].s) and \
                    PositionPredicateCollection.same_lane(ego_vehicle_lanelets, veh_o.lanelet_assignment[time_step]) \
                    and veh_o.states_lon[time_step].v - v_ego < self._traffic_rule_param.get("min_velocity_dif"):
                if a_min_other is None or veh_o.states_lon[time_step].a < a_min_other:
                    a_min_other = veh_o.states_lon[time_step].a

        if a_min_other is None and (a_ego < self._traffic_rule_param.get("a_abrupt") or
                                    j_ego < self._traffic_rule_param.get("j_abrupt")):
            # no leading vehicle
            return True
        elif a_min_other is not None and j_ego < self._traffic_rule_param.get("j_abrupt") and \
                a_ego - a_min_other < self._traffic_rule_param.get("a_abrupt"):
            return True
        else:
            return False

    def velocity_reduction_necessary(self, velocity: float):
        """
        Predicate to check whether a velocity reduction is necessary caused of safety reasons (currently only maximum
        velocity based on field of view and road conditions is evaluated, but active emergency maneuver or other
        information could also be considered)

        :param velocity: velocity of ego vehicle
        :return: boolean indicating satisfaction
        """
        v_max = min(self._ego_vehicle_param.get("road_condition_speed_limit"),
                    self._ego_vehicle_param.get("fov_speed_limit"))
        if v_max < velocity:
            return True
        else:
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
        """
        Finds minimum speed limit on provided lanelets

        :param lanelet_ids: set of lanelets which should be considered
        :returns minimum speed limit or None if no speed limit exists
        """
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
        :returns Boolean indicating satisfaction
        """
        if self._ego_vehicle_param.get("braking_speed_limit") < velocity:
            return False
        else:
            return True

    def _keeps_road_condition_speed_limit(self, velocity: float) -> bool:
        """
        Predicate for road condition speed limit evaluation

        :param velocity: Velocity of vehicle
        :returns Boolean indicating satisfaction
        """
        if self._ego_vehicle_param.get("road_condition_speed_limit") < velocity:
            return False
        else:
            return True

    @staticmethod
    def leading_vehicle(ego_vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
        """
        Predicate for velocity limit to ensure comfortable braking for speed limits

        :param ego_vehicle: ego vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction
        """
        pass
        # for veh_o in other_vehicles:
        #     if PositionPredicateCollection.in_front_of(s_veh, veh_o.states_lon[time_step].s) and \
        #             PositionPredicateCollection.same_lane(lanelets_veh, veh_o.lanelet_assignment[time_step]) and \
        #             veh_o.states_lon[time_step].v - v_veh < self._traffic_rule_param.get("min_velocity_dif"):
        #         return True
        #front_vehicles = PositionPredicateCollection.in_front_of(ego_vehicle.states_lon[time_step].s,
        #                                                         other_vehicles.states_lon[time_step].s,
        #                                                         time_step)

        #return front_vehicles[0]

    def slow_leading_vehicle(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int):
        """
        Predicate which evaluates if a slow leading vehicle exists if front of a vehicle

        :param vehicle: considered vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction
        """
        v_veh = vehicle.states_lon[time_step].v
        s_veh = vehicle.states_lon[time_step].s
        lanelets_veh = vehicle.lanelet_assignment[time_step]
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if PositionPredicateCollection.in_front_of(s_veh, veh_o.states_lon[time_step].s) and \
                    PositionPredicateCollection.same_lane(lanelets_veh, veh_o.lanelet_assignment[time_step]) and \
                    veh_o.states_lon[time_step].v - v_veh < self._traffic_rule_param.get("min_velocity_dif"):
                return True

        return False

    def _preserves_traffic_flow(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Predicate for minimum speed limit evaluation

        :param ego_vehicle: ego vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction
        """
        if self.slow_leading_vehicle(ego_vehicle, other_vehicles, time_step):
            return True
        else:
            v_max_lane = self.active_speed_limit(ego_vehicle.lanelet_assignment[time_step])
            if v_max_lane is None:
                v_max = min(self._ego_vehicle_param.get("road_condition_speed_limit"),
                            self._ego_vehicle_param.get("fov_speed_limit"),
                            self._ego_vehicle_param.get("braking_speed_limit"),
                            self._traffic_rule_param.get("desired_highway_velocity"))
            else:
                v_max = min(self._ego_vehicle_param.get("road_condition_speed_limit"),
                            self._ego_vehicle_param.get("fov_speed_limit"),
                            self._ego_vehicle_param.get("braking_speed_limit"),
                            self.active_speed_limit(ego_vehicle.lanelet_assignment[time_step]))
            if v_max - ego_vehicle.states_lon[time_step].v > self._traffic_rule_param.get("min_velocity_dif"):
                return False
            else:
                return True

    def _keeps_sign_min_speed_limit(self, velocity: float, lanelet_ids: Set[int]) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param velocity: Velocity of vehicle
        :param lanelet_ids: IDs of lanelets the vehicle is on
        :returns Boolean indicating satisfaction
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
        if max(speed_limits) > velocity:
            return False
        else:
            return True

    def _keeps_fov_speed_limit(self, velocity: float) -> bool:
        """
        Predicate for field of view speed limit evaluation

        :param velocity: Velocity of vehicle
        :returns boolean indicating satisfaction
        """
        if self._ego_vehicle_param.get("fov_speed_limit") < velocity:
            return False
        else:
            return True

    def _in_standstill(self, velocity: float):
        """
        Evaluation if vehicle is standing

        :param velocity: velocity of vehicle
        :returns boolean indicating satisfaction
        """
        if self._traffic_rule_param.get("standstill_error") < velocity < \
                self._traffic_rule_param.get("standstill_error"):
            return True
        else:
            return False

    @staticmethod
    def _safe_distance(v_follow: float, v_lead: float, a_min_follow: float,
                       a_min_lead: float, t_react_follow: float) -> float:
        """
        Calculates safe distance based on analytic formula

        :param v_follow: velocity of following vehicle
        :param v_lead: velocity of leading vehicle
        :param a_min_follow: minimum acceleration of following vehicle
        :param a_min_lead: minimum acceleration of leading vehicle
        :param t_react_follow: reaction time of following vehicle
        :returns boolean indicating satisfaction
        """
        u_max_follow = (v_follow**2) / (2 * abs(a_min_follow)) + v_follow * t_react_follow
        v_lead_t_react = v_lead + a_min_lead * t_react_follow
        t_stop_lead_remain = v_lead_t_react / abs(a_min_lead)
        delta_s_lead = v_lead * t_react_follow + 0.5 * a_min_lead * t_react_follow**2
        t_stop_ego = v_follow / abs(a_min_follow)

        precondition = (delta_s_lead <= u_max_follow and abs(a_min_lead) < abs(a_min_follow)
                        and v_lead_t_react < v_follow and t_stop_ego < t_stop_lead_remain)

        d_safe_1 = \
            (v_lead - abs(a_min_lead) * t_react_follow - v_follow) / \
            (-2 * (abs(a_min_lead) - abs(a_min_follow))) - \
            v_lead * t_react_follow + 0.5 * abs(a_min_lead) * t_react_follow**2

        d_safe_2 = \
            (v_lead**2) / (-2 * abs(a_min_lead)) - (v_follow**2) / \
            (-2 * abs(a_min_follow)) + v_follow * t_react_follow

        if precondition:
            return d_safe_1
        else:
            return d_safe_2

    def _keeps_safe_distance(self, s_follow: float, s_lead: float, v_follow: float, v_lead: float,
                             a_min_follow: float, a_min_lead: float, t_react_follow: float) -> bool:
        """
        Evaluates if safe distance is kept by following vehicle

        :param s_follow: longitudinal position of following vehicle
        :param s_lead: longitudinal position of leading vehicle
        :param v_follow: velocity of following vehicle
        :param v_lead: velocity of leading vehicle
        :param a_min_follow: minimum acceleration of following vehicle
        :param a_min_lead: minimum acceleration of leading vehicle
        :param t_react_follow: reaction time of following vehicle
        :returns boolean indicating satisfaction
        """
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
                           "preserves_traffic_flow": {ego_vehicle.id: {}},
                           "keeps_sign_min_speed_limit": {ego_vehicle.id: {}},
                           "keeps_braking_speed_limit": {ego_vehicle.id: {}},
                           "keeps_road_condition_speed_limit": {ego_vehicle.id: {}},
                           "unnecessary_braking": {ego_vehicle.id: {}},
                           "keeps_safe_distance": {}}

        for idx in range(len(ego_vehicle.state_list_cr)):
            time_step = ego_vehicle.state_list_cr[idx].time_step
            predicate_trace["keeps_lane_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_lane_speed_limit(ego_vehicle.states_lon[idx].v, ego_vehicle.lanelet_assignment[idx])
            predicate_trace["keeps_fov_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_fov_speed_limit(ego_vehicle.states_lon[idx].v)
            predicate_trace["keeps_braking_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_braking_speed_limit(ego_vehicle.states_lon[idx].v)
            predicate_trace["keeps_road_condition_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_road_condition_speed_limit(ego_vehicle.states_lon[idx].v)
            predicate_trace["preserves_traffic_flow"][ego_vehicle.id][time_step] = \
                self._preserves_traffic_flow(ego_vehicle, other_vehicles, idx)
            predicate_trace["keeps_sign_min_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_sign_min_speed_limit(ego_vehicle.states_lon[idx].v, ego_vehicle.lanelet_assignment[idx])
            predicate_trace["unnecessary_braking"][ego_vehicle.id][time_step] = \
                self._unnecessary_braking(ego_vehicle, other_vehicles, idx)

        for other_vehicle in other_vehicles:
            predicate_trace["keeps_safe_distance"][other_vehicle.id] = {}
            for idx in range(len(other_vehicle.state_list_cr)):
                time_step = other_vehicle.state_list_cr[idx].time_step
                if idx >= len(ego_vehicle.states_lon):
                    break
                predicate_trace["keeps_safe_distance"][other_vehicle.id][time_step] = \
                    self._keeps_safe_distance(ego_vehicle.states_lon[idx].s, other_vehicle.states_lon[idx].s,
                                              ego_vehicle.states_lon[idx].v, other_vehicle.states_lon[idx].v,
                                              self._ego_vehicle_param.get("a_min"),
                                              self._other_vehicles_param.get("a_min"),
                                              self._ego_vehicle_param.get("t_react"))
        return predicate_trace
