from typing import List, Dict
from predicates.predicate_collection import PredicateCollection
from common.road_network import RoadNetwork
from common.vehicle import Vehicle
from predicates.position_predicates import PositionPredicateCollection


class BrakingPredicateCollection(PredicateCollection):
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
                    PositionPredicateCollection.same_lane(
                        self._road_network.find_lane_ids_by_lanelets(ego_vehicle_lanelets),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])) \
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

    @staticmethod
    def safe_distance(v_follow: float, v_lead: float, a_min_follow: float,
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
        predicate_trace = {"unnecessary_braking": {ego_vehicle.id: {}},
                           "keeps_safe_distance": {}}

        for time_step in ego_vehicle.states_lon.keys():
            predicate_trace["unnecessary_braking"][ego_vehicle.id][time_step] = \
                self._unnecessary_braking(ego_vehicle, other_vehicles, time_step)

        for other_vehicle in other_vehicles:
            predicate_trace["keeps_safe_distance"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    predicate_trace["keeps_safe_distance"][other_vehicle.id][time_step] = True
                    continue
                predicate_trace["keeps_safe_distance"][other_vehicle.id][time_step] = \
                    self._keeps_safe_distance(ego_vehicle.states_lon[time_step].s,
                                              other_vehicle.states_lon[time_step].s,
                                              ego_vehicle.states_lon[time_step].v,
                                              other_vehicle.states_lon[time_step].v,
                                              self._ego_vehicle_param.get("a_min"),
                                              self._other_vehicles_param.get("a_min"),
                                              self._ego_vehicle_param.get("t_react"))
        return predicate_trace
