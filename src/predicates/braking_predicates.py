from typing import List, Dict, Set

from src.predicates.predicate_collection import PredicateCollection
from src.common.road_network import RoadNetwork
from src.common.vehicle import Vehicle
from src.predicates.position_predicates import PositionPredicateCollection


class BrakingPredicateCollection(PredicateCollection):
    def __init__(self, road_network: RoadNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, traffic_rules_param: Dict, necessary_predicates: Set[str]):
        """
        :param road_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        """
        super().__init__(road_network, simulation_param, ego_vehicle_param, other_vehicles_param,
                         traffic_rules_param, necessary_predicates)

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
            if PositionPredicateCollection.is_in_front_of(ego_vehicle, veh_o, time_step) and \
                    PositionPredicateCollection.is_in_same_lane(
                        self._road_network.find_lane_ids_by_lanelets(ego_vehicle_lanelets),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])) \
                    and veh_o.states_lon[time_step].v - v_ego < self._traffic_rules_param.get("min_velocity_dif"):
                if a_min_other is None or veh_o.states_lon[time_step].a < a_min_other:
                    a_min_other = veh_o.states_lon[time_step].a

        if a_min_other is None and (a_ego < self._traffic_rules_param.get("a_abrupt") or
                                    j_ego < self._traffic_rules_param.get("j_abrupt")):
            # no leading vehicle
            return True
        elif a_min_other is not None and j_ego < self._traffic_rules_param.get("j_abrupt") and \
                a_ego - a_min_other < self._traffic_rules_param.get("a_abrupt"):
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
                      a_min_lead: float, a_max_follow: float, t_react_follow: float) -> float:
        """
        Calculates safe distance analytically

        :param v_follow: velocity of following vehicle
        :param v_lead: velocity of leading vehicle
        :param a_min_follow: minimum acceleration of following vehicle
        :param a_min_lead: minimum acceleration of leading vehicle
        :param a_max_follow: maximum acceleration of following vehicle
        :param t_react_follow: reaction time of following vehicle
        :returns boolean indicating satisfaction
        """
        v_r = v_follow + a_max_follow * t_react_follow
        d_safe = \
            (v_lead**2) / (-2 * abs(a_min_lead)) - (v_r**2) / (-2 * abs(a_min_follow)) \
            + v_follow * t_react_follow + 0.5 * a_max_follow * t_react_follow**2

        return d_safe

    def _keeps_safe_distance_prec(self, vehicle_follow: Vehicle, vehicle_lead: Vehicle, a_min_follow: float,
                                  a_min_lead: float, a_max_follow: float, t_react_follow: float,
                                  time_step: int) -> bool:
        """
        Evaluates if safe distance is kept by following vehicle

        :param vehicle_follow: following vehicle
        :param vehicle_lead: leading vehicle
        :param a_min_follow: minimum acceleration of following vehicle
        :param a_min_lead: minimum acceleration of leading vehicle
        :param a_max_follow: maximum acceleration of following vehicle
        :param t_react_follow: reaction time of following vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if 0 < vehicle_lead.rear_position(time_step) - vehicle_follow.front_position(time_step) \
                < self.safe_distance(vehicle_follow.states_lon[time_step].v, vehicle_lead.states_lon[time_step].v,
                                     a_min_follow, a_min_lead, a_max_follow, t_react_follow):
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
                           "keeps_safe_distance_prec": {}}

        for time_step in ego_vehicle.states_lon.keys():
            if "unnecessary_braking" in self._necessary_predicates:
                predicate_trace["unnecessary_braking"][ego_vehicle.id][time_step] = \
                    self._unnecessary_braking(ego_vehicle, other_vehicles, time_step)

        for other_vehicle in other_vehicles:
            predicate_trace["keeps_safe_distance_prec"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    predicate_trace["keeps_safe_distance_prec"][other_vehicle.id][time_step] = True
                    continue
                if "keeps_safe_distance_prec" in self._necessary_predicates:
                    predicate_trace["keeps_safe_distance_prec"][other_vehicle.id][time_step] = \
                        self._keeps_safe_distance_prec(ego_vehicle, other_vehicle,
                                                       self._ego_vehicle_param.get("a_min"),
                                                       self._other_vehicles_param.get("a_min"),
                                                       self._ego_vehicle_param.get("a_max"),
                                                       self._ego_vehicle_param.get("t_react"), time_step)
        return predicate_trace
