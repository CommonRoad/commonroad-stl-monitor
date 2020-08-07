from typing import List, Dict, Set, Union, Tuple

from crmonitor.common.helper import OperatingMode
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle
from crmonitor.predicates.position_predicates import PositionPredicateCollection
from crmonitor.predicates.predicate_collection import PredicateCollection


class BrakingPredicateCollection(PredicateCollection):
    def __init__(self, road_network: RoadNetwork, simulation_param: Dict,
                 traffic_rules_param: Dict, necessary_predicates: Set[str], traffic_sign_interpreter):
        """
        :param road_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        :param traffic_sign_interpreter: CommonRoad traffic sign interpreter
        """
        super().__init__(road_network, simulation_param, traffic_rules_param,
                         necessary_predicates, traffic_sign_interpreter)

    def unnecessary_braking(self, time_step: int, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            operating_mode: OperatingMode) -> bool:
        """ Predicate to check whether an obstacle brakes abruptly

        :param ego_vehicle: ego vehicle
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint value, or robustness value
        """
        a_ego = ego_vehicle.states_lon[time_step].a
        if a_ego >= 0 and operating_mode is operating_mode.MONITOR:
            return False

        ego_vehicle_lanelets = ego_vehicle.lanelet_assignment[time_step]
        same_lane_front_vehicle = False
        robustness_values = []
        constraint_values = []
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if PositionPredicateCollection.in_front_of(time_step, ego_vehicle, veh_o, OperatingMode.MONITOR) and \
                    PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(ego_vehicle_lanelets),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])) \
                    and self.keeps_safe_distance_prec(time_step, ego_vehicle, veh_o, OperatingMode.MONITOR):
                same_lane_front_vehicle = True

                if operating_mode is operating_mode.CONSTRAINT:
                    constraint_values.append(veh_o.states_lon[time_step].a
                                             - abs(self._traffic_rules_param.get("a_abrupt")))
                elif operating_mode is operating_mode.MONITOR:
                    if (a_ego - veh_o.states_lon[time_step].a) < self._traffic_rules_param.get("a_abrupt"):
                        return True
                elif operating_mode is operating_mode.ROBUSTNESS:
                    robustness_values.append(
                        self._traffic_rules_param.get("a_abrupt") - a_ego + veh_o.states_lon[time_step].a)

        if operating_mode is operating_mode.MONITOR:
            if same_lane_front_vehicle is False and a_ego < self._traffic_rules_param.get("a_abrupt"):
                # no leading vehicle
                return True
            else:
                return False
        elif operating_mode is operating_mode.CONSTRAINT:
            if same_lane_front_vehicle is False:
                return self._traffic_rules_param.get("a_abrupt")
            else:
                return max(constraint_values)
        elif operating_mode is operating_mode.ROBUSTNESS:
            if same_lane_front_vehicle is False:
                return max(a_ego, -a_ego + self._traffic_rules_param.get("a_abrupt"))
            else:
                return max(robustness_values)

    @staticmethod
    def safe_distance(v_follow: float, v_lead: float, a_min_follow: float, a_min_lead: float,
                      t_react_follow: float) -> float:
        """
        Calculates safe distance analytically

        :param v_follow: velocity of following vehicle
        :param v_lead: velocity of leading vehicle
        :param a_min_follow: minimum acceleration of following vehicle
        :param a_min_lead: minimum acceleration of leading vehicle
        :param t_react_follow: reaction time of following vehicle
        :returns boolean indicating satisfaction
        """
        assert a_min_follow and 0 > a_min_lead, \
            '<BrakingPredicateCollection/safe_distance>: acceleration is not valid'
        d_safe = \
            (v_lead ** 2) / (-2 * abs(a_min_lead)) - (v_follow ** 2) / (-2 * abs(a_min_follow)) \
            + v_follow * t_react_follow

        return d_safe

    @staticmethod
    def keeps_safe_distance_prec(time_step: int, vehicle_follow: Vehicle, vehicle_lead: Vehicle,
                                 operating_mode: OperatingMode) -> Union[bool, float]:
        """
        Evaluates if safe distance is kept by following vehicle

        :param vehicle_follow: following vehicle
        :param vehicle_lead: leading vehicle
        :param time_step: time step of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint value, or robustness value
        """
        a_min_follow = vehicle_follow.vehicle_param.get("a_min")
        a_min_lead = vehicle_lead.vehicle_param.get("a_min")
        t_react_follow = vehicle_follow.vehicle_param.get("t_react")
        safe_distance = BrakingPredicateCollection.safe_distance(vehicle_follow.states_lon[time_step].v,
                                                                 vehicle_lead.states_lon[time_step].v,
                                                                 a_min_follow, a_min_lead, t_react_follow)

        if operating_mode is OperatingMode.CONSTRAINT:
            return safe_distance

        delta_s = vehicle_lead.rear_s(time_step) - vehicle_follow.front_s(time_step)
        if operating_mode is OperatingMode.MONITOR:
            if 0 <= delta_s < safe_distance:
                return False
            else:
                return True
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return delta_s - safe_distance

    @staticmethod
    def brakes_stronger(time_step: int, vehicle_k: Vehicle, vehicle_p: Vehicle,
                        operating_mode: OperatingMode) -> Union[bool, float]:
        """
        Predicate which checks if the kth vehicle brakes stronger (has lower acceleration) than the pth vehicle.
        If the kth vehicle has a positive acceleration the predicate evaluates always to false since
        the kth vehicle does not brake at all.

        :param vehicle_p: the pth vehicle
        :param vehicle_k: the kth vehicle
        :param time_step: time step of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint value, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if vehicle_k.states_lon[time_step].a < vehicle_p.states_lon[time_step].a \
                    and vehicle_k.states_lon[time_step].a < 0:
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:  # returns upper bound for acceleration
            return min(vehicle_p.states_lon[time_step].a, 0)
        elif operating_mode is OperatingMode.ROBUSTNESS:  # returns difference to upper bound defined by constraint
            return min(vehicle_p.states_lon[time_step].a, 0) - vehicle_k.states_lon[time_step].a

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"unnecessary_braking__x_ego": {ego_vehicle.id: {}},
                           "keeps_safe_distance_prec__x_ego__x_o": {},
                           "keeps_safe_distance_prec__x_o__x_ego": {},
                           "brakes_stronger__x_ego__x_o": {}}

        for time_step in ego_vehicle.states_lon.keys():
            if "unnecessary_braking__x_ego" in self._necessary_predicates:
                predicate_trace["unnecessary_braking__x_ego"][ego_vehicle.id][time_step] = \
                    self.unnecessary_braking(time_step, ego_vehicle, other_vehicles, OperatingMode.MONITOR)

        for other_vehicle in other_vehicles:
            predicate_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id] = {}
            predicate_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id] = {}
            for time_step in range(time_interval[0], time_interval[1] + 1):
                if other_vehicle.states_lon.get(time_step) is None:
                    predicate_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id][time_step] = True
                    predicate_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id][time_step] = True
                    predicate_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id][time_step] = True
                    continue
                if "keeps_safe_distance_prec__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.keeps_safe_distance_prec(time_step, ego_vehicle, other_vehicle, OperatingMode.MONITOR)
                if "keeps_safe_distance_prec__x_o__x_ego" in self._necessary_predicates:
                    predicate_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id][time_step] = \
                        self.keeps_safe_distance_prec(time_step, other_vehicle, ego_vehicle, OperatingMode.MONITOR)
                if "brakes_stronger__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.brakes_stronger(time_step, ego_vehicle, other_vehicle, OperatingMode.MONITOR)

        return predicate_trace

    def evaluate_constraints(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                             time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts constraints for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with trace of bool values for each predicate
        """
        constraint_trace = {"unnecessary_braking__x_ego": {ego_vehicle.id: {}},
                            "keeps_safe_distance_prec__x_ego__x_o": {},
                            "keeps_safe_distance_prec__x_o__x_ego": {},
                            "brakes_stronger__x_ego__x_o": {}}

        for time_step in ego_vehicle.states_lon.keys():
            if "unnecessary_braking__x_ego" in self._necessary_predicates:
                constraint_trace["unnecessary_braking__x_ego"][ego_vehicle.id][time_step] = \
                    self.unnecessary_braking(time_step, ego_vehicle, other_vehicles, OperatingMode.CONSTRAINT)

        for other_vehicle in other_vehicles:
            constraint_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id] = {}
            constraint_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id] = {}
            constraint_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id] = {}
            for time_step in range(time_interval[0], time_interval[1] + 1):
                if other_vehicle.states_lon.get(time_step) is None:
                    constraint_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id][time_step] = True
                    constraint_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id][time_step] = True
                    constraint_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id][time_step] = True
                    continue
                if "keeps_safe_distance_prec__x_ego__x_o" in self._necessary_predicates:
                    constraint_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.keeps_safe_distance_prec(time_step, ego_vehicle, other_vehicle, OperatingMode.CONSTRAINT)
                if "keeps_safe_distance_prec__x_o__x_ego" in self._necessary_predicates:
                    constraint_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id][time_step] = \
                        self.keeps_safe_distance_prec(time_step, other_vehicle, ego_vehicle, OperatingMode.CONSTRAINT)
                if "brakes_stronger__x_ego__x_o" in self._necessary_predicates:
                    constraint_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.brakes_stronger(time_step, ego_vehicle, other_vehicle, OperatingMode.CONSTRAINT)

        return constraint_trace

    def evaluate_robustness(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Extracts constraints for a vehicle

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with trace of real values for each predicate
        """
        robustness_trace = {"unnecessary_braking__x_ego": {ego_vehicle.id: {}},
                            "keeps_safe_distance_prec__x_ego__x_o": {},
                            "keeps_safe_distance_prec__x_o__x_ego": {},
                            "brakes_stronger__x_ego__x_o": {}}

        for time_step in ego_vehicle.states_lon.keys():
            if "unnecessary_braking__x_ego" in self._necessary_predicates:
                robustness_trace["unnecessary_braking__x_ego"][ego_vehicle.id][time_step] = \
                    self.unnecessary_braking(time_step, ego_vehicle, other_vehicles, OperatingMode.ROBUSTNESS)

        for other_vehicle in other_vehicles:
            robustness_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id] = {}
            robustness_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id] = {}
            robustness_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id] = {}
            for time_step in range(time_interval[0], time_interval[1] + 1):
                if other_vehicle.states_lon.get(time_step) is None:
                    robustness_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id][time_step] = True
                    robustness_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id][time_step] = True
                    robustness_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id][time_step] = True
                    continue
                if "keeps_safe_distance_prec__x_ego__x_o" in self._necessary_predicates:
                    robustness_trace["keeps_safe_distance_prec__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.keeps_safe_distance_prec(time_step, ego_vehicle, other_vehicle, OperatingMode.ROBUSTNESS)
                if "keeps_safe_distance_prec__x_o__x_ego" in self._necessary_predicates:
                    robustness_trace["keeps_safe_distance_prec__x_o__x_ego"][other_vehicle.id][time_step] = \
                        self.keeps_safe_distance_prec(time_step, other_vehicle, ego_vehicle, OperatingMode.ROBUSTNESS)
                if "brakes_stronger__x_ego__x_o" in self._necessary_predicates:
                    robustness_trace["brakes_stronger__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.brakes_stronger(time_step, ego_vehicle, other_vehicle, OperatingMode.ROBUSTNESS)

        return robustness_trace
