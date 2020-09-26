import math
from typing import List, Dict, Set, Tuple, Union

from commonroad.scenario.obstacle import ObstacleType

from crmonitor.common.helper import OperatingMode
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle
from crmonitor.predicates.python.position_predicates import PositionPredicateCollection
from crmonitor.predicates.python.predicate_collection import PredicateCollection, Constraint, \
    ConstraintRepresentation, ConstraintType


class VelocityPredicateCollection(PredicateCollection):
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

    def _speed_limit_suggested(self, time_step: int, vehicle: Vehicle) -> float:
        """
        Speed limit considering suggested speed

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns speed limit
        """
        v_max_lane = self._traffic_sign_interpreter.speed_limit(frozenset(vehicle.lanelet_assignment[time_step]))
        if v_max_lane is None or v_max_lane == float("inf"):
            return self._traffic_rules_param.get("desired_interstate_velocity")
        else:
            return min(self._traffic_rules_param.get("desired_interstate_velocity"), v_max_lane)

    def slow_leading_vehicle(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]) -> bool:
        """
        Predicate which evaluates if a slow leading vehicle exists if front of a vehicle

        :param vehicle: considered vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelets_veh = vehicle.lanelet_assignment[time_step]
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if not PositionPredicateCollection.in_front_of(time_step, vehicle, veh_o, OperatingMode.MONITOR) or \
                    not PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(lanelets_veh),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])):
                continue
            v_max_lane = self._speed_limit_suggested(time_step, veh_o)
            v_type = self._get_type_speed_limit(veh_o.obstacle_type)
            v_max = min(vehicle.vehicle_param.get("road_condition_speed_limit"), v_max_lane, v_type)
            if v_max - veh_o.states_lon[time_step].v >= self._traffic_rules_param.get("min_velocity_dif"):
                return True

        return False

    def preserves_traffic_flow(self, time_step: int, vehicle: Vehicle,
                               operating_mode: OperatingMode) -> [bool, Constraint, float]:
        """
        Predicate for minimum speed limit evaluation

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        v_max_lane = self._speed_limit_suggested(time_step, vehicle)
        v_type = self._get_type_speed_limit(vehicle.obstacle_type)
        v_max = min(vehicle.vehicle_param.get("road_condition_speed_limit"),
                    vehicle.vehicle_param.get("fov_speed_limit"),
                    vehicle.vehicle_param.get("braking_speed_limit"), v_max_lane, v_type)
        if operating_mode is OperatingMode.MONITOR:
            if v_max - vehicle.states_lon[time_step].v < self._traffic_rules_param.get("min_velocity_dif"):
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:
            return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER,
                              v_max - self._traffic_rules_param.get("min_velocity_dif"))
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return vehicle.states_lon[time_step].v - v_max + self._traffic_rules_param.get("min_velocity_dif") - 1e-17

    def in_standstill(self, time_step: int, vehicle: Vehicle, operating_mode: OperatingMode) \
            -> Union[bool, List[Constraint], float]:
        """
        Evaluation if vehicle is standing

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if -self._traffic_rules_param.get("standstill_error") < vehicle.states_lon[time_step].v < \
                    self._traffic_rules_param.get("standstill_error"):
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:
            return [Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER,
                               -self._traffic_rules_param.get("standstill_error")),
                    Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER,
                               self._traffic_rules_param.get("standstill_error"))]
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return abs(self._traffic_rules_param.get("standstill_error")) - abs(vehicle.states_lon[time_step].v)

    def exist_standing_leading_vehicle(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]) -> bool:
        """
        Predicate which checks if a standing leading vehicle exist in front of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelets_veh = vehicle.lanelet_assignment[time_step]
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if not PositionPredicateCollection.in_front_of(time_step, vehicle, veh_o, OperatingMode.MONITOR) or \
                    not PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(lanelets_veh),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])):
                continue
            if self.in_standstill(time_step, veh_o, OperatingMode.MONITOR):
                return True
        return False

    def drives_with_slightly_higher_speed(self, time_step: int, vehicle_k: Vehicle, vehicle_p: Vehicle,
                                          operating_mode: OperatingMode) -> Union[bool,
                                                                                  List[Constraint], float]:
        """
        Predicate which checks if the kth vehicle drives maximum with slightly higher speed than the pth vehicle

        :param vehicle_k: vehicle object
        :param vehicle_p: list of other vehicles
        :param time_step: time step of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if 0 < vehicle_k.states_lon[time_step].v - vehicle_p.states_lon[time_step].v \
                    < self._traffic_rules_param.get("slightly_higher_speed_difference"):
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:
            return [Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER,
                               vehicle_p.states_lon[time_step].v),
                    Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER,
                               vehicle_p.states_lon[time_step].v +
                               self._traffic_rules_param.get("slightly_higher_speed_difference"))]
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return min(
                vehicle_p.states_lon[time_step].v + self._traffic_rules_param.get("slightly_higher_speed_difference") -
                vehicle_k.states_lon[time_step].v,
                vehicle_k.states_lon[time_step].v - vehicle_p.states_lon[time_step].v) - 1e-17

    @staticmethod
    def drives_faster(time_step: int, vehicle_k: Vehicle, vehicle_p: Vehicle,
                      operating_mode: OperatingMode) -> Union[bool, float, Constraint]:
        """
        Predicate which checks if the kth vehicle drives faster than the pth vehicle

        :param vehicle_p: the pth vehicle
        :param vehicle_k: the kth vehicle
        :param time_step: time step of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if vehicle_p.states_lon[time_step].v < vehicle_k.states_lon[time_step].v:
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:
            return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER,
                              vehicle_p.states_lon[time_step].v)
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return vehicle_k.states_lon[time_step].v - vehicle_p.states_lon[time_step].v - 1.e-17

    def reverses(self, time_step: int, vehicle: Vehicle,
                 operating_mode: OperatingMode) -> Union[bool, float, Constraint]:
        """
        Evaluates if a vehicle drives backwards

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if vehicle.states_lon[time_step].v < -self._traffic_rules_param.get("standstill_error"):
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:
            return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER,
                              -self._traffic_rules_param.get("standstill_error"))
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return -self._traffic_rules_param.get("standstill_error") - vehicle.states_lon[time_step].v - -1e-17

    @staticmethod
    def _get_type_speed_limit(vehicle_type: ObstacleType) -> float:
        """
        Evaluates speed limit for a vehicle type

        :param vehicle_type: type of vehicle, e.g. truck
        :returns speed limit
        """
        if vehicle_type is ObstacleType.TRUCK:
            return 22.22
        else:
            return math.inf

    def keeps_sign_min_speed_limit(self, time_step: int, vehicle: Vehicle,
                                   operating_mode: OperatingMode) -> Union[bool, float, Constraint]:
        """
        Predicate for lanelet required speed evaluation

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        required_speed = self._traffic_sign_interpreter.required_speed(frozenset(lanelet_ids))
        if operating_mode is OperatingMode.MONITOR:
            if required_speed is None:
                return True
            if required_speed <= vehicle.states_lon[time_step].v:
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:
            if required_speed is None:
                return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER, 0)
            else:
                return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.LOWER, required_speed)
        elif operating_mode is OperatingMode.ROBUSTNESS:
            if required_speed is None:
                return math.inf
            else:
                return vehicle.states_lon[time_step].v - required_speed

    @staticmethod
    def keeps_fov_speed_limit(time_step: int, vehicle: Vehicle,
                              operating_mode: OperatingMode) -> Union[bool, float, Constraint]:
        """
        Predicate for field of view speed limit evaluation. This is necessary to ensure that a vehicle is able to react
        to a standing vehicle at the border of the field of view.

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if vehicle.vehicle_param.get("fov_speed_limit") < vehicle.states_lon[time_step].v:
                return False
            else:
                return True
        elif operating_mode is OperatingMode.CONSTRAINT:
            return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER,
                              vehicle.vehicle_param.get("fov_speed_limit"))
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return vehicle.vehicle_param.get("fov_speed_limit") - vehicle.states_lon[time_step].v

    @staticmethod
    def keeps_braking_speed_limit(time_step: int, vehicle: Vehicle,
                                  operating_mode: OperatingMode) -> Union[bool, float, Constraint]:
        """
        Predicate for braking speed limit evaluation. This is necessary to ensure that a vehicle is able to react
        without unnecessary braking to an upcoming speed limit.

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if vehicle.vehicle_param.get("braking_speed_limit") < vehicle.states_lon[time_step].v:
                return False
            else:
                return True
        elif operating_mode is OperatingMode.CONSTRAINT:
            return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER,
                              vehicle.vehicle_param.get("braking_speed_limit"))
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return vehicle.vehicle_param.get("braking_speed_limit") - vehicle.states_lon[time_step].v

    def keeps_type_speed_limit(self, time_step: int, vehicle: Vehicle,
                               operating_mode: OperatingMode) -> Union[bool, float, Constraint]:
        """
        Predicate for type speed limit evaluation

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        if operating_mode is OperatingMode.MONITOR:
            if vehicle.states_lon[time_step].v <= self._get_type_speed_limit(vehicle.obstacle_type):
                return True
            else:
                return False
        elif operating_mode is OperatingMode.CONSTRAINT:
            return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER,
                              self._get_type_speed_limit(vehicle.obstacle_type))
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return self._get_type_speed_limit(vehicle.obstacle_type) - vehicle.states_lon[time_step].v

    def keeps_lane_speed_limit(self, time_step: int, vehicle: Vehicle,
                               operating_mode: OperatingMode) -> Union[bool, float, Constraint]:
        """
        Predicate for lanelet speed limit evaluation

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :param operating_mode: specifies operating mode (one of robustness, constraint, or monitor)
        :returns boolean indicating satisfaction, constraint values, or robustness value
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        speed_limit = self._traffic_sign_interpreter.speed_limit(frozenset(lanelet_ids))
        if operating_mode is OperatingMode.MONITOR:
            if speed_limit is None:
                return True
            elif speed_limit < vehicle.states_lon[time_step].v:
                return False
            else:
                return True
        elif operating_mode is OperatingMode.CONSTRAINT:
            if speed_limit is None:
                return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER, math.inf)
            else:
                return Constraint([ConstraintType.VELOCITY], ConstraintRepresentation.UPPER, speed_limit)
        elif operating_mode is OperatingMode.ROBUSTNESS:
            if speed_limit is None:
                return math.inf
            else:
                return speed_limit - vehicle.states_lon[time_step].v

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int],
                            operating_mode: OperatingMode) -> Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :param operating_mode: operating mode which should be used for evaluation (monitor, constraint, or robustness)
        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"keeps_lane_speed_limit__x_ego": {ego_vehicle.id: {}},
                           "keeps_fov_speed_limit__x_ego": {ego_vehicle.id: {}},
                           "preserves_traffic_flow__x_ego": {ego_vehicle.id: {}},
                           "slow_leading_vehicle__x_ego": {ego_vehicle.id: {}},
                           "keeps_sign_min_speed_limit__x_ego": {ego_vehicle.id: {}},
                           "keeps_braking_speed_limit__x_ego": {ego_vehicle.id: {}},
                           "keeps_type_speed_limit__x_ego": {ego_vehicle.id: {}},
                           "exist_standing_leading_vehicle__x_ego": {ego_vehicle.id: {}},
                           "in_standstill__x_ego": {ego_vehicle.id: {}},
                           "drives_with_slightly_higher_speed__x_ego__x_o": {},
                           "drives_faster__x_ego__x_o": {},
                           "drives_faster__x_o__x_ego": {},
                           "reverses__x_ego": {ego_vehicle.id: {}}}

        for time_step in ego_vehicle.states_lon.keys():
            if "keeps_lane_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_lane_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_lane_speed_limit(time_step, ego_vehicle, operating_mode)
            if "keeps_fov_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_fov_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_fov_speed_limit(time_step, ego_vehicle, operating_mode)
            if "keeps_braking_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_braking_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_braking_speed_limit(time_step, ego_vehicle, operating_mode)
            if "preserves_traffic_flow__x_ego" in self._necessary_predicates:
                predicate_trace["preserves_traffic_flow__x_ego"][ego_vehicle.id][time_step] = \
                    self.preserves_traffic_flow(time_step, ego_vehicle, operating_mode)
            if "slow_leading_vehicle__x_ego" in self._necessary_predicates:
                predicate_trace["slow_leading_vehicle__x_ego"][ego_vehicle.id][time_step] = \
                    self.slow_leading_vehicle(time_step, ego_vehicle, other_vehicles)
            if "keeps_type_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_type_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_type_speed_limit(time_step, ego_vehicle, operating_mode)
            if "keeps_sign_min_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_sign_min_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_sign_min_speed_limit(time_step, ego_vehicle, operating_mode)
            if "exist_standing_leading_vehicle__x_ego" in self._necessary_predicates:
                predicate_trace["exist_standing_leading_vehicle__x_ego"][ego_vehicle.id][time_step] = \
                    self.exist_standing_leading_vehicle(time_step, ego_vehicle, other_vehicles)
            if "in_standstill__x_ego" in self._necessary_predicates:
                predicate_trace["in_standstill__x_ego"][ego_vehicle.id][time_step] = \
                    self.in_standstill(time_step, ego_vehicle, operating_mode)
            if "reverses__x_ego" in self._necessary_predicates:
                predicate_trace["reverses__x_ego"][ego_vehicle.id][time_step] = \
                    self.reverses(time_step, ego_vehicle, operating_mode)

        for other_vehicle in other_vehicles:
            predicate_trace["drives_faster__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["drives_faster__x_o__x_ego"][other_vehicle.id] = {}
            predicate_trace["drives_with_slightly_higher_speed__x_ego__x_o"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    continue
                if "drives_faster__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["drives_faster__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.drives_faster(time_step, ego_vehicle, other_vehicle, operating_mode)
                if "drives_faster__x_o__x_ego" in self._necessary_predicates:
                    predicate_trace["drives_faster__x_o__x_ego"][other_vehicle.id][time_step] = \
                        self.drives_faster(time_step, other_vehicle, ego_vehicle, operating_mode)
                if "drives_with_slightly_higher_speed__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["drives_with_slightly_higher_speed__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.drives_with_slightly_higher_speed(time_step, ego_vehicle, other_vehicle, operating_mode)
        return predicate_trace
