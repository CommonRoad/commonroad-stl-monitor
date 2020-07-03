from typing import List, Dict, Set, Tuple, Union

from commonroad.scenario.obstacle import ObstacleType

from src.predicates.predicate_collection import PredicateCollection
from src.common.road_network import RoadNetwork
from src.common.vehicle import Vehicle
from src.predicates.position_predicates import PositionPredicateCollection
from src.common.helper import OperatingMode


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
        super().__init__(road_network, simulation_param,  traffic_rules_param,
                         necessary_predicates, traffic_sign_interpreter)

    def _speed_limit_suggested(self, vehicle: Vehicle, time_step: int) -> float:
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

    def slow_leading_vehicle(self, time_step: int,  vehicle: Vehicle, other_vehicles: List[Vehicle]):
        """
        Predicate which evaluates if a slow leading vehicle exists if front of a vehicle

        :param vehicle: considered vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        lanelets_veh = vehicle.lanelet_assignment[time_step]
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if not PositionPredicateCollection.in_front_of(time_step, vehicle, veh_o) or \
                    not PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(lanelets_veh),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])):
                continue
            v_max_lane = self._speed_limit_suggested(veh_o, time_step)
            v_type = self._get_type_speed_limit(veh_o.obstacle_type)
            v_max = min(vehicle.vehicle_param.get("road_condition_speed_limit"), v_max_lane, v_type)
            if v_max - veh_o.states_lon[time_step].v >= self._traffic_rules_param.get("min_velocity_dif"):
                return True

        return False

    def preserves_traffic_flow(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Predicate for minimum speed limit evaluation

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        v_max_lane = self._speed_limit_suggested(vehicle, time_step)
        v_type = self._get_type_speed_limit(vehicle.obstacle_type)
        v_max = min(vehicle.vehicle_param.get("road_condition_speed_limit"),
                    vehicle.vehicle_param.get("fov_speed_limit"),
                    vehicle.vehicle_param.get("braking_speed_limit"), v_max_lane, v_type)
        if v_max - vehicle.states_lon[time_step].v < self._traffic_rules_param.get("min_velocity_dif"):
            return True
        else:
            return False

    def in_standstill(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Evaluation if vehicle is standing

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if -self._traffic_rules_param.get("standstill_error") < vehicle.states_lon[time_step].v < \
                self._traffic_rules_param.get("standstill_error"):
            return True
        else:
            return False

    def exist_standing_leading_vehicle(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]) -> bool:
        """
        Predicate which checks if a standing leading vehicle exist in front of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        lanelets_veh = vehicle.lanelet_assignment[time_step]
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if not PositionPredicateCollection.in_front_of(time_step, vehicle, veh_o) or \
                    not PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(lanelets_veh),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])):
                continue
            if self.in_standstill(time_step, veh_o):
                return True
        return False

    def drives_with_slightly_higher_speed(self, time_step: int, vehicle_k: Vehicle, vehicle_p: Vehicle,
                                          operating_mode: OperatingMode) -> Union[bool, float, Tuple[float, float]]:
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
            return vehicle_p.states_lon[time_step].v + self._traffic_rules_param.get("slightly_higher_speed_difference")
        elif operating_mode is OperatingMode.ROBUSTNESS:
            return vehicle_p.states_lon[time_step].v \
                   + self._traffic_rules_param.get("slightly_higher_speed_difference") \
                   - vehicle_k.states_lon[time_step].v

    @staticmethod
    def drives_faster(time_step: int, vehicle_k: Vehicle, vehicle_p: Vehicle) -> bool:
        """
        Predicate which checks if the kth vehicle drives faster than the pth vehicle

        :param vehicle_p: the pth vehicle
        :param vehicle_k: the kth vehicle
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if vehicle_p.states_lon[time_step].v < vehicle_k.states_lon[time_step].v:
            return True
        else:
            return False

    def reverses(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Evaluation if a vehicle drives backwards

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if vehicle.states_lon[time_step].v < -self._traffic_rules_param.get("standstill_error"):
            return True
        else:
            return False

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
            return 80.0

    def keeps_sign_min_speed_limit(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        required_speed = self._traffic_sign_interpreter.required_speed(frozenset(lanelet_ids))
        if required_speed is None:
            return True
        elif required_speed >= min(vehicle.vehicle_param.get("fov_speed_limit"),
                                   self._get_type_speed_limit(vehicle.obstacle_type),
                                   vehicle.vehicle_param.get("road_condition_speed_limit")):
            return False
        elif required_speed > vehicle.states_lon[time_step].v:
            return False
        else:
            return True

    @staticmethod
    def keeps_fov_speed_limit(time_step: int, vehicle: Vehicle) -> bool:
        """
        Predicate for field of view speed limit evaluation

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :returns boolean indicating satisfaction
        """
        if vehicle.vehicle_param.get("fov_speed_limit") < vehicle.states_lon[time_step].v:
            return False
        else:
            return True

    @staticmethod
    def keeps_braking_speed_limit(time_step: int, vehicle: Vehicle) -> bool:
        """
        Predicate for velocity limit to ensure comfortable braking for speed limits

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :returns boolean indicating satisfaction
        """
        if vehicle.vehicle_param.get("braking_speed_limit") < vehicle.states_lon[time_step].v:
            return False
        else:
            return True

    def keeps_type_speed_limit(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :returns boolean indicating satisfaction
        """
        if vehicle.states_lon[time_step].v <= self._get_type_speed_limit(vehicle.obstacle_type):
            return True
        else:
            return False

    def keeps_lane_speed_limit(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        speed_limit = self._traffic_sign_interpreter.speed_limit(frozenset(lanelet_ids))
        if speed_limit is None:
            return True
        elif speed_limit < vehicle.states_lon[time_step].v:
            return False
        else:
            return True

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
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
                    self.keeps_lane_speed_limit(time_step, ego_vehicle)
            if "keeps_fov_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_fov_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_fov_speed_limit(time_step, ego_vehicle)
            if "keeps_braking_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_braking_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_braking_speed_limit(time_step, ego_vehicle)
            if "preserves_traffic_flow__x_ego" in self._necessary_predicates:
                predicate_trace["preserves_traffic_flow__x_ego"][ego_vehicle.id][time_step] = \
                    self.preserves_traffic_flow(time_step, ego_vehicle)
            if "slow_leading_vehicle__x_ego" in self._necessary_predicates:
                predicate_trace["slow_leading_vehicle__x_ego"][ego_vehicle.id][time_step] = \
                    self.slow_leading_vehicle(time_step, ego_vehicle, other_vehicles)
            if "keeps_type_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_type_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_type_speed_limit(time_step, ego_vehicle)
            if "keeps_sign_min_speed_limit__x_ego" in self._necessary_predicates:
                predicate_trace["keeps_sign_min_speed_limit__x_ego"][ego_vehicle.id][time_step] = \
                    self.keeps_sign_min_speed_limit(time_step, ego_vehicle)
            if "exist_standing_leading_vehicle__x_ego" in self._necessary_predicates:
                predicate_trace["exist_standing_leading_vehicle__x_ego"][ego_vehicle.id][time_step] = \
                    self.exist_standing_leading_vehicle(time_step, ego_vehicle, other_vehicles)
            if "in_standstill__x_ego" in self._necessary_predicates:
                predicate_trace["in_standstill__x_ego"][ego_vehicle.id][time_step] = \
                    self.in_standstill(time_step, ego_vehicle)
            if "reverses__x_ego" in self._necessary_predicates:
                predicate_trace["reverses__x_ego"][ego_vehicle.id][time_step] = \
                    self.reverses(time_step, ego_vehicle)

        for other_vehicle in other_vehicles:
            predicate_trace["drives_faster__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["drives_faster__x_o__x_ego"][other_vehicle.id] = {}
            predicate_trace["drives_with_slightly_higher_speed__x_ego__x_o"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    continue
                if "drives_faster__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["drives_faster__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.drives_faster(time_step, ego_vehicle, other_vehicle)
                if "drives_faster__x_o__x_ego" in self._necessary_predicates:
                    predicate_trace["drives_faster__x_o__x_ego"][other_vehicle.id][time_step] = \
                        self.drives_faster(time_step, other_vehicle, ego_vehicle)
                if "drives_with_slightly_higher_speed__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["drives_with_slightly_higher_speed__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.drives_with_slightly_higher_speed(time_step, ego_vehicle, other_vehicle)
        return predicate_trace

    def evaluate_robustness(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        pass

    def evaluate_constraints(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                             time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        pass
