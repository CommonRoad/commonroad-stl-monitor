from typing import List, Dict, Set, Union
from predicates.predicate_collection import PredicateCollection
from common.road_network import RoadNetwork
from common.vehicle import Vehicle
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry, TrafficSignIDGermany
from predicates.position_predicates import PositionPredicateCollection
from commonroad.scenario.obstacle import ObstacleType


class VelocityPredicateCollection(PredicateCollection):
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

    def _speed_limit_max(self, lanelet_ids: Set[int]) -> Union[float, None]:
        """
        Finds minimum speed limit on provided lanelets

        :param lanelet_ids: set of lanelets which should be considered
        :returns speed limit of lanelets vehicle occupies or None if no speed limit exists
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
                        # TODO add other country options
        if len(speed_limits) == 0:
            return None
        else:
            return min(speed_limits)

    def _speed_limit_suggested(self, vehicle: Vehicle, time_step: int) -> float:
        """
        Speed limit considering suggested speed

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns speed limit
        """
        v_max_lane = self._speed_limit_max(vehicle.lanelet_assignment[time_step])
        if v_max_lane is None or v_max_lane == float("inf"):
            return self._traffic_rules_param.get("desired_highway_velocity")
        else:
            return min(self._traffic_rules_param.get("desired_highway_velocity"), v_max_lane)

    def _slow_leading_vehicle(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int):
        """
        Predicate which evaluates if a slow leading vehicle exists if front of a vehicle

        :param vehicle: considered vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction
        """
        lanelets_veh = vehicle.lanelet_assignment[time_step]
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if not PositionPredicateCollection.is_in_front_of(vehicle, veh_o, time_step) or \
                    not PositionPredicateCollection.is_in_same_lane(
                        self._road_network.find_lane_ids_by_lanelets(lanelets_veh),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])):
                continue
            v_max_lane = self._speed_limit_suggested(veh_o, time_step)
            v_type = self._get_type_speed_limit(veh_o.obstacle_type)
            v_max = min(self._ego_vehicle_param.get("road_condition_speed_limit"), v_max_lane, v_type)
            if v_max - veh_o.states_lon[time_step].v > self._traffic_rules_param.get("min_velocity_dif"):
                return True

        return False

    def _preserves_traffic_flow(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Predicate for minimum speed limit evaluation

        :param vehicle: vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction
        """
        if self._slow_leading_vehicle(vehicle, other_vehicles, time_step):
            return True
        else:
            v_max_lane = self._speed_limit_suggested(vehicle, time_step)
            v_type = self._get_type_speed_limit(vehicle.obstacle_type)
            v_max = min(self._ego_vehicle_param.get("road_condition_speed_limit"),
                        self._ego_vehicle_param.get("fov_speed_limit"),
                        self._ego_vehicle_param.get("braking_speed_limit"), v_max_lane, v_type)
            if v_max - vehicle.states_lon[time_step].v > self._traffic_rules_param.get("min_velocity_dif"):
                return False
            else:
                return True

    def _speed_min(self, lanelet_ids: Set[int]) -> float:
        """
        Extracts the maximum required speed a vehicle has to drive on a set of occupied lanelets

        :param lanelet_ids: IDs of lanelets the vehicle is on
        :returns minimum required speed of occupied lanelets
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

        return max(speed_limits)

    def _in_standstill(self, velocity: float):
        """
        Evaluation if vehicle is standing

        :param velocity: velocity of vehicle
        :returns boolean indicating satisfaction
        """
        if -self._traffic_rules_param.get("standstill_error") < velocity < \
                self._traffic_rules_param.get("standstill_error"):
            return True
        else:
            return False

    def _exist_standing_leading_vehicle(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Predicate which checks if a standing leading vehicle exist in front of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction
        """
        lanelets_veh = vehicle.lanelet_assignment[time_step]
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:
                continue
            if not PositionPredicateCollection.is_in_front_of(vehicle, veh_o, time_step) or \
                    not PositionPredicateCollection.is_in_same_lane(
                        self._road_network.find_lane_ids_by_lanelets(lanelets_veh),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])):
                continue
            if self._in_standstill(veh_o.states_lon[time_step].v):
                return True
        return False

    def _drives_with_slightly_higher_speed(self, vehicle_k: Vehicle, vehicle_p: Vehicle, time_step: int) -> bool:
        """
        Predicate which checks if the kth vehicle drives maximum with slightly higher speed than the pth vehicle

        :param vehicle_k: vehicle object
        :param vehicle_p: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction (True is default return)
        """
        if vehicle_k.states_lon.get(time_step) is None or vehicle_p.states_lon.get(time_step) is None:
            return True
        if 0 < vehicle_k.states_lon[time_step].v - vehicle_p.states_lon[time_step].v \
                < self._traffic_rules_param.get("slightly_higher_speed_difference"):
            return True
        else:
            return False

    @staticmethod
    def _drives_faster_than_vehicle_left(vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Predicate which checks if a vehicle drives faster than any vehicle on its left side

        :param vehicle: vehicle object
        :param other_vehicles: list of other vehicles
        :param time_step: time step of interest
        :returns Boolean indicating speed limit satisfaction
        """
        vehicles_left = PositionPredicateCollection.vehicles_left(vehicle, other_vehicles, time_step)
        for veh_l in vehicles_left:
            if vehicle.states_lon[time_step].v > veh_l.states_lon[time_step].v:
                return True
        return False

    def _reverses(self, velocity: float):
        """
        Evaluation if a vehicle drives backwards

        :param velocity: velocity of vehicle
        :returns boolean indicating satisfaction
        """
        if velocity < -self._traffic_rules_param.get("standstill_error"):
            return True
        else:
            return False

    def _get_type_speed_limit(self, vehicle_type: ObstacleType) -> float:
        """
        Evaluates speed limit for a vehicle type

        :param vehicle_type: type of vehicle, e.g. truck
        :returns speed limit
        """
        if vehicle_type is ObstacleType.TRUCK:
            return 22.22
        else:
            return 80.0

    def _keeps_sign_min_speed_limit(self, velocity: float, lanelet_ids: Set[int], vehicle_type: ObstacleType) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param velocity: Velocity of vehicle
        :param lanelet_ids: IDs of lanelets the vehicle is on
        :param vehicle_type: type of vehicle, e.g, truck
        :returns Boolean indicating satisfaction
        """
        required_speed = self._speed_min(lanelet_ids)
        if required_speed >= min(self._ego_vehicle_param.get("fov_speed_limit"),
                                 self._get_type_speed_limit(vehicle_type),
                                 self._ego_vehicle_param.get("road_condition_speed_limit")):
            return False
        if required_speed > velocity:
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

    def _keeps_type_speed_limit(self, velocity: float, vehicle_type: ObstacleType) -> bool:
        """
        Predicate for lanelet speed limit evaluation

        :param velocity: Velocity of vehicle
        :param lanelet_ids: IDs of lanelets the vehicle is on
        :param vehicle_type: type of vehicle, e.g. truck
        :returns Boolean indicating speed limit satisfaction
        """
        if velocity <= self._get_type_speed_limit(vehicle_type):
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
        speed_limit = self._speed_limit_max(lanelet_ids)
        if speed_limit is None:
            return True
        elif speed_limit < velocity:
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
                           "keeps_type_speed_limit": {ego_vehicle.id: {}}}

        for time_step in ego_vehicle.states_lon.keys():
            predicate_trace["keeps_lane_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_lane_speed_limit(ego_vehicle.states_lon[time_step].v,
                                             ego_vehicle.lanelet_assignment[time_step])
            predicate_trace["keeps_fov_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_fov_speed_limit(ego_vehicle.states_lon[time_step].v)
            predicate_trace["keeps_braking_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_braking_speed_limit(ego_vehicle.states_lon[time_step].v)
            predicate_trace["keeps_road_condition_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_road_condition_speed_limit(ego_vehicle.states_lon[time_step].v)
            predicate_trace["preserves_traffic_flow"][ego_vehicle.id][time_step] = \
                self._preserves_traffic_flow(ego_vehicle, other_vehicles, time_step)
            predicate_trace["keeps_type_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_type_speed_limit(ego_vehicle.states_lon[time_step].v, ego_vehicle.obstacle_type)
            predicate_trace["keeps_sign_min_speed_limit"][ego_vehicle.id][time_step] = \
                self._keeps_sign_min_speed_limit(ego_vehicle.states_lon[time_step].v,
                                                 ego_vehicle.lanelet_assignment[time_step], ego_vehicle.obstacle_type)

        return predicate_trace
