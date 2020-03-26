from typing import List, Dict, Set, Union

from commonroad.scenario.lanelet import LaneletType, LineMarking, Lanelet

from src.predicates.predicate_collection import PredicateCollection
from src.common.vehicle import Vehicle
from src.common.road_network import RoadNetwork


class PositionPredicateCollection(PredicateCollection):
    def __init__(self, road_network: RoadNetwork, simulation_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, traffic_rules_param: Dict, necessary_predicates: Set[str],
                 traffic_sign_interpreter):
        """
        :param road_network: CommonRoad lanelet network
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rules_param: dictionary with parameters of traffic rule parameters
        :param necessary_predicates: set with all predicates which should be evaluated
        :param traffic_sign_interpreter: CommonRoad traffic sign interpreter
        """
        super().__init__(road_network, simulation_param, ego_vehicle_param, other_vehicles_param,
                         traffic_rules_param, necessary_predicates, traffic_sign_interpreter)

    @staticmethod
    def is_in_front_of(vehicle_p: Vehicle, vehicle_k: Vehicle, time_step: int) -> bool:
        """
        Evaluates if the kth vehicle is in front of the pth vehicle

        :param vehicle_p: pth vehicle
        :param vehicle_k: kth vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if vehicle_p.front_s(time_step) < vehicle_k.rear_s(time_step):
            return True
        else:
            return False

    def is_left_of(self, vehicle_p: Vehicle, vehicle_k: Vehicle, time_step: int) -> bool:
        """
        Evaluates if the kth vehicle is left of the pth vehicle

        :param vehicle_p: pth vehicle
        :param vehicle_k: kth vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if not self._in_left_lane(vehicle_p, vehicle_k, time_step):
            return False
        else:
            if vehicle_p.rear_s(time_step) <= vehicle_k.front_s(time_step) <= \
                    vehicle_p.front_s(time_step):
                return True
            if vehicle_p.rear_s(time_step) < vehicle_k.rear_s(time_step) < \
                    vehicle_p.front_s(time_step):
                return True
            if vehicle_k.rear_s(time_step) < vehicle_p.rear_s(time_step) \
                    and vehicle_p.front_s(time_step) < vehicle_k.front_s(time_step):
                return True
            else:
                return False

    def _in_left_lane(self, vehicle_p: Vehicle, vehicle_k: Vehicle, time_step: int) -> bool:
        """
        Evaluates if the kth vehicle is in any lane left of the pth vehicle

        :param vehicle_p: pth vehicle
        :param vehicle_k: kth vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelets_k = vehicle_k.lanelet_assignment[time_step]
        lanelets_left_of_p = self._ids_lanelets_left_of_vehicle(vehicle_p, time_step)
        if any(lanelet_id in lanelets_left_of_p for lanelet_id in lanelets_k):
            return True
        else:
            return False

    def exist_leading_vehicle(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Predicate which evaluates if leading vehicle exists

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns bool indicating satisfaction
        """
        for veh in other_vehicles:
            if self.is_in_same_lane(vehicle.lanelet_assignment[time_step], veh.lanelet_assignment[time_step]) \
                    and self.is_in_front_of(vehicle, veh, time_step):
                return True
        return False

    @staticmethod
    def is_in_same_lane(lane_ids_k: Set[int], lane_ids_p: Set[int]) -> bool:
        """
        Evaluates if the kth vehicle is in the same lane as the pth vehicle

        :param lane_ids_k: lane IDs of lanes the kth vehicle is on
        :param lane_ids_p: lane IDs of lanes the pth vehicle is on
        :returns boolean indicating satisfaction
        """
        for lane_id in lane_ids_k:
            if lane_id in lane_ids_p:
                return True
        return False

    def _on_access_ramp(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle is on an access ramp

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
                return True
        return False

    def _on_exit_ramp(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle is on an exit ramp

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                return True
        return False

    def _on_shoulder(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle is on an shoulder lane

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.SHOULDER in lanelet.lanelet_type:
                return True
        return False

    def _on_main_carriage_way(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle is on an main carriage way

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type:
                return True
        return False

    def _right_of_broad_lane_marking(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle is right of a broad lane marking

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED or \
                    lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID:
                return False

        lanelets_left_of_veh = self._lanelets_left_of_vehicle(vehicle, time_step)
        for lanelet in lanelets_left_of_veh:
            if lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED or \
                    lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID:
                return True
        return False

    def _left_of_broad_lane_marking(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle is left of a broad lane marking

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids_occ = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids_occ:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED or \
                    lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID:
                return False

        lanelets_right_of_veh = self._lanelets_right_of_vehicle(vehicle, time_step)
        for lanelet in lanelets_right_of_veh:
            if lanelet.line_marking_left_vertices is LineMarking.BROAD_DASHED or \
                    lanelet.line_marking_left_vertices is LineMarking.BROAD_SOLID:
                return True
        return False

    def _lanelets_left_of_vehicle(self, vehicle: Vehicle, time_step: int) -> Set[Lanelet]:
        """
        Extracts all lanelets left of a vehicle

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns set of lanelet objects
        """
        left_lanelets = set()
        occupied_lanelets = vehicle.lanelet_assignment[time_step]
        for occ_l in occupied_lanelets:
            new_lanelets = self._lanelets_left_of_lanelet(self._road_network.lanelet_network.find_lanelet_by_id(occ_l))
            for l in new_lanelets:
                left_lanelets.add(l)

        return left_lanelets

    def _ids_lanelets_left_of_vehicle(self, vehicle: Vehicle, time_step: int) -> Set[int]:
        """
        Extracts all IDs of lanelets left of a vehicle

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns set of lanelet objects
        """
        left_lanelets = set()
        occupied_lanelets = vehicle.lanelet_assignment[time_step]
        for occ_l in occupied_lanelets:
            new_lanelets = self._lanelets_left_of_lanelet(self._road_network.lanelet_network.find_lanelet_by_id(occ_l))
            for l in new_lanelets:
                left_lanelets.add(l.lanelet_id)

        return left_lanelets

    def _lanelets_right_of_vehicle(self, vehicle: Vehicle, time_step: int) -> Set[Lanelet]:
        """
        Extracts all lanelets right of a vehicle

        :param vehicle: vehicle of interest
        :param time_step: time step of interest
        :returns set of lanelet objects
        """
        right_lanelets = set()
        occupied_lanelets = vehicle.lanelet_assignment[time_step]
        for occ_l in occupied_lanelets:
            new_lanelets = self._lanelets_right_of_lanelet(self._road_network.lanelet_network.find_lanelet_by_id(occ_l))
            for l in new_lanelets:
                right_lanelets.add(l)

        return right_lanelets

    def _lanelets_left_of_lanelet(self, lanelet: Lanelet) -> Set[Lanelet]:
        """
        Extracts all lanelet IDs left of a given lanelet based on adjacency relations

        :param lanelet: given lanelet
        :returns set of lanelet objects
        """
        left_lanelets = set()
        l = lanelet
        while l.adj_left is not None:
            l = self._road_network.lanelet_network.find_lanelet_by_id(l.adj_left)
            left_lanelets.add(l)

        return left_lanelets

    def _lanelets_right_of_lanelet(self, lanelet: Lanelet) -> Set[Lanelet]:
        """
        Extracts all lanelet IDs right of a given lanelet based on adjacency relations

        :param lanelet: given lanelet
        :returns set of lanelet objects
        """
        right_lanelets = set()
        l = lanelet
        while l.adj_right is not None:
            l = self._road_network.lanelet_network.find_lanelet_by_id(l.adj_right)
            right_lanelets.add(l)

        return right_lanelets

    def _vehicles_left(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
        """
        Searches for vehicles left of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns list of vehicles left of a vehicle
        """
        # TODO: consider orientation
        vehicles_adj = self._vehicles_adjacent(vehicle, other_vehicles, time_step)
        vehicles_left = [veh for veh in vehicles_adj if veh.states_lat[time_step].d > vehicle.states_lat[time_step].d]
        return vehicles_left

    def _vehicles_right(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
        """
        Searches for vehicles right of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns list of vehicles right of a vehicle
        """
        # TODO: consider orientation
        vehicles_adj = self._vehicles_adjacent(vehicle, other_vehicles, time_step)
        vehicles_right = [veh for veh in vehicles_adj if veh.states_lat[time_step].d < vehicle.states_lat[time_step].d]
        return vehicles_right

    @staticmethod
    def _vehicles_adjacent(vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
        """
        Searches for vehicles adjacent to a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns list of adjacent vehicles of a vehicle
        """
        vehicles_adj = []
        for veh in other_vehicles:
            if veh.rear_s(time_step) < vehicle.front_s(time_step) < veh.front_s(time_step):
                vehicles_adj.append(veh)
                continue
            if veh.rear_s(time_step) < vehicle.rear_s(time_step) < veh.front_s(time_step):
                vehicles_adj.append(veh)
                continue
            if vehicle.rear_s(time_step) <= veh.rear_s(time_step) \
                    and veh.front_s(time_step) <= vehicle.front_s(time_step):
                vehicles_adj.append(veh)
                continue
        return vehicles_adj

    def _in_leftmost_lane(self, lanelet_ids: Set[int]):
        """
        Evaluates if a vehicle is in the leftmost lane

        :param lanelet_ids: lanelet IDs the vehicle is on
        :returns boolean indicating satisfaction
        """
        for l_id in lanelet_ids:
            if self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_left_same_direction is None:
                return True
        return False

    def _in_rightmost_lane(self, lanelet_ids: Set[int]):
        """
        Evaluates if a vehicle is in the rightmost lane (excluding shoulder lane)

        :param lanelet_ids: lanelet IDs the vehicle is on
        :returns boolean indicating satisfaction
        """
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.adj_right_same_direction is None or LaneletType.SHOULDER in \
                    self._road_network.lanelet_network.find_lanelet_by_id(lanelet.adj_right).lanelet_type:
                return True
        return False

    def _vehicle_directly_right(self, vehicle: Vehicle, other_vehicles: List[Vehicle],
                              time_step: int) -> Union[Vehicle, None]:
        vehicles_right = self._vehicles_right(vehicle, other_vehicles, time_step)
        if len(vehicles_right) == 0:
            return None
        elif len(vehicles_right) == 1:
            return vehicles_right[0]
        else:
            vehicle_directly_right = vehicles_right[0]
            for veh in vehicles_right:
                if veh.states_lat[time_step].d < vehicle_directly_right.states_lat[time_step].d:
                    vehicle_directly_right = veh
            return vehicle_directly_right

    def _vehicle_directly_left(self, vehicle: Vehicle, other_vehicles: List[Vehicle],
                              time_step: int) -> Union[Vehicle, None]:
        vehicles_left = self._vehicles_left(vehicle, other_vehicles, time_step)
        if len(vehicles_left) == 0:
            return None
        elif len(vehicles_left) == 1:
            return vehicles_left[0]
        else:
            vehicle_directly_left = vehicles_left[0]
            for veh in vehicles_left:
                if veh.states_lat[time_step].d < vehicle_directly_left.states_lat[time_step].d:
                    vehicle_directly_left = veh
            return vehicle_directly_left

    def _drives_right_lane_rightmost(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle drives in right lane rightmost

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        # TODO consider orientation
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        if self._in_rightmost_lane(vehicle.lanelet_assignment[time_step]) is False:
            return False
        else:
            right_position = vehicle.right_position(time_step)
            s_ego = vehicle.states_lon[time_step].s
            lanes = self._road_network.find_lanes_by_lanelets(occupied_lanelet_ids)
            for lane in lanes:
                one_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(list(lane.contained_lanelets)[0])
                if one_lanelet.adj_right_same_direction is None or LaneletType.SHOULDER in \
                        self._road_network.lanelet_network.find_lanelet_by_id(one_lanelet.adj_right).lanelet_type:
                    continue
                else:
                    return False
            for lane in lanes:
                if 0.5 * lane.width(s_ego) + right_position > self._traffic_rules_param.get("close_to_lane_border"):
                    return False
                else:
                    return True

    def _drives_rightmost(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Evaluates if a vehicle drives rightmost

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        # TODO consider orientation
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        vehicle_directly_right = self._vehicle_directly_right(vehicle, other_vehicles, time_step)
        if vehicle_directly_right is not None:
            if vehicle.right_position(time_step) - vehicle_directly_right.left_position(time_step) < \
                    self._traffic_rules_param.get("close_to_other_vehicle"):
                return True
            else:
                return False
        elif self._in_rightmost_lane(vehicle.lanelet_assignment[time_step]) is False:
            return False
        else:
            right_position = vehicle.right_position(time_step)
            s_ego = vehicle.states_lon[time_step].s
            lanes = self._road_network.find_lanes_by_lanelets(occupied_lanelet_ids)
            for lane in lanes:
                one_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(list(lane.contained_lanelets)[0])
                if one_lanelet.adj_right_same_direction is None or LaneletType.SHOULDER in \
                        self._road_network.lanelet_network.find_lanelet_by_id(one_lanelet.adj_right).lanelet_type:
                    if 0.5 * lane.width(s_ego) + right_position > self._traffic_rules_param.get("close_to_lane_border"):
                        return False
                    else:
                        return True

    def _drives_leftmost(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> bool:
        """
        Evaluates if a vehicle drives leftmost

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        # TODO consider orientation
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        vehicle_directly_left = self._vehicle_directly_left(vehicle, other_vehicles, time_step)
        if vehicle_directly_left is not None:
            if vehicle.left_position(time_step) - vehicle_directly_left.right_position(time_step) < \
                    self._traffic_rules_param.get("close_to_other_vehicle"):
                return True
            else:
                return False
        elif self._in_leftmost_lane(vehicle.lanelet_assignment[time_step]) is False:
            return False
        else:
            left_position = vehicle.left_position(time_step)
            s_ego = vehicle.states_lon[time_step].s
            lanes = self._road_network.find_lanes_by_lanelets(occupied_lanelet_ids)
            for lane in lanes:
                one_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(list(lane.contained_lanelets)[0])
                if one_lanelet.adj_left_same_direction is None or LaneletType.SHOULDER in \
                        self._road_network.lanelet_network.find_lanelet_by_id(one_lanelet.adj_left).lanelet_type:
                    if 0.5 * lane.width(s_ego) - left_position > self._traffic_rules_param.get("close_to_lane_border"):
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
        predicate_trace = {"is_in_same_lane__x_ego__x_o": {},
                           "is_in_front_of__x_ego__x_o": {},
                           "is_left_of__x_ego__x_o": {},
                           "on_access_ramp__x_o": {},
                           "on_main_carriage_way__x_o": {},
                           "in_rightmost_lane__x_ego": {ego_vehicle.id: {}},
                           "in_leftmost_lane__x_ego": {ego_vehicle.id: {}},
                           "on_access_ramp__x_ego": {ego_vehicle.id: {}},
                           "drives_leftmost__x_ego": {ego_vehicle.id: {}},
                           "drives_rightmost__x_ego": {ego_vehicle.id: {}},
                           "on_main_carriage_way__x_ego": {ego_vehicle.id: {}},
                           "right_of_broad_lane_marking__x_ego": {ego_vehicle.id: {}},
                           "left_of_broad_lane_marking__x_o": {},
                           "on_shoulder__x_ego": {ego_vehicle.id: {}},
                           "drives_right_lane_rightmost__x_ego": {ego_vehicle.id: {}}}

        for time_step in ego_vehicle.states_lon.keys():
            if "on_access_ramp__x_ego" in self._necessary_predicates:
                predicate_trace["on_access_ramp__x_ego"][ego_vehicle.id][time_step] = \
                    self._on_access_ramp(ego_vehicle, time_step)
            if "on_main_carriage_way__x_ego" in self._necessary_predicates:
                predicate_trace["on_main_carriage_way__x_ego"][ego_vehicle.id][time_step] = \
                    self._on_main_carriage_way(ego_vehicle, time_step)
            if "on_shoulder__x_ego" in self._necessary_predicates:
                predicate_trace["on_shoulder__x_ego"][ego_vehicle.id][time_step] = \
                    self._on_shoulder(ego_vehicle, time_step)
            if "in_rightmost_lane__x_ego" in self._necessary_predicates:
                predicate_trace["in_rightmost_lane__x_ego"][ego_vehicle.id][time_step] = \
                    self._in_rightmost_lane(ego_vehicle.lanelet_assignment[time_step])
            if "in_leftmost_lane__x_ego" in self._necessary_predicates:
                predicate_trace["in_leftmost_lane__x_ego"][ego_vehicle.id][time_step] = \
                    self._in_leftmost_lane(ego_vehicle.lanelet_assignment[time_step])
            if "right_of_broad_lane_marking__x_ego" in self._necessary_predicates:
                predicate_trace["right_of_broad_lane_marking__x_ego"][ego_vehicle.id][time_step] = \
                    self._right_of_broad_lane_marking(ego_vehicle, time_step)
            if "drives_leftmost__x_ego" in self._necessary_predicates:
                predicate_trace["drives_leftmost__x_ego"][ego_vehicle.id][time_step] = \
                    self._drives_leftmost(ego_vehicle, other_vehicles, time_step)
            if "drives_rightmost__x_ego" in self._necessary_predicates:
                predicate_trace["drives_rightmost__x_ego"][ego_vehicle.id][time_step] = \
                    self._drives_rightmost(ego_vehicle, other_vehicles, time_step)
            if "drives_right_lane_rightmost__x_ego" in self._necessary_predicates:
                predicate_trace["drives_right_lane_rightmost__x_ego"][ego_vehicle.id][time_step] = \
                    self._drives_right_lane_rightmost(ego_vehicle, time_step)

        for other_vehicle in other_vehicles:
            predicate_trace["is_in_same_lane__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["is_in_front_of__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["left_of_broad_lane_marking__x_o"][other_vehicle.id] = {}
            predicate_trace["is_left_of__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["on_access_ramp__x_o"][other_vehicle.id] = {}
            predicate_trace["on_main_carriage_way__x_o"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    continue
                if "is_in_same_lane__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["is_in_same_lane__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.is_in_same_lane(
                            self._road_network.find_lane_ids_by_lanelets(ego_vehicle.lanelet_assignment[time_step]),
                            self._road_network.find_lane_ids_by_lanelets(other_vehicle.lanelet_assignment[time_step]))
                if "is_in_front_of__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["is_in_front_of__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.is_in_front_of(ego_vehicle, other_vehicle, time_step)
                if "is_left_of__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["is_left_of__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.is_left_of(ego_vehicle, other_vehicle, time_step)
                if "left_of_broad_lane_marking__x_o" in self._necessary_predicates:
                    predicate_trace["left_of_broad_lane_marking__x_o"][other_vehicle.id][time_step] = \
                        self._left_of_broad_lane_marking(other_vehicle, time_step)
                if "on_access_ramp__x_o" in self._necessary_predicates:
                    predicate_trace["on_access_ramp__x_o"][other_vehicle.id][time_step] = \
                        self._on_access_ramp(other_vehicle, time_step)
                if "on_access_ramp__x_o" in self._necessary_predicates:
                    predicate_trace["on_access_ramp__x_o"][other_vehicle.id][time_step] = \
                        self._on_access_ramp(other_vehicle, time_step)
                if "on_main_carriage_way__x_o" in self._necessary_predicates:
                    predicate_trace["on_main_carriage_way__x_o"][other_vehicle.id][time_step] = \
                        self._on_main_carriage_way(other_vehicle, time_step)
        return predicate_trace
