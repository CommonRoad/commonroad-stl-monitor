from typing import List, Dict, Set, Union

from commonroad.scenario.lanelet import LaneletType, LineMarking, Lanelet

from src.predicates.predicate_collection import PredicateCollection
from src.common.vehicle import Vehicle
from src.common.road_network import RoadNetwork


class PositionPredicateCollection(PredicateCollection):
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

    @staticmethod
    def in_front_of(time_step: int, vehicle_p: Vehicle, vehicle_k: Vehicle) -> bool:
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

    def left_of(self, time_step: int, vehicle_p: Vehicle, vehicle_k: Vehicle) -> bool:
        """
        Evaluates if the kth vehicle is left of the pth vehicle

        :param vehicle_p: pth vehicle
        :param vehicle_k: kth vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if not self.in_lane_further_left(time_step, vehicle_p, vehicle_k):
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

    def in_lane_further_left(self, time_step: int, vehicle_p: Vehicle, vehicle_k: Vehicle) -> bool:
        """
        Evaluates if the kth vehicle is in any lane left of the pth vehicle

        :param vehicle_p: pth vehicle
        :param vehicle_k: kth vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanelets_k = vehicle_k.lanelet_assignment[time_step]
        lanelets_left_of_p = self._lanelets_ids_left_of_vehicle(time_step, vehicle_p)
        if any(lanelet_id in lanelets_left_of_p for lanelet_id in lanelets_k):
            return True
        else:
            return False

    def exist_leading_vehicle(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]) -> bool:
        """
        Predicate which evaluates if leading vehicle exists

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns bool indicating satisfaction
        """
        for veh in other_vehicles:
            if veh.states_lon.get(time_step) is None:
                continue
            if self.in_same_lane(time_step, vehicle, veh) and self.in_front_of(time_step, vehicle, veh):
                return True
        return False

    def in_same_lane(self, time_step: int, vehicle_p: Vehicle, vehicle_k: Vehicle) -> bool:
        """
        Evaluates if the kth vehicle is in the same lane as the pth vehicle

        :param time_step: time step of interest
        :param vehicle_k: kth vehicle
        :param vehicle_p: pth vehicle
        :returns boolean indicating satisfaction
        """
        lane_ids_k = self._road_network.find_lanes_by_lanelets(vehicle_k.lanelet_assignment[time_step])
        lane_ids_p = self._road_network.find_lanes_by_lanelets(vehicle_p.lanelet_assignment[time_step])
        for lane_id in lane_ids_k:
            if lane_id in lane_ids_p:
                return True
        return False

    @staticmethod
    def in_same_lane_classmethod(lane_ids_k: Set[int], lane_ids_p: Set[int]) -> bool:
        """
        Evaluates if the kth vehicle is in the same lane as the pth vehicle

        :param lane_ids_k: lane ids of the kth vehicle
        :param lane_ids_p: lane ids of the pth vehicle
        :returns boolean indicating satisfaction
        """
        for lane_id in lane_ids_k:
            if lane_id in lane_ids_p:
                return True
        return False

    def on_access_ramp(self, time_step: int, vehicle: Vehicle) -> bool:
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

    def on_exit_ramp(self, time_step: int, vehicle: Vehicle) -> bool:
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

    def on_shoulder(self, time_step: int, vehicle: Vehicle) -> bool:
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

    def on_main_carriage_way(self, time_step: int, vehicle: Vehicle) -> bool:
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

    def right_of_broad_lane_marking(self, time_step: int, vehicle: Vehicle) -> bool:
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

        lanelets_left_of_veh = self._lanelets_left_of_vehicle(time_step, vehicle)
        for lanelet in lanelets_left_of_veh:
            if lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED or \
                    lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID:
                return True
        return False

    def left_of_broad_lane_marking(self, time_step: int, vehicle: Vehicle) -> bool:
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

    def _lanelets_left_of_vehicle(self, time_step: int, vehicle: Vehicle) -> Set[Lanelet]:
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
            for lanelet in new_lanelets:
                left_lanelets.add(lanelet)

        return left_lanelets

    def _lanelets_ids_left_of_vehicle(self, time_step: int, vehicle: Vehicle) -> Set[int]:
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
            for lanelet in new_lanelets:
                left_lanelets.add(lanelet.lanelet_id)

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
            for lanelet in new_lanelets:
                right_lanelets.add(lanelet)

        return right_lanelets

    def _lanelets_left_of_lanelet(self, lanelet: Lanelet) -> Set[Lanelet]:
        """
        Extracts all lanelet IDs left of a given lanelet based on adjacency relations

        :param lanelet: given lanelet
        :returns set of lanelet objects
        """
        left_lanelets = set()
        tmp_lanelet = lanelet
        while tmp_lanelet.adj_left is not None:
            tmp_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_left)
            left_lanelets.add(tmp_lanelet)

        return left_lanelets

    def _lanelets_right_of_lanelet(self, lanelet: Lanelet) -> Set[Lanelet]:
        """
        Extracts all lanelet IDs right of a given lanelet based on adjacency relations

        :param lanelet: given lanelet
        :returns set of lanelet objects
        """
        right_lanelets = set()
        tmp_lanelet = lanelet
        while tmp_lanelet.adj_right is not None:
            tmp_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_right)
            right_lanelets.add(tmp_lanelet)

        return right_lanelets

    def _vehicles_left(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
        """
        Searches for vehicles left of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns list of vehicles left of a vehicle
        """
        vehicles_adj = self._vehicles_adjacent(vehicle, other_vehicles, time_step)
        vehicles_left = [veh for veh in vehicles_adj
                         if veh.right_position(time_step) > vehicle.left_position(time_step)]
        return vehicles_left

    def _vehicles_right(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
        """
        Searches for vehicles right of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns list of vehicles right of a vehicle
        """
        vehicles_adj = self._vehicles_adjacent(vehicle, other_vehicles, time_step)
        vehicles_right = [veh for veh in vehicles_adj
                          if veh.left_position(time_step) < vehicle.right_position(time_step)]
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
            if veh.states_lon.get(time_step) is None:
                continue
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

    def in_leftmost_lane(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Evaluates if a vehicle is in the leftmost lane

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            if self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_left_same_direction is None:
                return True
        return False

    def in_rightmost_lane_of_same_type(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Evaluates if a vehicle is in the rightmost lane of with the same type

        :param time_step: time step of interest
        :param vehicle: vehicle of interest
        :returns boolean indicating satisfaction
        """
        lanelet_ids = vehicle.lanelet_assignment[time_step]
        for l_id in lanelet_ids:
            lanelet = self._road_network.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.adj_right_same_direction is None:
                return True
            if any(lanelet_type not in
                   self._road_network.lanelet_network.find_lanelet_by_id(lanelet.adj_right).lanelet_type
                   for lanelet_type in lanelet.lanelet_type):
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

    def drives_right_lane_rightmost(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Evaluates if a vehicle drives in right lane rightmost

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        if self.in_rightmost_lane_of_same_type(time_step, vehicle) is False:
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

    def drives_rightmost(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]) -> bool:
        """
        Evaluates if a vehicle drives rightmost

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :param other_vehicles: list of other vehicles
        :returns boolean indicating satisfaction
        """
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        vehicle_directly_right = self._vehicle_directly_right(vehicle, other_vehicles, time_step)
        if vehicle_directly_right is not None:
            if vehicle.right_position(time_step) - vehicle_directly_right.left_position(time_step) < \
                    self._traffic_rules_param.get("close_to_other_vehicle"):
                return True
            else:
                return False
        else:
            right_position = vehicle.right_position(time_step)
            s_ego = vehicle.states_lon[time_step].s
            lanes = self._road_network.find_lanes_by_lanelets(occupied_lanelet_ids)
            for lane in lanes:
                if 0.5 * lane.width(s_ego) + right_position > self._traffic_rules_param.get("close_to_lane_border"):
                    return False
            return True

    def drives_leftmost(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]) -> bool:
        """
        Evaluates if a vehicle drives leftmost

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :param other_vehicles: list of other vehicles
        :returns boolean indicating satisfaction
        """
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        vehicle_directly_left = self._vehicle_directly_left(vehicle, other_vehicles, time_step)
        if vehicle_directly_left is not None:
            if vehicle.left_position(time_step) - vehicle_directly_left.right_position(time_step) < \
                    self._traffic_rules_param.get("close_to_other_vehicle"):
                return True
            else:
                return False
        else:
            left_position = vehicle.left_position(time_step)
            s_ego = vehicle.states_lon[time_step].s
            lanes = self._road_network.find_lanes_by_lanelets(occupied_lanelet_ids)
            for lane in lanes:
                if 0.5 * lane.width(s_ego) - left_position > self._traffic_rules_param.get("close_to_lane_border"):
                    return False
            return True

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"in_same_lane__x_ego__x_o": {},
                           "in_front_of__x_ego__x_o": {},
                           "left_of__x_ego__x_o": {},
                           "on_access_ramp__x_o": {},
                           "on_main_carriage_way__x_o": {},
                           "in_rightmost_lane_of_same_type__x_ego": {ego_vehicle.id: {}},
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
                    self.on_access_ramp(time_step, ego_vehicle)
            if "on_main_carriage_way__x_ego" in self._necessary_predicates:
                predicate_trace["on_main_carriage_way__x_ego"][ego_vehicle.id][time_step] = \
                    self.on_main_carriage_way(time_step, ego_vehicle)
            if "on_shoulder__x_ego" in self._necessary_predicates:
                predicate_trace["on_shoulder__x_ego"][ego_vehicle.id][time_step] = \
                    self.on_shoulder(time_step, ego_vehicle)
            if "in_rightmost_lane_of_same_type__x_ego" in self._necessary_predicates:
                predicate_trace["in_rightmost_lane_of_same_type__x_ego"][ego_vehicle.id][time_step] = \
                    self.in_rightmost_lane_of_same_type(time_step, ego_vehicle)
            if "in_leftmost_lane__x_ego" in self._necessary_predicates:
                predicate_trace["in_leftmost_lane__x_ego"][ego_vehicle.id][time_step] = \
                    self.in_leftmost_lane(time_step, ego_vehicle)
            if "right_of_broad_lane_marking__x_ego" in self._necessary_predicates:
                predicate_trace["right_of_broad_lane_marking__x_ego"][ego_vehicle.id][time_step] = \
                    self.right_of_broad_lane_marking(time_step, ego_vehicle)
            if "drives_leftmost__x_ego" in self._necessary_predicates:
                predicate_trace["drives_leftmost__x_ego"][ego_vehicle.id][time_step] = \
                    self.drives_leftmost(time_step, ego_vehicle, other_vehicles)
            if "drives_rightmost__x_ego" in self._necessary_predicates:
                predicate_trace["drives_rightmost__x_ego"][ego_vehicle.id][time_step] = \
                    self.drives_rightmost(time_step, ego_vehicle, other_vehicles)
            if "drives_right_lane_rightmost__x_ego" in self._necessary_predicates:
                predicate_trace["drives_right_lane_rightmost__x_ego"][ego_vehicle.id][time_step] = \
                    self.drives_right_lane_rightmost(time_step, ego_vehicle)

        for other_vehicle in other_vehicles:
            predicate_trace["in_same_lane__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["in_front_of__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["left_of_broad_lane_marking__x_o"][other_vehicle.id] = {}
            predicate_trace["left_of__x_ego__x_o"][other_vehicle.id] = {}
            predicate_trace["on_access_ramp__x_o"][other_vehicle.id] = {}
            predicate_trace["on_main_carriage_way__x_o"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    continue
                if "in_same_lane__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["in_same_lane__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.in_same_lane(time_step, ego_vehicle, other_vehicle)
                if "in_front_of__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["in_front_of__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.in_front_of(time_step, ego_vehicle, other_vehicle)
                if "left_of__x_ego__x_o" in self._necessary_predicates:
                    predicate_trace["left_of__x_ego__x_o"][other_vehicle.id][time_step] = \
                        self.left_of(time_step, ego_vehicle, other_vehicle)
                if "left_of_broad_lane_marking__x_o" in self._necessary_predicates:
                    predicate_trace["left_of_broad_lane_marking__x_o"][other_vehicle.id][time_step] = \
                        self.left_of_broad_lane_marking(time_step, other_vehicle)
                if "on_access_ramp__x_o" in self._necessary_predicates:
                    predicate_trace["on_access_ramp__x_o"][other_vehicle.id][time_step] = \
                        self.on_access_ramp(time_step, other_vehicle)
                if "on_access_ramp__x_o" in self._necessary_predicates:
                    predicate_trace["on_access_ramp__x_o"][other_vehicle.id][time_step] = \
                        self.on_access_ramp(time_step, other_vehicle)
                if "on_main_carriage_way__x_o" in self._necessary_predicates:
                    predicate_trace["on_main_carriage_way__x_o"][other_vehicle.id][time_step] = \
                        self.on_main_carriage_way(time_step, other_vehicle)
        return predicate_trace
