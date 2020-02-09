from typing import List, Dict, Set
from predicates.predicate_collection import PredicateCollection
from common.vehicle import Vehicle
from common.road_network import RoadNetwork
from commonroad.scenario.lanelet import LaneletType, LineMarking, Lanelet


class PositionPredicateCollection(PredicateCollection):
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

    @staticmethod
    def is_in_front_of(vehicle_p: Vehicle, vehicle_k: Vehicle, time_step: int) -> bool:
        """
        Evaluates if the kth vehicle is in front of the pth vehicle

        :param vehicle_p: pth vehicle
        :param vehicle_k: kth vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if vehicle_p.front_position(time_step) < vehicle_k.rear_position(time_step):
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

    def _is_on_access_ramp(self, vehicle: Vehicle, time_step: int) -> bool:
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

    def _is_on_exit_ramp(self, vehicle: Vehicle, time_step: int) -> bool:
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

    def _is_on_shoulder(self, vehicle: Vehicle, time_step: int) -> bool:
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

    def _is_on_main_carriage_way(self, vehicle: Vehicle, time_step: int) -> bool:
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
            if not (lanelet.line_marking_right_vertices is LineMarking.BROAD_DASHED or
                    lanelet.line_marking_right_vertices is LineMarking.BROAD_SOLID):
                return False

        lanelets_left_of_veh = self._lanelets_left_of_vehicle(vehicle, time_step)
        for lanelet in lanelets_left_of_veh:
            if lanelet.adj_left_same_direction is True:
                return True

    def _lanelets_left_of_vehicle(self, vehicle: Vehicle, time_step: int) -> Set[Lanelet]:
        """
        Evaluates if a vehicle is right of a broad lane marking

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

    @staticmethod
    def vehicles_left(vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
        """
        Searches for vehicles left of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns list of vehicles left of an vehicle
        """
        vehicles_left = []
        for veh in other_vehicles:
            if veh.rear_position(time_step) < vehicle.front_position(time_step) < veh.front_position(time_step):
                vehicles_left.append(veh)
                continue
            if veh.rear_position(time_step) < vehicle.rear_position(time_step) < veh.front_position(time_step):
                vehicles_left.append(veh)
                continue
            if vehicle.rear_position(time_step) < veh.rear_position(time_step) \
                    and veh.front_position(time_step) < vehicle.front_position(time_step):
                vehicles_left.append(veh)
                continue
        return vehicles_left

    @classmethod
    def vehicle_is_left(cls, vehicle_k: Vehicle, vehicle_p: Vehicle, time_step: int) -> bool:
        """
        Evaluates if the kth vehicle is left of the pth vehicle

        :param vehicle_k: the kth vehicle
        :param vehicle_p: the pth vehicle
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        if (vehicle_p.rear_position(time_step) <= vehicle_k.rear_position(time_step) <=
            vehicle_p.front_position(time_step)) or \
                (vehicle_p.rear_position(time_step) <= vehicle_k.front_position(time_step) <=
                 vehicle_p.front_position(time_step)) or \
                (vehicle_p.front_position(time_step) < vehicle_k.front_position(time_step) and
                 vehicle_k.rear_position(time_step) < vehicle_p.rear_position(time_step)):
            return True
        else:
            return False

    def _is_on_left_most_lane(self, lanelet_ids: Set[int]):
        """
        Evaluates if a vehicle is on the left most lane

        :param lanelet_ids: lanelet IDs the vehicle is on
        :returns boolean indicating satisfaction
        """
        for l_id in lanelet_ids:
            if self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_left_same_direction is None:
                return True
        return False

    def _is_on_right_most_lane(self, lanelet_ids: Set[int]):
        """
        Evaluates if a vehicle is on the right most lane

        :param lanelet_ids: lanelet IDs the vehicle is on
        :returns boolean indicating satisfaction
        """
        for l_id in lanelet_ids:
            if self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_right_same_direction is None:
                return True
        return False

    def _drives_left_most(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a vehicle drives leftmost in its occupied lanelets

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        d = vehicle.states_lat[time_step].d
        s = vehicle.states_lon[time_step].s
        lanes = self._road_network.find_lanes_by_lanelets(occupied_lanelet_ids)
        width = vehicle.shape.width
        for lane in lanes:
            if 0.5 * lane.width(s) + d - 0.5 * width > self._traffic_rules_param.get("close_to_lane_border"):
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
        predicate_trace = {"same_lane_as_ego_vehicle": {},
                           "in_front_of_ego_vehicle": {}}

        for other_vehicle in other_vehicles:
            predicate_trace["same_lane_as_ego_vehicle"][other_vehicle.id] = {}
            predicate_trace["in_front_of_ego_vehicle"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    continue
                predicate_trace["same_lane_as_ego_vehicle"][other_vehicle.id][time_step] = \
                    self.is_in_same_lane(
                        self._road_network.find_lane_ids_by_lanelets(ego_vehicle.lanelet_assignment[time_step]),
                        self._road_network.find_lane_ids_by_lanelets(other_vehicle.lanelet_assignment[time_step]))
                predicate_trace["in_front_of_ego_vehicle"][other_vehicle.id][time_step] = \
                    self.is_in_front_of(ego_vehicle.states_lon[time_step].s, other_vehicle.states_lon[time_step].s)
        return predicate_trace
