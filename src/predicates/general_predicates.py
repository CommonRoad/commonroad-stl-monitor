from typing import List, Dict, Set, Tuple

from commonroad.scenario.lanelet import Lanelet

from src.predicates.predicate_collection import PredicateCollection
from src.predicates.position_predicates import PositionPredicateCollection
from src.common.vehicle import Vehicle
from src.common.road_network import RoadNetwork


class GeneralPredicateCollection(PredicateCollection):
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

    def in_congestion(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]):
        """
        Evaluates if a vehicle is in a congestion

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        num_vehicles = 0
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:  # in some datasets trajectories do not
                # start at the first time step
                continue
            if PositionPredicateCollection.in_front_of(time_step, vehicle, veh_o) and \
                    PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(vehicle.lanelet_assignment[time_step]),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])) and \
                    veh_o.states_lon[time_step].v <= self._traffic_rules_param.get("max_congestion_velocity"):
                num_vehicles += 1
        if num_vehicles >= self._traffic_rules_param.get("num_veh_congestion"):
            return True
        else:
            return False

    def in_slow_moving_traffic(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]):
        """
        Evaluates if a vehicle is part of slow moving traffic

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        num_vehicles = 0
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:  # in some datasets trajectories do not
                # start at the first time step
                continue
            if PositionPredicateCollection.in_front_of(time_step, vehicle, veh_o) and \
                    PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(vehicle.lanelet_assignment[time_step]),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])) and \
                    veh_o.states_lon[time_step].v <= self._traffic_rules_param.get("max_slow_moving_traffic_velocity"):
                num_vehicles += 1
        if num_vehicles >= self._traffic_rules_param.get("num_veh_slow_moving_traffic"):
            return True
        else:
            return False

    def in_queue_of_vehicles(self, time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]):
        """
        Evaluates if a vehicle is part of a queue of vehicles

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        num_vehicles = 0
        for veh_o in other_vehicles:
            if veh_o.states_lon.get(time_step) is None:  # in some datasets trajectories do not
                # start at the first time step
                continue
            if PositionPredicateCollection.in_front_of(time_step, vehicle, veh_o) and \
                    PositionPredicateCollection.in_same_lane_classmethod(
                        self._road_network.find_lane_ids_by_lanelets(vehicle.lanelet_assignment[time_step]),
                        self._road_network.find_lane_ids_by_lanelets(veh_o.lanelet_assignment[time_step])) and \
                    veh_o.states_lon[time_step].v <= self._traffic_rules_param.get("max_slow_moving_traffic_velocity"):
                num_vehicles += 1
        if num_vehicles >= self._traffic_rules_param.get("num_veh_queue_of_vehicles"):
            return True
        else:
            return False

    def _adjacent_lanelets(self, lanelet: Lanelet) -> Set[Lanelet]:
        """
        Returns all lanelet which are adjacent to a lanelet and the lanelet itself

        :param lanelet: CommonRoad lanelet
        :returns set of adjacent lanelets
        """
        lanelets = {lanelet}
        la = lanelet
        while la is not None and la.adj_left is not None:
            la = self._road_network.lanelet_network.find_lanelet_by_id(la.adj_left)
            lanelets.add(la)
        la = lanelet
        while la is not None and la.adj_right is not None:
            la = self._road_network.lanelet_network.find_lanelet_by_id(la.adj_right)
            lanelets.add(la)
        return lanelets

    def makes_u_turn(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Predicate which evaluates if vehicle makes U-turn

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        lanes = self._road_network.find_lanes_by_lanelets(vehicle.lanelet_assignment[time_step])
        for la in lanes:
            if self._traffic_rules_param.get("u_turn") <= \
                    abs(vehicle.states_lat[time_step].theta - la.orientation(vehicle.states_lon[time_step].s)):
                return True
        return False

    def _road_width(self, lanelet: Lanelet, position: float) -> float:
        """
        Calculates width of road given a lanelet and a longitudinal position

        :param lanelet: CommonRoad lanelet
        :param position: longitudinal position
        :returns road witdh
        """
        adj_lanelets = self._adjacent_lanelets(lanelet)
        road_width = 0.0
        for lanelet in list(adj_lanelets):
            road_width += self._road_network.find_lane_by_lanelet(lanelet.lanelet_id).width(position)
        return road_width

    def interstate_broad_enough(self, time_step: int, vehicle: Vehicle) -> bool:
        """
        Evaluates if a interstate is broad enough to build a standard emergency lane

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        s = vehicle.states_lon[time_step].s
        for l_id in occupied_lanelet_ids:
            if self._road_width(self._road_network.lanelet_network.find_lanelet_by_id(l_id), s) \
                    <= self._traffic_rules_param.get("min_interstate_width"):
                return False
        return True

    def cut_in(self, time_step: int, vehicle_k: Vehicle, vehicle_p: Vehicle) -> bool:
        """
        Predicate which checks if the kth vehicle performs a cut-in into the pth vehicles lane

        :param vehicle_p: the pth vehicle
        :param vehicle_k: the kth vehicle
        :param time_step: time step of interest
        :returns Boolean indicating satisfaction
        """
        if len(self._road_network.find_lanes_by_lanelets(vehicle_k.lanelet_assignment[time_step])) == 1:
            return False
        if not PositionPredicateCollection.in_same_lane_classmethod(
                self._road_network.find_lane_ids_by_lanelets(vehicle_k.lanelet_assignment[time_step]),
                self._road_network.find_lane_ids_by_lanelets(vehicle_p.lanelet_assignment[time_step])):
            return False
        if vehicle_k.states_lat[time_step].d < vehicle_p.states_lat[time_step].d \
                and vehicle_k.states_lat[time_step].theta < 0 or \
                vehicle_k.states_lat[time_step].d > vehicle_p.states_lat[time_step].d \
                and vehicle_k.states_lat[time_step].theta > 0:
            return True
        else:
            return False

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :param time_interval: time interval for which the predicates should be evaluated
        :returns dictionary with trace of bool values for each predicate
        """
        predicate_trace = {"in_congestion__x_ego": {ego_vehicle.id: {}},
                           "in_congestion__x_o": {},
                           "in_slow_moving_traffic__x_ego": {ego_vehicle.id: {}},
                           "in_slow_moving_traffic__x_o": {},
                           "in_queue_of_vehicles__x_ego": {ego_vehicle.id: {}},
                           "in_queue_of_vehicles__x_o": {},
                           "cut_in__x_o__x_ego": {},
                           "makes_u_turn__x_ego": {ego_vehicle.id: {}},
                           "interstate_broad_enough__x_ego": {ego_vehicle.id: {}}}

        for time_step in ego_vehicle.states_lon.keys():
            if "in_congestion__x_ego" in self._necessary_predicates:
                predicate_trace["in_congestion__x_ego"][ego_vehicle.id][time_step] = \
                    self.in_congestion(time_step, ego_vehicle, other_vehicles)
            if "in_slow_moving_traffic__x_ego" in self._necessary_predicates:
                predicate_trace["in_slow_moving_traffic__x_ego"][ego_vehicle.id][time_step] = \
                    self.in_slow_moving_traffic(time_step, ego_vehicle, other_vehicles)
            if "in_queue_of_vehicles__x_ego" in self._necessary_predicates:
                predicate_trace["in_queue_of_vehicles__x_ego"][ego_vehicle.id][time_step] = \
                    self.in_queue_of_vehicles(time_step, ego_vehicle, other_vehicles)
            if "makes_u_turn__x_ego" in self._necessary_predicates:
                predicate_trace["makes_u_turn__x_ego"][ego_vehicle.id][time_step] = \
                    self.makes_u_turn(time_step, ego_vehicle)
            if "interstate_broad_enough__x_ego" in self._necessary_predicates:
                predicate_trace["interstate_broad_enough__x_ego"][ego_vehicle.id][time_step] = \
                    self.interstate_broad_enough(time_step, ego_vehicle)

        for other_vehicle in other_vehicles:
            predicate_trace["in_congestion__x_o"][other_vehicle.id] = {}
            predicate_trace["in_slow_moving_traffic__x_o"][other_vehicle.id] = {}
            predicate_trace["in_queue_of_vehicles__x_o"][other_vehicle.id] = {}
            predicate_trace["cut_in__x_o__x_ego"][other_vehicle.id] = {}
            for time_step in ego_vehicle.states_lon.keys():
                if other_vehicle.states_lon.get(time_step) is None:
                    continue
                if "in_congestion__x_o" in self._necessary_predicates:
                    predicate_trace["in_congestion__x_o"][other_vehicle.id][time_step] = \
                        self.in_congestion(time_step, other_vehicle, other_vehicles)
                if "in_slow_moving_traffic__x_o" in self._necessary_predicates:
                    predicate_trace["in_slow_moving_traffic__x_o"][other_vehicle.id][time_step] = \
                        self.in_slow_moving_traffic(time_step, other_vehicle, other_vehicles)
                if "in_queue_of_vehicles__x_o" in self._necessary_predicates:
                    predicate_trace["in_queue_of_vehicles__x_o"][other_vehicle.id][time_step] = \
                        self.in_queue_of_vehicles(time_step, other_vehicle, other_vehicles)
                if "cut_in__x_o__x_ego" in self._necessary_predicates:
                    predicate_trace["cut_in__x_o__x_ego"][other_vehicle.id][time_step] = \
                        self.cut_in(time_step, other_vehicle, ego_vehicle)

        return predicate_trace

    def evaluate_constraints(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                             time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        pass

    def evaluate_robustness(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle],
                            time_interval: Tuple[int, int]) -> Dict[str, Dict[int, Dict[int, float]]]:
        pass