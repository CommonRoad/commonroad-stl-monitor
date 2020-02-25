from typing import List, Dict, Set

from commonroad.scenario.lanelet import Lanelet

from src.predicates.predicate_collection import PredicateCollection
from src.predicates.position_predicates import PositionPredicateCollection
from src.common.vehicle import Vehicle
from src.common.road_network import RoadNetwork


class GeneralPredicateCollection(PredicateCollection):
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

    def _in_congestion(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int):
        """
        Evaluates if a vehicles is in a congestion

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        vehicle_lanelets = vehicle.lanelet_assignment[time_step]
        num_vehicles = 0
        for veh_o in other_vehicles:
            veh_o_lanelets = veh_o.lanelet_assignment[time_step]
            if PositionPredicateCollection.is_in_front_of(vehicle, veh_o, time_step) and \
                    PositionPredicateCollection.is_in_same_lane(
                        self._road_network.find_lane_ids_by_lanelets(vehicle_lanelets),
                        self._road_network.find_lane_ids_by_lanelets(veh_o_lanelets)) and \
                    veh_o.states_lon[time_step].v <= self._traffic_rules_param.get("max_congestion_velocity"):
                num_vehicles += 1
        if num_vehicles >= self._traffic_rules_param.get("num_veh_congestion"):
            return True
        else:
            return False

    def remove_vehicle_from_list(self, vehicle_list: List[Vehicle], id: int):
        for veh in vehicle_list:
            if veh.id == id:
                vehicle_list.remove(veh)
                return

    def _congestion_left(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int):
        """
        Evaluates if a congestion exists in the left lane of a vehicle

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        for idx, veh_o in enumerate(other_vehicles):
            other_vehicles_updated = other_vehicles[:idx] + other_vehicles[idx+1:]
            self.remove_vehicle_from_list(other_vehicles_updated, veh_o.id)
            if PositionPredicateCollection.vehicle_is_left(veh_o, vehicle, time_step) and \
                    self._in_congestion(veh_o, other_vehicles_updated, time_step):
                return True
        return False

    def _adjacent_lanelets(self, lanelet: Lanelet) -> Set[Lanelet]:
        """
        Returns all lanelet which are adjacent to a lanelet and the lanelet itself

        :param lanelet: CommonRoad lanelet
        :returns set of adjacent lanelets
        """
        lanelets = set()
        l = lanelet
        while l is not None and l.adj_left is not None:
            l = self._road_network.lanelet_network.find_lanelet_by_id(l.adj_left)
            lanelets.add(l)
        l = lanelet
        while l is not None and l.adj_right is not None:
            l = self._road_network.lanelet_network.find_lanelet_by_id(l.adj_right)
            lanelets.add(l)
        return lanelets

    def _makes_u_turn(self, vehicle: Vehicle, time_step: int) -> bool:
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
        :param lane: lane lanelet belongs to
        :param position: longitudinal position
        :returns road witdh
        """
        adj_lanelets = self._adjacent_lanelets(lanelet)
        road_width = 0.0
        for lanelet in list(adj_lanelets):
            road_width += self._road_network.find_lane_by_lanelet(lanelet.lanelet_id).width(position)
        return road_width

    def _interstate_broad_enough(self, vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluates if a interstate is broad enough to build a standard emergency lane

        :param vehicle: vehicle object
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        occupied_lanelet_ids = vehicle.lanelet_assignment[time_step]
        s = vehicle.states_lon[time_step].s
        for id in occupied_lanelet_ids:
            if self._road_width(self._road_network.lanelet_network.find_lanelet_by_id(id), s) \
                    <= self._traffic_rules_param.get("min_interstate_width"):
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
        predicate_trace = {"in_congestion": {ego_vehicle.id: {}},
                           "congestion_left": {ego_vehicle.id: {}},
                           "makes_u_turn": {ego_vehicle.id: {}},
                           "interstate_broad_enough": {ego_vehicle.id: {}}}

        for time_step in ego_vehicle.states_lon.keys():
            predicate_trace["in_congestion"][ego_vehicle.id][time_step] = \
                self._in_congestion(ego_vehicle, other_vehicles, time_step)
            predicate_trace["congestion_left"][ego_vehicle.id][time_step] = \
                self._congestion_left(ego_vehicle, other_vehicles, time_step)
            predicate_trace["makes_u_turn"][ego_vehicle.id][time_step] = \
                self._makes_u_turn(ego_vehicle, time_step)
            predicate_trace["interstate_broad_enough"][ego_vehicle.id][time_step] = \
                self._interstate_broad_enough(ego_vehicle, time_step)

        return predicate_trace
