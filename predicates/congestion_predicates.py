from typing import List, Dict, Set
from predicates.predicate_collection import PredicateCollection
from predicates.position_predicates import PositionPredicateCollection
from common.vehicle import Vehicle
from common.road_network import RoadNetwork
from commonroad.scenario.lanelet import Lanelet


class CongestionPredicateCollection(PredicateCollection):
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

    def _is_vehicle_in_congestion(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int):
        """
        Evaluates if a vehicles is in a congestion

        :param vehicle: vehicle object
        :param other_vehicles: other vehicles in scenario
        :param time_step: time step of interest
        :returns boolean indicating satisfaction
        """
        front_vehicles_all_vehicle_lanes = PositionPredicateCollection.front_vehicle_same_lane(vehicle, other_vehicles,
                                                                                               time_step)
        if vehicle.states_lon[time_step].v > self._traffic_rules_param.get("max_congestion_velocity"):
            return False

        for lane_vehicles in front_vehicles_all_vehicle_lanes:
            if len(lane_vehicles) < self._traffic_rules_param.get("num_veh_congestion"):
                return False
            for veh in lane_vehicles:
                if veh.states_lon[time_step].v > self._traffic_rules_param.get("max_congestion_velocity"):
                    return False
        return True

    def _adjacent_lanelets(self, lanelet: Lanelet) -> Set[Lanelet]:
        """
        Returns all lanelet which are adjacent to a lanelet and the lanelet itself

        :param lanelet: CommonRoad lanelet
        :returns set of adjacent lanelets
        """
        lanelets = set()
        l = lanelet
        while l.adj_left is not None:
            l = self._road_network.lanelet_network.find_lanelet_by_id(l.adj_left)
            lanelets.add(l)
        l = lanelet
        while l.adj_right is not None:
            l = self._road_network.lanelet_network.find_lanelet_by_id(l.adj_left)
            lanelets.add(l)
        return lanelets

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
        for lanelet in adj_lanelets:
            road_width +=  self._road_network.find_lane_by_lanelet(lanelet.lanelet_id).width(position)
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
        pass
