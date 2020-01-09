from typing import List, Dict, Set
from predicates.predicate_collection import PredicateCollection
from predicates.position_predicates import PositionPredicateCollection
from common.vehicle import Vehicle
from common.road_network import RoadNetwork


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

    def _vehicles_left(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int) -> List[Vehicle]:
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

    def _vehicle_in_congestion(self, vehicle: Vehicle, other_vehicles: List[Vehicle], time_step: int):
        front_vehicles_all_vehicle_lanes = PositionPredicateCollection.front_vehicle_same_lane(vehicle, other_vehicles,
                                                                                               time_step)
        if vehicle.states_lon[time_step].v > self._traffic_rule_param.get("max_congestion_velocity"):
            return False

        for lane_vehicles in front_vehicles_all_vehicle_lanes:
            if len(lane_vehicles) < self._traffic_rule_param.get("num_veh_congestion"):
                return False
            for veh in lane_vehicles:
                if veh.states_lon[time_step].v > self._traffic_rule_param.get("max_congestion_velocity"):
                    return False
        return True

    def _street_contains_two_lanes(self, lanelet_ids: Set[int]):
        for l_id in lanelet_ids:
            num_lanes = 0
            l_id_tmp = l_id
            while self._road_network.lanelet_network.find_lanelet_by_id(l_id_tmp).adj_right_same_direction:
                num_lanes +=1
                if num_lanes == 2:
                    return True
                l_id_tmp = self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_right
            l_id_tmp = l_id
            while self._road_network.lanelet_network.find_lanelet_by_id(l_id_tmp).adj_left_same_direction:
                num_lanes +=1
                if num_lanes == 2:
                    return True
                l_id_tmp = self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_left

    def _is_on_left_most_lane(self, lanelet_ids: Set[int]):
        for l_id in lanelet_ids:
            if self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_left_same_direction is None:
                return True
        return False

    def _is_on_right_most_lane(self, lanelet_ids: Set[int]):
        for l_id in lanelet_ids:
            if self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_right_same_direction is None:
                return True
        return False

    def _is_on_middle_lane(self, lanelet_ids: Set[int]):
        for l_id in lanelet_ids:
            if self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_right_same_direction is not None \
                    and self._road_network.lanelet_network.find_lanelet_by_id(l_id).adj_left_same_lane is not None:
                return True
        return False

    def evaluate_predicates(self, ego_vehicle: Vehicle, other_vehicles: List[Vehicle]) -> \
            Dict[str, Dict[int, Dict[int, bool]]]:
        """
        Evaluates trajectory for safety predicate compliance

        :param ego_vehicle: ego vehicle object containing trajectory and other relevant information
        :param other_vehicles: other vehicle objects containing trajectory and other relevant information
        :returns dictionary with trace of bool values for each predicate
        """
        pass