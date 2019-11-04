from typing import List, Dict, Tuple
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
import common.util as util
from common.vehicle import Vehicle, Maneuver, StateLongitudinal
from acc.safety_layer import SafetyLayer
from common.configuration import LaneCategory


class ObstacleSelection:
    """
    Class to extract dynamic obstacles in the field of view of the ego vehicle within the left,
    same, and right lane.
    """
    def __init__(self, scenario: Scenario, curvilinear_cosy_ego_lane, ego_lane: Lanelet, ego_lanelet: Lanelet,
                 left_lane: Lanelet, right_lane: Lanelet,  safety_layer: SafetyLayer, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict):
        """
        :param scenario: CommonRoad scenario
        :param curvilinear_cosy_ego_lane: curvilinear coordinate system based on the ego vehicle's lane
        :param ego_lane: the ego vehicle's lane
        :param right_lane: the right adjacent lane of the ego vehicle's lane
        :param left_lane: the left adjacent lane of the ego vehicle's lane
        :param ego_lane: the current lanelet of the ego vehicle
        :param safety_layer: safety layer object
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        """
        self._scenario = scenario
        self._curvilinear_cosy_ego_lane = curvilinear_cosy_ego_lane
        self._ego_lane = ego_lane
        self._ego_lanelet = ego_lanelet
        self._right_lane = right_lane
        self._left_lane = left_lane
        self._safety_layer = safety_layer
        self._fov = ego_vehicle_param.get("fov")
        self._th_theta = other_vehicles_param.get("th_theta")
        self._th_offset = other_vehicles_param.get("th_offset")
        self._n_cutin = other_vehicles_param.get("n_cutin")
        self._vehicle_dict = {}

        self._lanelet_network = LaneletNetwork.create_from_lanelet_list([ego_lane])
        if self._right_lane is not None:
            self._curvilinear_cosy_right = util.create_curvilinear_coordinate_system_from_lanelet(
                right_lane.center_vertices)
            self._lanelet_network.add_lanelet(self._right_lane)
        else:
            self._curvilinear_cosy_right = None
        if self._left_lane is not None:
            self._curvilinear_cosy_left = util.create_curvilinear_coordinate_system_from_lanelet(
                left_lane.center_vertices)
            self._lanelet_network.add_lanelet(self._left_lane)
        else:
            self._curvilinear_cosy_left = None

    @property
    def vehicle_dict(self) -> Dict[int, Vehicle]:
        return self._vehicle_dict

    @vehicle_dict.setter
    def vehicle_dict(self, vehicle_dict: Dict[int, Vehicle]):
        self._vehicle_dict = vehicle_dict

    def update_vehicle_dict(self, new_obstacles: List[Dict], time_step: int):
        """
        Update vehicle dictionary with new obstacle information from current time step
        :param new_obstacles: list containing dictionaries with information of obstacle from current time step
        :param time_step: current time step
        """
        for data in new_obstacles:
            if self._vehicle_dict.get(data["id"]) is not None:
                self._vehicle_dict[data["id"]].append_state_lon(data["state_lon"], time_step)
                self._vehicle_dict[data["id"]].append_state_lat(data["state_lat"], time_step)
                self._vehicle_dict[data["id"]].append_state_cr(data["state_cr"], time_step)
                self._vehicle_dict[data["id"]].append_lane_number(data["lane_number"], time_step)
            else:
                vehicle = Vehicle(data["state_lon"], data["state_lat"], data["shape"],
                                  data["lane_number"], data["state_cr"], data["id"])
                self._vehicle_dict[data["id"]] = vehicle

    def cutin_prev(self, vehicle: Vehicle, time_step: int):
        """
        Evaluation if vehicle in adjacent lane performs cut-in already for n_cutin time steps
        :param time_step: current time step
        :param vehicle: vehicle in adjacent lane
        :return: Boolean indicating if vehicle performed cut-in for enough time steps
        """
        count = 0
        for idx in range(self._n_cutin):
            try:
                if vehicle.maneuver_list[time_step - idx] == Maneuver.CUTIN:
                    count += 1
            except:
                print("stop")
        if count == self._n_cutin:
            return True
        else:
            return False

    def detect_cutin_adjacent(self, time_step: int, vehicle_ids_right: List[int], vehicle_ids_left: List[int],
                              ego_state_lon: StateLongitudinal) -> List[int]:
        """
        Detects cut-in vehicles in adjacent lanes and updates vehicle maneuver
        :param time_step: current time step
        :param vehicle_ids_right: vehicle IDs of vehicles in ego vehicle's right lane
        :param vehicle_ids_left: vehicle IDs of vehicles in ego vehicle's left lane
        :param ego_state_lon: ego vehicle object
        :return: lists with IDs of vehicles in adjacent lane performing a cut-in
        """
        vehicles_adjacent = []
        for idx, veh_id in enumerate(vehicle_ids_left):
            if abs(self._vehicle_dict[veh_id].state_list_lat[time_step].theta) > self._th_theta \
                    and abs(self._vehicle_dict[veh_id].state_list_lat[time_step].d) > self._th_offset:
                self._vehicle_dict[veh_id].set_maneuver(Maneuver.CUTIN, time_step)
                if self.cutin_prev(self._vehicle_dict[veh_id], time_step):
                    vehicles_adjacent.append(veh_id)
                    state_lon = self.vehicle_dict[veh_id].state_list_lon[time_step]
                    safe_distance = self._safety_layer.safe_distance_profile_based(state_lon.s, state_lon.v,
                                                                                   state_lon.a,
                                                                                   ego_state_lon.s, ego_state_lon.v,
                                                                                   ego_state_lon.a)
                    self.vehicle_dict[veh_id].append_safe_distance(safe_distance, time_step)

        for idx, veh_id in enumerate(vehicle_ids_right):
            if abs(self._vehicle_dict[veh_id].state_list_lat[time_step].theta) < self._th_theta \
                    and abs(self._vehicle_dict[veh_id].state_list_lat[time_step].d) < self._th_offset:
                self._vehicle_dict[veh_id].set_maneuver(Maneuver.CUTIN, time_step)
                if self.cutin_prev(self._vehicle_dict[veh_id], time_step):
                    vehicles_adjacent.append(veh_id)
                    state_lon = self.vehicle_dict[veh_id].state_list_lon[time_step]
                    safe_distance = self._safety_layer.safe_distance_profile_based(state_lon.s, state_lon.v,
                                                                                   state_lon.a,
                                                                                   ego_state_lon.s, ego_state_lon.v,
                                                                                   ego_state_lon.a)
                    self.vehicle_dict[veh_id].append_safe_distance(safe_distance, time_step)

        return vehicles_adjacent

    def detect_cutin_same(self, time_step: int, vehicle_ids_same_lane: List[int], vehicles_cutin: List[int],
                          ego_vehicle: Vehicle) -> Tuple[List[int], List[int]]:
        """
        Updates vehicle maneuver
        :param time_step: current time step
        :param vehicle_ids_same_lane: vehicle IDs of vehicles in ego vehicle's lane
        :param vehicles_cutin: vehicle IDs of vehicles in ego vehicle's adjacent lanes
        :param ego_vehicle: ego vehicle object
        :return: lists with IDs of vehicles performing cut-in and driving in front of ego vehicle
        """
        vehicle_ids_same_lane_updated = []
        for idx, veh_id in enumerate(vehicle_ids_same_lane):
            ego_state_lon = self.vehicle_dict[veh_id].state_list_lon[time_step]
            state_lon = self.vehicle_dict[veh_id].state_list_lon[time_step]
            safe_distance = self._safety_layer.safe_distance_profile_based(state_lon.s, state_lon.v,
                                                                           state_lon.a,
                                                                           ego_state_lon.s, ego_state_lon.v,
                                                                           ego_state_lon.a)
            self.vehicle_dict[veh_id].append_safe_distance(safe_distance, time_step)
            if self._vehicle_dict[veh_id].rear_position(time_step) - ego_vehicle.front_position(time_step) \
                    <= self._vehicle_dict[veh_id].safe_distance_list[time_step]:
                if self.cutin_prev(self._vehicle_dict[veh_id], time_step):
                    self._vehicle_dict[veh_id].append_maneuver(Maneuver.CUTIN, time_step)
                    vehicles_cutin.append(veh_id)
                else:
                    self._vehicle_dict[veh_id].append_maneuver(Maneuver.LANE_FOLLOWING, time_step)
                    vehicle_ids_same_lane_updated.append(veh_id)
            else:
                self._vehicle_dict[veh_id].append_maneuver(Maneuver.LANE_FOLLOWING, time_step)
                vehicle_ids_same_lane_updated.append(veh_id)

        return vehicle_ids_same_lane_updated, vehicles_cutin

    @staticmethod
    def vehicle_ids_at_time_step(new_obstacles):
        """
        Extracts list of IDs of vehicles at provided time step
        :param new_obstacles: list containing dictionaries with information of obstacle from current time step
        :return: list for each lane with IDs of vehicles existing at current time step
        """
        vehicle_ids_left, vehicle_ids_right, vehicle_ids_same = [], [], []
        for obs in new_obstacles:
            if obs["lane_number"] == LaneCategory.LEFT:
                vehicle_ids_left.append(obs["id"])
            elif obs["lane_number"] == LaneCategory.RIGHT:
                vehicle_ids_right.append(obs["id"])
            else:
                vehicle_ids_same.append(obs["id"])

        return vehicle_ids_left, vehicle_ids_right, vehicle_ids_same

    def vehicles_fov(self, time_step: int, ego_s_position: float, vehicle_ids_left: List[int],
                     vehicle_ids_right: List[int],
                     vehicle_ids_same: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Extracts list with IDs of vehicles within the field of view of the ego vehicle at specific time step
        :param time_step: current time step
        :param ego_s_position: ego vehicle s-coordinate
        :param vehicle_ids_left: list of vehicles in left lane
        :param vehicle_ids_right: list of vehicles in right lane
        :param vehicle_ids_same: list of vehicles in ego vehicle's lane
        :return: lists with vehicle IDs for left, right, and ego vehicle's lane
        """
        vehicle_ids_left_updated = \
            [veh_id for veh_id in vehicle_ids_left
             if ego_s_position <= self._vehicle_dict[veh_id].state_list_lon[time_step].s <= ego_s_position + self._fov]
        vehicle_ids_right_updated = \
            [veh_id for veh_id in vehicle_ids_right
             if ego_s_position <= self._vehicle_dict[veh_id].state_list_lon[time_step].s <= ego_s_position + self._fov]
        vehicle_ids_same_updated = \
            [veh_id for veh_id in vehicle_ids_same
             if ego_s_position <= self._vehicle_dict[veh_id].state_list_lon[time_step].s <= ego_s_position + self._fov]

        return vehicle_ids_left_updated, vehicle_ids_right_updated, vehicle_ids_same_updated

    def get_vehicles_from_ids(self, vehicle_ids: List[int]):
        """
        Extract vehicle objects from IDs
        :param vehicle_ids: list of vehicle IDs
        :return: list with vehicle objects
        """
        vehicles = []
        for veh_id in vehicle_ids:
            vehicles.append(self._vehicle_dict[veh_id])
        return vehicles

    def extract_vehicles(self, time_step: int, ego_vehicle: Vehicle,
                         obstacles: List[Dict]) -> Tuple[List[Vehicle], List[Vehicle]]:
        """
        Returns lists for vehicles to consider for cut-in reaction and ACC calculation
        in front of the ego vehicle within the field of view
        :param time_step: current time step
        :param ego_vehicle: ego vehicle object
        :param obstacles: list containing dictionaries with information of obstacle from current time step
        :return: list for left, right and ego lane containing vehicles to consider in the next process steps
        """
        # Update vehicle dictionary with new vehicle information from current time step
        self.update_vehicle_dict(obstacles, time_step)

        # Extract vehicle IDS of vehicles existing at current time step based on their lane membership
        vehicle_ids_left, vehicle_ids_right, vehicle_ids_same = self.vehicle_ids_at_time_step(obstacles)

        # Select all vehicles in front of the ego vehicle within field of view at specific time step
        vehicle_ids_left, vehicle_ids_right, vehicle_ids_same = \
            self.vehicles_fov(time_step, ego_vehicle.state_list_lon[time_step].s, vehicle_ids_left,
                              vehicle_ids_right, vehicle_ids_same)

        # Cut-in detection and safe distance calculation for left and right lane
        vehicle_ids_adjacent = self.detect_cutin_adjacent(time_step, vehicle_ids_right, vehicle_ids_left,
                                                          ego_vehicle.state_list_lon[time_step])
        # Cut-in detection and safe distance calculation for same lane
        vehicle_ids_same_lane, vehicle_ids_cutin = self.detect_cutin_same(time_step, vehicle_ids_same,
                                                                          vehicle_ids_adjacent, ego_vehicle)
        # Get vehicle objects from IDs
        vehicles_same_lane = self.get_vehicles_from_ids(vehicle_ids_same_lane)
        vehicle_ids_cutin = self.get_vehicles_from_ids(vehicle_ids_cutin)

        return vehicles_same_lane, vehicle_ids_cutin
