import logging
from collections.abc import Collection
from typing import Iterable, List, Set, Union

import numpy as np
from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.intersection import IntersectionIncomingElement
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from shapely.geometry import LineString, Polygon

from crmonitor.common.road_network import Lane, RoadNetwork
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

_LOGGER = logging.getLogger(__name__)


def distance_to_bounds(
    vehicle: Vehicle, lanelet_ids: Iterable[int], world: World, time_step: int
) -> tuple[np.ndarray, np.ndarray]:
    state = vehicle.get_cr_state(time_step)
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)

    # Map the lanelets to lanes.
    target_lanes = {
        world.road_network.find_lane_by_lanelet(lanelet_id) for lanelet_id in lanelet_ids
    }

    ds_left = []
    ds_right = []
    for lane in target_lanes:
        for point in occ_points:
            d_left = -lane.distance_to_left(*point)
            d_right = lane.distance_to_right(*point)

            ds_left.append(d_left)
            ds_right.append(d_right)

    return np.array(ds_left), np.array(ds_right)


def distance_to_lanes(
    vehicle: Vehicle, lanelet_ids: Collection[int], world: World, time_step: int
) -> float:
    """
    Determine the minimum distance the vehicle would not to move, to either no longer occupy the lanes or to occupy at least one of the lanes.

    :param vehicle: The vehicle for which the distance to the lanes should be determined.
    :param lanelet_ids: Collection of lanelets, to which the distance should be determined.
    :param world: The world which contains vehicle and the lanelets.
    :param time_step: The time step at which the distance of the vehicle should be evaluated.

    :returns:
    """
    if len(lanelet_ids) == 0:
        return np.nan

    # Use the vehicle state to determine which coordinates the vehicle occupies at the time step.
    state = vehicle.get_cr_state(time_step)
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)

    # Map the lanelets to lanes.
    target_lanes = {
        world.road_network.find_lane_by_lanelet(lanelet_id) for lanelet_id in lanelet_ids
    }

    # Determine whether the vehicle is in at least one of the target lanes.
    # This is necessary, because we need to differentiate between the cases where the vehicle
    # is in one of the target lanes and when it is not in any target lane.
    occupied_lanes = vehicle.lanes_at_time_step(time_step)
    occupied_target_lanes = occupied_lanes.intersection(target_lanes)

    if len(occupied_target_lanes) > 0:
        # If the vehicle occupies at least one target lane the result is the distance
        # that is required such that any of its occupancy points is outside all target lanes.
        min_distance = np.inf
        for lane in occupied_target_lanes:
            _, max_dist_left = lane.min_max_distance_to_left(occ_points)
            _, max_dist_right = lane.min_max_distance_to_right(occ_points)
            min_distance = min(max_dist_left, max_dist_right, min_distance)

        return min_distance
    else:
        # If the vehicle occupies no target lane the result is the distance
        # that is required such that any of its occupancy points enters any target lane.
        min_distance = np.inf
        for lane in target_lanes:
            # Determine the minimum distance for each lane individually.
            lane_min_distance = np.inf
            for point in occ_points:
                d_left = lane.distance_to_left(*point)
                d_right = lane.distance_to_right(*point)

                # Use the magnitude of the distance vector to determine the minimum.
                if abs(d_left) <= abs(d_right):
                    point_min_distance = d_left
                else:
                    point_min_distance = d_right

                if abs(point_min_distance) <= abs(lane_min_distance):
                    lane_min_distance = point_min_distance

            min_distance = min(lane_min_distance, min_distance)

        return min_distance


def lanelets_left_of_lanelet(lanelet: Lanelet, lanelet_network: LaneletNetwork) -> Set[Lanelet]:
    """
    Extracts all lanelet IDs left of a given lanelet based on adjacency relations

    :param lanelet: given lanelet
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    left_lanelets = set()
    tmp_lanelet = lanelet
    while tmp_lanelet.adj_left is not None:
        tmp_lanelet = lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_left)
        left_lanelets.add(tmp_lanelet)

    return left_lanelets


def lanelets_right_of_lanelet(lanelet: Lanelet, lanelet_network: LaneletNetwork) -> Set[Lanelet]:
    """
    Extracts all lanelet IDs right of a given lanelet based on adjacency relations

    :param lanelet: given lanelet
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    right_lanelets = set()
    tmp_lanelet = lanelet
    while tmp_lanelet.adj_right is not None:
        tmp_lanelet = lanelet_network.find_lanelet_by_id(tmp_lanelet.adj_right)
        right_lanelets.add(tmp_lanelet)

    return right_lanelets


def lanelets_left_of_vehicle(
    time_step: int, vehicle: Vehicle, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Extracts all lanelets left of a vehicle

    :param vehicle: vehicle of interest
    :param time_step: time step of interest
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    left_lanelets = set()
    occupied_lanelets = vehicle.lanelet_ids_at_time_step(time_step)
    for occ_l in occupied_lanelets:
        new_lanelets = lanelets_left_of_lanelet(
            lanelet_network.find_lanelet_by_id(occ_l), lanelet_network
        )
        for lanelet in new_lanelets:
            left_lanelets.add(lanelet)

    return left_lanelets


def lanelets_right_of_vehicle(
    time_step: int, vehicle: Vehicle, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Extracts all lanelets right of a vehicle

    :param vehicle: vehicle of interest
    :param time_step: time step of interest
    :param lanelet_network: lanelet network
    :returns set of lanelet objects
    """
    right_lanelets = set()
    occupied_lanelets = vehicle.lanelet_ids_at_time_step(time_step)
    for occ_l in occupied_lanelets:
        new_lanelets = lanelets_right_of_lanelet(
            lanelet_network.find_lanelet_by_id(occ_l), lanelet_network
        )
        for lanelet in new_lanelets:
            right_lanelets.add(lanelet)

    return right_lanelets


def vehicles_adjacent(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> List[Vehicle]:
    """
     Searches for vehicles adjacent to a vehicle

    :param vehicle: vehicle object
    :param other_vehicles: other vehicles in scenario
    :param time_step: time step of interest
    :returns list of adjacent vehicles of a vehicle
    """
    vehicles_adj = []
    lane_share = vehicle.lane_at_time_step(time_step)
    for veh in other_vehicles:
        if veh.get_lon_state(time_step, lane_share) is None:
            continue
        if (
            veh.rear_s(time_step, lane_share)
            < vehicle.front_s(time_step, lane_share)
            < veh.front_s(time_step, lane_share)
        ):
            vehicles_adj.append(veh)
            continue
        if (
            veh.rear_s(time_step, lane_share)
            < vehicle.rear_s(time_step, lane_share)
            < veh.front_s(time_step, lane_share)
        ):
            vehicles_adj.append(veh)
            continue
        if vehicle.rear_s(time_step, lane_share) <= veh.rear_s(
            time_step, lane_share
        ) and veh.front_s(time_step, lane_share) <= vehicle.front_s(time_step, lane_share):
            vehicles_adj.append(veh)
            continue
    return vehicles_adj


def vehicles_left(time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]) -> List[Vehicle]:
    """
    Searches for vehicles left of a vehicle

    :param vehicle: vehicle object
    :param other_vehicles: other vehicles in scenario
    :param time_step: time step of interest
    :returns list of vehicles left of a vehicle
    """
    vehicles_adj = vehicles_adjacent(time_step, vehicle, other_vehicles)
    lane_share = vehicle.lane_at_time_step(time_step)
    vehicles_left = [
        veh
        for veh in vehicles_adj
        if veh.right_d(time_step, lane_share) > vehicle.left_d(time_step, lane_share)
    ]
    return vehicles_left


def vehicle_directly_left(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> Union[Vehicle, None]:
    vehicle_left = vehicles_left(time_step, vehicle, other_vehicles)
    if len(vehicle_left) == 0:
        return None
    elif len(vehicle_left) == 1:
        return vehicle_left[0]
    else:
        vehicle_directly_left = vehicle_left[0]
        for veh in vehicle_left:
            lane_share = veh.lane_at_time_step(time_step)
            if (
                veh.get_lat_state(time_step, lane_share).d
                < vehicle_directly_left.get_lat_state(time_step, lane_share).d
            ):
                vehicle_directly_left = veh
        return vehicle_directly_left


def vehicles_right(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> List[Vehicle]:
    """
    Searches for vehicles right of a vehicle

    :param vehicle: vehicle object
    :param other_vehicles: other vehicles in scenario
    :param time_step: time step of interest
    :returns list of vehicles left of a vehicle
    """
    vehicles_adj = vehicles_adjacent(time_step, vehicle, other_vehicles)
    lane_share = vehicle.lane_at_time_step(time_step)
    vehicles_right = [
        veh
        for veh in vehicles_adj
        if veh.left_d(time_step, lane_share) < vehicle.right_d(time_step, lane_share)
    ]
    return vehicles_right


def vehicle_directly_right(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> Union[Vehicle, None]:
    vehicle_right = vehicles_right(time_step, vehicle, other_vehicles)
    if len(vehicle_right) == 0:
        return None
    elif len(vehicle_right) == 1:
        return vehicle_right[0]
    else:
        vehicle_directly_right = vehicle_right[0]
        for veh in vehicle_right:
            lane_share = veh.lane_at_time_step(time_step)
            if (
                veh.get_lat_state(time_step, lane_share).d
                > vehicle_directly_right.get_lat_state(time_step, lane_share).d
            ):
                vehicle_directly_right = veh
        return vehicle_directly_right


def _adjacent_lanelets(lanelet: Lanelet, lanelet_network: LaneletNetwork) -> Set[Lanelet]:
    """
    Returns all lanelet which are adjacent to a lanelet and the lanelet itself

    :param lanelet: CommonRoad lanelet
    :returns set of adjacent lanelets
    """
    lanelets = {lanelet}
    la = lanelet
    while la is not None and la.adj_left is not None:
        la = lanelet_network.find_lanelet_by_id(la.adj_left)
        if la is not None:
            lanelets.add(la)
    la = lanelet
    while la is not None and la.adj_right is not None:
        la = lanelet_network.find_lanelet_by_id(la.adj_right)
        if la is not None:
            lanelets.add(la)
    return lanelets


def cal_road_width(lanelet: Lanelet, road_network: RoadNetwork, position: float) -> float:
    """
    Calculates width of road given a lanelet and a longitudinal position
    """
    adj_lanelets = _adjacent_lanelets(lanelet, road_network.lanelet_network)
    road_width = 0.0
    for lanelet in list(adj_lanelets):
        road_width += road_network.find_lane_by_lanelet(lanelet.lanelet_id).width(position)
    return road_width


def traffic_sign_type(lanelet_ids, road_network: RoadNetwork):
    """

    :param lanelet_id:
    :param road_network:
    :return: the set of traffic sign types assigned to a lanelet
    """
    traffic_sign_list = list()
    for lanelet_id in lanelet_ids:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        ts_element_ids = lanelet.traffic_signs
        for ts_element_id in ts_element_ids:
            traffic_sign_object = road_network.lanelet_network.find_traffic_sign_by_id(
                ts_element_id
            )
            for ts_element in traffic_sign_object.traffic_sign_elements:
                if ts_element.traffic_sign_element_id not in traffic_sign_list:
                    traffic_sign_list.append(ts_element.traffic_sign_element_id)
    return traffic_sign_list


def traffic_sign(lanelet_id: int, given_traffic_sign_id, road_network: RoadNetwork):
    """
    :param lanelet_id:
    :param given_traffic_sign_id:
    :param road_network:
    :return: the traffic sign element of a given type assigned to a lanelet
    """
    traffic_sign_elements = list()
    lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
    ts_element_ids = lanelet.traffic_signs
    for ts_element_id in ts_element_ids:
        traffic_sign_object = road_network.lanelet_network.find_traffic_sign_by_id(ts_element_id)
        for ts_element in traffic_sign_object.traffic_sign_elements:
            if ts_element.traffic_sign_element_id == given_traffic_sign_id:
                traffic_sign_elements.append(traffic_sign_object)
    if len(traffic_sign_elements) == 0:
        return None
    assert len(traffic_sign_elements) == 1, (
        "TODO: Only works for one traffic sign type per lanelet!"
    )
    return traffic_sign_elements[0]


def get_lanelet_start_line(lanelet: Lanelet):
    right_start_vertice = lanelet.right_vertices[0, :]
    left_start_vertice = lanelet.left_vertices[0, :]
    return np.array([left_start_vertice, right_start_vertice])


def get_lanelet_end_line(lanelet: Lanelet):
    right_start_vertice = lanelet.right_vertices[-1, :]
    left_start_vertice = lanelet.left_vertices[-1, :]
    return np.array([left_start_vertice, right_start_vertice])


def get_incoming_multi_intersections(vehicle: Vehicle, time_step, road_network: RoadNetwork):
    """
    get all incoming elements and distance to these incoming elements in different intersections
    by given a vehicle and current time step
    """
    # get all lanelets which can be occupied by current vehicle
    lanelets_dir_vehicle = np.array(vehicle.lanelets_dir)
    lanelets_dir_pre = road_network.lanelet_reach_pre(lanelets_dir_vehicle[0])
    lanelets_dir_suc = road_network.lanelet_reach_suc(lanelets_dir_vehicle[-1])
    lanelets_dir_vehicle = np.append(lanelets_dir_vehicle, lanelets_dir_pre)
    lanelets_dir_vehicle = np.append(lanelets_dir_vehicle, lanelets_dir_suc)
    occupied_lanelets_possible = np.unique(lanelets_dir_vehicle)
    # get front- and rear-most point of vehicle along reference lane
    front_s = vehicle.front_s(time_step, vehicle.ref_path_lane)
    rear_s = vehicle.rear_s(time_step, vehicle.ref_path_lane)
    incoming_elements = list()
    distance_to_incomings = list()
    for intersection in road_network.lanelet_network.intersections:
        incoming_intersection = list()
        start_incoming_s = list()
        for incoming in intersection.incomings:
            if len(incoming.incoming_lanelets.intersection(occupied_lanelets_possible)) != 0:
                incoming_intersection.append(incoming)
        if len(incoming_intersection) > 1:
            for incoming in incoming_intersection:
                if (
                    len(
                        incoming.incoming_lanelets.intersection(
                            vehicle.ref_path_lane.contained_lanelets
                        )
                    )
                    != 0
                ):
                    incoming_intersection = incoming
                    incoming_ids = list(
                        incoming.incoming_lanelets.intersection(occupied_lanelets_possible)
                    )
                    start_incoming_s = get_lanelets_start_s(
                        vehicle.ref_path_lane, incoming_ids, road_network
                    )
        else:
            incoming_intersection = incoming_intersection[0]
            incoming_ids = list(
                incoming_intersection.incoming_lanelets.intersection(occupied_lanelets_possible)
            )
            start_incoming_s = get_lanelets_start_s(
                vehicle.ref_path_lane, incoming_ids, road_network
            )
        incoming_elements.append(incoming_intersection)
        incoming_successor = set.union(
            incoming_intersection.successors_right,
            incoming_intersection.successors_straight,
            incoming_intersection.successors_left,
        )
        successor_possible = incoming_successor.intersection(set(occupied_lanelets_possible))
        end_intersection_s = get_lanelets_end_s(
            vehicle.ref_path_lane, successor_possible, road_network
        )
        # lanelets in front of vehicle
        if (front_s - start_incoming_s) < 0 < (end_intersection_s - rear_s):
            distance_to_incomings.append(front_s - start_incoming_s)
        # vehicle in front of lanelets
        elif (end_intersection_s - rear_s) <= 0 <= (front_s - start_incoming_s):
            distance_to_incomings.append(end_intersection_s - rear_s)
        # vehicle inside lanelets
        else:
            distance_to_incomings.append(
                min(front_s - start_incoming_s, end_intersection_s - rear_s)
            )
    return incoming_elements, distance_to_incomings


def get_right_turning_lane_by_lanelets(
    lanelets_id, road_network: RoadNetwork
) -> (IntersectionIncomingElement, Lane):
    """
    find the incoming according to current lanelet assignments which includes the right turning lanelet of the
    searched incoming
    """
    for incoming_id, lanes_incoming in road_network.lanes_incoming.items():
        if lanes_incoming[0].contained_lanelets.intersection(lanelets_id):
            return road_network.incoming[incoming_id], lanes_incoming[0]
    return None, None


def get_left_turning_lane_by_lanelets(
    lanelets_id, road_network: RoadNetwork
) -> (IntersectionIncomingElement, Lane):
    """
    find the incoming according to current lanelet assignments which includes the left turning lanelet of the
    searched incoming
    """
    for incoming_id, lanes_incoming in road_network.lanes_incoming.items():
        if lanes_incoming[2].contained_lanelets.intersection(lanelets_id):
            return road_network.incoming[incoming_id], lanes_incoming[2]
    return None, None


def get_straight_going_lane_by_lanelets(
    lanelets_id, road_network: RoadNetwork
) -> (IntersectionIncomingElement, Lane):
    """
    find the incoming according to current lanelet assignments which includes the straight going lanelet of the
    searched incoming
    """
    for incoming_id, lanes_incoming in road_network.lanes_incoming.items():
        if lanes_incoming[1].contained_lanelets.intersection(lanelets_id):
            return road_network.incoming[incoming_id], lanes_incoming[1]
    return None, None


def get_lanelets_start_s(
    reference_lane: Lane, lanelets_ids, road_network: RoadNetwork
) -> (float, float):
    """
    get start point of a set of lanelets along a reference lane
    """
    lanelets_start_s = np.inf
    for lanelet_id in lanelets_ids:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        start_s = reference_lane.clcs.convert_to_curvilinear_coords(
            *get_lanelet_start_line(lanelet)[0]
        )[0]
        lanelets_start_s = min(lanelets_start_s, start_s)
    return lanelets_start_s


def get_lanelets_end_s(
    reference_lane: Lane, lanelets_ids, road_network: RoadNetwork
) -> (float, float):
    """
    get end point of a set of lanelets along a reference lane
    """
    lanelets_end_s = -np.inf
    for lanelet_id in lanelets_ids:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        end_s = reference_lane.clcs.convert_to_curvilinear_coords(
            *get_lanelet_end_line(lanelet)[0]
        )[0]
        lanelets_end_s = max(lanelets_end_s, end_s)
    return lanelets_end_s


def distance_to_left_bounds_clcs(vehicle: Vehicle, lane: Lane, time_step):
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)
    distance = list()
    for point in occ_points:
        d_left = lane.distance_to_left(*point)
        distance.append(d_left)
    return distance


def distance_to_right_bounds_clcs(vehicle: Vehicle, lane: Lane, time_step):
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)
    distance = list()
    for point in occ_points:
        d_right = lane.distance_to_right(*point)
        distance.append(d_right)
    return distance


def get_priority(lanelet_ids, road_network, direction, traffic_sign_priority):
    """
    Determine the priority of a vehicle at an intersection based on traffic signs.

    Parameters:
    - lanelet_ids: List of IDs of the lanelets where the vehicle is located.
    - road_network: The road network data structure.
    - direction: The direction the vehicle intends to go ('right', 'straight', or 'left').
    - traffic_sign_priority: A mapping from traffic sign types to priority configurations.

    Returns:
    - The priority value for the given direction based on traffic signs.

    Raises:
    - ValueError: If the provided direction is invalid.
    """
    # Get the types of traffic signs associated with the given lanelets
    ts_types = traffic_sign_type(lanelet_ids, road_network)

    # Filter traffic signs to those that affect priority
    applicable_ts_types = [
        traffic_sign_priority[ts] for ts in ts_types if ts in traffic_sign_priority
    ]

    # If no relevant traffic signs are found, default to 'right before left' rule
    if not applicable_ts_types:
        applicable_ts_types = [
            traffic_sign_priority[TrafficSignIDGermany.WARNING_RIGHT_BEFORE_LEFT]
        ]

    # Select the traffic sign with the highest priority (lowest evaluation index)
    selected_ts_type = min(applicable_ts_types, key=lambda ts: ts.evaluation_idx)

    # Return the priority value based on the direction
    if direction in ["right", "straight", "left"]:
        return getattr(selected_ts_type, direction)
    else:
        raise ValueError(f"Invalid direction: {direction}")


def find_longest_lane_by_intersection_lanelet(lanelet_id: int, road_network: RoadNetwork) -> Lane:
    longest_lane = None
    num_contained_lanelets = 0
    for lane in road_network.lanes:
        if lanelet_id in lane.contained_lanelets:
            if len(lane.contained_lanelets) > num_contained_lanelets:
                longest_lane = lane
                num_contained_lanelets = len(lane.contained_lanelets)
    return longest_lane


def find_conflict_points(line, conflict_polygon: Polygon):
    """
    find intersection points between a line and polygon
    """
    conflict_line_points = list()
    # Create curved line
    curved_line = LineString(line)
    # Get intersection of line and polygon
    intersection = curved_line.intersection(conflict_polygon)
    if intersection.geom_type == "Point":
        conflict_line_points.append(intersection)
    elif intersection.geom_type == "LineString" or intersection.geom_type == "LinearRing":
        for point in intersection.coords:
            conflict_line_points.append(np.array(point))
    elif intersection.geom_type == "MultiPoint" or intersection.geom_type == "MultiLineString":
        for geom in intersection.geoms:
            for point in geom.coords:
                conflict_line_points.append(point)
    if len(conflict_line_points) == 0:
        conflict_points = None
    else:
        conflict_points = [conflict_line_points[0], conflict_line_points[-1]]
    return conflict_points


def get_long_distance_stop_lines_from_lane(
    road_network: "RoadNetwork", lane: "Lane"
) -> "List[float]":
    s_stop_line_list = list()
    for lanelet_id in lane.contained_lanelets:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        if lanelet.stop_line is not None:
            s_stop_line = min(
                lane.clcs.convert_to_curvilinear_coords(*lanelet.stop_line.start)[0],
                lane.clcs.convert_to_curvilinear_coords(*lanelet.stop_line.end)[0],
            )
            s_stop_line_list.append(s_stop_line)
        else:
            continue
    return s_stop_line_list


def check_in_intersection(road_network: "RoadNetwork", lanelets_id):
    for lanelet_id in lanelets_id:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        if LaneletType.INTERSECTION in lanelet.lanelet_type:
            return True
    return False
