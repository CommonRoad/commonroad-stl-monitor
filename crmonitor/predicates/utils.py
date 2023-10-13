import enum
import logging
import math
from typing import Iterable, List, Optional, Set, Tuple, Union

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.planning.goal import GoalRegion
from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import CustomState

import numpy as np

from shapely.geometry import Polygon, LineString
from commonroad.common.util import subtract_orientations
from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.intersection import IntersectionIncomingElement
from commonroad.scenario.lanelet import (
    Intersection,
    Lanelet,
    LaneletNetwork,
    LaneletType,
    StopLine,
)
from commonroad.scenario.traffic_sign import TrafficSignIDGermany

from crmonitor.common.helper import cartesian_to_curvilinear
from crmonitor.common.road_network import Lane, RoadNetwork
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

from commonroad_dc.geometry.util import (
    chaikins_corner_cutting,
    compute_curvature_from_polyline,
    resample_polyline,
    compute_pathlength_from_polyline,
    compute_orientation_from_polyline,
)

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def distance_to_left_bounds(
    vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step
):
    state = vehicle_i.states_cr[time_step]
    occ_points = rotate_translate(
        vehicle_i.shape.vertices[:-1], state.position, state.orientation
    )
    lanelets = [
        world.road_network.lanelet_network.find_lanelet_by_id(i) for i in lanelet_ids
    ]
    left_bounds = tuple(
        [
            l.left_vertices
            for l in lanelets
            if l.adj_left is None
            or l.adj_left not in lanelet_ids
            and not l.adj_left_same_direction
        ]
    )
    if len(left_bounds) > 0:
        d_left = np.array(cartesian_to_curvilinear(left_bounds, occ_points))[
            ..., 1
        ].ravel()
        d_left = d_left[~np.isnan(d_left)]
    else:
        d_left = np.array([])
    return d_left


def distance_to_right_bounds(
    vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step
):
    state = vehicle_i.states_cr[time_step]
    occ_points = rotate_translate(
        vehicle_i.shape.vertices[:-1], state.position, state.orientation
    )
    lanelets = [
        world.road_network.lanelet_network.find_lanelet_by_id(i) for i in lanelet_ids
    ]
    right_bounds = tuple(
        [
            l.right_vertices
            for l in lanelets
            if l.adj_right is None
            or l.adj_right not in lanelet_ids
            and not l.adj_left_same_direction
        ]
    )
    if len(right_bounds) > 0:
        d_right = np.array(cartesian_to_curvilinear(right_bounds, occ_points))[
            ..., 1
        ].ravel()
        d_right = d_right[~np.isnan(d_right)]
    else:
        d_right = np.array([])
    return d_right


def distance_to_bounds(
    vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step
):
    state = vehicle_i.states_cr[time_step]
    occ_points = rotate_translate(
        vehicle_i.shape.vertices[:-1], state.position, state.orientation
    )
    lanelets = [
        world.road_network.lanelet_network.find_lanelet_by_id(i) for i in lanelet_ids
    ]
    left_bounds = tuple(
        [
            l.left_vertices
            for l in lanelets
            if l.adj_left is not None and l.adj_left not in lanelet_ids
        ]
    )
    right_bounds = tuple(
        [
            l.right_vertices
            for l in lanelets
            if l.adj_right is not None and l.adj_right not in lanelet_ids
        ]
    )
    if len(left_bounds) > 0:
        d_left = np.array(cartesian_to_curvilinear(left_bounds, occ_points))[
            ..., 1
        ].ravel()
        d_left = d_left[~np.isnan(d_left)]
    else:
        d_left = np.array([])
    if len(right_bounds) > 0:
        d_right = np.array(cartesian_to_curvilinear(right_bounds, occ_points))[
            ..., 1
        ].ravel()
        d_right = d_right[~np.isnan(d_right)]
    else:
        d_right = np.array([])

    return d_left, d_right


def distance_to_lanes(vehicle_i: Vehicle, lanelet_ids: Iterable[int], world, time_step):
    d_left, d_right = distance_to_bounds(vehicle_i, lanelet_ids, world, time_step)
    d_left = -np.min(d_left) if d_left.size > 0 else np.inf
    d_right = np.max(d_right) if d_right.size > 0 else np.inf
    return np.fmin(d_left, d_right)


def distance_veh_center_to_lane_boundaries(
    vehicle_i: Vehicle, lane: Lane, time_step: int
):
    """
    Distance of the vehicle center to the boundaries of the lane
    """
    veh_position = vehicle_i.states_cr[time_step].position
    try:
        dis_to_left = -lane.clcs_left.convert_to_curvilinear_coords(
            veh_position[0], veh_position[1]
        )[1]
    except:
        dis_to_left = -lane.clcs_left_large_step.convert_to_curvilinear_coords(
            veh_position[0], veh_position[1]
        )[1]
    try:
        dis_to_right = lane.clcs_right.convert_to_curvilinear_coords(
            veh_position[0], veh_position[1]
        )[1]
    except:
        dis_to_right = lane.clcs_right_large_step.convert_to_curvilinear_coords(
            veh_position[0], veh_position[1]
        )[1]
    return dis_to_left, dis_to_right


def lanelets_left_of_lanelet(
    lanelet: Lanelet, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
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


def lanelets_right_of_lanelet(
    lanelet: Lanelet, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
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
    occupied_lanelets = vehicle.lanelet_assignment[time_step]
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
    occupied_lanelets = vehicle.lanelet_assignment[time_step]
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
    lane_share = vehicle.get_lane(time_step)
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
        ) and veh.front_s(time_step, lane_share) <= vehicle.front_s(
            time_step, lane_share
        ):
            vehicles_adj.append(veh)
            continue
    return vehicles_adj


def vehicles_left(
    time_step: int, vehicle: Vehicle, other_vehicles: List[Vehicle]
) -> List[Vehicle]:
    """
    Searches for vehicles left of a vehicle

    :param vehicle: vehicle object
    :param other_vehicles: other vehicles in scenario
    :param time_step: time step of interest
    :returns list of vehicles left of a vehicle
    """
    vehicles_adj = vehicles_adjacent(time_step, vehicle, other_vehicles)
    lane_share = vehicle.get_lane(time_step)
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
            lane_share = veh.get_lane(time_step)
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
    lane_share = vehicle.get_lane(time_step)
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
            lane_share = veh.get_lane(time_step)
            if (
                veh.get_lat_state(time_step, lane_share).d
                > vehicle_directly_right.get_lat_state(time_step, lane_share).d
            ):
                vehicle_directly_right = veh
        return vehicle_directly_right


def _adjacent_lanelets(
    lanelet: Lanelet, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
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


def cal_road_width(
    lanelet: Lanelet, road_network: RoadNetwork, position: float
) -> float:
    """
    Calculates width of road given a lanelet and a longitudinal position
    """
    adj_lanelets = _adjacent_lanelets(lanelet, road_network.lanelet_network)
    road_width = 0.0
    for lanelet in list(adj_lanelets):
        road_width += road_network.find_lane_by_lanelet(lanelet.lanelet_id).width(
            position
        )
    return road_width


def bool_to_num(bool_value):
    return 1 if bool_value else -1


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
        traffic_sign_object = road_network.lanelet_network.find_traffic_sign_by_id(
            ts_element_id
        )
        for ts_element in traffic_sign_object.traffic_sign_elements:
            if ts_element.traffic_sign_element_id == given_traffic_sign_id:
                traffic_sign_elements.append(traffic_sign_object)
    if len(traffic_sign_elements) == 0:
        return None
    assert len(traffic_sign_elements) == 1, (
        "TODO: Only works for one " "traffic sign type per lanelet!"
    )
    return traffic_sign_elements[0]


def distance_start_lanelet(
    vehicle: Vehicle, lanelet_id: int, road_network: RoadNetwork, time_step
):
    """

    :param vehicle:
    :param lanelet_id:
    :param road_network:
    :param time_step:
    :return: the distance between start of lanelet and the front of vehicle
    """
    lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
    lanlet_start_line = get_lanelet_start_line(lanelet)
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(
        vehicle.shape.vertices[:-1], state.position, state.orientation
    )
    d_start_lanelet = np.array(
        cartesian_to_curvilinear(tuple([lanlet_start_line]), occ_points)
    )[..., 1].ravel()
    d_start_lanelet = d_start_lanelet[~np.isnan(d_start_lanelet)]
    return np.max(d_start_lanelet)


def get_lanelet_start_line(lanelet: Lanelet):
    right_start_vertice = lanelet.right_vertices[0, :]
    left_start_vertice = lanelet.left_vertices[0, :]
    return np.array([left_start_vertice, right_start_vertice])


def get_lanelet_end_line(lanelet: Lanelet):
    right_start_vertice = lanelet.right_vertices[-1, :]
    left_start_vertice = lanelet.left_vertices[-1, :]
    return np.array([left_start_vertice, right_start_vertice])


def active_tls_by_lanelet(lanelet: Lanelet, road_network: RoadNetwork):
    assert len(lanelet.traffic_lights) == 1, (
        "TODO: Only works for one " "traffic light per lanelet!"
    )
    tl = road_network.lanelet_network.find_traffic_light_by_id(
        list(lanelet.traffic_lights)[0]
    )
    if tl.active:
        return True
    return False


def get_incoming(lanelets_id, road_network: RoadNetwork) -> IntersectionIncomingElement:
    incoming = None
    for lanelet_id in lanelets_id:
        lanelet_pre = road_network.reach_pre(lanelet_id)
        for incoming_element in road_network.lanelet_network.intersections[0].incomings:
            if (
                len(incoming_element.incoming_lanelets.intersection(set(lanelet_pre)))
                > 0
            ):
                incoming = incoming_element
                break
        if incoming is not None:
            break
    return incoming


def get_incoming_multi_intersections(
    vehicle: Vehicle, time_step, road_network: RoadNetwork
):
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
            if (
                len(incoming.incoming_lanelets.intersection(occupied_lanelets_possible))
                != 0
            ):
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
                        incoming.incoming_lanelets.intersection(
                            occupied_lanelets_possible
                        )
                    )
                    start_incoming_s = get_lanelets_start_s(
                        vehicle.ref_path_lane, incoming_ids, road_network
                    )
        else:
            incoming_intersection = incoming_intersection[0]
            incoming_ids = list(
                incoming_intersection.incoming_lanelets.intersection(
                    occupied_lanelets_possible
                )
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
        successor_possible = incoming_successor.intersection(
            set(occupied_lanelets_possible)
        )
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


def get_right_turn_lane(
    road_network: RoadNetwork, incoming: IntersectionIncomingElement
) -> Lane:
    """
    get the right-turning lane by given an incoming element
    """
    incoming_lanelets_ids = incoming.incoming_lanelets
    right_turn_lanelets_ids = incoming.successors_right
    right_turn_lane = list()
    for lane in road_network.lanes:
        # choose the lane which contains both incoming and right-turning successors
        if (
            len(lane.contained_lanelets.intersection(right_turn_lanelets_ids)) > 0
            and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0
        ):
            right_turn_lane.append(lane)
    assert (
        len(right_turn_lane) == 1
    ), "Something not correct, OR there are more than one lanelets before intersection."
    return right_turn_lane[0]


def get_left_turn_lane(
    road_network: RoadNetwork, incoming: IntersectionIncomingElement
) -> Lane:
    """
    get the left-turning lane by given an incoming element
    """
    incoming_lanelets_ids = incoming.incoming_lanelets
    left_turn_lanelets_ids = incoming.successors_left
    left_turn_lane = list()
    for lane in road_network.lanes:
        # choose the lane which contains both incoming and left-turning successors
        if (
            len(lane.contained_lanelets.intersection(left_turn_lanelets_ids)) > 0
            and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0
        ):
            left_turn_lane.append(lane)
    assert (
        len(left_turn_lane) == 1
    ), "Something not correct, OR there are more than one lanelets before intersection."
    return left_turn_lane[0]


def get_straight_going_lane(
    road_network: RoadNetwork, incoming: IntersectionIncomingElement
) -> Lane:
    """
    get the straight going lane by given an incoming element
    """
    incoming_lanelets_ids = incoming.incoming_lanelets
    straight_going_lanelets_ids = incoming.successors_straight
    straight_going_lane = list()
    for lane in road_network.lanes:
        if (
            len(lane.contained_lanelets.intersection(straight_going_lanelets_ids)) > 0
            and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0
        ):
            straight_going_lane.append(lane)
    assert (
        len(straight_going_lane) == 1
    ), "Something not correct, OR there are more than one lanelets before intersection."
    return straight_going_lane[0]


def get_lanelets_start_end_s(
    reference_lane: Lane, lanelets_ids, road_network: RoadNetwork
) -> (float, float):
    """
    get start and end point of a set of lanelets along a reference lane
    """
    lanelets_start_s = np.inf
    lanelets_end_s = -np.inf
    for lanelet_id in lanelets_ids:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        start_s = reference_lane.clcs.convert_to_curvilinear_coords(
            *get_lanelet_start_line(lanelet)[0]
        )[0]
        end_s = reference_lane.clcs.convert_to_curvilinear_coords(
            *get_lanelet_end_line(lanelet)[0]
        )[0]
        lanelets_start_s = min(lanelets_start_s, start_s)
        lanelets_end_s = max(lanelets_end_s, end_s)
    return lanelets_start_s, lanelets_end_s


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
    occ_points = rotate_translate(
        vehicle.shape.vertices[:-1], state.position, state.orientation
    )
    distance = list()
    for point in occ_points:
        try:
            d_left = lane.clcs_left.convert_to_curvilinear_coords(*point)[1]
        except:
            d_left = lane.clcs_left_large_step.convert_to_curvilinear_coords(*point)[1]
        distance.append(d_left)
    return distance


def distance_to_right_bounds_clcs(vehicle: Vehicle, lane: Lane, time_step):
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(
        vehicle.shape.vertices[:-1], state.position, state.orientation
    )
    distance = list()
    for point in occ_points:
        try:
            d_right = lane.clcs_right.convert_to_curvilinear_coords(*point)[1]
        except:
            d_right = lane.clcs_right_large_step.convert_to_curvilinear_coords(*point)[
                1
            ]
        distance.append(d_right)
    return distance


def longitudinal_distance_to_lane(vehicle: Vehicle, lane: Lane, time_step):
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(
        vehicle.shape.vertices[:-1], state.position, state.orientation
    )
    distance = list()
    for point in occ_points:
        d_right = lane.clcs.convert_to_curvilinear_coords(*point)[0]
        distance.append(d_right)
    return distance


def get_priority(
    lanelet_ids, road_network: RoadNetwork, direction, traffic_sign_priority
):
    ts_types = traffic_sign_type(lanelet_ids, road_network)
    ts_types_intersection = [traffic_sign_priority[ts] for ts in ts_types]
    if len(ts_types_intersection) == 0:
        ts_types_intersection = [
            traffic_sign_priority[TrafficSignIDGermany.WARNING_RIGHT_BEFORE_LEFT]
        ]
    eval_idx_list = list()
    for ts_type in ts_types_intersection:
        eval_idx_list.append(ts_type.evaluation_idx)
    argmin_s = np.argmin(eval_idx_list)
    if direction == "right":
        return ts_types_intersection[argmin_s].right
    elif direction == "straight":
        return ts_types_intersection[argmin_s].straight
    elif direction == "left":
        return ts_types_intersection[argmin_s].left


def inc_la_left_of(
    incoming: IntersectionIncomingElement, road_network: RoadNetwork
) -> IntersectionIncomingElement:
    left_of_incoming = incoming.left_of
    for incoming_element in road_network.lanelet_network.intersections[0].incomings:
        if incoming_element.incoming_id == left_of_incoming:
            return incoming_element


def find_longest_lane_by_intersection_lanelet(
    lanelet_id: int, road_network: RoadNetwork
) -> Lane:
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
    elif (
        intersection.geom_type == "LineString" or intersection.geom_type == "LinearRing"
    ):
        for point in intersection.coords:
            conflict_line_points.append(np.array(point))
    elif (
        intersection.geom_type == "MultiPoint"
        or intersection.geom_type == "MultiLineString"
    ):
        for geom in intersection.geoms:
            for point in geom.coords:
                conflict_line_points.append(point)
    if len(conflict_line_points) == 0:
        conflict_points = None
    else:
        conflict_points = [conflict_line_points[0], conflict_line_points[-1]]
    return conflict_points


def get_vehicle_front_points(vehicle: Vehicle, time_step):
    center_position = vehicle.states_cr[time_step].position
    orientation = vehicle.states_cr[time_step].orientation
    l = vehicle.shape.length
    w = vehicle.shape.width
    front_right = center_position + np.array(
        [
            l / 2 * np.cos(orientation) + w / 2 * np.sin(orientation),
            l / 2 * np.sin(orientation) - w / 2 * np.cos(orientation),
        ]
    )
    front_left = center_position + np.array(
        [
            l / 2 * np.cos(orientation) - w / 2 * np.sin(orientation),
            l / 2 * np.sin(orientation) + w / 2 * np.cos(orientation),
        ]
    )
    return np.array([front_right, front_left])


def get_vehicle_rear_points(vehicle: Vehicle, time_step):
    center_position = vehicle.states_cr[time_step].position
    orientation = vehicle.states_cr[time_step].orientation
    l = vehicle.shape.length
    w = vehicle.shape.width
    rear_right = center_position + np.array(
        [
            -l / 2 * np.cos(orientation) + w / 2 * np.sin(orientation),
            -l / 2 * np.sin(orientation) - w / 2 * np.cos(orientation),
        ]
    )
    rear_left = center_position + np.array(
        [
            -l / 2 * np.cos(orientation) - w / 2 * np.sin(orientation),
            -l / 2 * np.sin(orientation) + w / 2 * np.cos(orientation),
        ]
    )
    return np.array([rear_right, rear_left])


def points_distance_to_right_bound(points, lane: Lane):
    distance = list()
    for point in points:
        d_right = lane.clcs_right.convert_to_curvilinear_coords(*point)[1]
        distance.append(d_right)
    return np.max(distance)


def points_distance_to_left_bound(points, lane: Lane):
    distance = list()
    for point in points:
        d_left = lane.clcs_left.convert_to_curvilinear_coords(*point)[1]
        distance.append(d_left)
    return np.min(distance)


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
