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
from commonroad.scenario.lanelet import (Intersection,
                                         Lanelet,
                                         LaneletNetwork,
                                         LaneletType,
                                         StopLine, )
from commonroad.scenario.traffic_sign import TrafficSignIDGermany
from crmonitor.common.road_network import Lane

from crmonitor.common.helper import cartesian_to_curvilinear
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

from commonroad_dc.geometry.util import (chaikins_corner_cutting, compute_curvature_from_polyline, resample_polyline,
                                         compute_pathlength_from_polyline, compute_orientation_from_polyline)

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


######################################################
### Our work starts here.
######################################################

# -------------------------------------------------------------------------------------------------------------------- #
# new from Mahdi Bayouli
def adjacent_lanelets_same_direction(
        lanelet: Lanelet, lanelet_network: LaneletNetwork
) -> Set[Lanelet]:
    """
    Returns all lanelet which are adjacent to a lanelet and the lanelet itself

    :param lanelet: CommonRoad lanelet
    :returns set of adjacent lanelets
    """
    lanelets = {lanelet}
    la = lanelet
    left_opp = None

    while la is not None and la.adj_left is not None:
        if la.adj_left_same_direction:
            la = lanelet_network.find_lanelet_by_id(la.adj_left)
            if la is not None:
                lanelets.add(la)
        else:
            left_opp = lanelet_network.find_lanelet_by_id(la.adj_left)
            if left_opp is not None:
                lanelets.add(left_opp)
            break

    while la is not None and la.adj_right is not None and la.adj_right_same_direction:
        la = lanelet_network.find_lanelet_by_id(la.adj_right)
        if la is not None:
            lanelets.add(la)

    while (
            left_opp is not None
            and left_opp.adj_right is not None
            and left_opp.adj_right_same_direction
    ):
        left_opp = lanelet_network.find_lanelet_by_id(left_opp.adj_right)
        if left_opp is not None:
            lanelets.add(left_opp)

    return lanelets


# def get_priority(
#     lanelets_dir_ids: List[int], road_network: RoadNetwork, direction: str
# ):
#
#     # limitations:
#     # -> "306 indicates priority until signs 205, 206, or 307" is not covered
#     # -> only 5 german traffic signs are covered (present in the paper)
#     # future work:
#     # -> 306 indicates priority until signs 205, 206, or 307.
#     # -> cover the rest of the german traffic signs
#
#     sign_id_priority = {
#         # sign_id :[prio_left, prio_straight, prio_right, evaluation_index]
#         "306": [4, 5, 4, 11],  # TrafficSignIDGermany.PRIORITY
#         "301": [4, 5, 4, 12],  # TrafficSignIDGermany.RIGHT_OF_WAY
#         "205": [2, 2, 2, 13],  # TrafficSignIDGermany.YIELD
#         "206": [1, 1, 1, 14],  # TrafficSignIDGermany.STOP
#         "102": [3, 3, 3, 15],  # TrafficSignIDGermany.WARNING_RIGHT_BEFORE_LEFT
#     }
#
#     direction_index_dic = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}
#     direction_index = direction_index_dic[direction.upper()]
#
#     for l_id in lanelets_dir_ids:
#         lanelet = road_network.lanelet_network.find_lanelet_by_id(l_id)
#         traffic_sign_ids = lanelet.traffic_signs
#         # traffic_sign_object = road_network.lanelet_network.find_traffic_sign_by_id(traffic_sign_id)
#
#         traffic_ids = list()
#         for ts_id in traffic_sign_ids:
#             traffic_sign_object = road_network.lanelet_network.find_traffic_sign_by_id(
#                 ts_id
#             )
#             traffic_sign_elements = traffic_sign_object.traffic_sign_elements
#             for ts_element in traffic_sign_elements:
#                 ts_element_id = ts_element.traffic_sign_element_id
#                 traffic_ids.append(ts_element_id.value)
#
#         if len(traffic_ids) == 0:
#             traffic_ids.append("102")
#
#         min_priority = 3  # 3 by default is the priority for '102'
#         min_evaluation_index = 15
#
#         for id in traffic_ids:
#             if sign_id_priority[id][3] < min_evaluation_index:
#                 min_evaluation_index = sign_id_priority[id][3]
#                 min_priority = sign_id_priority[id][direction_index]
#
#         return min_priority


def get_robustness_inside_lanelet(
        vehicle: Vehicle, time_step, lanelet: Lanelet, only_successors: False
):
    # limitation: euclidean distance from car state position.
    # not a good indicator of how well the car is inside the lanelet
    # a good indicator would be the lateral and longitudinal distance to the vehicle shape boundaries
    endpoint = get_lanelet_center_endpoint(lanelet)
    startpoint = get_lanelet_center_startpoint(lanelet)
    vehicle_state_position = vehicle.state_list_cr[time_step].position
    d_end = distance_between_two_points(vehicle_state_position, endpoint)
    s_start = distance_between_two_points(vehicle_state_position, startpoint)
    return np.minimum(d_end, s_start)


def get_robustness_wrt_lanelet_type(
        world,
        time_step,
        vehicle_ids: List[int],
        lanelet_type,
        inside_intersection: bool,
        only_successors: bool = False,
) -> float:
    """
    get robustness based on desired lanelet type.
    :param: inside intersection: is intersection type addtionally required for the predicate ?
    """
    # abs: true if a lanelet of lanelet_type is found in the successors
    # this guarantees the correctness of the robustness of relevant_traffic_light

    abs = False
    on_type = False
    rnet = world.road_network
    vehicle_k = world.vehicle_by_id(vehicle_ids[0])
    lanelet_of_type = None

    lanelets_dir_k = lanelets_dir(vehicle_k, time_step, rnet)

    for l_id in lanelets_dir_k:
        lanelet = world.road_network.lanelet_network.find_lanelet_by_id(l_id)
        # lanelet must be of type lanelet_type and also in intersection
        if is_lanelet_of_type(lanelet, lanelet_type, world.road_network):
            on_type = True
            lanelet_of_type = lanelet

    # vehicle is on a lanelet with type lanelet_type.
    # lanelet must also be of type intersection

    if on_type:
        if inside_intersection and not is_lanelet_of_type(
                lanelet_of_type, LaneletType.INTERSECTION, world.road_network
        ):
            _, min_dist, abs = get_closest_lanelet_of_type(
                vehicle_k, time_step, LaneletType.INTERSECTION, rnet
            )
            rob = (-1) * min_dist
            # if there is a successor that has active traffic light, robustness should be positive
            if abs and lanelet_type is HelperLaneletTypes.RELEVANT_TRAFFIC_LIGHT:
                rob = np.abs(rob)
            return rob
        else:
            # current lanelet has desired type.
            return get_robustness_inside_lanelet(
                vehicle_k, time_step, lanelet_of_type, only_successors
            )
    else:  # not on type lanelet_type
        if inside_intersection:
            _, min_dist, abs = get_closest_lanelet_of_type(
                vehicle_k, time_step, LaneletType.INTERSECTION, rnet
            )
            rob = (-1) * min_dist
            # if there is a successor that has active traffic light, robustness should be positive
            if abs and lanelet_type is HelperLaneletTypes.RELEVANT_TRAFFIC_LIGHT:
                rob = np.abs(rob)
            return rob
        else:
            _, min_dist, abs = get_closest_lanelet_of_type(
                vehicle_k, time_step, lanelet_type, rnet
            )
            rob = (-1) * min_dist
            # if there is a successor that has active traffic light, robustness should be positive
            if abs and lanelet_type is HelperLaneletTypes.RELEVANT_TRAFFIC_LIGHT:
                rob = np.abs(rob)
            return rob


# def get_incoming(
#     lanelet: Lanelet, lanelet_network: LaneletNetwork
# ) -> Optional[Tuple[Intersection, IntersectionIncomingElement]]:
#     """Get the incoming element of a lanelet.
#     :returns: Optional[Tuple[intersection to which lanelet belongs, IncomingElement to which lanelet belongs]]
#     """
#
#     # get the intersection to which our lanelet is an incoming element
#     intersection = lanelet_network.map_inc_lanelets_to_intersections.get(
#         lanelet.lanelet_id
#     )
#
#     # if our lanelet is not an incoming element -> return none
#     if intersection is None:
#         return None
#
#     # Tuple[Intersection to which lanelet belongs, IncomingElement to which lanelet belongs]
#     return intersection, intersection.map_incoming_lanelets[lanelet.lanelet_id]


# def inc_la_left_of(lanelet: Lanelet, lanelet_network: LaneletNetwork) -> Set[int]:
#     """
#     returns setof lanelets located on the left of the passed lanalet
#     """
#     # Implementation by Luis
#     intersection_incoming = get_incoming(lanelet, lanelet_network)
#
#     # return empty set if lanelet is no incoming element.
#     if intersection_incoming is None:
#         return set()
#
#     # Intersection to which lanelet belongs, IncomingElement to which lanelet belongs
#     intersection, incoming = intersection_incoming
#
#     # get all IncomingElements leftof our lanelet's IncomingElement
#     # TODO: Q: why list[0] if incoming.left_of returns only one id anyway ? why not get it directly?
#     left_incoming = [
#         inc for inc in intersection.incomings if inc.incoming_id == incoming.left_of
#     ][0]
#
#     return left_incoming.incoming_lanelets  # returns set of IDs of incoming lanelets


def get_latest_predecessors_path(
        lanelet: Lanelet, lanelet_network: LaneletNetwork, predecessors=[]
) -> List[int]:
    if len(lanelet.predecessor == 1):
        predecessors.extend(lanelet.predecessor)
        return get_latest_predecessors_path(
            lanelet_network.find_lanelet_by_id(lanelet.predecessor[0]),
            lanelet_network,
            predecessors,
        )
    else:
        return predecessors


def get_lanelet_paths(lanelet: Lanelet, road_network: RoadNetwork) -> List[List[int]]:
    """
    return a list of all possible paths in which lanelet exists
    """

    succ_paths = reach_succ(lanelet, road_network.lanelet_network)
    pre_paths = reach_pre(lanelet, road_network.lanelet_network)
    paths = []

    for pre_path in pre_paths:
        for succ_path in succ_paths:
            paths.append(pre_path + [lanelet.lanelet_id] + succ_path)
    return paths


def same_incom(lanelet_k: Lanelet, lanelet_p: Lanelet, rnet: RoadNetwork) -> bool:
    """
    returns true if two lanelets belong to the same incoming element.
    """

    # limitations: reach_pre is not deterministic, returns all possible predecessor paths
    # current hack: just iterate over all possible reachable predecessors
    # TODO: access past time steps to fix reach_pre ??
    reach_pre_k = reach_pre(lanelet_k, rnet.lanelet_network)

    reach_pre_p = reach_pre(lanelet_p, rnet.lanelet_network)
    reach_pre_k_flat: List[int] = [
                                      pre for reach_pre_path in reach_pre_k for pre in reach_pre_path
                                  ] + [lanelet_k.lanelet_id]

    reach_pre_p_flat: List[int] = [
                                      pre for reach_pre_path in reach_pre_p for pre in reach_pre_path
                                  ] + [lanelet_p.lanelet_id]

    for lak in reach_pre_k_flat:
        if not is_lanelet_of_type(
                rnet.lanelet_network.find_lanelet_by_id(lak),
                HelperLaneletTypes.INCOMING,
                rnet,
        ):
            continue
        adj_lanelets_lak_lanelets = adjacent_lanelets_same_direction(
            rnet.lanelet_network.find_lanelet_by_id(lak), rnet.lanelet_network
        )

        adj_lanelets_lak = list(
            map(lambda lanelet: lanelet.lanelet_id, adj_lanelets_lak_lanelets)
        )
        if set(adj_lanelets_lak).intersection(set(reach_pre_p_flat)):
            return True

    return False


def orientation_of_lanelet_center_veritices(
        lanelet: Lanelet, road_network: RoadNetwork
) -> np.ndarray:
    lanes = road_network.find_lanes_by_lanelets([lanelet.lanelet_id])

    for lane in lanes:
        # size = int(len(center_v) / 2)
        angles = lane._compute_orientation_from_polyline(lanelet.center_vertices)
        return angles


def mean_orientation_of_lanelet_center(
        lanelet: Lanelet, road_network: RoadNetwork
) -> float:
    angles = orientation_of_lanelet_center_veritices(lanelet, road_network)
    size = len(angles)
    if size == 1:
        return angles[0]

    sum = 0.0

    # TODO: why first orientation sometimes 0 ??
    for i in range(1, size):
        sum += angles[i]

    mean = sum / (size - 1)
    return mean


def get_stop_line_from_incoming(
        vehicle: Vehicle,
        incoming: IntersectionIncomingElement,
        lanelet_network: LaneletNetwork,
        time_step,
):
    """
    finds all stop lines in an intersection incoming element, and returns the stop line that is closest to the passed vehicle
    """
    # get the incoming lanelets as Set[int]
    lanelets = incoming.incoming_lanelets
    stop_lines = set()
    closest_stop_line = None
    min_distance = -1.0

    for lanelet in lanelets:
        lanelet_obj = lanelet_network.find_lanelet_by_id(lanelet)
        if lanelet_obj.stop_line != None:
            stop_lines.add(lanelet_obj.stop_line)
            if (
                    min_distance == -1.0
                    or distance_vehicle_to_stop_line(
                vehicle, lanelet_obj.stop_line, time_step
            )
                    < min_distance
            ):
                closest_stop_line = lanelet_obj.stop_line
    return (closest_stop_line, min_distance)


def lanelets_same_direction(
        lanelet1: Lanelet, lanelet2: Lanelet, road_network: RoadNetwork
) -> bool:
    e = 10
    mean_angle1 = math.degrees(
        mean_orientation_of_lanelet_center(lanelet1, road_network)
    )
    mean_angle2 = math.degrees(
        mean_orientation_of_lanelet_center(lanelet2, road_network)
    )
    diff = np.abs(mean_angle1 - mean_angle2)
    return diff <= e or diff >= (360 - e)


def lanelets_opposite_direction(
        lanelet1: Lanelet, lanelet2: Lanelet, road_network: RoadNetwork
) -> bool:
    e = 10
    mean_angle1 = math.degrees(
        mean_orientation_of_lanelet_center(lanelet1, road_network)
    )
    mean_angle2 = math.degrees(
        mean_orientation_of_lanelet_center(lanelet2, road_network)
    )
    diff = np.abs(mean_angle1 - mean_angle2)
    return diff >= 180 - e and diff <= 180 + e


def oncom(incoming: Lanelet, road_network: RoadNetwork) -> Set[int]:
    """
    returning the set of oncoming lanelets belonging to an incoming lanelet
    """
    # TODO: old implementation of reach succ
    # TODO: check if parameter lanelet is incoming first, otherwise return empty set

    oncom = set()
    if incoming.stop_line == None:
        return oncom

    # checks if

    lanelet_network = road_network.lanelet_network
    possible_successors = reach_succ(incoming, lanelet_network)
    merged_possible_successors = set().union(*possible_successors)
    for succ in merged_possible_successors:
        opp_adj_of_succ = indirect_opposite_adjacents(
            lanelet_network.find_lanelet_by_id(succ), lanelet_network
        )
        for opp in opp_adj_of_succ:
            if lanelets_opposite_direction(
                    incoming, lanelet_network.find_lanelet_by_id(opp), road_network
            ):
                oncom.add(opp)
    return oncom


def get_lanelet_center_endpoint(lanelet: Lanelet):
    center_vertices = lanelet.center_vertices
    return center_vertices[len(center_vertices) - 1]


def get_lanelet_center_startpoint(lanelet: Lanelet):
    center_vertices = lanelet.center_vertices
    return center_vertices[0]


def distance_between_two_points(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def distance_between_vehicles(
        vehicle_k: Vehicle, vehicle_p: Vehicle, time_step
) -> float:
    # TODO: Find a better way to calculate the distance_between_vehicles.
    p1 = np.array(vehicle_k.states_cr[time_step].position)
    p2 = np.array(vehicle_p.states_cr[time_step].position)
    return distance_between_two_points(p1, p2)


def distance_vehicle_to_stop_line(
        vehicle: Vehicle, stop_line: StopLine, time_step
) -> float:
    """
    calculates the euclidean distance from a vehicle position to the center point of a stop line
    """
    vehicle_position = vehicle.state_list_cr[time_step].position
    stop_line_center = [
        (stop_line.start[0] + stop_line.end[0]) / 2,
        (stop_line.start[1] + stop_line.end[1]) / 2,
    ]

    return distance_between_two_points(stop_line_center, vehicle_position)


def distance_lanelet_front_to_stop_line(lanelet: Lanelet, stop_line: StopLine) -> float:
    center_vertices = lanelet.center_vertices
    final_center_point = center_vertices[len(center_vertices) - 1]
    stop_line_center = [
        (stop_line.start[0] + stop_line.end[0]) / 2,
        (stop_line.start[1] + stop_line.end[1]) / 2,
    ]
    # second idea: distance from vehicle to the center point of the stop line.
    return distance_between_two_points(stop_line_center, final_center_point)


def indirect_opposite_adjacents(
        lanelet1: Lanelet, lanelet_network: LaneletNetwork
) -> List[int]:
    """
    returns a list for all adjacent lanelets and their adjacent lanelets, having opposite direction to parameter lanelet
    """
    current = lanelet1.adj_left
    left = False
    if lanelet1.adj_left_same_direction:
        left = True

    adj_dir = []
    adj_opp = []

    while current != None:
        current_lanelet = lanelet_network.find_lanelet_by_id(current)
        if left:
            adj_dir.append(current)
            if not current_lanelet.adj_left_same_direction:
                left = False
            current = current_lanelet.adj_left
        else:
            adj_opp.append(current)
            current = current_lanelet.adj_right

    return adj_opp


def get_closest_stop_line_from_lanelet(
        lanelet: Lanelet, road_network: RoadNetwork
) -> Tuple[StopLine, float]:
    if lanelet.stop_line != None:
        return lanelet.stop_line, distance_lanelet_front_to_stop_line(
            lanelet, lanelet.stop_line
        )

    min_distance = math.inf
    min_stopline = None
    pre_paths = reach_pre(lanelet, road_network.lanelet_network)
    succ_paths = reach_succ(lanelet, road_network.lanelet_network)
    merged_succ = set().union(*succ_paths)
    merged_pre = set().union(*pre_paths)
    lanelets = merged_pre.union(merged_succ)

    for l in lanelets:
        l_obj = road_network.lanelet_network.find_lanelet_by_id(l)
        if l_obj.stop_line != None:
            distance = distance_lanelet_front_to_stop_line(lanelet, l_obj.stop_line)
            if distance < min_distance:
                min_distance = distance
                min_stopline = l_obj.stop_line
    return min_stopline, min_distance


class HelperLaneletTypes(enum.Enum):
    """
    Enum describing useful types of lanelets, not included in LaneletType
    """

    INCOMING = "incoming"
    LEFT_TURNING = "left_turning"
    RIGHT_TURING = "right_turning"
    STRAIGHT_GOING = "straight_going"
    RELEVANT_TRAFFIC_LIGHT = "relevant_traffic_light"


def get_closest_lanelet_of_type(
        vehicle: Vehicle,
        time_step: int,
        lanelet_type: HelperLaneletTypes,
        rnet: RoadNetwork,
) -> Optional[Tuple[Lanelet, float, bool]]:
    """
    finds the closest lanelet of type lanelet_type by searching ref_path_lanelets (possible successors and predecessors)
    returns none if none is found, or a tuple[ closest_lanelet, distance, exists_in_successors]
    """
    # this var tells us if a lanelet having the property was found in successors
    exists_in_successors = False

    l_dir = lanelets_dir(vehicle, time_step, rnet)

    # TODO: how to decide which lanelet to search generally ?
    # current: just pop one element from the set of lanelets_dir
    # idea: most occupied lanelet.
    l_id = l_dir.pop()

    # PS: succ_paths and
    succ_paths = reach_succ(
        rnet.lanelet_network.find_lanelet_by_id(l_id), rnet.lanelet_network
    )
    pre_paths = reach_pre(
        rnet.lanelet_network.find_lanelet_by_id(l_id), rnet.lanelet_network
    )

    # find the nearest left turning successor lanelet
    # TODO: find better way to get the nearest lanelet:
    # current implementation

    closest_lanelet = None
    min_dist = math.inf
    for succ_path in succ_paths:
        for succ in succ_path:
            succ_lanelet = rnet.lanelet_network.find_lanelet_by_id(succ)
            if is_lanelet_of_type(succ_lanelet, lanelet_type, rnet):
                # TODO: find better way to calculate distance !!!
                # current: euclidean distance between vehicle state position and first center vertex of successor
                # idea: project distance along the path
                dist = distance_between_two_points(
                    vehicle.state_list_cr[time_step].position,
                    succ_lanelet.center_vertices[0],
                )
                if dist < min_dist:
                    exists_in_successors = True
                    closest_lanelet = succ_lanelet
                    min_dist = dist

    # find the nearest left turning predecessor lanelet
    for pred_path in pre_paths:
        for pred in pred_path:
            pred_lanelet = rnet.lanelet_network.find_lanelet_by_id(pred)
            if is_lanelet_of_type(pred_lanelet, lanelet_type, rnet):
                dist = distance_between_two_points(
                    vehicle.state_list_cr[time_step].position,
                    pred_lanelet.center_vertices[0],
                )
                if dist < min_dist:
                    closest_lanelet = pred_lanelet
                    min_dist = dist

    return closest_lanelet, min_dist, exists_in_successors


def has_active_light(lanelet: Lanelet, rnet: RoadNetwork) -> bool:
    """
    returns true if lanelet has an active traffic light
    """
    light_ids = lanelet.traffic_lights
    for light_id in light_ids:
        tl = rnet.lanelet_network.find_traffic_light_by_id(light_id)
        if tl.active:
            return True
    return False


def is_lanelet_of_type(
        lanelet: Lanelet,
        lanelet_type: Union[HelperLaneletTypes, LaneletType],
        rnet: RoadNetwork,
) -> bool:
    """ "
    evaluates to true, if lanelet has type lanelet_type
    """
    if lanelet_type is LaneletType.INTERSECTION:
        return lanelet_type in lanelet.lanelet_type
    elif lanelet_type is HelperLaneletTypes.LEFT_TURNING:
        return left_turning_lanelet(lanelet, rnet)
    elif lanelet_type is HelperLaneletTypes.STRAIGHT_GOING:
        return straight_going_lanelet(lanelet, rnet)
    elif lanelet_type is HelperLaneletTypes.RIGHT_TURING:
        return right_turning_lanelet(lanelet, rnet)
    elif lanelet_type is HelperLaneletTypes.INCOMING:
        return get_incoming(lanelet, rnet.lanelet_network) is not None
    elif lanelet_type is HelperLaneletTypes.RELEVANT_TRAFFIC_LIGHT:
        return has_active_light(lanelet, rnet)


def has_type_intersection(lanelet: Lanelet) -> bool:
    return LaneletType.INTERSECTION in lanelet.lanelet_type


def right_turning_lanelet(lanelet: Lanelet, rnet: RoadNetwork) -> bool:
    """
    returns whether a lanelet is turning right by computing the orientations of center vertices.
    i.e. orientations are decreasing (curvature has a Clockwise direction)
    """
    lane = rnet.find_lane_by_lanelet(lanelet.lanelet_id)
    orientations = lane._compute_orientation_from_polyline(lanelet.center_vertices)

    # TODO: solution for first orientation, always has a problem
    current = orientations[1]
    for i in range(2, len(orientations) - 1):
        if orientations[i] >= current:
            return False
        current = orientations[i]
    return True


def left_turning_lanelet(lanelet: Lanelet, rnet: RoadNetwork) -> bool:
    """
    returns whether a lanelet is turning left by computing the orientations of center vertices.
    i.e. orientations are increasing. (curvature has a counterclockwise direction)
    """
    lane = rnet.find_lane_by_lanelet(lanelet.lanelet_id)
    orientations = lane._compute_orientation_from_polyline(lanelet.center_vertices)

    # TODO: find a solution for this:
    # first orientation is skipped, always set to 0 has a problem (most likely because of crdesigner)
    current = orientations[1]
    for i in range(2, len(orientations) - 1):
        if orientations[i] <= current:
            return False
        current = orientations[i]
    return True


def straight_going_lanelet(lanelet: Lanelet, rnet: RoadNetwork) -> bool:
    """
    returns whether a lanelet is going straight by computing the orientations of center vertices.
    i.e. orientations are constant (no curvature)
    """
    # TODO: set a meaningful error. 0.1 is chosen wihtout any convincing reason.
    e = 0.2
    lane = rnet.find_lane_by_lanelet(lanelet.lanelet_id)
    orientations = lane._compute_orientation_from_polyline(lanelet.center_vertices)

    # TODO: find a solution for this:
    # first orientation is skipped, always set to 0 has a problem (most likely because of crdesigner)
    current = orientations[1]
    for i in range(2, len(orientations) - 1):

        if np.abs(orientations[i] - current) > e:
            return False
        current = orientations[i]
    return True


def orientation_difference_vehicle_lanelet(
        vehicle: Vehicle, lanelet: Lanelet, rnet: RoadNetwork, time_step
) -> float:
    """
    finds closest center point of lanelet to vehicle, and returns orientation difference in Rad between this point and the vehicle.
    """
    center_vertices = lanelet.center_vertices
    lane = rnet.find_lane_by_lanelet(lanelet.lanelet_id)
    orientations = lane._compute_orientation_from_polyline(center_vertices)
    vehicle_position = vehicle.state_list_cr[time_step].position
    vehicle_orientation = vehicle.state_list_cr[time_step].orientation

    min_index = -1
    min_distance = math.inf
    for i in range(len(center_vertices) - 1):
        distance = distance_between_two_points(center_vertices[i], vehicle_position)
        if distance < min_distance:
            min_distance = distance
            min_index = i

    orientation_of_closest_point = orientations[min_index]
    orientation_difference = np.abs(
        subtract_orientations(orientation_of_closest_point, vehicle_orientation)
    )
    return orientation_difference


# ---------------------------------------------------------------------------------------------------------------------#

# def distance_to_stop_line(vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step):
#     stop_line = get_stop_line(lanelet_ids, world.road_network)
#     """
#     find the distance between stop line and the front of vehicle
#     """
#     if stop_line is None:
#         return None
#     stop_line_path = np.array([stop_line.start, stop_line.end])
#     state = vehicle_i.states_cr[time_step]
#     occ_points = rotate_translate(vehicle_i.shape.vertices[:-1], state.position, state.orientation)
#     d_stop_line = np.array(cartesian_to_curvilinear(tuple([stop_line_path]), occ_points))[
#             ..., 1
#         ].ravel()
#     d_stop_line = d_stop_line[~np.isnan(d_stop_line)]
#     return np.max(d_stop_line)
#
#
# def get_stop_line(
#     lanelet_ids: Iterable[int], road_network: RoadNetwork
# ) -> StopLine:
#     """
#     find the stop line according to occupied lanelet
#     """
#     for lanelet_id in lanelet_ids:
#         lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
#         if lanelet.stop_line is not None:
#             return lanelet.stop_line
#     return None


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
            traffic_sign_object = road_network.lanelet_network.find_traffic_sign_by_id(ts_element_id)
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
    assert len(traffic_sign_elements) == 1, "TODO: Only works for one " "traffic sign type per lanelet!"
    return traffic_sign_elements[0]


def distance_start_lanelet(vehicle: Vehicle, lanelet_id: int, road_network: RoadNetwork, time_step):
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
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)
    d_start_lanelet = np.array(cartesian_to_curvilinear(tuple([lanlet_start_line]), occ_points))[..., 1].ravel()
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
    assert len(lanelet.traffic_lights) == 1, "TODO: Only works for one " "traffic light per lanelet!"
    tl = road_network.lanelet_network.find_traffic_light_by_id(list(lanelet.traffic_lights)[0])
    if tl.active:
        return True
    return False


# TODO: set limits for searching successors
def reach_suc(lanelet_id, road_network: RoadNetwork) -> np.array:
    paths = lanes_suc(lanelet_id, road_network)
    return np.unique(paths)


def reach_pre(lanelet_id, road_network: RoadNetwork) -> np.array:
    paths = lanes_pre(lanelet_id, road_network)
    return np.unique(paths)


# TODO: set limits for searching predecessors
def lanes_pre(lanelet_id: int, road_network: RoadNetwork) -> List[List[int]]:
    lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
    lanelet_ids = set()
    lanelet_ids.add(lanelet_id)
    lanelet_network = road_network.lanelet_network
    predecessors = lanelet.predecessor
    if len(predecessors) == 0:
        return [[lanelet_id]]
    paths = []
    for pre in predecessors:
        pre_lanelet = lanelet_network.find_lanelet_by_id(pre)
        pre_paths = lanes_pre(pre_lanelet.lanelet_id, road_network)
        for pre_path in pre_paths:
            pre_path.insert(0, pre)
            paths.append(list(lanelet_ids.union(set(pre_path))))
    return paths


def lanes_suc(lanelet_id, road_network: RoadNetwork) -> List[List[int]]:
    lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
    lanelet_ids = set()
    lanelet_ids.add(lanelet_id)
    lanelet_network = road_network.lanelet_network
    successors = lanelet.successor
    if len(successors) == 0:
        return [[lanelet_id]]
    paths = []
    for suc in successors:
        suc_lanelet = lanelet_network.find_lanelet_by_id(suc)
        suc_paths = lanes_suc(suc_lanelet.lanelet_id, road_network)
        for suc_path in suc_paths:
            suc_path.insert(0, suc)
            paths.append(list(lanelet_ids.union(set(suc_path))))
    return paths


def get_incoming(lanelets_id, road_network: RoadNetwork) -> IntersectionIncomingElement:
    incoming = None
    for lanelet_id in lanelets_id:
        lanelet_pre = reach_pre(lanelet_id, road_network)
        for incoming_element in road_network.lanelet_network.intersections[0].incomings:
            if len(incoming_element.incoming_lanelets.intersection(set(lanelet_pre))) > 0:
                incoming = incoming_element
                break
        if incoming is not None:
            break
    return incoming


def get_incoming_multi_intersections(vehicle: Vehicle, time_step, road_network: RoadNetwork):
    """
    get all incoming elements and distance to these incoming elements in different intersections
    by given a vehicle and current time step
    """
    # get all lanelets which can be occupied by current vehicle
    lanelets_dir_vehicle = np.array(vehicle.lanelets_dir)
    lanelets_dir_pre = reach_pre(lanelets_dir_vehicle[0], road_network)
    lanelets_dir_suc = reach_suc(lanelets_dir_vehicle[-1], road_network)
    lanelets_dir_vehicle = np.append(lanelets_dir_vehicle, lanelets_dir_pre)
    lanelets_dir_vehicle = np.append(lanelets_dir_vehicle, lanelets_dir_suc)
    occupied_lanelets_possible = np.unique(lanelets_dir_vehicle)
    # get front- and rear-most point of vehicle along reference lane
    front_s = vehicle.front_s(time_step, vehicle.ref_path_lane)
    rear_s = vehicle.rear_s(time_step, vehicle.ref_path_lane)
    incoming_elements = list()
    distance_to_incomings = list()
    for intersection in road_network.lanelet_network.intersections:
        for incoming in intersection.incomings:
            incoming_ids = list(incoming.incoming_lanelets.intersection(set(occupied_lanelets_possible)))
            if len(incoming_ids) != 0:
                incoming_elements.append(incoming)
                start_incoming_s = get_lanelets_start_s(vehicle.ref_path_lane, incoming_ids, road_network)
                incoming_successor = set.union(incoming.successors_right, incoming.successors_straight,
                                               incoming.successors_left)
                successor_possible = incoming_successor.intersection(set(occupied_lanelets_possible))
                end_intersection_s = get_lanelets_end_s(vehicle.ref_path_lane, successor_possible, road_network)
                # lanelets in front of vehicle
                if (front_s - start_incoming_s) < 0 < (end_intersection_s - rear_s):
                    distance_to_incomings.append(front_s - start_incoming_s)
                # vehicle in front of lanelets
                elif (end_intersection_s - rear_s) <= 0 <= (front_s - start_incoming_s):
                    distance_to_incomings.append(end_intersection_s - rear_s)
                # vehicle inside lanelets
                else:
                    distance_to_incomings.append(min(front_s - start_incoming_s, end_intersection_s - rear_s))
    return incoming_elements, distance_to_incomings


def get_right_turning_lane_by_lanelets(lanelets_id, road_network: RoadNetwork) -> (IntersectionIncomingElement, Lane):
    """
    find the incoming according to current lanelet assignments which includes the right turning lanelet of the
    searched incoming
    """
    for incoming_id, lanes_incoming in road_network.lanes_incoming.items():
        if lanes_incoming[0].contained_lanelets.intersection(lanelets_id):
            return road_network.incoming[incoming_id], lanes_incoming[0]
    return None, None


def get_left_turning_lane_by_lanelets(lanelets_id, road_network: RoadNetwork) -> (IntersectionIncomingElement, Lane):
    """
    find the incoming according to current lanelet assignments which includes the left turning lanelet of the
    searched incoming
    """
    for incoming_id, lanes_incoming in road_network.lanes_incoming.items():
        if lanes_incoming[2].contained_lanelets.intersection(lanelets_id):
            return road_network.incoming[incoming_id], lanes_incoming[2]
    return None, None


def get_straight_going_lane_by_lanelets(lanelets_id, road_network: RoadNetwork) -> (IntersectionIncomingElement, Lane):
    """
    find the incoming according to current lanelet assignments which includes the straight going lanelet of the
    searched incoming
    """
    for incoming_id, lanes_incoming in road_network.lanes_incoming.items():
        if lanes_incoming[1].contained_lanelets.intersection(lanelets_id):
            return road_network.incoming[incoming_id], lanes_incoming[1]
    return None, None


def get_right_turn_lane(road_network: RoadNetwork, incoming: IntersectionIncomingElement) -> Lane:
    """
    get the right-turning lane by given an incoming element
    """
    incoming_lanelets_ids = incoming.incoming_lanelets
    right_turn_lanelets_ids = incoming.successors_right
    right_turn_lane = list()
    for lane in road_network.lanes:
        # choose the lane which contains both incoming and right-turning successors
        if (len(lane.contained_lanelets.intersection(right_turn_lanelets_ids)) > 0
                and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0):
            right_turn_lane.append(lane)
    assert len(right_turn_lane) == 1, 'Something not correct, OR there are more than one lanelets before intersection.'
    return right_turn_lane[0]


def get_left_turn_lane(road_network: RoadNetwork, incoming: IntersectionIncomingElement) -> Lane:
    """
    get the left-turning lane by given an incoming element
    """
    incoming_lanelets_ids = incoming.incoming_lanelets
    left_turn_lanelets_ids = incoming.successors_left
    left_turn_lane = list()
    for lane in road_network.lanes:
        # choose the lane which contains both incoming and left-turning successors
        if (len(lane.contained_lanelets.intersection(left_turn_lanelets_ids)) > 0
                and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0):
            left_turn_lane.append(lane)
    assert len(left_turn_lane) == 1, 'Something not correct, OR there are more than one lanelets before intersection.'
    return left_turn_lane[0]


def get_straight_going_lane(road_network: RoadNetwork, incoming: IntersectionIncomingElement) -> Lane:
    """
    get the straight going lane by given an incoming element
    """
    incoming_lanelets_ids = incoming.incoming_lanelets
    straight_going_lanelets_ids = incoming.successors_straight
    straight_going_lane = list()
    for lane in road_network.lanes:
        if (len(lane.contained_lanelets.intersection(straight_going_lanelets_ids)) > 0
                and len(lane.contained_lanelets.intersection(incoming_lanelets_ids)) > 0):
            straight_going_lane.append(lane)
    assert len(
        straight_going_lane) == 1, 'Something not correct, OR there are more than one lanelets before intersection.'
    return straight_going_lane[0]


def get_lanelets_start_end_s(reference_lane: Lane, lanelets_ids, road_network: RoadNetwork) -> (float, float):
    """
    get start and end point of a set of lanelets along a reference lane
    """
    lanelets_start_s = np.inf
    lanelets_end_s = -np.inf
    for lanelet_id in lanelets_ids:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        start_s = reference_lane.clcs.convert_to_curvilinear_coords(*get_lanelet_start_line(lanelet)[0])[0]
        end_s = reference_lane.clcs.convert_to_curvilinear_coords(*get_lanelet_end_line(lanelet)[0])[0]
        lanelets_start_s = min(lanelets_start_s, start_s)
        lanelets_end_s = max(lanelets_end_s, end_s)
    return lanelets_start_s, lanelets_end_s


def get_lanelets_start_s(reference_lane: Lane, lanelets_ids, road_network: RoadNetwork) -> (float, float):
    """
    get start point of a set of lanelets along a reference lane
    """
    lanelets_start_s = np.inf
    for lanelet_id in lanelets_ids:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        start_s = reference_lane.clcs.convert_to_curvilinear_coords(*get_lanelet_start_line(lanelet)[0])[0]
        lanelets_start_s = min(lanelets_start_s, start_s)
    return lanelets_start_s


def get_lanelets_end_s(reference_lane: Lane, lanelets_ids, road_network: RoadNetwork) -> (float, float):
    """
    get end point of a set of lanelets along a reference lane
    """
    lanelets_end_s = -np.inf
    for lanelet_id in lanelets_ids:
        lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
        end_s = reference_lane.clcs.convert_to_curvilinear_coords(*get_lanelet_end_line(lanelet)[0])[0]
        lanelets_end_s = max(lanelets_end_s, end_s)
    return lanelets_end_s


def get_oncoming(vehicle: Vehicle, road_network: RoadNetwork):
    incoming_vehicle = get_incoming(vehicle.lanelets_dir, road_network)
    successors_straight = incoming_vehicle.successors_straight
    oncoming_straight_list = list()
    for suc_straight in successors_straight:
        straight_lanelet = road_network.lanelet_network.find_lanelet_by_id(suc_straight)
        oncoming_straight_sublist = list()
        adj_direction = "left"
        current_lanelet = straight_lanelet
        while True:
            if adj_direction == "left":
                if current_lanelet.adj_left is None:
                    break
                adj_left_lanelet = road_network.lanelet_network.find_lanelet_by_id(current_lanelet.adj_left)
                if not current_lanelet.adj_left_same_direction:
                    oncoming_straight_sublist.append(current_lanelet.adj_left)
                    adj_direction = "right"
                    current_lanelet = adj_left_lanelet
                elif current_lanelet.adj_left_same_direction and len(oncoming_straight_sublist) == 0:
                    current_lanelet = adj_left_lanelet
                else:
                    assert False
            elif adj_direction == "right":
                if current_lanelet.adj_right is None:
                    break
                adj_right_lanelet = road_network.lanelet_network.find_lanelet_by_id(current_lanelet.adj_right)
                if current_lanelet.adj_right_same_direction:
                    oncoming_straight_sublist.append(current_lanelet.adj_right)
                    current_lanelet = adj_right_lanelet
                elif not current_lanelet.adj_right_same_direction:
                    break
                else:
                    assert False
        oncoming_straight_list = oncoming_straight_list + oncoming_straight_sublist
    oncoming_list = list()
    for oncoming_straight in oncoming_straight_list:
        incoming = road_network.find_incoming_intersection([oncoming_straight])
        oncoming_list = oncoming_list + list(incoming.incoming_lanelets)
    return oncoming_list


def distance_to_left_bounds_clcs(vehicle: Vehicle, lane: Lane, time_step):
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)
    distance = list()
    for point in occ_points:
        d_left = lane.clcs_left.convert_to_curvilinear_coords(*point)[1]
        distance.append(d_left)
    return distance


def distance_to_right_bounds_clcs(vehicle: Vehicle, lane: Lane, time_step):
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)
    distance = list()
    for point in occ_points:
        d_right = lane.clcs_right.convert_to_curvilinear_coords(*point)[1]
        distance.append(d_right)
    return distance


def longitudinal_distance_to_lane(vehicle: Vehicle, lane: Lane, time_step):
    state = vehicle.states_cr[time_step]
    occ_points = rotate_translate(vehicle.shape.vertices[:-1], state.position, state.orientation)
    distance = list()
    for point in occ_points:
        d_right = lane.clcs.convert_to_curvilinear_coords(*point)[0]
        distance.append(d_right)
    return distance


def get_priority(lanelet_ids, road_network: RoadNetwork, direction, traffic_sign_priority):
    ts_types = traffic_sign_type(lanelet_ids, road_network)
    ts_types_intersection = [traffic_sign_priority[ts] for ts in ts_types]
    if len(ts_types_intersection) == 0:
        ts_types_intersection = [traffic_sign_priority[TrafficSignIDGermany.WARNING_RIGHT_BEFORE_LEFT]]
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


def inc_la_left_of(incoming: IntersectionIncomingElement, road_network: RoadNetwork) -> IntersectionIncomingElement:
    left_of_incoming = incoming.left_of
    for incoming_element in road_network.lanelet_network.intersections[0].incomings:
        if incoming_element.incoming_id == left_of_incoming:
            return incoming_element


def adjacent_lanelets(lanelets: Set[Lanelet], lanelet_network: LaneletNetwork) -> Set[Lanelet]:
    """
    find all adjacent lanelets by given a lanelet or lanelets
    """
    for lanelet in lanelets:
        la = lanelet
        while la is not None and la.adj_left is not None:
            if la.adj_left_same_direction:
                la = lanelet_network.find_lanelet_by_id(la.adj_left)
                if la is not None:
                    lanelets.add(la)
            else:
                la = None

        la = lanelet
        while la is not None and la.adj_right is not None:
            if la.adj_right_same_direction:
                la = lanelet_network.find_lanelet_by_id(la.adj_right)
                if la is not None:
                    lanelets.add(la)
            else:
                la = None
    return lanelets


def find_longest_lane_by_intersection_lanelet(lanelet_id: int, road_network: RoadNetwork) -> Lane:
    longest_lane = None
    num_contained_lanelets = 0
    for lane in road_network.lanes:
        if lanelet_id in lane.contained_lanelets:
            if len(lane.contained_lanelets) > num_contained_lanelets:
                longest_lane = lane
                num_contained_lanelets = len(lane.contained_lanelets)
    return longest_lane


def find_conflict_bounds(current_lanelet: Lanelet, conflict_lanelet: Lanelet):
    """
    find bounds of conflict_lanelet, which have point in current lanelet
    """
    conflict_bounds = list()
    right_bound = conflict_lanelet.right_vertices[1:-1]
    for right_bound_point in right_bound:
        if current_lanelet.polygon.contains_point(right_bound_point):
            conflict_bounds.append('right')
            break
    left_bound = conflict_lanelet.left_vertices[1:-1]
    for left_bound_point in left_bound:
        if current_lanelet.polygon.contains_point(left_bound_point):
            conflict_bounds.append('left')
            break
    start_bound_conflict = get_lanelet_start_line(conflict_lanelet)
    start_bound_current = get_lanelet_start_line(current_lanelet)
    if np.max(abs(start_bound_conflict - start_bound_current)) <= 0.01:
        conflict_bounds.append('start')
    end_bound_conflict = get_lanelet_end_line(conflict_lanelet)
    end_bound_current = get_lanelet_end_line(current_lanelet)
    dif = end_bound_current - end_bound_conflict
    max_dif = np.max(dif)
    if np.max(abs(end_bound_conflict - end_bound_current)) <= 0.01:
        conflict_bounds.append('end')
    return conflict_bounds


def find_conflict_points(line, conflict_polygon: Polygon):
    """
    find intersection points between a line and polygon
    """
    conflict_line_points = list()
    # Create curved line
    curved_line = LineString(line)
    # Get intersection of line and polygon
    intersection = curved_line.intersection(conflict_polygon)
    if intersection.geom_type == 'Point':
        conflict_line_points.append(intersection)
    elif intersection.geom_type == 'LineString' or intersection.geom_type == 'LinearRing':
        for point in intersection.coords:
            conflict_line_points.append(np.array(point))
    elif intersection.geom_type == 'MultiPoint' or intersection.geom_type == 'MultiLineString':
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
    front_right = center_position + np.array([l / 2 * np.cos(orientation) + w / 2 * np.sin(orientation),
                                              l / 2 * np.sin(orientation) - w / 2 * np.cos(orientation)])
    front_left = center_position + np.array([l / 2 * np.cos(orientation) - w / 2 * np.sin(orientation),
                                             l / 2 * np.sin(orientation) + w / 2 * np.cos(orientation)])
    return np.array([front_right, front_left])


def get_vehicle_rear_points(vehicle: Vehicle, time_step):
    center_position = vehicle.states_cr[time_step].position
    orientation = vehicle.states_cr[time_step].orientation
    l = vehicle.shape.length
    w = vehicle.shape.width
    rear_right = center_position + np.array([- l / 2 * np.cos(orientation) + w / 2 * np.sin(orientation),
                                             - l / 2 * np.sin(orientation) - w / 2 * np.cos(orientation)])
    rear_left = center_position + np.array([- l / 2 * np.cos(orientation) - w / 2 * np.sin(orientation),
                                            - l / 2 * np.sin(orientation) + w / 2 * np.cos(orientation)])
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
