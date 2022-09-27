import logging
import numpy as np
from typing import List, Tuple, Set, Iterable, Dict, Callable

from commonroad.geometry.transform import rotate_translate
from commonroad.scenario.lanelet import LaneletType, LineMarking, Lanelet, LaneletNetwork

from crmonitor.common.helper import cartesian_to_curvilinear
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

logger = logging.getLogger(__name__)


def distance_to_bounds(vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step):
    state = vehicle_i.states_cr[time_step]
    occ_points = rotate_translate(vehicle_i.shape.vertices[:-1], state.position, state.orientation)
    lanelets = [world.road_network.lanelet_network.find_lanelet_by_id(i) for i in lanelet_ids]
    left_bounds = tuple([l.left_vertices for l in lanelets if l.adj_left is not None and l.adj_left not in lanelet_ids])
    right_bounds = tuple(
            [l.right_vertices for l in lanelets if l.adj_right is not None and l.adj_right not in lanelet_ids])
    if len(left_bounds) > 0:
        d_left = np.array(cartesian_to_curvilinear(left_bounds, occ_points))[..., 1].ravel()
        d_left = d_left[~np.isnan(d_left)]
    else:
        d_left = np.array([])
    if len(right_bounds) > 0:
        d_right = np.array(cartesian_to_curvilinear(right_bounds, occ_points))[..., 1].ravel()
        d_right = d_right[~np.isnan(d_right)]
    else:
        d_right = np.array([])

    return d_left, d_right

def distance_to_lanes(vehicle_i: Vehicle, lanelet_ids: Iterable[int], world, time_step):
    d_left, d_right = distance_to_bounds(vehicle_i, lanelet_ids, world, time_step)
    d_left = -np.min(d_left) if d_left.size > 0 else np.inf
    d_right = np.max(d_right) if d_right.size > 0 else np.inf
    return np.fmin(d_left, d_right)

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


def lanelets_left_of_vehicle(time_step: int, vehicle: Vehicle, lanelet_network: LaneletNetwork) -> Set[Lanelet]:
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
        new_lanelets = lanelets_left_of_lanelet(lanelet_network.find_lanelet_by_id(occ_l), lanelet_network)
        for lanelet in new_lanelets:
            left_lanelets.add(lanelet)

    return left_lanelets


def lanelets_right_of_vehicle(time_step: int, vehicle: Vehicle, lanelet_network: LaneletNetwork) -> Set[Lanelet]:
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
        new_lanelets = lanelets_right_of_lanelet(lanelet_network.find_lanelet_by_id(occ_l), lanelet_network)
        for lanelet in new_lanelets:
            right_lanelets.add(lanelet)

    return right_lanelets