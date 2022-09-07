
import logging
from typing import Iterable
import numpy as np
from commonroad.geometry.transform import rotate_translate

from crmonitor.common.helper import cartesian_to_curvilinear
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World

logger = logging.getLogger(__name__)


def distance_to_bounds(vehicle_i: Vehicle, lanelet_ids: Iterable[int], world: World, time_step):
    state = vehicle_i.states_cr[time_step]
    occ_points = rotate_translate(
            vehicle_i.shape.vertices[:-1], state.position, state.orientation
        )
    lanelets = [
        world.road_network.lanelet_network.find_lanelet_by_id(i)
        for i in lanelet_ids
    ]
    left_bounds = tuple([
        l.left_vertices for l in lanelets if l.adj_left is not None and l.adj_left not in lanelet_ids
    ])
    right_bounds = tuple([
        l.right_vertices
        for l in lanelets
        if l.adj_right is not None and l.adj_right not in lanelet_ids
    ])
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