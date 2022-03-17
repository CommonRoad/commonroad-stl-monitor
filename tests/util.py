from typing import List

import numpy as np
from commonroad.scenario.lanelet import Lanelet


def parallel_lanes(num_lanes) -> List[Lanelet]:
    """
    Defines 3 parallel lanes with width 4 and length 90
    Lane ids are 1-indexed!
    :return: List of 3 lanelets
    """
    lane_width = 4
    lane_length = 90
    lon_step = 10
    lanelets = []
    for i in range(num_lanes):
        # Lanes from right to left
        right_y = i * lane_width
        left_y = (i + 1) * lane_width
        center_y = (i + 0.5) * lane_width
        x_points = np.arange(start=0, stop=lane_length + lon_step, step=lon_step)
        ones = np.ones((x_points.shape[0]))
        right_vertices_lane = np.stack((x_points, ones * right_y), axis=1)
        left_vertices_lane = np.stack((x_points, ones * left_y), axis=1)
        center_vertices_lane = np.stack((x_points, ones * center_y), axis=1)
        if i == 0:
            adjacent_right = None
        else:
            adjacent_right = i
        if i == num_lanes - 1:
            adjacent_left = None
        else:
            adjacent_left = i + 2
        lanelets.append(
            Lanelet(
                left_vertices_lane,
                center_vertices_lane,
                right_vertices_lane,
                lanelet_id=i + 1,
                adjacent_left=adjacent_left,
                adjacent_right=adjacent_right,
                adjacent_right_same_direction=True,
                adjacent_left_same_direction=True,
            )
        )
    return lanelets