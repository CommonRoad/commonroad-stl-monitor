import unittest
from typing import List
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.trajectory import State
from crmonitor.common.evaluation import RuleSetEvaluator
from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import StateLongitudinal, StateLateral, Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.rule import Rule


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
        right_y = i * lane_width
        left_y = (i + 1) * lane_width
        center_y = (i + 0.5) * lane_width
        x_points = np.arange(start=0, stop=lane_length + lon_step,
                             step=lon_step)
        ones = np.ones((x_points.shape[0]))
        right_vertices_lane = np.stack((x_points, ones * right_y), axis=1)
        left_vertices_lane = np.stack((x_points, ones * left_y), axis=1)
        center_vertices_lane = np.stack((x_points, ones * center_y), axis=1)
        if i == num_lanes - 1:
            adjacent_left = None
        else:
            adjacent_left = i + 1
        lanelets.append(Lanelet(left_vertices_lane, center_vertices_lane,
                                right_vertices_lane, lanelet_id=i + 1,
                                adjacent_left=adjacent_left,
                                adjacent_left_same_direction=True))
    return lanelets

class TestEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = "crmonitor/config.yaml"
        self.config = load_yaml(config_path)

    def test_single_vehicle(self):
        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(1)
        lanelet_network.add_lanelet(lanelets[0])
        road_network = RoadNetwork(lanelet_network,
                                   self.config.get("road_network_param"))

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        # ego vehicle
        state_list_lon_ego = {
            0: StateLongitudinal(s=0, v=10),
            1: StateLongitudinal(s=10, v=4),
            2: StateLongitudinal(s=14, v=10),
            3: StateLongitudinal(s=24, v=5),
            4: StateLongitudinal(s=29, v=5)}
        state_list_lat_ego = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0, theta=0),
            2: StateLateral(d=0, theta=0),
            3: StateLateral(d=0, theta=0),
            4: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {
            0: State(position=0, time_step=0),
            1: State(position=10, time_step=1),
            2: State(position=14, time_step=2),
            3: State(position=24, time_step=3),
            4: State(position=29, time_step=4)}
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego,
                              Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, ego_vehicle_param,
                              lanelet_assignments_ego, None, None, None)


        state_list_lon_other_1 = {
            0: StateLongitudinal(s=8, v=2),
            1: StateLongitudinal(s=10, v=2),
            2: StateLongitudinal(s=12, v=2),
            3: StateLongitudinal(s=14, v=2)}
        state_list_lat_other_1 = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0, theta=0),
            2: StateLateral(d=0, theta=0),
            3: StateLateral(d=0, theta=0)}
        cr_state_list_other_1 = {
            0: State(position=10, time_step=1),
            1: State(position=10, time_step=1),
            2: State(position=20, time_step=2),
            3: State(position=30, time_step=3)}
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1,
                                  state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 41, ObstacleType.CAR,
                                  ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        world_state = WorldState(ego_vehicle, [other_vehicle_1], road_network)

        rule_str = "in_front_of__a0_a1"
        rule = Rule(rule_str, {"traffic_rules_param": {}})
        rule_eval = RuleSetEvaluator([rule])
        rob, preds = rule_eval.evaluate_all_rules_all_timesteps_floating(world_state)
        self.assertEqual(rob.shape[0], 4)

if __name__ == "__main__":
    unittest.main()