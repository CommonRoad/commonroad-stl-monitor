import math
import unittest
from typing import List

import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.trajectory import State
from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import StateLongitudinal, StateLateral, Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.predicate import PredCutIn, \
    PredInSameLane, \
    PredSafeDistPrec, \
    PredInFrontOf, \
    PredSingleLane


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


class TestPredicate(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = "crmonitor/config.yaml"
        self.config = load_yaml(config_path)

    def test_cut_in(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # before cut-in -> ego vehicle occupies only single lane
        exp_sol_monitor_mode_2 = True  # during cut-in
        exp_sol_monitor_mode_3 = False  # after cut-in
        exp_sol_monitor_mode_4 = False  # driving back to initial lane
        exp_sol_monitor_mode_5 = False  # during cut-in -> but other vehicles is in another lane

        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(3)
        for l in lanelets:
            lanelet_network.add_lanelet(l)
        road_network = RoadNetwork(lanelet_network,
                                   self.config.get("road_network_param"))

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        # ego vehicle
        # Constant velocity, Lane switches 1 -> 1, 2 -> 2 -> 2, 1 with 45 degree
        state_list_lon_ego = {
            0: StateLongitudinal(s=10, v=10),
            1: StateLongitudinal(s=20, v=10),
            2: StateLongitudinal(s=30, v=10),
            3: StateLongitudinal(s=40, v=10)}
        state_list_lat_ego = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=2, theta=(1 / 4) * math.pi),
            2: StateLateral(d=4, theta=0),
            3: StateLateral(d=2, theta=-(1 / 4) * math.pi)}
        cr_state_list_ego = {
            0: State(position=(10, 2), time_step=0),
            1: State(position=(20, 4), time_step=1),
            2: State(position=(30, 6), time_step=2),
            3: State(position=(40, 4), time_step=3)}
        lanelet_assignments_ego = {0: {1}, 1: {1, 2}, 2: {1}, 3: {1, 2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego,
                              Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, ego_vehicle_param,
                              lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        # Constant velocity, lane keeping on lane 2
        state_list_lon_other_1 = {
            0: StateLongitudinal(s=0, v=10),
            1: StateLongitudinal(s=10, v=10),
            2: StateLongitudinal(s=20, v=10),
            3: StateLongitudinal(s=30, v=10)}
        state_list_lat_other_1 = {
            0: StateLateral(d=4, theta=0),
            1: StateLateral(d=4, theta=0),
            2: StateLateral(d=4, theta=0),
            3: StateLateral(d=4, theta=0)}
        cr_state_list_other_1 = {
            0: State(position=(0, 6), time_step=0),
            1: State(position=(10, 6), time_step=1),
            2: State(position=(20, 6), time_step=2),
            3: State(position=(30, 6), time_step=3)}
        lanelet_assignments_other_1 = {0: {2}, 1: {2}, 2: {2}, 3: {2}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1,
                                  state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 1, ObstacleType.CAR,
                                  ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        # Constant velocity, lane keeping on lane 3
        # Only defined at time step 1
        state_list_lon_other_2 = {1: StateLongitudinal(s=0, v=10)}
        state_list_lat_other_2 = {1: StateLateral(d=8, theta=0)}
        cr_state_list_other_2 = {1: State(position=(10, 10), time_step=1)}
        lanelet_assignments_other_2 = {1: {3}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2,
                                  state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 2, ObstacleType.CAR,
                                  ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        world_state = WorldState(ego_vehicle,
                                 [other_vehicle_1, other_vehicle_2],
                                 road_network)

        pred = PredCutIn(self.config)

        sol_monitor_mode_1 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.step()
        sol_monitor_mode_2 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.step()
        sol_monitor_mode_3 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.step()
        sol_monitor_mode_4 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.time_step = 1
        sol_monitor_mode_5 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_2.id])

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

    def test_same_lane(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True  # vehicles completely on same lane
        exp_sol_monitor_mode_2 = True  # ego vehicle partially in another lane
        exp_sol_monitor_mode_3 = True  # other vehicle partially in another lane
        exp_sol_monitor_mode_4 = False  # vehicles not in same lane
        exp_sol_monitor_mode_5 = True  # vehicles completely on same lane, but other vehicle is behind

        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(2)
        for l in lanelets:
            lanelet_network.add_lanelet(l)
        road_network = RoadNetwork(lanelet_network,
                                   self.config.get("road_network_param"))

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        # ego vehicle
        state_list_lon_ego = {
            0: StateLongitudinal(s=0, v=10),
            1: StateLongitudinal(s=10, v=10),
            2: StateLongitudinal(s=20, v=10),
            3: StateLongitudinal(s=30, v=10),
            4: StateLongitudinal(s=40, v=10)}
        state_list_lat_ego = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=2, theta=0),
            2: StateLateral(d=0, theta=0),
            3: StateLateral(d=0, theta=0),
            4: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {
            0: State(position=(0, 2), time_step=0),
            1: State(position=(10, 2), time_step=1),
            2: State(position=(20, 2), time_step=2),
            3: State(position=(30, 2), time_step=3),
            4: State(position=(40, 2), time_step=4)}
        lanelet_assignments_ego = {0: {1}, 1: {1, 2}, 2: {1}, 3: {1}, 4: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego,
                              Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, ego_vehicle_param,
                              lanelet_assignments_ego, None, None, None)

        # other vehicle 1
        state_list_lon_other_1 = {
            0: StateLongitudinal(s=10, v=10),
            1: StateLongitudinal(s=20, v=10),
            2: StateLongitudinal(s=30, v=10),
            3: StateLongitudinal(s=40, v=10)}
        state_list_lat_other_1 = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0, theta=0),
            2: StateLateral(d=2, theta=0),
            3: StateLateral(d=4, theta=0)}
        cr_state_list_other_1 = {
            0: State(position=(10, 2), time_step=0),
            1: State(position=(20, 2), time_step=1),
            2: State(position=(30, 4), time_step=2),
            3: State(position=(40, 6), time_step=3)}
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1, 2}, 3: {2}}
        other_vehicle_1 = Vehicle(state_list_lon_other_1,
                                  state_list_lat_other_1, Rectangle(5, 2),
                                  cr_state_list_other_1, 1, ObstacleType.CAR,
                                  ego_vehicle_param,
                                  lanelet_assignments_other_1, None, None, None)

        # other vehicle 2
        state_list_lon_other_2 = {4: StateLongitudinal(s=20, v=10)}
        state_list_lat_other_2 = {4: StateLateral(d=0, theta=0)}
        cr_state_list_other_2 = {4: State(position=(20, 2), time_step=4)}
        lanelet_assignments_ego = {4: {1}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2,
                                  state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 2, ObstacleType.CAR,
                                  ego_vehicle_param, lanelet_assignments_ego,
                                  None, None, None)

        world_state = WorldState(ego_vehicle,
                                 [other_vehicle_1, other_vehicle_2],
                                 road_network)

        pred = PredInSameLane(self.config)

        sol_monitor_mode_1 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.step()
        sol_monitor_mode_2 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.step()
        sol_monitor_mode_3 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.step()
        sol_monitor_mode_4 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_1.id])
        world_state.step()
        sol_monitor_mode_5 = pred.evaluate_boolean(world_state, [ego_vehicle.id,
                                                                 other_vehicle_2.id])

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)

    def test_safe_distance(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = False
        exp_sol_robustness_mode_1 = 9.0
        exp_sol_robustness_mode_2 = -21.0

        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(1)
        for l in lanelets:
            lanelet_network.add_lanelet(l)
        road_network = RoadNetwork(lanelet_network,
                                   self.config.get("road_network_param"))

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        state_list_lon_ego = {
            0: StateLongitudinal(s=0, v=20),
            1: StateLongitudinal(s=20, v=20)}
        state_list_lat_ego = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0, theta=0)}
        cr_state_list_ego = {
            0: State(acceleration=-1, time_step=0),
            1: State(acceleration=0, time_step=1)}
        lanelet_assignments_ego = {0: {1}, 1: {1}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego,
                              Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, ego_vehicle_param,
                              lanelet_assignments_ego, None, None, None)

        state_list_lon_other = {
            0: StateLongitudinal(s=20, v=20),
            1: StateLongitudinal(s=30, v=0)}
        state_list_lat_other = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0, theta=0)}
        cr_state_list_other = {
            0: State(acceleration=-1, time_step=0),
            1: State(acceleration=0, time_step=1)}
        lanelet_assignments_other = {0: {1}, 1: {1}}
        other_vehicle = Vehicle(state_list_lon_other, state_list_lat_other,
                                Rectangle(5, 2), cr_state_list_other, 1,
                                ObstacleType.CAR, ego_vehicle_param,
                                lanelet_assignments_other, None, None, None)

        world_state = WorldState(ego_vehicle, [other_vehicle], road_network)

        pred = PredSafeDistPrec(self.config)

        vehicle_ids = [ego_vehicle.id, other_vehicle.id]
        sol_monitor_mode_1 = pred.evaluate_boolean(world_state, vehicle_ids)
        world_state.step()
        sol_monitor_mode_2 = pred.evaluate_boolean(world_state, vehicle_ids)

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        world_state.time_step = 0
        sol_robustness_mode_1 = pred.evaluate_robustness(world_state,
                                                         vehicle_ids)
        world_state.step()
        sol_robustness_mode_2 = pred.evaluate_robustness(world_state,
                                                         vehicle_ids)

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)

    def test_front_of(self):
        # expected solutions
        exp_sol_monitor_mode_1 = False  # ego vehicle behind
        exp_sol_monitor_mode_2 = False  # ego vehicle and other vehicle have same occupancy
        exp_sol_monitor_mode_3 = False  # ego vehicle is not completely in front
        exp_sol_monitor_mode_4 = True  # ego vehicle is in front in same lane
        exp_sol_monitor_mode_5 = True  # ego vehicle is in front in another lane

        exp_sol_robustness_mode_1 = -13.0
        exp_sol_robustness_mode_2 = -5.0
        exp_sol_robustness_mode_3 = -3.0
        exp_sol_robustness_mode_4 = 5.0
        exp_sol_robustness_mode_5 = 14.0

        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(2)
        for l in lanelets:
            lanelet_network.add_lanelet(l)
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

        # other vehicle 1
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

        # other vehicle 2
        state_list_lon_other_2 = {4: StateLongitudinal(s=10, v=10)}
        state_list_lat_other_2 = {4: StateLateral(d=4, theta=0)}
        cr_state_list_other_2 = {4: State(position=10, time_step=4)}
        lanelet_assignments_other_2 = {4: {2}}
        other_vehicle_2 = Vehicle(state_list_lon_other_2,
                                  state_list_lat_other_2, Rectangle(5, 2),
                                  cr_state_list_other_2, 42, ObstacleType.CAR,
                                  ego_vehicle_param,
                                  lanelet_assignments_other_2, None, None, None)

        world_state = WorldState(ego_vehicle,
                                 [other_vehicle_1, other_vehicle_2],
                                 road_network)

        pred = PredInFrontOf(self.config)

        vehicle_ids = [other_vehicle_1.id, ego_vehicle.id]
        sol_monitor_mode = []
        sol_robustness_mode = []
        for i in range(4):
            sol_monitor_mode.append(
                    pred.evaluate_boolean(world_state, vehicle_ids))
            sol_robustness_mode.append(
                    pred.evaluate_robustness(world_state, vehicle_ids))
            world_state.step()
        vehicle_ids = [other_vehicle_2.id, ego_vehicle.id]
        sol_monitor_mode.append(pred.evaluate_boolean(world_state, vehicle_ids))
        sol_robustness_mode.append(
                pred.evaluate_robustness(world_state, vehicle_ids))

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode[0])
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode[1])
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode[2])
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode[3])
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode[4])

        self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode[0])
        self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode[1])
        self.assertEqual(exp_sol_robustness_mode_3, sol_robustness_mode[2])
        self.assertEqual(exp_sol_robustness_mode_4, sol_robustness_mode[3])
        self.assertEqual(exp_sol_robustness_mode_5, sol_robustness_mode[4])

    def test_same_lane(self):
        # expected solutions
        exp_sol_monitor_mode_1 = True
        exp_sol_monitor_mode_2 = True
        exp_sol_monitor_mode_3 = True
        exp_sol_monitor_mode_4 = False
        exp_sol_monitor_mode_5 = True
        exp_sol_monitor_mode_6 = False

        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(3)
        for l in lanelets:
            lanelet_network.add_lanelet(l)
        road_network = RoadNetwork(lanelet_network,
                                   self.config.get("road_network_param"))

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        # ego vehicle
        # ego vehicle
        state_list_lon_ego = {
            0: StateLongitudinal(s=0, v=10),
            1: StateLongitudinal(s=10, v=10),
            2: StateLongitudinal(s=20, v=10),
            3: StateLongitudinal(s=30, v=10),
            4: StateLongitudinal(s=40, v=10),
            5: StateLongitudinal(s=50, v=10)}
        state_list_lat_ego = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0.25, theta=0),
            2: StateLateral(d=1, theta=0),
            3: StateLateral(d=2, theta=0),
            4: StateLateral(d=0, theta=0),
            5: StateLateral(d=-2, theta=0)}
        cr_state_list_ego = {
            0: State(position=0, time_step=0),
            1: State(position=10, time_step=1),
            2: State(position=20, time_step=2),
            3: State(position=30, time_step=3),
            4: State(position=40, time_step=4),
            5: State(position=50, time_step=4)}
        lanelet_assignments_ego = {
            0: {2},
            1: {2},
            2: {2},
            3: {2, 3},
            4: {1},
            5: {1, 2}}
        ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego,
                              Rectangle(5, 2), cr_state_list_ego, 0,
                              ObstacleType.CAR, ego_vehicle_param,
                              lanelet_assignments_ego, None, None,
                              road_network.lanes[1])

        world_state = WorldState(ego_vehicle, [], road_network)

        pred = PredSingleLane(self.config)

        vehicle_ids = [ego_vehicle.id]
        sol_monitor_mode = []
        for i in range(6):
            sol_monitor_mode.append(
                    pred.evaluate_boolean(world_state, vehicle_ids))
            world_state.step()

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode[0])
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode[1])
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode[2])
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode[3])
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode[4])
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode[5])


if __name__ == "__main__":
    unittest.main()
