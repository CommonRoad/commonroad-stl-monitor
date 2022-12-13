import math
import unittest
from pathlib import Path
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import State, ObstacleType

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.priority import (PredRelevantTrafficLight)


class TestPriorityPredicates(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = False
        right_vertices_lane_1 = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [
                                         50, 0], [60, 0], [70, 0], [80, 1], [90, 0]])
        left_vertices_lane_1 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [
                                        50, 4], [60, 4], [70, 4], [80, 1], [90, 0]])
        center_vertices_lane_1 = np.array([[0, 2], [10, 2], [20, 2], [30, 2], [40, 2], [
                                          50, 2], [60, 2], [70, 2], [80, 1], [90, 0]])
        self._lanelet_1 = Lanelet(left_vertices_lane_1, center_vertices_lane_1, right_vertices_lane_1, lanelet_id=1,
                                  adjacent_left=2, adjacent_left_same_direction=True)
        right_vertices_lane_2 = np.array([[0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4], [60, 4], [70, 4],
                                          [80, 4], [90, 4]])
        left_vertices_lane_2 = np.array([[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8],
                                         [80, 8], [90, 8]])
        center_vertices_lane_2 = np.array([[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12],
                                           [70, 12], [80, 12], [90, 12]])
        self._lanelet_2 = Lanelet(left_vertices_lane_2, center_vertices_lane_2, right_vertices_lane_2, lanelet_id=2,
                                  adjacent_left=3, adjacent_left_same_direction=True,
                                  adjacent_right=1, adjacent_right_same_direction=True)
        right_vertices_lane_3 = np.array([[0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8], [60, 8], [70, 8],
                                          [80, 8], [90, 8]])
        left_vertices_lane_3 = np.array([[0, 12], [10, 12], [20, 12], [30, 12], [40, 12], [50, 12], [60, 12], [70, 12],
                                         [80, 12], [90, 12]])
        center_vertices_lane_3 = np.array([[0, 10], [10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [60, 10],
                                           [70, 10], [80, 10], [90, 10]])
        self._lanelet_3 = Lanelet(left_vertices_lane_3, center_vertices_lane_3, right_vertices_lane_3, lanelet_id=3,
                                  adjacent_right=2, adjacent_right_same_direction=True)
        right_vertices_lane_4 = np.array([[0, 14], [10, 14], [20, 14], [30, 14], [40, 14], [50, 14], [60, 14], [70, 14],
                                          [80, 14], [90, 14]])
        left_vertices_lane_4 = np.array([[0, 18], [10, 18], [20, 18], [30, 18], [40, 18], [50, 18], [60, 18], [70, 18],
                                         [80, 18], [90, 18]])
        center_vertices_lane_4 = np.array([[0, 16], [10, 16], [20, 16], [30, 16], [40, 16], [50, 16], [60, 16],
                                           [70, 16], [80, 16], [90, 16]])
        self._lanelet_4 = Lanelet(
            left_vertices_lane_4, center_vertices_lane_4, right_vertices_lane_4, lanelet_id=4)
        lanelet_network = LaneletNetwork()
        lanelet_network.add_lanelet(self._lanelet_1)
        lanelet_network.add_lanelet(self._lanelet_2)
        lanelet_network.add_lanelet(self._lanelet_3)
        lanelet_network.add_lanelet(self._lanelet_4)
        self.road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param"))



    def test_relevant_traffic_light(self):

        #TODO
        exp_sol_monitor_mode_1 = False  
        exp_sol_monitor_mode_2 = False 
        exp_sol_monitor_mode_3 = True  
        exp_sol_monitor_mode_4 = True     

        #TODO
        # ego vehicle
        cr_state_list_ego = {0: State(position=[0, 0], time_step=0, orientation=0, velocity=15),
                             1: State(position=[10, 0], time_step=1, orientation=(1 / 8) * math.pi, velocity=15),
                             2: State(position=[20, 0], time_step=2, orientation=(1 / 2) * math.pi, velocity=15),
                             3: State(position=[30, 0], time_step=3, orientation=(3 / 4) * math.pi, velocity=15)}
        
        #TODO
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        
        ego_vehicle_param = self.config.get("ego_vehicle_param")
        ego_vehicle = Vehicle(0, ObstacleType.CAR, ego_vehicle_param, Rectangle(5, 2), cr_state_list_ego, None,
                              CurvilinearStateManager(self.road_network), lanelet_assignments_ego)

        pred = PredRelevantTrafficLight(self.config)
        vehicle_ids = [ego_vehicle.id]

        world = World({ego_vehicle}, self.road_network)

        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, vehicle_ids)
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

        sol_monitor_mode_4 = pred.evaluate_boolean(world, 3, vehicle_ids)
        sol_robustness_monitor_mode_4 = pred.evaluate_robustness(world, 3, vehicle_ids)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 > 0)