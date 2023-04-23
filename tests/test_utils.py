import math
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import (LaneletNetwork, LineMarking, Lanelet, LaneletType, )
from commonroad.scenario.obstacle import State, ObstacleType

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from crmonitor.common.helper import merge_dicts_recursively

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.position import (PredRightOfBroadLaneMarking, PredLeftOfBroadLaneMarking,
                                           PredOnLaneletWithTypeIntersection, PredInIntersectionConflictArea,
                                           PredOnIncomingLeftOf, PredOnOncomOf, PredStopLineInFront)
from crmonitor.predicates.utils import (lanelets_dir, ref_path_lanelets)


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_sl"] = 1.0

    def testLaneletsDir(self):
        scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-1_1_T-1.xml")).open(
            lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestLaneletsdir-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurnRight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_Ffb-2_2_I-1-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_test_turn_left_5.xml")).open(
        #         lanelet_assignment=True)
        world = World.create_from_scenario(scenario)

        road_network = RoadNetwork(scenario.lanelet_network, self.config.get("road_network_param"))
        ego_vehicle = world.vehicle_by_id(1000)

        for time in range(ego_vehicle.end_time + 1):
            lanelets_dir_id = lanelets_dir(ego_vehicle, time, road_network)
            ref_path = ref_path_lanelets(ego_vehicle, road_network, time)
            print('-------------------')
            print('time = ', time)
            print(lanelets_dir_id)
            print(ref_path.contained_lanelets)
            fig = plt.figure()
            ax = fig.gca()
            rnd = MPRenderer()
            # set time step in draw_params
            rnd.draw_params.time_begin = time
            rnd.draw_params.dynamic_obstacle['show_label'] = True
            # plot the scenario at different time step

            scenario.draw(rnd)
            rnd.render()
            for lanelet_id in lanelets_dir_id:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(patches.Polygon(polygon, edgecolor='#ea1d1d', fill=False, linewidth=1, zorder=1000000))
            for lanelet_id in ref_path.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(patches.Polygon(polygon, edgecolor='blue', alpha=0.2, fill=True, linewidth=0.5, zorder=10000))
            plt.show()



