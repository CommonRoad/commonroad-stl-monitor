import math
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import (
    LaneletNetwork,
    LineMarking,
    Lanelet,
    LaneletType,
)
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
from crmonitor.predicates.position import (
    PredRightOfBroadLaneMarking,
    PredLeftOfBroadLaneMarking,
    PredOnLaneletWithTypeIntersection,
    PredInIntersectionConflictArea,
    PredOnIncomingLeftOf,
    PredOnOncomOf,
    PredStopLineInFront,
)
from crmonitor.predicates.utils import (
    get_priority,
    get_lanelet_start_line,
    get_lanelet_end_line,
)


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = True
        self.config["d_sl"] = 1.0

    def testLaneletsDir(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-1_1_T-1.xml")).open(
        #     lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-3_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestLaneletsdir-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurnRight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_Ffb-2_2_I-1-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_test_turn_left_5.xml")).open(
        #         lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml"
            )
        ).open(lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(
        #     str("../scenarios/test_intersection/DEU_TestGoingStraight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        world = World.create_from_scenario(scenario)

        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        ego_vehicle = world.vehicle_by_id(32)
        position_ego = np.array([state.position for state in ego_vehicle.state_list_cr])
        target_vehicle = world.vehicle_by_id(30)
        position_target = np.array(
            [state.position for state in target_vehicle.state_list_cr]
        )

        for time in range(40, ego_vehicle.end_time + 1):
            lanelets_dir_ego = ego_vehicle.lanelets_dir
            ref_path_ego = ego_vehicle.ref_path_lane
            lanelets_dir_target = target_vehicle.lanelets_dir
            ref_path_target = target_vehicle.ref_path_lane
            print("-------------------")
            print("time = ", time)
            fig = plt.figure()
            ax = fig.gca()
            rnd = MPRenderer()
            # set time step in draw_params
            rnd.draw_params.time_begin = time
            rnd.draw_params.dynamic_obstacle["show_label"] = True
            # plot the scenario at different time step

            scenario.draw(rnd)
            rnd.render()
            for lanelet_id in [14]:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="blue",
                        fill=False,
                        linewidth=2,
                        zorder=1000000,
                    )
                )
                ax.plot(
                    lanelet.right_vertices[:, 0],
                    lanelet.right_vertices[:, 1],
                    color="red",
                    linewidth=5,
                    zorder=10000000000,
                )
                ax.plot(
                    lanelet.left_vertices[:, 0],
                    lanelet.left_vertices[:, 1],
                    color="red",
                    linewidth=5,
                    zorder=10000000000,
                )
            for lanelet_id in ref_path_ego.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="blue",
                        alpha=0.2,
                        fill=True,
                        linewidth=0.5,
                        zorder=10,
                    )
                )
                ax.plot(
                    position_ego[time, 0],
                    position_ego[time, 1],
                    marker="x",
                    color="red",
                    markersize=5,
                    linewidth=1.5,
                    zorder=10000,
                )
            for lanelet_id in ref_path_target.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="red",
                        facecolor="red",
                        alpha=0.2,
                        fill=True,
                        linewidth=0.5,
                        zorder=10,
                    )
                )
            ax.plot(
                position_ego[:, 0],
                position_ego[:, 1],
                marker="x",
                markersize=5,
                linewidth=1.5,
                zorder=1000,
            )
            # ax.plot(position_target[:, 0], position_target[:, 1], marker='x', markersize=5, linewidth=1.5, zorder=1000)
            plt.show()

    def test_draw_scenario(self):
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-1_1_T-1.xml")).open(
        #     lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-3_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestLaneletsdir-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurnRight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_Ffb-2_2_I-1-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_test_turn_left_5.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(
        #     str("../scenarios/test_intersection/DEU_TestGoingStraight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        scenario, _ = CommonRoadFileReader(
            str(
                "../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml"
            )
        ).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)

        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        ego_vehicle = world.vehicle_by_id(30)
        position_ego = np.array([state.position for state in ego_vehicle.state_list_cr])
        target_vehicle = world.vehicle_by_id(31)
        position_target = np.array(
            [state.position for state in target_vehicle.state_list_cr]
        )

        for time in range(25, ego_vehicle.end_time + 1):
            lanelets_dir_ego = ego_vehicle.lanelets_dir
            ref_path_ego = ego_vehicle.ref_path_lane
            lanelets_dir_target = target_vehicle.lanelets_dir
            ref_path_target = target_vehicle.ref_path_lane
            print("-------------------")
            print("time = ", time)
            fig = plt.figure()
            ax = fig.gca()
            rnd = MPRenderer()
            # set time step in draw_params
            rnd.draw_params.time_begin = time
            rnd.draw_params.dynamic_obstacle["show_label"] = True
            # plot the scenario at different time step

            scenario.draw(rnd)
            rnd.render()
            for lanelet_id in [12]:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="blue",
                        fill=False,
                        linewidth=2,
                        zorder=1000000,
                    )
                )
                # ax.plot(lanelet.right_vertices[:, 0], lanelet.right_vertices[:, 1], color='red', linewidth=5, zorder=10000000000)
                ax.plot(
                    lanelet.left_vertices[:, 0],
                    lanelet.left_vertices[:, 1],
                    color="red",
                    linewidth=5,
                    zorder=10000000000,
                )
                # start_vertices = get_lanelet_start_line(lanelet)
                # ax.plot(start_vertices[:, 0], start_vertices[:, 1], color='red', linewidth=5, zorder=10000000000)
                end_vertices = get_lanelet_end_line(lanelet)
                ax.plot(
                    end_vertices[:, 0],
                    end_vertices[:, 1],
                    color="red",
                    linewidth=5,
                    zorder=10000000000,
                )
            for lanelet_id in ref_path_ego.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="blue",
                        alpha=0.2,
                        fill=True,
                        linewidth=0.5,
                        zorder=10,
                    )
                )
                ax.plot(
                    position_ego[time, 0],
                    position_ego[time, 1],
                    marker="x",
                    color="red",
                    markersize=5,
                    linewidth=1.5,
                    zorder=10000,
                )
            for lanelet_id in ref_path_target.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="red",
                        facecolor="red",
                        alpha=0.2,
                        fill=True,
                        linewidth=0.5,
                        zorder=10,
                    )
                )
            ax.plot(
                position_ego[:, 0],
                position_ego[:, 1],
                marker="x",
                markersize=5,
                linewidth=1.5,
                zorder=1000,
            )
            # ax.plot(position_target[:, 0], position_target[:, 1], marker='x', markersize=5, linewidth=1.5, zorder=1000)
            plt.show()

    def test_draw_scenario_single(self):
        scenario, _ = CommonRoadFileReader(
            str("../scenarios/test_intersection/DEU_TestRIN1-1_1_T-1.xml")
        ).open(lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-2_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestRIN1-3_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestLaneletsdir-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestTurnRight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_Ffb-2_2_I-1-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_test_turn_left_5.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(str("../scenarios/test_intersection/DEU_TestIntersectionInteract-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(
        #     str("../scenarios/test_intersection/DEU_TestGoingStraight-1_1_T-1.xml")).open(
        #         lanelet_assignment=True)
        # scenario, _ = CommonRoadFileReader(
        #         str("../scenarios/test_intersection/DEU_TestIntersectionInteract-2_1_T-1.xml")).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)

        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        ego_vehicle = world.vehicle_by_id(31)
        position_ego = np.array([state.position for state in ego_vehicle.state_list_cr])

        for time in range(25, ego_vehicle.end_time + 1):
            lanelets_dir_ego = ego_vehicle.lanelets_dir
            ref_path_ego = ego_vehicle.ref_path_lane
            print("-------------------")
            print("time = ", time)
            fig = plt.figure()
            ax = fig.gca()
            rnd = MPRenderer()
            # set time step in draw_params
            rnd.draw_params.time_begin = time
            rnd.draw_params.dynamic_obstacle["show_label"] = True
            # plot the scenario at different time step

            scenario.draw(rnd)
            rnd.render()
            for lanelet_id in [10, 12, 4]:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="red",
                        fill=False,
                        linewidth=2,
                        zorder=1000000,
                    )
                )
                # start_vertices = get_lanelet_start_line(lanelet)
                # ax.plot(start_vertices[:, 0], start_vertices[:, 1], color='red', linewidth=5, zorder=10000000000)
                # end_vertices = get_lanelet_end_line(lanelet)
                # ax.plot(end_vertices[:, 0], end_vertices[:, 1], color='red', linewidth=5, zorder=10000000000)
            for lanelet_id in ref_path_ego.contained_lanelets:
                lanelet = road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                polygon = lanelet.polygon.vertices
                ax.add_patch(
                    patches.Polygon(
                        polygon,
                        edgecolor="blue",
                        alpha=0.2,
                        fill=True,
                        linewidth=0.5,
                        zorder=10,
                    )
                )
                ax.plot(
                    position_ego[time, 0],
                    position_ego[time, 1],
                    marker="x",
                    color="red",
                    markersize=5,
                    linewidth=1.5,
                    zorder=10000,
                )
            ax.plot(
                position_ego[:, 0],
                position_ego[:, 1],
                marker="x",
                markersize=5,
                linewidth=1.5,
                zorder=1000,
            )
            # ax.plot(position_target[:, 0], position_target[:, 1], marker='x', markersize=5, linewidth=1.5, zorder=1000)
            plt.show()

    def testGetPriority(self):
        scenario, _ = CommonRoadFileReader(
            str("../scenarios/test_intersection/DEU_TestRIN1-3_1_T-1.xml")
        ).open(lanelet_assignment=True)
        world = World.create_from_scenario(scenario)
        road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )
        ego_vehicle = world.vehicle_by_id(31)
        priority = list()
        for lanelet_id in ego_vehicle.lanelets_dir:
            priority.append(get_priority(lanelet_id, road_network, "right"))
        print(priority)
