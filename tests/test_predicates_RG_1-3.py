import math
import unittest
from pathlib import Path

import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import (
    TrafficSign,
    TrafficSignIDGermany,
    TrafficSignElement,
)
from commonroad.scenario.trajectory import State

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import (
    StateLongitudinal,
    StateLateral,
    Vehicle,
    CurvilinearStateManager,
)
from crmonitor.common.world import World
from crmonitor.predicates.position import (
    PredInSameLane,
    PredSingleLane,
    PredPreceding,
    PredSafeDistPrec,
    PredInFrontOf,
)
from crmonitor.predicates.velocity import PredLaneSpeedLimit
from crmonitor.predicates.general import PredCutIn
from tests.util import parallel_lanes


class TestPredicate(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = False

    def test_cut_in(self):
        # expected solutions
        exp_sol_monitor_mode_1 = (
            False
            # before cut-in -> ego vehicle occupies only single lane
        )
        exp_sol_monitor_mode_2 = True  # during cut-in
        exp_sol_monitor_mode_3 = False  # after cut-in
        exp_sol_monitor_mode_4 = False  # driving back to initial lane
        exp_sol_monitor_mode_5 = (
            False
            # during cut-in -> but other vehicles is in another lane
        )

        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(3)
        for l in lanelets:
            lanelet_network.add_lanelet(l)
        road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param")
        )

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        # ego vehicle
        # Constant velocity, Lane switches 1 -> 1, 2 -> 2 -> 2, 1 with 45 degree
        cr_state_list_ego = {
            0: State(position=(10, 2), velocity=10, orientation=0, time_step=0),
            1: State(
                position=(20, 4),
                velocity=10,
                orientation=(1 / 4) * math.pi,
                time_step=1,
            ),
            2: State(position=(30, 6), velocity=10, orientation=0, time_step=2),
            3: State(
                position=(40, 4),
                velocity=10,
                orientation=-(1 / 4) * math.pi,
                time_step=3,
            ),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1, 2}, 2: {1}, 3: {1, 2}}
        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_ego,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_ego,
        )

        # other vehicle 1
        # Constant velocity, lane keeping on lane 2
        cr_state_list_other_1 = {
            0: State(position=(0, 6), orientation=0, velocity=0, time_step=0),
            1: State(position=(10, 6), orientation=0, velocity=0, time_step=1),
            2: State(position=(20, 6), orientation=0, velocity=0, time_step=2),
            3: State(position=(30, 6), orientation=0, velocity=0, time_step=3),
        }
        lanelet_assignments_other_1 = {0: {2}, 1: {2}, 2: {2}, 3: {2}}
        other_vehicle_1 = Vehicle(
            1,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_1,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_other_1,
        )

        # other vehicle 2
        # Constant velocity, lane keeping on lane 3
        # Only defined at time step 1
        cr_state_list_other_2 = {
            1: State(position=(10, 10), velocity=10, orientation=0, time_step=1)
        }
        lanelet_assignments_other_2 = {1: {3}}
        other_vehicle_2 = Vehicle(
            2,
            ObstacleType.CAR,
            ego_vehicle_param,
            Rectangle(5, 2),
            cr_state_list_other_2,
            None,
            CurvilinearStateManager(road_network),
            lanelet_assignments_other_2,
        )

        world = World({ego_vehicle, other_vehicle_1, other_vehicle_2}, road_network)

        pred = PredCutIn(self.config)

        sol_monitor_mode_1 = pred.evaluate_boolean(
            world, 0, [ego_vehicle.id, other_vehicle_1.id]
        )
        sol_monitor_mode_2 = pred.evaluate_boolean(
            world, 1, [ego_vehicle.id, other_vehicle_1.id]
        )
        sol_monitor_mode_3 = pred.evaluate_boolean(
            world, 2, [ego_vehicle.id, other_vehicle_1.id]
        )
        sol_monitor_mode_4 = pred.evaluate_boolean(
            world, 3, [ego_vehicle.id, other_vehicle_1.id]
        )
        sol_monitor_mode_5 = pred.evaluate_boolean(
            world, 1, [ego_vehicle.id, other_vehicle_2.id]
        )

        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)


#     def test_same_lane(self):
#         # expected solutions
#         exp_sol_monitor_mode_1 = 3.0  # vehicles completely on same lane
#         exp_sol_monitor_mode_2 = 1.5  # ego vehicle partially in left lane
#         exp_sol_monitor_mode_3 = 1.0  # other vehicle partially in another lane
#         exp_sol_monitor_mode_4 = -1.0  # vehicles not in same lane
#         exp_sol_monitor_mode_5 = (
#             3.0  # vehicles completely on same lane, but other vehicle is behind
#         )
#         exp_sol_monitor_mode_6 = np.inf  # both vehicles in two lanes
#         exp_sol_monitor_mode_7 = 0.5  # ego vehicle less in right lane
#         exp_sol_monitor_mode_8 = 1.5  # ego vehicle more in right lane

#         lanelet_network = LaneletNetwork()
#         lanelets = parallel_lanes(2)
#         for l in lanelets:
#             lanelet_network.add_lanelet(l)
#         road_network = RoadNetwork(
#             lanelet_network, self.config.get("road_network_param")
#         )

#         ego_vehicle_param = self.config.get("ego_vehicle_param")

#         # ego vehicle
#         cr_state_list_ego = {
#             0: State(position=(0, 2), time_step=0, velocity=10, orientation=0),
#             1: State(position=(10, 3.5), time_step=1, velocity=10, orientation=0),
#             2: State(position=(20, 2), time_step=2, velocity=10, orientation=0),
#             3: State(position=(30, 2), time_step=3, velocity=10, orientation=0),
#             4: State(position=(40, 2), time_step=4, velocity=10, orientation=0),
#             5: State(position=(50, 3.5), time_step=5, velocity=10, orientation=0),
#             6: State(position=(60, 3.5), time_step=6, velocity=10, orientation=0),
#             7: State(position=(70, 4.5), time_step=7, velocity=10, orientation=0),
#         }
#         lanelet_assignments_ego = {
#             0: {1},
#             1: {1, 2},
#             2: {1},
#             3: {1},
#             4: {1},
#             5: {1, 2},
#             6: {1, 2},
#             7: {1, 2},
#         }
#         ego_vehicle = Vehicle(
#             0,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_ego,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_ego,
#         )

#         # other vehicle 1
#         cr_state_list_other_1 = {
#             0: State(position=(10, 2), time_step=0, velocity=10, orientation=0),
#             1: State(position=(20, 2), time_step=1, velocity=10, orientation=0),
#             2: State(position=(30, 4), time_step=2, velocity=10, orientation=0),
#             3: State(position=(40, 6), time_step=3, velocity=10, orientation=0),
#         }
#         lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1, 2}, 3: {2}}
#         other_vehicle_1 = Vehicle(
#             1,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_other_1,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_other_1,
#         )

#         # other vehicle 2
#         cr_state_list_other_2 = {
#             4: State(position=(20, 2), time_step=4, velocity=10, orientation=0),
#             5: State(position=(30, 4), time_step=5, velocity=10, orientation=0),
#             6: State(position=(40, 6), time_step=6, velocity=10, orientation=0),
#             7: State(position=(50, 6), time_step=7, velocity=10, orientation=0),
#         }
#         lanelet_assignments_other_2 = {4: {1}, 5: {1, 2}, 6: {2}, 7: {2}}
#         other_vehicle_2 = Vehicle(
#             2,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_other_2,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_other_2,
#         )

#         world = World({ego_vehicle, other_vehicle_1, other_vehicle_2}, road_network)

#         pred = PredInSameLane(self.config)

#         sol_monitor_mode_1 = pred.evaluate_robustness(
#             world, 0, [ego_vehicle.id, other_vehicle_1.id]
#         )
#         sol_monitor_mode_2 = pred.evaluate_robustness(
#             world, 1, [ego_vehicle.id, other_vehicle_1.id]
#         )
#         sol_monitor_mode_3 = pred.evaluate_robustness(
#             world, 2, [ego_vehicle.id, other_vehicle_1.id]
#         )
#         sol_monitor_mode_4 = pred.evaluate_robustness(
#             world, 3, [ego_vehicle.id, other_vehicle_1.id]
#         )
#         sol_monitor_mode_5 = pred.evaluate_robustness(
#             world, 4, [ego_vehicle.id, other_vehicle_2.id]
#         )
#         sol_monitor_mode_6 = pred.evaluate_robustness(
#             world, 5, [ego_vehicle.id, other_vehicle_2.id]
#         )
#         sol_monitor_mode_7 = pred.evaluate_robustness(
#             world, 6, [ego_vehicle.id, other_vehicle_2.id]
#         )
#         sol_monitor_mode_8 = pred.evaluate_robustness(
#             world, 7, [ego_vehicle.id, other_vehicle_2.id]
#         )

#         self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
#         self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
#         self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
#         self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
#         self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
#         self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)
#         self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode_7)
#         self.assertEqual(exp_sol_monitor_mode_8, sol_monitor_mode_8)

#     def test_safe_distance(self):
#         # expected solutions
#         exp_sol_monitor_mode_1 = True
#         exp_sol_monitor_mode_2 = False
#         exp_sol_robustness_mode_1 = 9.0
#         exp_sol_robustness_mode_2 = -21.0

#         lanelet_network = LaneletNetwork()
#         lanelets = parallel_lanes(1)
#         for l in lanelets:
#             lanelet_network.add_lanelet(l)
#         road_network = RoadNetwork(
#             lanelet_network, self.config.get("road_network_param")
#         )

#         ego_vehicle_param = self.config.get("ego_vehicle_param")

#         cr_state_list_ego = {
#             0: State(
#                 acceleration=-1,
#                 time_step=0,
#                 orientation=0,
#                 velocity=20,
#                 position=[0, 0],
#             ),
#             1: State(
#                 acceleration=0,
#                 time_step=1,
#                 orientation=0,
#                 velocity=20,
#                 position=[20, 0],
#             ),
#         }
#         lanelet_assignments_ego = {0: {1}, 1: {1}}
#         ego_vehicle = Vehicle(
#             0,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_ego,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_ego,
#         )

#         state_list_lon_other = {
#             0: StateLongitudinal(s=20, v=20),
#             1: StateLongitudinal(s=30, v=0),
#         }
#         state_list_lat_other = {
#             0: StateLateral(d=0, theta=0),
#             1: StateLateral(d=0, theta=0),
#         }
#         cr_state_list_other = {
#             0: State(
#                 acceleration=-1,
#                 time_step=0,
#                 position=[20, 0],
#                 velocity=20,
#                 orientation=0,
#             ),
#             1: State(
#                 acceleration=0, time_step=1, position=[30, 0], velocity=0, orientation=0
#             ),
#         }
#         lanelet_assignments_other = {0: {1}, 1: {1}}
#         other_vehicle = Vehicle(
#             1,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_other,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_other,
#         )

#         world = World({ego_vehicle, other_vehicle}, road_network)
#         pred = PredSafeDistPrec(self.config)

#         vehicle_ids = [ego_vehicle.id, other_vehicle.id]
#         sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, vehicle_ids)

#         sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, vehicle_ids)

#         self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
#         self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
#         sol_robustness_mode_1 = pred.evaluate_robustness(world, 0, vehicle_ids)

#         sol_robustness_mode_2 = pred.evaluate_robustness(world, 1, vehicle_ids)

#         # self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode_1)
#         # self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode_2)

#     def test_front_of(self):
#         # expected solutions
#         exp_sol_monitor_mode_1 = False  # ego vehicle behind
#         exp_sol_monitor_mode_2 = (
#             False
#             # ego vehicle and other vehicle have same occupancy
#         )
#         exp_sol_monitor_mode_3 = False  # ego vehicle is not completely in front
#         exp_sol_monitor_mode_4 = True  # ego vehicle is in front in same lane
#         exp_sol_monitor_mode_5 = True  # ego vehicle is in front in another lane

#         exp_sol_robustness_mode_1 = -13.0
#         exp_sol_robustness_mode_2 = -5.0
#         exp_sol_robustness_mode_3 = -3.0
#         exp_sol_robustness_mode_4 = 5.0
#         exp_sol_robustness_mode_5 = 14.0

#         lanelet_network = LaneletNetwork()
#         lanelets = parallel_lanes(2)
#         for l in lanelets:
#             lanelet_network.add_lanelet(l)
#         road_network = RoadNetwork(
#             lanelet_network, self.config.get("road_network_param")
#         )

#         ego_vehicle_param = self.config.get("ego_vehicle_param")

#         # ego vehicle
#         cr_state_list_ego = {
#             0: State(position=[0, 0], velocity=10, orientation=0, time_step=0),
#             1: State(position=[10, 0], velocity=4, orientation=0, time_step=1),
#             2: State(position=[14, 0], velocity=10, orientation=0, time_step=2),
#             3: State(position=[24, 0], velocity=5, orientation=0, time_step=3),
#             4: State(position=[29, 0], velocity=5, orientation=0, time_step=4),
#         }
#         lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}}
#         ego_vehicle = Vehicle(
#             0,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_ego,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_ego,
#         )

#         # other vehicle 1
#         cr_state_list_other_1 = {
#             0: State(position=[8, 0], orientation=0, velocity=2, time_step=1),
#             1: State(position=[10, 0], orientation=0, velocity=2, time_step=1),
#             2: State(position=[12, 0], orientation=0, velocity=2, time_step=2),
#             3: State(position=[14, 0], orientation=0, velocity=2, time_step=3),
#         }
#         lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
#         other_vehicle_1 = Vehicle(
#             1,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_other_1,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_other_1,
#         )

#         # other vehicle 2
#         cr_state_list_other_2 = {
#             4: State(position=[10, 4], velocity=10, orientation=0, time_step=4)
#         }
#         lanelet_assignments_other_2 = {4: {2}}
#         other_vehicle_2 = Vehicle(
#             2,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_other_2,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_other_2,
#         )

#         world = World({ego_vehicle, other_vehicle_1, other_vehicle_2}, road_network)
#         pred = PredInFrontOf(self.config)

#         vehicle_ids = [other_vehicle_1.id, ego_vehicle.id]
#         sol_monitor_mode = []
#         sol_robustness_mode = []
#         for i in range(4):
#             sol_monitor_mode.append(pred.evaluate_boolean(world, i, vehicle_ids))
#             sol_robustness_mode.append(pred.evaluate_robustness(world, i, vehicle_ids))

#         vehicle_ids = [other_vehicle_2.id, ego_vehicle.id]
#         sol_monitor_mode.append(pred.evaluate_boolean(world, 4, vehicle_ids))
#         sol_robustness_mode.append(pred.evaluate_robustness(world, 4, vehicle_ids))

#         self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode[0])
#         self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode[1])
#         self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode[2])
#         self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode[3])
#         self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode[4])

#         # self.assertEqual(exp_sol_robustness_mode_1, sol_robustness_mode[0])
#         # self.assertEqual(exp_sol_robustness_mode_2, sol_robustness_mode[1])
#         # self.assertEqual(exp_sol_robustness_mode_3, sol_robustness_mode[2])
#         # self.assertEqual(exp_sol_robustness_mode_4, sol_robustness_mode[3])
#         # self.assertEqual(exp_sol_robustness_mode_5, sol_robustness_mode[4])

#     def test_single_lane(self):
#         # expected solutions
#         exp_sol_monitor_mode_1 = True
#         exp_sol_monitor_mode_2 = True
#         exp_sol_monitor_mode_3 = True
#         exp_sol_monitor_mode_4 = False
#         exp_sol_monitor_mode_5 = False
#         exp_sol_monitor_mode_6 = False
#         exp_sol_monitor_mode_7 = False

#         lanelet_network = LaneletNetwork()
#         lanelets = parallel_lanes(3)
#         for l in lanelets:
#             lanelet_network.add_lanelet(l)
#         road_network = RoadNetwork(
#             lanelet_network, self.config.get("road_network_param")
#         )

#         ego_vehicle_param = self.config.get("ego_vehicle_param")

#         # ego vehicle
#         # ego vehicle
#         cr_state_list_ego = {
#             0: State(position=(0, 1), time_step=0, velocity=10, orientation=0),
#             1: State(position=(10, 2), time_step=1, velocity=10, orientation=0),
#             2: State(position=(20, 3), time_step=2, velocity=10, orientation=0),
#             3: State(position=(30, 3.5), time_step=3, velocity=10, orientation=0),
#             4: State(position=(40, 4), time_step=4, velocity=10, orientation=0),
#             5: State(position=(50, 4.5), time_step=5, velocity=10, orientation=0),
#             6: State(position=(50, 100), time_step=5, velocity=10, orientation=0),
#         }
#         lanelet_assignments_ego = {
#             0: {1},
#             1: {1},
#             2: {1},
#             3: {1, 2},
#             4: {1, 2},
#             5: {1, 2},
#             6: {1},
#         }
#         ego_vehicle = Vehicle(
#             0,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_ego,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_ego,
#         )

#         world = World({ego_vehicle}, road_network)

#         pred = PredSingleLane(self.config)

#         vehicle_ids = [ego_vehicle.id]
#         sol_monitor_mode = []
#         assert len(cr_state_list_ego) == len(lanelet_assignments_ego)
#         for i in range(len(cr_state_list_ego)):
#             sol_monitor_mode.append(pred.evaluate_robustness(world, i, vehicle_ids))

#         self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode[0] >= 0)
#         self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode[1] >= 0)
#         self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode[2] >= 0)
#         self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode[3] >= 0)
#         self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode[4] >= 0)
#         self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode[5] >= 0)
#         self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode[6] >= 0)

#     def test_speed_limit(self):
#         # expected solutions
#         exp_sol_monitor_mode_1 = True  # ego vehicle drives with lower velocity
#         exp_sol_monitor_mode_2 = True  # ego vehicle drives exactly with the
#         # max speed
#         exp_sol_monitor_mode_3 = False  # ego vehicle drives too fast
#         exp_sol_monitor_mode_4 = True  # there exists no speed limit
#         exp_sol_robustness_mode_1 = 5.0
#         exp_sol_robustness_mode_2 = 0
#         exp_sol_robustness_mode_3 = -5.0
#         exp_sol_robustness_mode_4 = math.inf

#         lanelets = parallel_lanes(2)
#         lanelet_network = LaneletNetwork()
#         lanelet_network.add_lanelet(lanelets[0])
#         traffic_sign_max_speed = TrafficSignElement(
#             TrafficSignIDGermany.MAX_SPEED, ["50"]
#         )
#         lanelet_network.add_traffic_sign(
#             TrafficSign(111, [traffic_sign_max_speed], {1}, np.array([0.0, 0.0])), {1}
#         )
#         lanelet_network.add_lanelet(lanelets[1])
#         road_network = RoadNetwork(
#             lanelet_network, self.config.get("road_network_param")
#         )

#         ego_vehicle_param = self.config.get("ego_vehicle_param")

#         # ego vehicle
#         state_list_lon_ego = {
#             0: StateLongitudinal(s=0, v=45),
#             1: StateLongitudinal(s=45, v=50),
#             2: StateLongitudinal(s=95, v=55),
#             3: StateLongitudinal(s=150, v=45),
#         }
#         state_list_lat_ego = {
#             0: StateLateral(d=0, theta=0),
#             1: StateLateral(d=0, theta=0),
#             2: StateLateral(d=0, theta=0),
#             3: StateLateral(d=4, theta=0),
#         }
#         cr_state_list_ego = {
#             0: State(position=(0, 0), orientation=0, velocity=45, time_step=0),
#             1: State(position=(45, 0), orientation=0, velocity=50, time_step=1),
#             2: State(position=(95, 0), orientation=0, velocity=55, time_step=2),
#             3: State(position=(150, 4), orientation=0, velocity=45, time_step=3),
#         }
#         lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {2}}
#         ego_vehicle = Vehicle(
#             0,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_ego,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_ego,
#         )

#         world = World({ego_vehicle}, road_network)

#         pred = PredLaneSpeedLimit({"country": "DEU"})

#         vehicle_ids = [ego_vehicle.id]
#         sol_monitor_mode = []
#         sol_robustness_mode = []
#         for i in range(4):
#             sol_monitor_mode.append(pred.evaluate_boolean(world, i, vehicle_ids))
#             sol_robustness_mode.append(pred.evaluate_robustness(world, i, vehicle_ids))

#         self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode[0])
#         self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode[1])
#         self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode[2])
#         self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode[3])

#     def test_precedes(self):
#         # Directly precedes t=0
#         # Ego offset t=1
#         # Ego and other offset  t=2
#         # Other behind t=3
#         # Other behind one in between t=4
#         # Other in front other in between t=5
#         # Other in other lane t=6
#         # Ego in other lane t=7
#         expected = [True, True, True, False, False, False, False, False, True]

#         lanelet_network = LaneletNetwork()
#         lanelets = parallel_lanes(2)
#         lanelet_network.add_lanelet(lanelets[0])
#         lanelet_network.add_lanelet(lanelets[1])
#         road_network = RoadNetwork(
#             lanelet_network, self.config.get("road_network_param")
#         )

#         lat_ego = [0, 1, 1, 0, 0, 0, 0, 4, 0]
#         lon_ego = [30, 30, 30, 30, 30, 30, 30, 30, 10]
#         lanelets_ego = [{1}, {1, 2}, {1, 2}, {1}, {1}, {1}, {1}, {2}, {1}]
#         ego_vehicle = self.create_vehicle(
#             0, lanelets_ego, lat_ego, lon_ego, road_network
#         )

#         lat_other = [0, 0, 1, 0, 0, 0, 4, 0, 2]
#         lon_other = [40, 40, 40, 20, 10, 50, 40, 40, 40]
#         lanelets_other = [{1}, {1}, {1, 2}, {1}, {1}, {1}, {2}, {1}, {1, 2}]
#         other_vehicle = self.create_vehicle(
#             1, lanelets_other, lat_other, lon_other, road_network
#         )

#         lat_other = [0, 0, 0, 0, 0, 0, 0, 0, 2]
#         lon_other = [10, 10, 10, 10, 20, 40, 10, 10, 20]
#         lanelets_other = [{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {2}]
#         other_vehicle_2 = self.create_vehicle(
#             2, lanelets_other, lat_other, lon_other, road_network
#         )

#         world = World({ego_vehicle, other_vehicle, other_vehicle_2}, road_network)
#         vehicle_ids = [ego_vehicle.id, other_vehicle.id]

#         pred = PredPreceding({})
#         for t, exp in enumerate(expected):
#             rob = pred.evaluate_robustness(world, t, vehicle_ids)
#             self.assertEqual(exp, rob >= 0.0, f"t={t}")

#     def create_vehicle(self, veh_id, lanelets_ego, lat_ego, lon_ego, road_network):
#         ego_vehicle_param = self.config.get("ego_vehicle_param")
#         cr_state_list_ego = {
#             t: State(position=(s, d + 0.5 * 4), time_step=t, orientation=0, velocity=45)
#             for t, (s, d, l) in enumerate(zip(lon_ego, lat_ego, lanelets_ego))
#         }
#         lanelet_assignments_ego = {t: l for t, l in enumerate(lanelets_ego)}
#         ego_vehicle = Vehicle(
#             veh_id,
#             ObstacleType.CAR,
#             ego_vehicle_param,
#             Rectangle(5, 2),
#             cr_state_list_ego,
#             None,
#             CurvilinearStateManager(road_network),
#             lanelet_assignments_ego,
#         )
#         return ego_vehicle


if __name__ == "__main__":
    unittest.main()
