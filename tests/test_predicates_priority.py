import math
import unittest
from pathlib import Path
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import State, ObstacleType
from commonroad.common.file_reader import CommonRoadFileReader


from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world import World
from crmonitor.predicates.priority import (
    PredRelevantTrafficLight,  # not covered
    PredRelevantTrafficLight,  # not covered
    PredHasPriorityRightRight,
    PredHasPriorityRightLeft,
    PredHasPriorityRightStraight,
    PredHasPriorityLeftRight,  # not covered
    PredHasPriorityLeftLeft,  # not covered
    PredHasPriorityLeftStraight,
    PredHasPriorityStraightRight,
    PredHasPriorityStraightLeft,  # not covered
    PredHasPriorityStraightStraight,
    PredSamePriorityRightRight,
    PredSamePriorityRightLeft,
    PredSamePriorityRightStraight,
    PredSamePriorityLeftRight,
    PredSamePriorityLeftLeft,  # not covered
    PredSamePriorityLeftStraight,
    PredSamePriorityStraightRight,
    PredSamePriorityStraightLeft,  # not covered
    PredSamePriorityStraightStraight,
)


class TestPriorityPredicates(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        config_path = Path(__file__).parents[1] / "crmonitor" / "config.yaml"
        self.config = load_yaml(str(config_path))
        self.config["scale_rob"] = False

        scenario, _ = CommonRoadFileReader(
            str(
                "scenarios/test_intersection/DEU_Intersectionwithlightsandsigns-1_1_T-1.xml"
            )
        ).open(True)

        self.scenario = scenario
        self.road_network = RoadNetwork(
            scenario.lanelet_network, self.config.get("road_network_param")
        )


    def test_relevant_traffic_light(self):

        world = World.create_from_scenario(self.scenario)

        exp_sol_monitor_mode_1 = False  # traffic light inactive
        exp_sol_monitor_mode_2 = True
        exp_sol_monitor_mode_3 = False  # incoming lanelet has no traffic light

        # ego vehicle
        cr_state_list_ego = {
            0: State(
                position=[42, 3],
                time_step=0,
                orientation=(1) * math.pi,
                velocity=15,
            ),
            1: State(
                position=[13, 0],
                time_step=1,
                # orientation=(1 / 2) * math.pi, #straight
                # orientation=(3 / 4) * math.pi, #left
                orientation=(0) * math.pi,  # right
                velocity=15,
            ),
            2: State(
                position=[30, 0],
                time_step=2,
                orientation=(0) * math.pi,
                velocity=15,
            ),
        }

        lanelet_assignments_ego = {0: {10}, 1: {1}, 2: {9}}

        ego_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_ego,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_ego,
        )

        # fixing lanelet assignement according to spatial occupancy
        for time, _ in ego_vehicle.lanelet_assignment.items():
            shape = ego_vehicle.shape
            state = ego_vehicle.states_cr[time]

            ego_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        pred = PredRelevantTrafficLight(self.config)

        world.add_vehicle(ego_vehicle)
        # world = World({ego_vehicle}, self.road_network)

        # ts = 0 : still in incoming => false
        sol_monitor_mode_1 = pred.evaluate_boolean(world, 0, [0])
        sol_robustness_monitor_mode_1 = pred.evaluate_robustness(world, 0, [0])
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 > 0)

        # ts = 1 : inside the intersection on a lanelet going right => true
        sol_monitor_mode_2 = pred.evaluate_boolean(world, 1, [0])
        sol_robustness_monitor_mode_2 = pred.evaluate_robustness(world, 1, [0])
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 > 0)

        # ts = 2 : outside incoming => false
        sol_monitor_mode_3 = pred.evaluate_boolean(world, 2, [0])
        sol_robustness_monitor_mode_3 = pred.evaluate_robustness(world, 2, [0])
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 > 0)

    def test_same_priority(self):

        world = World.create_from_scenario(self.scenario)

        exp_sol_monitor_mode_1 = False  # rightright
        exp_sol_monitor_mode_2 = True  # rightright
        exp_sol_monitor_mode_3 = False  # right_straight
        exp_sol_monitor_mode_4 = True  # right_straight
        exp_sol_monitor_mode_5 = False  # left_straight
        exp_sol_monitor_mode_6 = True  # left_straight
        exp_sol_monitor_mode_7 = False  # straight_right
        exp_sol_monitor_mode_8 = True  # straight_right
        exp_sol_monitor_mode_9 = False  # right_left
        exp_sol_monitor_mode_10 = True  # right_left
        exp_sol_monitor_mode_11 = False  # straight_straight
        exp_sol_monitor_mode_12 = True  # straight_straight

        right_right = PredSamePriorityRightRight(self.config)
        right_straight = PredSamePriorityRightStraight(self.config)
        left_straight = PredSamePriorityLeftStraight(self.config)
        straight_right = PredSamePriorityStraightRight(self.config)
        right_left = PredSamePriorityRightLeft(self.config)
        straight_straight = PredSamePriorityStraightStraight(self.config)

        # k vehicle
        cr_state_list_k = {
            0: State(
                position=[40, 3],
                time_step=0,
                orientation=(1) * math.pi,
                velocity=15,
            ),
            1: State(
                position=[40, 3],
                time_step=1,
                orientation=1 * math.pi,
                velocity=15,
            ),
            2: State(
                position=[40, 3],
                time_step=2,
                orientation=1 * math.pi,
                velocity=15,
            ),
            3: State(
                position=[26.5, -10],
                time_step=3,
                orientation=0 * math.pi,
                velocity=15,
            ),
            4: State(
                position=[40, 3],
                time_step=4,
                orientation=1 * math.pi,
                velocity=15,
            ),
            5: State(
                position=[26.5, -10],
                time_step=5,
                orientation=0 * math.pi,
                velocity=15,
            ),
            6: State(
                position=[40, 3],
                time_step=6,
                orientation=1 * math.pi,
                velocity=15,
            ),
            7: State(
                position=[26.5, -10],
                time_step=7,
                orientation=0 * math.pi,
                velocity=15,
            ),
            8: State(
                position=[40, 3],
                time_step=8,
                orientation=1 * math.pi,
                velocity=15,
            ),
            9: State(
                position=[40, 3],
                time_step=9,
                orientation=1 * math.pi,
                velocity=15,
            ),
            10: State(
                position=[40, 3],
                time_step=10,
                orientation=1 * math.pi,
                velocity=15,
            ),
            11: State(
                position=[10, -0.5],
                time_step=11,
                orientation=0 * math.pi,
                velocity=15,
            ),
        }

        # p vehicle
        cr_state_list_p = {
            0: State(
                position=[23.5, 13],
                time_step=0,
                orientation=(1) * math.pi,
                velocity=15,
            ),
            1: State(
                position=[10, -0.5],
                time_step=1,
                orientation=1 * math.pi,
                velocity=15,
            ),
            2: State(
                position=[23.5, 13],
                time_step=2,
                orientation=1 * math.pi,
                velocity=15,
            ),
            3: State(
                position=[26.5, -10],
                time_step=3,
                orientation=0 * math.pi,
                velocity=15,
            ),
            4: State(
                position=[23.5, 13],
                time_step=4,
                orientation=1 * math.pi,
                velocity=15,
            ),
            5: State(
                position=[26.5, -10],
                time_step=5,
                orientation=0 * math.pi,
                velocity=15,
            ),
            6: State(
                position=[23.5, 13],
                time_step=6,
                orientation=1 * math.pi,
                velocity=15,
            ),
            7: State(
                position=[26.5, -10],
                time_step=7,
                orientation=0 * math.pi,
                velocity=15,
            ),
            8: State(
                position=[23.5, 13],
                time_step=8,
                orientation=1 * math.pi,
                velocity=15,
            ),
            9: State(
                position=[10, -0.5],
                time_step=9,
                orientation=1 * math.pi,
                velocity=15,
            ),
            10: State(
                position=[23.5, 13],
                time_step=10,
                orientation=1 * math.pi,
                velocity=15,
            ),
            11: State(
                position=[40, 3],
                time_step=11,
                orientation=0 * math.pi,
                velocity=15,
            ),
        }

        lanelet_assignments_k = {
            0: {10},
            1: {10},
            2: {10},
            3: {16},
            4: {10},
            5: {16},
            6: {10},
            7: {16},
            8: {10},
            9: {10},
            10: {10},
            11: {1},
        }

        lanelet_assignments_p = {
            0: {8},
            1: {1},
            2: {8},
            3: {16},
            4: {8},
            5: {16},
            6: {8},
            7: {16},
            8: {8},
            9: {1},
            10: {8},
            11: {10},
        }

        k_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_k,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_k,
        )

        p_vehicle = Vehicle(
            1,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_p,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_p,
        )

        # fixing lanelet assignement according to spatial occupancy
        for time, _ in p_vehicle.lanelet_assignment.items():
            shape = p_vehicle.shape
            state = p_vehicle.states_cr[time]

            p_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        for time, _ in k_vehicle.lanelet_assignment.items():
            shape = k_vehicle.shape
            state = k_vehicle.states_cr[time]

            k_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        world.add_vehicle(k_vehicle)

        world.add_vehicle(p_vehicle)

        vehicles = [0, 1]

        sol_monitor_mode_1 = right_right.evaluate_boolean(world, 0, vehicles)
        sol_robustness_monitor_mode_1 = right_right.evaluate_robustness(
            world, 0, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 >= 0)

        sol_monitor_mode_2 = right_right.evaluate_boolean(world, 1, vehicles)
        sol_robustness_monitor_mode_2 = right_right.evaluate_robustness(
            world, 1, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 >= 0)

        sol_monitor_mode_3 = right_straight.evaluate_boolean(world, 2, vehicles)
        sol_robustness_monitor_mode_3 = right_straight.evaluate_robustness(
            world, 2, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 >= 0)

        sol_monitor_mode_4 = right_straight.evaluate_boolean(world, 3, vehicles)
        sol_robustness_monitor_mode_4 = right_straight.evaluate_robustness(
            world, 3, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 >= 0)

        sol_monitor_mode_5 = left_straight.evaluate_boolean(world, 4, vehicles)
        sol_robustness_monitor_mode_5 = left_straight.evaluate_robustness(
            world, 4, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 >= 0)

        sol_monitor_mode_6 = left_straight.evaluate_boolean(world, 5, vehicles)
        sol_robustness_monitor_mode_6 = left_straight.evaluate_robustness(
            world, 5, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_6, sol_robustness_monitor_mode_6 >= 0)

        sol_monitor_mode_7 = straight_right.evaluate_boolean(world, 6, vehicles)
        sol_robustness_monitor_mode_7 = straight_right.evaluate_robustness(
            world, 6, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode_7)
        self.assertEqual(exp_sol_monitor_mode_7, sol_robustness_monitor_mode_7 >= 0)

        sol_monitor_mode_8 = straight_right.evaluate_boolean(world, 7, vehicles)
        sol_robustness_monitor_mode_8 = straight_right.evaluate_robustness(
            world, 7, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_8, sol_monitor_mode_8)
        self.assertEqual(exp_sol_monitor_mode_8, sol_robustness_monitor_mode_8 >= 0)

        sol_monitor_mode_9 = right_left.evaluate_boolean(world, 8, vehicles)
        sol_robustness_monitor_mode_9 = right_left.evaluate_robustness(
            world, 8, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_9, sol_monitor_mode_9)
        self.assertEqual(exp_sol_monitor_mode_9, sol_robustness_monitor_mode_9 >= 0)

        sol_monitor_mode_10 = right_left.evaluate_boolean(world, 9, vehicles)
        sol_robustness_monitor_mode_10 = right_left.evaluate_robustness(
            world, 9, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_10, sol_monitor_mode_10)
        self.assertEqual(exp_sol_monitor_mode_10, sol_robustness_monitor_mode_10 >= 0)

        sol_monitor_mode_11 = straight_straight.evaluate_boolean(world, 10, vehicles)
        sol_robustness_monitor_mode_11 = straight_straight.evaluate_robustness(
            world, 10, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_11, sol_monitor_mode_11)
        self.assertEqual(exp_sol_monitor_mode_11, sol_robustness_monitor_mode_11 >= 0)

        sol_monitor_mode_12 = straight_straight.evaluate_boolean(world, 11, vehicles)
        sol_robustness_monitor_mode_12 = straight_straight.evaluate_robustness(
            world, 11, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_12, sol_monitor_mode_12)
        self.assertEqual(exp_sol_monitor_mode_12, sol_robustness_monitor_mode_12 >= 0)

    def test_has_priority(self):

        world = World.create_from_scenario(self.scenario)

        exp_sol_monitor_mode_1 = True  # right_right
        exp_sol_monitor_mode_2 = False  # right_right
        exp_sol_monitor_mode_3 = True  # right_straight
        exp_sol_monitor_mode_4 = False  # right_straight
        exp_sol_monitor_mode_5 = True  # left_straight
        exp_sol_monitor_mode_6 = False  # left_straight
        exp_sol_monitor_mode_7 = True  # straight_right
        exp_sol_monitor_mode_8 = False  # straight_right
        exp_sol_monitor_mode_9 = True  # right_left
        exp_sol_monitor_mode_10 = False  # right_left
        exp_sol_monitor_mode_11 = True  # straight_straight
        exp_sol_monitor_mode_12 = False  # straight_straight

        right_right = PredHasPriorityRightRight(self.config)
        right_straight = PredHasPriorityRightStraight(self.config)
        left_straight = PredHasPriorityLeftStraight(self.config)
        straight_right = PredHasPriorityStraightRight(self.config)
        right_left = PredHasPriorityRightLeft(self.config)
        straight_straight = PredHasPriorityStraightStraight(self.config)

        # k vehicle
        cr_state_list_k = {
            0: State(
                position=[40, 3],
                time_step=0,
                orientation=(1) * math.pi,
                velocity=15,
            ),
            1: State(
                position=[40, 3],
                time_step=1,
                orientation=1 * math.pi,
                velocity=15,
            ),
            2: State(
                position=[40, 3],
                time_step=2,
                orientation=1 * math.pi,
                velocity=15,
            ),
            3: State(
                position=[26.5, -10],
                time_step=3,
                orientation=0 * math.pi,
                velocity=15,
            ),
            4: State(
                position=[40, 3],
                time_step=4,
                orientation=1 * math.pi,
                velocity=15,
            ),
            5: State(
                position=[26.5, -10],
                time_step=5,
                orientation=0 * math.pi,
                velocity=15,
            ),
            6: State(
                position=[40, 3],
                time_step=6,
                orientation=1 * math.pi,
                velocity=15,
            ),
            7: State(
                position=[26.5, -10],
                time_step=7,
                orientation=0 * math.pi,
                velocity=15,
            ),
            8: State(
                position=[40, 3],
                time_step=8,
                orientation=1 * math.pi,
                velocity=15,
            ),
            9: State(
                position=[40, 3],
                time_step=9,
                orientation=1 * math.pi,
                velocity=15,
            ),
            10: State(
                position=[40, 3],
                time_step=10,
                orientation=1 * math.pi,
                velocity=15,
            ),
            11: State(
                position=[10, -0.5],
                time_step=11,
                orientation=0 * math.pi,
                velocity=15,
            ),
        }

        # p vehicle
        cr_state_list_p = {
            0: State(
                position=[23.5, 13],
                time_step=0,
                orientation=(1) * math.pi,
                velocity=15,
            ),
            1: State(
                position=[10, -0.5],
                time_step=1,
                orientation=1 * math.pi,
                velocity=15,
            ),
            2: State(
                position=[23.5, 13],
                time_step=2,
                orientation=1 * math.pi,
                velocity=15,
            ),
            3: State(
                position=[26.5, -10],
                time_step=3,
                orientation=0 * math.pi,
                velocity=15,
            ),
            4: State(
                position=[23.5, 13],
                time_step=4,
                orientation=1 * math.pi,
                velocity=15,
            ),
            5: State(
                position=[26.5, -10],
                time_step=5,
                orientation=0 * math.pi,
                velocity=15,
            ),
            6: State(
                position=[23.5, 13],
                time_step=6,
                orientation=1 * math.pi,
                velocity=15,
            ),
            7: State(
                position=[26.5, -10],
                time_step=7,
                orientation=0 * math.pi,
                velocity=15,
            ),
            8: State(
                position=[23.5, 13],
                time_step=8,
                orientation=1 * math.pi,
                velocity=15,
            ),
            9: State(
                position=[10, -0.5],
                time_step=9,
                orientation=1 * math.pi,
                velocity=15,
            ),
            10: State(
                position=[23.5, 13],
                time_step=10,
                orientation=1 * math.pi,
                velocity=15,
            ),
            11: State(
                position=[40, 3],
                time_step=11,
                orientation=0 * math.pi,
                velocity=15,
            ),
        }

        lanelet_assignments_k = {
            0: {10},
            1: {10},
            2: {10},
            3: {16},
            4: {10},
            5: {16},
            6: {10},
            7: {16},
            8: {10},
            9: {10},
            10: {10},
            11: {1},
        }

        lanelet_assignments_p = {
            0: {8},
            1: {1},
            2: {8},
            3: {16},
            4: {8},
            5: {16},
            6: {8},
            7: {16},
            8: {8},
            9: {1},
            10: {8},
            11: {10},
        }

        k_vehicle = Vehicle(
            0,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_k,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_k,
        )

        p_vehicle = Vehicle(
            1,
            ObstacleType.CAR,
            None,
            Rectangle(5, 2),
            cr_state_list_p,
            None,
            CurvilinearStateManager(self.road_network),
            lanelet_assignments_p,
        )

        # fixing lanelet assignement according to spatial occupancy
        for time, _ in p_vehicle.lanelet_assignment.items():
            shape = p_vehicle.shape
            state = p_vehicle.states_cr[time]

            p_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        for time, _ in k_vehicle.lanelet_assignment.items():
            shape = k_vehicle.shape
            state = k_vehicle.states_cr[time]

            k_vehicle.lanelet_assignment[
                time
            ] = self.road_network.lanelet_network.find_lanelet_by_shape(
                shape.rotate_translate_local(state.position, state.orientation)
            )

        world.add_vehicle(k_vehicle)

        world.add_vehicle(p_vehicle)

        vehicles = [0, 1]

        sol_monitor_mode_1 = right_right.evaluate_boolean(world, 0, vehicles)
        sol_robustness_monitor_mode_1 = right_right.evaluate_robustness(
            world, 0, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1)
        self.assertEqual(exp_sol_monitor_mode_1, sol_robustness_monitor_mode_1 >= 0)

        sol_monitor_mode_2 = right_right.evaluate_boolean(world, 1, vehicles)
        sol_robustness_monitor_mode_2 = right_right.evaluate_robustness(
            world, 1, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2)
        self.assertEqual(exp_sol_monitor_mode_2, sol_robustness_monitor_mode_2 >= 0)

        sol_monitor_mode_3 = right_straight.evaluate_boolean(world, 2, vehicles)
        sol_robustness_monitor_mode_3 = right_straight.evaluate_robustness(
            world, 2, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3)
        self.assertEqual(exp_sol_monitor_mode_3, sol_robustness_monitor_mode_3 >= 0)

        sol_monitor_mode_4 = right_straight.evaluate_boolean(world, 3, vehicles)
        sol_robustness_monitor_mode_4 = right_straight.evaluate_robustness(
            world, 3, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4)
        self.assertEqual(exp_sol_monitor_mode_4, sol_robustness_monitor_mode_4 >= 0)

        sol_monitor_mode_5 = left_straight.evaluate_boolean(world, 4, vehicles)
        sol_robustness_monitor_mode_5 = left_straight.evaluate_robustness(
            world, 4, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5)
        self.assertEqual(exp_sol_monitor_mode_5, sol_robustness_monitor_mode_5 >= 0)

        sol_monitor_mode_6 = left_straight.evaluate_boolean(world, 5, vehicles)
        sol_robustness_monitor_mode_6 = left_straight.evaluate_robustness(
            world, 5, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_6, sol_monitor_mode_6)
        self.assertEqual(exp_sol_monitor_mode_6, sol_robustness_monitor_mode_6 >= 0)

        sol_monitor_mode_7 = straight_right.evaluate_boolean(world, 6, vehicles)
        sol_robustness_monitor_mode_7 = straight_right.evaluate_robustness(
            world, 6, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_7, sol_monitor_mode_7)
        self.assertEqual(exp_sol_monitor_mode_7, sol_robustness_monitor_mode_7 >= 0)

        sol_monitor_mode_8 = straight_right.evaluate_boolean(world, 7, vehicles)
        sol_robustness_monitor_mode_8 = straight_right.evaluate_robustness(
            world, 7, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_8, sol_monitor_mode_8)
        self.assertEqual(exp_sol_monitor_mode_8, sol_robustness_monitor_mode_8 >= 0)

        sol_monitor_mode_9 = right_left.evaluate_boolean(world, 8, vehicles)
        sol_robustness_monitor_mode_9 = right_left.evaluate_robustness(
            world, 8, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_9, sol_monitor_mode_9)
        self.assertEqual(exp_sol_monitor_mode_9, sol_robustness_monitor_mode_9 >= 0)

        sol_monitor_mode_10 = right_left.evaluate_boolean(world, 9, vehicles)
        sol_robustness_monitor_mode_10 = right_left.evaluate_robustness(
            world, 9, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_10, sol_monitor_mode_10)
        self.assertEqual(exp_sol_monitor_mode_10, sol_robustness_monitor_mode_10 >= 0)

        sol_monitor_mode_11 = straight_straight.evaluate_boolean(world, 10, vehicles)
        sol_robustness_monitor_mode_11 = straight_straight.evaluate_robustness(
            world, 10, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_11, sol_monitor_mode_11)
        self.assertEqual(exp_sol_monitor_mode_11, sol_robustness_monitor_mode_11 >= 0)

        sol_monitor_mode_12 = straight_straight.evaluate_boolean(world, 11, vehicles)
        sol_robustness_monitor_mode_12 = straight_straight.evaluate_robustness(
            world, 11, vehicles
        )
        self.assertEqual(exp_sol_monitor_mode_12, sol_monitor_mode_12)
        self.assertEqual(exp_sol_monitor_mode_12, sol_robustness_monitor_mode_12 >= 0)
