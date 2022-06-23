import logging
import os
import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.trajectory import State

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle, CurvilinearStateManager
from crmonitor.common.world_state import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.predicates.rule import parse_rule, RuleNode, PredicateNode, ExistNode, AllNode
from tests.util import parallel_lanes

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


class RuleTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = root_path / "config.yaml"
        self.config = load_yaml(str(config_path))
        rules_path = root_path / "traffic_rules_rtamt.yaml"
        self.traffic_rules = load_yaml(str(rules_path))
        self.scenario_root_path = root_path.parent / "scenarios"

    def test_single_vehicle(self):
        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(1)
        lanelet_network.add_lanelet(lanelets[0])
        road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param")
        )

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        # ego vehicle
        cr_state_list_ego = {
            0: State(position=(0, 0), orientation=0, velocity= 10, time_step=0),
            1: State(position=(10, 0), orientation=0, velocity=4 , time_step=1),
            2: State(position=(14, 0), orientation=0, velocity=10, time_step=2),
            3: State(position=(24, 0), orientation=0, velocity=5 , time_step=3),
            4: State(position=(29, 0), orientation=0, velocity=5 , time_step=4),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}}
        ego_vehicle = Vehicle(0, ObstacleType.CAR, ego_vehicle_param, Rectangle(5, 2), cr_state_list_ego, None, CurvilinearStateManager(road_network),
                              lanelet_assignments_ego)

        cr_state_list_other_1 = {
            0: State(position=(10, 0), orientation=0, velocity=2, time_step=0),
            1: State(position=(10, 0), orientation=0, velocity=2, time_step=1),
            2: State(position=(20, 0), orientation=0, velocity=2, time_step=2),
            3: State(position=(30, 0), orientation=0, velocity=2, time_step=3),
        }
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(1, ObstacleType.CAR, ego_vehicle_param, Rectangle(5, 2), cr_state_list_other_1, None,
                                CurvilinearStateManager(road_network), lanelet_assignments_other_1)

        world_state = World({ego_vehicle, other_vehicle_1}, road_network)

        rule_str = "A a1: (in_front_of__a0_a1)"
        rule = parse_rule(rule_str, {"traffic_rules_param": {}})
        rule_eval = RuleEvaluator(rule, ego_vehicle, world_state)
        rule_robustness = rule_eval.evaluate()
        preds = rule_eval.get_predicates()
        self.assertEqual(rule_robustness[4], 1.0)
        np.testing.assert_allclose(np.array(list(preds.values())), 1.0)

        rule_str = "E a1: (in_front_of__a0_a1)"
        rule = parse_rule(rule_str, {"traffic_rules_param": {}})
        rule_eval = RuleEvaluator(rule, ego_vehicle, world_state)
        rule_robustness = []
        for i in range(ego_vehicle.end_time + 1):
            rob = rule_eval.update()
            rule_robustness.append(rob)
        rule_robustness = np.array(rule_robustness)
        self.assertEqual(rule_robustness[4], -1.0)
        preds = rule_eval.get_predicates()
        np.testing.assert_allclose(np.array(list(preds.values())), -1.0)

    def test_safe_distance(self):
        # one vehicles which has no leading vehicle (1001)
        # two vehicles which violate safe distance to directly leading vehicle (1003, 1004)
        # one vehicle which violates safe distance to two leading vehicles (1002)
        # one vehicle which violates safe distance partially (1000)
        # one vehicle which always keeps safe distance (1005)
        # one vehicle which keeps safe distance to vehicle which minimally occupies lane (1006)
        # one vehicle which has no leading vehicles and drives in two lanes (1007)
        # one vehicle which leaves lane (1009)
        # one vehicle which violates safe distance to leading vehicle which leaves lane and
        #   recaptures safe distance to vehicle which enters lane (1008)
        # one vehicle which performs illegal cut-in (1010)
        all_exp_result = [
            (
                1000,
                {
                    1001: False,
                    1002: True,
                    1003: True,
                    1004: True,
                    1005: True,
                    1006: True,
                    1007: True,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1001,
                {
                    1000: True,
                    1002: True,
                    1003: True,
                    1004: True,
                    1005: True,
                    1006: True,
                    1007: True,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1002,
                {
                    1000: True,
                    1001: True,
                    1003: False,
                    1004: False,
                    1005: True,
                    1006: True,
                    1007: False,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1003,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1004: False,
                    1005: True,
                    1006: True,
                    1007: False,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1004,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1003: True,
                    1005: True,
                    1006: True,
                    1007: False,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1005,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1003: True,
                    1004: True,
                    1006: True,
                    1007: True,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1006,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1003: True,
                    1004: True,
                    1005: True,
                    1007: True,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1007,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1003: True,
                    1004: True,
                    1005: True,
                    1006: True,
                    1008: True,
                    1009: True,
                    1010: True,
                },
            ),
            (
                1008,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1003: True,
                    1004: True,
                    1005: True,
                    1006: True,
                    1007: True,
                    1009: False,
                    1010: True,
                },
            ),
            (
                1009,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1003: True,
                    1004: True,
                    1005: True,
                    1006: True,
                    1007: True,
                    1008: True,
                    1010: True,
                },
            ),
            (
                1010,
                {
                    1000: True,
                    1001: True,
                    1002: True,
                    1003: True,
                    1004: True,
                    1005: True,
                    1006: True,
                    1007: True,
                    1008: True,
                    1009: True,
                },
            ),
        ]

        exp_floating = [(ego, all(val.values())) for ego, val in all_exp_result]

        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_safe_distance.xml")
        scenario, _ = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)

        # standard robustness
        # TODO: Repair test for defined other agent
        # for ego_id, o_ids in exp_result:
        #     world_state = World.create_from_scenario(scenario, ego_id,
        #                                                   self.config)
        #     for o_id, exp_violation in o_ids.items():
        #         rob_values, _ = rule_eval.evaluate_all_rules_all_timesteps(
        #                 world_state, (o_id,))
        #         self.assertEqual(exp_violation, rob_values[0][-1][1] >= 0.0,
        #                          f"Test failed for ego_id={ego_id} and o_id={o_id}")

        for ego_id, exp_violation in exp_floating:
            world_state = World.create_from_scenario(scenario)
            ego_vehicle = world_state.vehicle_by_id(ego_id)
            rule_eval = RuleEvaluator.create_from_config(world_state, ego_vehicle, "R_G1")
            rule = rule_eval._rule
            self.assertTrue(isinstance(rule, AllNode))
            self.assertEqual(len(rule.children), 1)
            self.assertTrue(isinstance(rule.children[0], RuleNode))
            self.assertTrue(any([isinstance(c, PredicateNode) for c in rule.children[0].children]))
            rule_robustness = []
            for i in range(ego_vehicle.end_time + 1):
                rob = rule_eval.update()
                rule_robustness.append(rob)
            rule_robustness = np.array(rule_robustness)
            bool_value = rule_robustness >= 0.0
            self.assertEqual(
                exp_violation, np.all(bool_value), f"Test failed for ego_id={ego_id}"
            )

        # output robustness
        # Todo: RuleEvaluator.create_from_rule_str
        # rule_str = "A a1: ((in_front_of_i__a0_a1 and in_same_lane_i__a0_a1 ) implies keeps_safe_distance_prec__a0_a1)"
        # rule_eval = RuleEvaluator.create_from_rule_str(rule_str, output_type=Semantics.OUTPUT_ROBUSTNESS)
        # for ego_id, exp_violation in exp_floating:
        #     world_state = World.create_from_scenario(scenario, ego_id)
        #     rule_robustness, predicate_robustness = rule_eval.evaluate_incremental(
        #         world_state, to_pandas=False
        #     )
        #     rob_value = [list(rule_dict.values())[0] for rule_dict in rule_robustness.values()]
        #
        #     # create expected values
        #     exp_rob = []
        #     for time_step in range(world_state.time_step + 1):
        #         in_front_of = world_state.predicate_values[time_step]["in_front_of"]
        #         in_same_lane = world_state.predicate_values[time_step]["in_same_lane"]
        #         keeps_safe_distance_prec = world_state.predicate_values[time_step]["keeps_safe_distance_prec"]
        #
        #         all_vehicle_results = []
        #         for vehicle_pair in in_front_of.keys():
        #             if in_front_of[vehicle_pair] >= 0. and in_same_lane[vehicle_pair] >= 0:
        #                 all_vehicle_results.append(keeps_safe_distance_prec[vehicle_pair])
        #             else:
        #                 all_vehicle_results.append(np.inf)
        #
        #         exp_rob.append(np.min(all_vehicle_results))
        #
        #     self.assertEqual(
        #         exp_rob, rob_value, f"Test failed for ego_id={ego_id}"
        #     )

    def test_unnecessary_braking(self):
        # one vehicle accelerates (1000)
        # one vehicle drives with constant velocity (1001)
        # one vehicle which has no leading vehicle violates acceleration constraint (1002)
        # two leading vehicle which brake only minimal (1005, 1007)
        # one vehicle following another vehicle which brakes normal (1006)
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_unnecessary_braking.xml")
        scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        exp_result = {
            1000: True,
            1001: True,
            1002: False,
            1005: True,
            1006: True,
            1007: True,
        }
        rule_str = self.traffic_rules["traffic_rules"]["R_G2"]
        self.traffic_rules["scale_rob"] = False
        rule = parse_rule(
                rule_str,
                self.traffic_rules,
                name="UnnecessaryBraking"
        )
        self.assertTrue(isinstance(rule, RuleNode))
        self.assertEqual(len(rule.children), 2)
        self.assertTrue(any([isinstance(c, PredicateNode) for c in rule.children]))
        self.assertTrue(any([isinstance(c, ExistNode) for c in rule.children]))
        world_state = World.create_from_scenario(scenario)
        for ego_id, exp_violation in exp_result.items():
            ego_vehicle = world_state.vehicle_by_id(ego_id)
            rule_eval = RuleEvaluator(rule, ego_vehicle, world_state)
            rule_robustness = []
            for i in range(ego_vehicle.end_time + 1):
                rob = rule_eval.update()
                rule_robustness.append(rob)
            rule_robustness = np.array(rule_robustness)
            bool_value = rule_robustness >= 0.0
            self.assertEqual(exp_violation, np.all(bool_value), f"Test failed for ego_id={ego_id}")

        # output robustness
        # Todo: RuleEvaluator.create_from_rule_str
        # rule_eval = RuleSetEvaluator([rule], dt=0.1, output_type=Semantics.OUTPUT_ROBUSTNESS)
        # for ego_id, exp_violation in exp_result.items():
        #     world_state = World.create_from_scenario(scenario, ego_id)
        #     df_rule, _ = rule_eval.evaluate_incremental(
        #         world_state
        #     )
        #     rob_value = [r for r in df_rule["robustness"].values]
        #
        #     # create expected values
        #     exp_rob = []
        #     for time_step in range(world_state.time_step + 1):
        #         precedes = world_state.predicate_values[time_step]["precedes"]
        #         keeps_safe_distance_prec = world_state.predicate_values[time_step]["keeps_safe_distance_prec"]
        #         brakes_abruptly = world_state.predicate_values[time_step]["brakes_abruptly"]
        #         rel_brakes_abruptly = world_state.predicate_values[time_step]["rel_brakes_abruptly"]
        #
        #         if brakes_abruptly[(ego_id,)] < 0.:
        #             exp_rob.append(np.inf)
        #         else:
        #             all_vehicle_results = []
        #             for vehicle_pair in precedes.keys():
        #                 all_vehicle_results.append(
        #                     np.min([
        #                         precedes[vehicle_pair],
        #                         np.max([
        #                             -keeps_safe_distance_prec[vehicle_pair],
        #                             -rel_brakes_abruptly[vehicle_pair]
        #                         ])
        #                     ])
        #                 )
        #             exp_rob.append(np.max(all_vehicle_results))
        #
        #     self.assertEqual(exp_rob, rob_value, msg=f"Test failed for ego_id={ego_id}")


    def test_speed_limit(self):
        # one vehicle which always violates speed limit (1002)
        # two vehicles which never violate speed limit (1001, 1003)
        # one vehicle which violates speed limit partially (1000)
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_max_speed_limit.xml")
        scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        exp_result = {1000: False, 1001: True, 1002: False, 1003: True}

        # standard robustness
        world_state = World.create_from_scenario(scenario)
        for ego_id, exp_violation in exp_result.items():
            ego_vehicle = world_state.vehicle_by_id(ego_id)
            rule_eval = RuleEvaluator.create_from_config(world_state, ego_vehicle, "R_G3")
            rule = rule_eval._rule
            self.assertTrue(isinstance(rule, RuleNode))
            self.assertEqual(len(rule.children), 4)
            self.assertTrue(all([isinstance(c, PredicateNode) for c in rule.children]))
            rule_robustness = []
            for i in range(ego_vehicle.end_time + 1):
                rob = rule_eval.update()
                rule_robustness.append(rob)
            rule_robustness = np.array(rule_robustness)
            bool_value = rule_robustness >= 0.0
            self.assertEqual(
                exp_violation, np.all(bool_value), f"Test failed for ego_id={ego_id}"
            )


if __name__ == "__main__":
    unittest.main()
