import logging
import os
import unittest
from typing import List, Tuple
import numpy as np

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.trajectory import State

from crmonitor.evaluation.evaluation import RuleSetEvaluator
from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import StateLongitudinal, StateLateral, Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.rule import parse_rule, RuleNode, PredicateNode, \
    ExistNode, AllNode
from crmonitor.tests.util import parallel_lanes

from rtamt.enumerations.options import Semantics

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def check_violation(rob_values: List[Tuple[float, float]]):
    bool_values = [r[1] >= 0.0 for r in rob_values]
    return all(bool_values)


class RuleTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        config_path = os.path.join(root_path, "config.yaml")
        self.config = load_yaml(config_path)
        rules_path = os.path.join(root_path, "traffic_rules_rtamt.yaml")
        self.traffic_rules = load_yaml(rules_path)
        self.scenario_root_path = os.path.join(root_path, "../scenarios")

    def test_single_vehicle(self):
        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(1)
        lanelet_network.add_lanelet(lanelets[0])
        road_network = RoadNetwork(
            lanelet_network, self.config.get("road_network_param")
        )

        ego_vehicle_param = self.config.get("ego_vehicle_param")

        # ego vehicle
        state_list_lon_ego = {
            0: StateLongitudinal(s=0, v=10),
            1: StateLongitudinal(s=10, v=4),
            2: StateLongitudinal(s=14, v=10),
            3: StateLongitudinal(s=24, v=5),
            4: StateLongitudinal(s=29, v=5),
        }
        state_list_lat_ego = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0, theta=0),
            2: StateLateral(d=0, theta=0),
            3: StateLateral(d=0, theta=0),
            4: StateLateral(d=0, theta=0),
        }
        cr_state_list_ego = {
            0: State(position=0, time_step=0),
            1: State(position=10, time_step=1),
            2: State(position=14, time_step=2),
            3: State(position=24, time_step=3),
            4: State(position=29, time_step=4),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}}
        ego_vehicle = Vehicle(
            state_list_lon_ego,
            state_list_lat_ego,
            Rectangle(5, 2),
            cr_state_list_ego,
            0,
            ObstacleType.CAR,
            ego_vehicle_param,
            lanelet_assignments_ego,
            None,
            None,
            None,
        )

        state_list_lon_other_1 = {
            0: StateLongitudinal(s=8, v=2),
            1: StateLongitudinal(s=10, v=2),
            2: StateLongitudinal(s=12, v=2),
            3: StateLongitudinal(s=14, v=2),
        }
        state_list_lat_other_1 = {
            0: StateLateral(d=0, theta=0),
            1: StateLateral(d=0, theta=0),
            2: StateLateral(d=0, theta=0),
            3: StateLateral(d=0, theta=0),
        }
        cr_state_list_other_1 = {
            0: State(position=10, time_step=1),
            1: State(position=10, time_step=1),
            2: State(position=20, time_step=2),
            3: State(position=30, time_step=3),
        }
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(
            state_list_lon_other_1,
            state_list_lat_other_1,
            Rectangle(5, 2),
            cr_state_list_other_1,
            41,
            ObstacleType.CAR,
            ego_vehicle_param,
            lanelet_assignments_other_1,
            None,
            None,
            None,
        )

        world_state = WorldState(ego_vehicle, [other_vehicle_1], road_network, ego_vehicle.end_time)

        rule_str = "A a1: (in_front_of__a0_a1)"
        rule = parse_rule(rule_str, {"traffic_rules_param": {}})
        rule_eval = RuleSetEvaluator([rule], dt=0.1)
        rob, preds = rule_eval.evaluate_incremental(world_state)
        rob = rob[rob.time_step == 4]["robustness"].values[0]
        preds = preds[preds.time_step == 4]["robustness"]
        self.assertEqual(rob, 1.0)
        self.assertTrue(np.all(preds.values == 1.0))

        rule_str = "E a1: (in_front_of__a0_a1)"
        rule = parse_rule(rule_str, {"traffic_rules_param": {}})
        rule_eval = RuleSetEvaluator([rule], dt=0.1)
        rob, preds = rule_eval.evaluate_incremental(world_state)
        rob = rob[rob.time_step == 4]["robustness"].values[0]
        preds = preds[preds.time_step == 4]["robustness"]
        self.assertEqual(rob, -1.0)
        self.assertTrue(np.all(preds.values == -1.0))

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
        rule_eval = RuleSetEvaluator.create_from_config(["R_G1"])
        rule = rule_eval.rules[0]
        self.assertTrue(isinstance(rule, AllNode))
        self.assertEqual(len(rule.children), 1)
        self.assertTrue(isinstance(rule.children[0], RuleNode))
        self.assertTrue(any([isinstance(c, PredicateNode) for c in rule.children[0].children]))
        # TODO: Repair test for defined other agent
        # for ego_id, o_ids in exp_result:
        #     world_state = WorldState.create_from_scenario(scenario, ego_id,
        #                                                   self.config)
        #     for o_id, exp_violation in o_ids.items():
        #         rob_values, _ = rule_eval.evaluate_all_rules_all_timesteps(
        #                 world_state, (o_id,))
        #         self.assertEqual(exp_violation, rob_values[0][-1][1] >= 0.0,
        #                          f"Test failed for ego_id={ego_id} and o_id={o_id}")

        for ego_id, exp_violation in exp_floating:
            world_state = WorldState.create_from_scenario(scenario, ego_id)
            df_rule, _ = rule_eval.evaluate_incremental(
                world_state
            )
            rob_value = all([r >= 0.0 for r in df_rule["robustness"].values])
            self.assertEqual(
                exp_violation, rob_value, f"Test failed for ego_id={ego_id}"
            )

        # output robustness
        rule_str = "A ((in_front_of_i__a0_a1>=0 and in_same_lane_i__a0_a1>=0 ) implies keeps_safe_distance_prec__a0_a1>=0)"
        rule = Rule(rule_str, {"traffic_rules_param": {}})
        rule_eval = RuleSetEvaluator([rule], output_type=Semantics.OUTPUT_ROBUSTNESS)
        for ego_id, exp_violation in exp_floating:
            world_state = WorldState.create_from_scenario(scenario, ego_id)
            df_rule, _ = rule_eval.evaluate_incremental(
                world_state
            )
            rob_value = [r for r in df_rule["robustness"].values]

            # create expected values
            exp_rob = []
            for time_step in range(world_state.time_step + 1):
                in_front_of = world_state.predicate_values[time_step]["in_front_of_i"]
                in_same_lane = world_state.predicate_values[time_step]["in_same_lane_i"]
                keeps_safe_distance_prec = world_state.predicate_values[time_step]["keeps_safe_distance_prec"]

                all_vehicle_results = []
                for vehicle_pair in in_front_of.keys():
                    if in_front_of[vehicle_pair] >= 0. and in_same_lane[vehicle_pair] >= 0:
                        all_vehicle_results.append(keeps_safe_distance_prec[vehicle_pair])
                    else:
                        all_vehicle_results.append(np.inf)

                exp_rob.append(np.min(all_vehicle_results))

            self.assertEqual(
                exp_rob, rob_value, f"Test failed for ego_id={ego_id}"
            )

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
        rule_eval = RuleSetEvaluator([rule], dt=0.1)
        for ego_id, exp_violation in exp_result.items():
            world_state = WorldState.create_from_scenario(scenario, ego_id, self.config)
            df_rule, _ = rule_eval.evaluate_incremental(
                world_state
            )
            rob_value = all([r >= 0.0 for r in df_rule["robustness"].values])
            self.assertEqual(
                exp_violation, rob_value, f"Test failed for ego_id={ego_id}"
            )

        # output robustness
        rule_str = "E ((accel_i__a0<=-2.) implies (precedes__a0_a1>=0.) and (not keeps_safe_distance_prec__a0_a1>=0 or accel__a0 - accel__a1 > -2.0))"
        rule = Rule(rule_str, {"traffic_rules_param": {}})
        rule_eval = RuleSetEvaluator([rule], output_type=Semantics.OUTPUT_ROBUSTNESS)
        for ego_id, exp_violation in exp_result.items():
            world_state = WorldState.create_from_scenario(scenario, ego_id)
            df_rule, _ = rule_eval.evaluate_incremental(
                world_state
            )
            rob_value = [r for r in df_rule["robustness"].values]

            # create expected values
            exp_rob = []
            for time_step in range(world_state.time_step + 1):
                precedes = world_state.predicate_values[time_step]["precedes"]
                keeps_safe_distance_prec = world_state.predicate_values[time_step]["keeps_safe_distance_prec"]
                accel = world_state.predicate_values[time_step]["accel"]

                all_vehicle_results = []
                ego_accel = world_state.predicate_values[time_step]["accel_i"][(ego_id, )]
                for vehicle_pair in precedes.keys():
                    other_id = vehicle_pair[1]
                    if ego_accel <= -2.:
                        # all_vehicle_results.append(precedes[vehicle_pair])
                        all_vehicle_results.append(
                            np.min([precedes[vehicle_pair],
                                   np.max([keeps_safe_distance_prec[vehicle_pair],
                                           ego_accel - accel[(other_id, )] + 2.])])
                        )
                    else:
                        all_vehicle_results.append(np.inf)

                exp_rob.append(np.max(all_vehicle_results))

            self.assertEqual(exp_rob, rob_value, msg=f"Test failed for ego_id={ego_id}")
            # for i, (e, r) in enumerate(zip(exp_rob, rob_value)):
            #     self.assertAlmostEqual(e, r, delta=1e-5, msg=f"Test failed for ego_id={ego_id}, item {i}")


    def test_speed_limit(self):
        # one vehicle which always violates speed limit (1002)
        # two vehicles which never violate speed limit (1001, 1003)
        # one vehicle which violates speed limit partially (1000)
        scenario_file = os.path.join(self.scenario_root_path, "test_interstate/DEU_test_max_speed_limit.xml")
        scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open(lanelet_assignment=True)
        exp_result = {1000: False, 1001: True, 1002: False, 1003: True}

        # standard robustness
        rule_eval = RuleSetEvaluator.create_from_config("R_G3")
        rule = rule_eval.rules[0]
        self.assertTrue(isinstance(rule, RuleNode))
        self.assertEqual(len(rule.children), 4)
        self.assertTrue(all([isinstance(c, PredicateNode) for c in rule.children]))
        for ego_id, exp_violation in exp_result.items():
            world_state = WorldState.create_from_scenario(scenario, ego_id)
            df_rule, _ = rule_eval.evaluate_incremental(
                world_state
            )
            rob_value = all([r >= 0.0 for r in df_rule["robustness"].values])
            self.assertEqual(
                exp_violation, rob_value, f"Test failed for ego_id={ego_id}"
            )


if __name__ == "__main__":
    unittest.main()
