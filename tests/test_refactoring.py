import math
import unittest
from typing import List, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.scenario.trajectory import State

from crmonitor.common.evaluation import \
    evaluate_rule
from crmonitor.common.helper import load_yaml, gather
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import StateLongitudinal, StateLateral, Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.predicate import PredCutIn
from crmonitor.predicates.python.rule import Rule
from crmonitor.monitor.rtamt_monitor_stl import TrafficRuleMonitorForwardSTL
import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)

def check_violation(rob_values: List[Tuple[float, float]]):
    bool_values = [r[1] >= 0.0 for r in rob_values]
    return all(bool_values)

class RefactoringTests(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = "crmonitor/rtamt/config.yaml"
        self.config = load_yaml(config_path)
        self.scenario_file = "scenarios/test_interstate/DEU_test_safe_distance.xml"
        self.rule_str = "always((in_front_of__a0_a1 and in_same_lane__a0_a1 and !once[0, 30](cut_in__a1_a0 and prev(not cut_in__a1_a0))) " \
                   "implies keeps_safe_distance_prec__a0_a1)"
        self.ego_id = 1005
        self.other_id = 1004

        scenario, _ = CommonRoadFileReader(self.scenario_file).open(
                lanelet_assignment=True)
        self.world_state = WorldState(scenario, self.ego_id, self.config)
        self.rule = Rule(self.rule_str, self.config)

    def test_one_rule_one_agent_offline_stepwise(self):
        assert self.rule.num_dependent_vehicles == 1
        rob_values, pred_values = evaluate_rule(self.world_state, self.other_id, self.rule)

    # def test_cut_in_pred(self):
    #     # expected solutions
    #     exp_sol_monitor_mode_1 = False  # before cut-in -> ego vehicle occupies only single lane
    #     exp_sol_monitor_mode_2 = True  # during cut-in
    #     exp_sol_monitor_mode_3 = False  # after cut-in
    #     exp_sol_monitor_mode_4 = False  # driving back to initial lane
    #     exp_sol_monitor_mode_5 = False  # during cut-in -> but other vehicles is in another lane
    #
    #     lanelet_network = LaneletNetwork()
    #     lanelet_network.add_lanelet(self._lanelet_1)
    #     lanelet_network.add_lanelet(self._lanelet_2)
    #     lanelet_network.add_lanelet(self._lanelet_3)
    #     road_network = RoadNetwork(lanelet_network, self._road_network_param)
    #     traffic_sign_interpreter = TrafficSigInterpreter(SupportedTrafficSignCountry["DEU"],
    #                                                      road_network.lanelet_network)
    #
    #     evaluator = PredCutIn(self.config)
    #
    #     # ego vehicle
    #     state_list_lon_ego = {0: StateLongitudinal(s=10, v=10), 1: StateLongitudinal(s=20, v=10),
    #                           2: StateLongitudinal(s=30, v=10), 3: StateLongitudinal(s=40, v=10)}
    #     state_list_lat_ego = {0: StateLateral(d=0, theta=0), 1: StateLateral(d=2, theta=(1/4)*math.pi),
    #                           2: StateLateral(d=4, theta=0), 3: StateLateral(d=2, theta=-(1/4)*math.pi)}
    #     cr_state_list_ego = {0: State(position=0, time_step=0), 1: State(position=10, time_step=1),
    #                          2: State(position=20, time_step=2), 3: State(position=30, time_step=3)}
    #     lanelet_assignments_ego = {0: {1}, 1: {1, 2}, 2: {1, 2}, 3: {1, 2}}
    #     ego_vehicle = Vehicle(state_list_lon_ego, state_list_lat_ego, Rectangle(5, 2), cr_state_list_ego, 0,
    #                           ObstacleType.CAR, self._ego_vehicle_param, lanelet_assignments_ego, None, None, None)
    #
    #     # other vehicle 1
    #     state_list_lon_other_1 = {0: StateLongitudinal(s=0, v=10), 1: StateLongitudinal(s=10, v=10),
    #                               2: StateLongitudinal(s=20, v=10), 3: StateLongitudinal(s=30, v=10)}
    #     state_list_lat_other_1 = {0: StateLateral(d=4, theta=0), 1: StateLateral(d=4, theta=0),
    #                               2: StateLateral(d=4, theta=0), 3: StateLateral(d=4, theta=0)}
    #     cr_state_list_other_1 = {0: State(position=10, time_step=0), 1: State(position=20, time_step=1),
    #                              2: State(position=30, time_step=2), 3: State(position=40, time_step=3)}
    #     lanelet_assignments_other_1 = {0: {2}, 1: {2}, 2: {2}, 3: {2}}
    #     other_vehicle_1 = Vehicle(state_list_lon_other_1, state_list_lat_other_1, Rectangle(5, 2),
    #                               cr_state_list_other_1, 0, ObstacleType.CAR, self._ego_vehicle_param,
    #                               lanelet_assignments_other_1, None, None, None)
    #
    #     # other vehicle 2
    #     state_list_lon_other_2 = {1: StateLongitudinal(s=0, v=10)}
    #     state_list_lat_other_2 = {1: StateLateral(d=10, theta=0)}
    #     cr_state_list_other_2 = {1: State(position=10, time_step=1)}
    #     lanelet_assignments_other_2 = {1: {3}}
    #     other_vehicle_2 = Vehicle(state_list_lon_other_2, state_list_lat_other_2, Rectangle(5, 2),
    #                               cr_state_list_other_2, 0, ObstacleType.CAR, self._ego_vehicle_param,
    #                               lanelet_assignments_other_2, None, None, None)
    #     world_state = WorldState()
    #     # Monitor-Mode
    #     sol_monitor_mode_1 = evaluator.evaluate_robustness(0, [ego_vehicle.id, other_vehicle_1.id])
    #     sol_monitor_mode_2 = general_predicates.cut_in(1, ego_vehicle, other_vehicle_1, OperatingMode.MONITOR)
    #     sol_monitor_mode_3 = general_predicates.cut_in(2, ego_vehicle, other_vehicle_1, OperatingMode.MONITOR)
    #     sol_monitor_mode_4 = general_predicates.cut_in(3, ego_vehicle, other_vehicle_1, OperatingMode.MONITOR)
    #     sol_monitor_mode_5 = evaluator.evaluate_robustness(0, [ego_vehicle.id, other_vehicle_2.id])
    #
    #     self.assertEqual(exp_sol_monitor_mode_1, sol_monitor_mode_1 >= 0.0)
    #     self.assertEqual(exp_sol_monitor_mode_2, sol_monitor_mode_2 >= 0.0)
    #     self.assertEqual(exp_sol_monitor_mode_3, sol_monitor_mode_3 >= 0.0)
    #     self.assertEqual(exp_sol_monitor_mode_4, sol_monitor_mode_4 >= 0.0)
    #     self.assertEqual(exp_sol_monitor_mode_5, sol_monitor_mode_5 >= 0.0)

    def test_result(self):
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
        exp_result = [(1000, {1001: False, 1002: True, 1003: True,
                              1004: True, 1005: True, 1006: True,
                              1007: True, 1008: True, 1009: True,
                              1010: True}),
                      (1001, {1000: True, 1002: True, 1003: True,
                              1004: True, 1005: True, 1006: True,
                              1007: True, 1008: True, 1009: True,
                              1010: True}),
                      (1002, {1000: True, 1001: True, 1003: False,
                              1004: False, 1005: True, 1006: True,
                              1007: False, 1008: True, 1009: True,
                              1010: True}),
                      (1003, {1000: True, 1001: True, 1002: True,
                              1004: False, 1005: True, 1006: True,
                              1007: False, 1008: True, 1009: True,
                              1010: True}),
                      (1004, {1000: True, 1001: True, 1002: True,
                              1003: True, 1005: True, 1006: True,
                              1007: False, 1008: True, 1009: True,
                              1010: True}),
                      (1005, {1000: True, 1001: True, 1002: True,
                              1003: True, 1004: True, 1006: True,
                              1007: True, 1008: True, 1009: True,
                              1010: True}),
                      (1006, {1000: True, 1001: True, 1002: True,
                              1003: True, 1004: True, 1005: True,
                              1007: True, 1008: True, 1009: True,
                              1010: True}),
                      (1007, {1000: True, 1001: True, 1002: True,
                              1003: True, 1004: True, 1005: True,
                              1006: True, 1008: True, 1009: True,
                              1010: True}),
                      (1008, {1000: True, 1001: True, 1002: True,
                              1003: True, 1004: True, 1005: True,
                              1006: True, 1007: True, 1009: False,
                              1010: True}),
                      (1009, {1000: True, 1001: True, 1002: True,
                              1003: True, 1004: True, 1005: True,
                              1006: True, 1007: True, 1008: True,
                              1010: True}),
                      (1010, {1000: True, 1001: True, 1002: True,
                              1003: True, 1004: True, 1005: True,
                              1006: True, 1007: True, 1008: True,
                              1009: True})
                      ]
        for ego_id, o_ids in exp_result:
            world_state = WorldState(self.world_state.scenario, ego_id, self.config)
            for o_id, violation in o_ids.items():
                rob_values, pred_values = evaluate_rule(world_state, o_id, self.rule)
                self.assertEqual(violation, rob_values[-1][1] >= 0.0, f"Test failed for ego_id={ego_id} and o_id={o_id}")


if __name__ == "__main__":
    unittest.main()
