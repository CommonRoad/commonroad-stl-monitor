import os
import unittest

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from crmonitor.common.helper import *
from crmonitor.common.road_network import RoadNetwork
from crmonitor.predicates.braking_predicates import BrakingPredicateCollection


class TestInterfaces(unittest.TestCase):
    def setUp(self):
        config_path = os.path.dirname(__file__) + "/../crmonitor/"
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 0.1, 'DEU')
        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._ego_vehicle_param = create_other_vehicles_param(config.get("ego_vehicle_param"))
        self._traffic_rule_param = traffic_rules.get("traffic_rules_param")
        self._road_network_param = config.get("road_network_param")
        self.test_scenario_dir = os.path.dirname(__file__) + "/../scenarios/test_interstate/"

    def test_operating_mode_monitor(self):
        exp_sol = {1001: {5: True, 6: False, 7: False, 8: False, 9: False, 10: False},
                   1002: {5: True, 6: True, 7: True, 8: True, 9: True, 10: True},
                   1003: {5: False, 6: True, 7: True, 8: True, 9: True, 10: True},
                   1004: {5: False, 6: False, 7: False, 8: False, 9: False, 10: False},
                   1005: {5: True, 6: True, 7: True, 8: True, 9: True, 10: True},
                   1006: {5: False, 6: False, 7: False, 8: False, 9: False, 10: False},
                   1007: {5: False, 6: False, 7: False, 8: False, 9: False, 10: False},
                   1008: {5: True, 6: True, 7: True, 8: True, 9: True, 10: True},
                   1009: {5: False, 6: False, 7: False, 8: False, 9: False, 10: False},
                   1010: {5: False, 6: False, 7: False, 8: False, 9: False, 10: False}}

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir + "DEU_test_safe_distance.xml"). \
            open(lanelet_assignment=True)
        ego_id = 1000
        road_network = RoadNetwork(scenario.lanelet_network, self._road_network_param)
        necessary_predicates = {"keeps_safe_distance_prec__x_ego__x_o"}

        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        braking_predicates = BrakingPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        necessary_predicates, traffic_sign_interpreter)
        ego_vehicle, other_vehicles = create_scenario_vehicles(self._simulation_param.get("dt"),
                                                               scenario.obstacle_by_id(ego_id),
                                                               self._ego_vehicle_param, self._other_vehicles_param,
                                                               road_network, scenario.dynamic_obstacles)
        time_interval = (5, 10)
        sol = braking_predicates.evaluate_predicates(ego_vehicle, other_vehicles, time_interval,
                                                     OperatingMode.MONITOR).get("keeps_safe_distance_prec__x_ego__x_o")
        self.assertEqual(exp_sol, sol)

    def test_operating_mode_constraint(self):
        exp_sol = {1001: {5: 48.28485714285713, 6: 48.28485714285713, 7: 47.43085714285715,
                          8: 47.43085714285715, 9: 46.58085714285714, 10: 46.58085714285714},
                   1002: {5: 48.28485714285713, 6: 48.28485714285713, 7: 47.43085714285715,
                          8: 47.43085714285715, 9: 46.58085714285714, 10: 46.58085714285714},
                   1003: {5: 48.28485714285713, 6: 48.28485714285713, 7: 47.43085714285715,
                          8: 47.43085714285715, 9: 46.58085714285714, 10: 46.58085714285714},
                   1004: {5: 72.09438095238093, 6: 72.09438095238093, 7: 71.24038095238095,
                          8: 71.24038095238095, 9: 70.39038095238094, 10: 70.39038095238094},
                   1005: {5: 72.09438095238093, 6: 72.09438095238093, 7: 71.24038095238095,
                          8: 71.24038095238095, 9: 70.39038095238094, 10: 70.39038095238094},
                   1006: {5: 72.09438095238093, 6: 72.09438095238093, 7: 71.24038095238095,
                          8: 71.24038095238095, 9: 70.39038095238094, 10: 70.39038095238094},
                   1007: {5: 72.09438095238093, 6: 72.09438095238093, 7: 71.24038095238095,
                          8: 71.24038095238095, 9: 70.39038095238094, 10: 70.39038095238094},
                   1008: {5: 48.28485714285713, 6: 48.28485714285713, 7: 47.43085714285715,
                          8: 47.43085714285715, 9: 46.58085714285714, 10: 46.58085714285714},
                   1009: {5: 72.09438095238093, 6: 72.09438095238093, 7: 71.24038095238095,
                          8: 71.24038095238095, 9: 70.39038095238094, 10: 70.39038095238094},
                   1010: {5: 72.09438095238093, 6: 72.09438095238093, 7: 71.24038095238095,
                          8: 71.24038095238095, 9: 70.39038095238094, 10: 70.39038095238094}}

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir + "DEU_test_safe_distance.xml"). \
            open(lanelet_assignment=True)
        ego_id = 1000
        road_network = RoadNetwork(scenario.lanelet_network, self._road_network_param)
        necessary_predicates = {"keeps_safe_distance_prec__x_ego__x_o"}
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        braking_predicates = BrakingPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        necessary_predicates, traffic_sign_interpreter)
        ego_vehicle, other_vehicles = create_scenario_vehicles(self._simulation_param.get("dt"),
                                                               scenario.obstacle_by_id(ego_id),
                                                               self._ego_vehicle_param, self._other_vehicles_param,
                                                               road_network, scenario.dynamic_obstacles)

        time_interval = (5, 10)
        sol = braking_predicates.evaluate_predicates(ego_vehicle, other_vehicles, time_interval,
                                                     OperatingMode.CONSTRAINT) \
            .get("keeps_safe_distance_prec__x_ego__x_o")
        for veh_id, pred in sol.items():
            for time, constraint in pred.items():
                sol[veh_id][time] = constraint.value
        self.assertEqual(exp_sol, sol)

    def test_operating_robustness(self):
        exp_sol = {1001: {5: 0.7371428571428638, 6: -0.2428571428571331, 7: -0.3588571428571541,
                          8: -1.318857142857155, 9: -1.4188571428571422, 10: -2.358857142857147},
                   1002: {5: -57.76285714285713, 6: -58.74285714285713, 7: -58.85885714285715,
                          8: -59.818857142857155, 9: -59.918857142857135, 10: -60.85885714285715},
                   1003: {5: -47.76285714285713, 6: -48.74285714285713, 7: -48.85885714285715,
                          8: -49.81885714285715, 9: -49.918857142857135, 10: -50.85885714285714},
                   1004: {5: -14.57238095238094, 6: -16.552380952380936, 7: -17.66838095238095,
                          8: -19.62838095238095, 9: -20.728380952380938, 10: -22.668380952380943},
                   1005: {5: -86.57238095238093, 6: -88.55238095238093, 7: -89.66838095238094,
                          8: -91.62838095238095, 9: -92.72838095238093, 10: -94.66838095238094},
                   1006: {5: -49.572380952380925, 6: -51.55238095238093, 7: -52.66838095238094,
                          8: -54.628380952380944, 9: -55.72838095238093, 10: -57.668380952380936},
                   1007: {5: -4.572380952380939, 6: -6.552380952380929, 7: -7.66838095238095,
                          8: -9.628380952380951, 9: -10.728380952380938, 10: -12.668380952380943},
                   1008: {5: -57.76285714285713, 6: -58.74285714285713, 7: -58.85885714285715,
                          8: -59.818857142857155, 9: -59.918857142857135, 10: -60.85885714285715},
                   1009: {5: -59.936150078483166, 6: -62.06719625929312, 7: -63.347047824343505,
                          8: -65.47637217073142, 9: -66.72225190700121, 10: -68.76517903452503},
                   1010: {5: -44.200661384158295, 6: -46.05900924707709, 7: -47.064428648017355,
                          8: -48.93297281434592, 9: -49.97625938216209, 10: -51.896304205777206}}

        scenario, planning_problem_set = CommonRoadFileReader(self.test_scenario_dir + "DEU_test_safe_distance.xml"). \
            open(lanelet_assignment=True)
        ego_id = 1000
        road_network = RoadNetwork(scenario.lanelet_network, self._road_network_param)
        necessary_predicates = {"keeps_safe_distance_prec__x_ego__x_o"}
        traffic_sign_interpreter = TrafficSigInterpreter(self._simulation_param.get("country"),
                                                         road_network.lanelet_network)
        braking_predicates = BrakingPredicateCollection(road_network, self._simulation_param, self._traffic_rule_param,
                                                        necessary_predicates, traffic_sign_interpreter)

        ego_vehicle, other_vehicles = create_scenario_vehicles(self._simulation_param.get("dt"),
                                                               scenario.obstacle_by_id(ego_id),
                                                               self._ego_vehicle_param, self._other_vehicles_param,
                                                               road_network, scenario.dynamic_obstacles)

        time_interval = (5, 10)
        sol = braking_predicates.evaluate_predicates(ego_vehicle, other_vehicles, time_interval,
                                                     OperatingMode.ROBUSTNESS). \
            get("keeps_safe_distance_prec__x_ego__x_o")
        self.assertEqual(exp_sol, sol)
