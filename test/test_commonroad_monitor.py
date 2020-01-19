import unittest
from common.configuration import *
from monitor.traffic_rule_dispatcher import TrafficRuleDispatcher
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import DynamicObstacle
from common.vehicle import Vehicle
from common.road_network import RoadNetwork
from typing import List, Dict, Tuple
import copy


class TestCommonRoadMonitor(unittest.TestCase):
    def setUp(self):
        config = load_yaml("./../config.yaml")
        self.simulation_param = create_simulation_param(config.get("simulation_param"), 0.1, 'DEU')
        self.other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self.traffic_rules_param = config.get("traffic_rule_monitoring").get("traffic_rules_param")
        self.ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), self.simulation_param,
                                                          self.traffic_rules_param)
        self.traffic_rule_sets = config.get("traffic_rule_monitoring").get("traffic_rule_sets")
        self.traffic_rules = config.get("traffic_rule_monitoring").get("traffic_rules")
        self.activated_traffic_rule_sets = config.get("traffic_rule_monitoring").get("activated_traffic_rule_sets")
        self.vehicle_dependent_rules = config.get("traffic_rule_monitoring").get("vehicle_dependent_rules")
        self.road_network_param = config.get("road_network_param")
        self.road_network = None  # updated in each test case

    def create_vehicle(self, obstacle: DynamicObstacle) -> Vehicle:
        lane = self.road_network.find_lane_by_obstacle(obstacle.obstacle_id, obstacle.initial_state.time_step)
        state_lon, state_lat = lane.create_curvilinear_states(obstacle.initial_state)
        vehicle = Vehicle(state_lon, state_lat, obstacle.obstacle_shape,
                          obstacle.initial_state, obstacle.obstacle_id, obstacle.obstacle_type,
                          obstacle.initial_lanelet_ids, obstacle.initial_signal_state)

        for state in obstacle.prediction.trajectory.state_list:
            lane = self.road_network.find_lane_by_obstacle(obstacle.obstacle_id, state.time_step)
            vehicle.append_state_cr(state, state.time_step)
            state_lon, state_lat = lane.create_curvilinear_states(state)
            vehicle.append_state_lon(state_lon, state.time_step)
            vehicle.append_state_lat(state_lat, state.time_step)
            vehicle.append_lanelet_assignment(obstacle.prediction.lanelet_assignment[state.time_step],
                                              state.time_step)
            vehicle.append_signal_state(obstacle.signal_state_at_time_step(state.time_step), state.time_step)

        return vehicle

    def test_keeps_max_lane_speed_limit(self):
        # one vehicle which always violates speed limit (1002)
        # two vehicles which never violate speed limit (1001, 1003)
        # one vehicle which violates speed limit partially (1000)

        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_max_speed_limit.xml").open()
        self.activated_traffic_rule_sets = [1]
        exp_result = [(1000, {'max_speed_limit': False}), (1001, {'max_speed_limit': True}),
                      (1002, {'max_speed_limit': False}), (1003, {'max_speed_limit': True})]
        result = self.execute_test(scenario)
        print("Max Lane Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_fov_speed_limit(self):
        # two vehicles which always violate speed limit (1001, 1002)
        # one vehicle which never violates speed limit (1003)
        # one vehicle which violates speed limit partially (1000)

        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_max_speed_limit.xml").open()
        self.activated_traffic_rule_sets = [1]
        exp_result = [(1000, {'max_speed_limit': False}), (1001, {'max_speed_limit': False}),
                      (1002, {'max_speed_limit': False}), (1003, {'max_speed_limit': True})]
        self.ego_vehicle_param["fov_speed_limit"] = 32
        result = self.execute_test(scenario)
        print("Max FOV Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_braking_speed_limit(self):
        # two vehicles which always violate speed limit (1001, 1002)
        # one vehicle which never violates speed limit (1003)
        # one vehicle which violates speed limit partially (1000)

        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_max_speed_limit.xml").open()
        self.activated_traffic_rule_sets = [1]
        exp_result = [(1000, {'max_speed_limit': False}), (1001, {'max_speed_limit': False}),
                      (1002, {'max_speed_limit': False}), (1003, {'max_speed_limit': True})]
        self.ego_vehicle_param["fov_speed_limit"] = 32
        result = self.execute_test(scenario)
        print("Max Braking Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_min_speed_limit(self):
        # two lanes with minimum speed limit sign
        # one vehicle which keeps minimum speed limit based on sign (1006)
        # one vehicle which violates minimum speed limit based on sign (1007)
        # two vehicles which preserves traffic flow (1001 ,1004)
        # two vehicles without following vehicle (1000, 1002)
        # one vehicle which does not preserve traffic flow with leading and following vehicle (1003)
        # one vehicle which drives to alone and slow on single lane -> according rule false (1005)

        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_min_speed_limit.xml").open()
        self.activated_traffic_rule_sets = [2]
        exp_result = [(1000, {'min_speed_limit': True}), (1001, {'min_speed_limit': True}),
                      (1002, {'min_speed_limit': True}), (1003, {'min_speed_limit': False}),
                      (1004, {'min_speed_limit': True}), (1005, {'min_speed_limit': False}),
                      (1006, {'min_speed_limit': False}), (1007, {'min_speed_limit': True})]
        result = self.execute_test(scenario)
        print("Min Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_safe_distance(self):
        # three vehicles which have no leading vehicle (1001, 1004, 1006)
        # one vehicle which violates safe distance to directly leading vehicle (1003)
        # one vehicle which violates safe distance to two leading vehicles (1002)
        # one vehicle which violates safe distance partially (1000)
        # one vehicle which always keeps safe distance (1005)
        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_safe_distance.xml").open()
        self.activated_traffic_rule_sets = [3]
        exp_result = [(1000, {'safe_distance_veh_1001': False}), (1001, {'safe_distance': True}),
                      (1002, {'safe_distance_veh_1003': False, 'safe_distance_veh_1004': False}),
                      (1003, {'safe_distance_veh_1004': False}), (1004, {'safe_distance': True}),
                      (1005, {'safe_distance_veh_1006': True}), (1006, {'safe_distance': True})]
        result = self.execute_test(scenario)
        print("Safe Distance Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_unnecessary_braking_1(self):
        # one vehicle accelerates (1000)
        # one vehicle drives with constant velocity (1001)
        # two leading vehicle which brake only minimal (1005, 1007)
        # one vehicle following another vehicle which brakes unnecessary strong (1004)
        # one vehicle following another vehicle which brakes normal (1006)
        # one vehicle which has no leading vehicle violates acceleration constraint (1002)
        # one vehicle which has no leading vehicle violates jerk constraint (1003)
        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_unnecessary_braking_1.xml").open()
        self.activated_traffic_rule_sets = [4]
        exp_result = [(1000, {'no_unnecessary_braking': True}), (1001, {'no_unnecessary_braking': True}),
                      (1002, {'no_unnecessary_braking': False}), (1003, {'no_unnecessary_braking': False}),
                      (1004, {'no_unnecessary_braking': False}), (1005, {'no_unnecessary_braking': True}),
                      (1006, {'no_unnecessary_braking': True}), (1007, {'no_unnecessary_braking': True})]
        result = self.execute_test(scenario)
        print("Unnecessary Braking Test 1:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_unnecessary_braking_2(self):
        # one vehicle which without leading vehicle which brakes very strong, because it is necessary (1000)
        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_unnecessary_braking_2.xml").open()
        self.activated_traffic_rule_sets = [4]
        self.ego_vehicle_param["fov_speed_limit"] = 5
        exp_result = [(1000, {'no_unnecessary_braking': True})]
        result = self.execute_test(scenario)
        print("Unnecessary Braking Test 2:")
        print(result)
        self.assertEqual(exp_result, result)

    def add_jerk(self, vehicle: Vehicle):
        for idx, state in enumerate(vehicle.state_list_cr):
            if vehicle.jerk_profile.get(state.time_step) is None:
                if idx + 1 < len(vehicle.state_list_cr):
                    jerk = (vehicle.state_list_cr[idx + 1].acceleration - state.acceleration) / \
                           self.simulation_param.get("dt")
                else:
                    jerk = 0
                vehicle.append_jerk(jerk, state.time_step)
        return vehicle

    def add_acceleration(self, vehicle: Vehicle):
        for idx, state in enumerate(vehicle.state_list_cr):
            if hasattr(state, "acceleration") is False:
                if idx + 1 < len(vehicle.state_list_cr):
                    acceleration = (vehicle.state_list_cr[idx + 1].velocity - state.velocity) / \
                           self.simulation_param.get("dt")
                else:
                    acceleration = 0
                vehicle.state_list_cr[idx].acceleration = acceleration
                vehicle.states_lon[state.time_step].a = acceleration
        return vehicle

    def execute_test(self, scenario) -> List[Tuple[int, Dict[str, bool]]]:
        self.road_network = RoadNetwork(scenario.lanelet_network, self.road_network_param)
        dispatcher = TrafficRuleDispatcher(self.traffic_rules, self.traffic_rule_sets, self.road_network,
                                           self.simulation_param, self.ego_vehicle_param, self.other_vehicles_param,
                                           self.traffic_rules_param, self.activated_traffic_rule_sets,
                                           self.vehicle_dependent_rules)
        vehicle_evaluation = []

        vehicles = []
        for obs in scenario.dynamic_obstacles:
            vehicles.append(self.create_vehicle(obs))

        for idx, ego_veh in enumerate(vehicles):
            other_vehicles = copy.deepcopy(vehicles)
            other_vehicles.pop(idx)
            self.add_acceleration(ego_veh)
            self.add_jerk(ego_veh)
            vehicle_evaluation.append((ego_veh.id, dispatcher.evaluate_trajectory(ego_veh, other_vehicles)))

        return vehicle_evaluation


if __name__ == '__main__':
    unittest.main()
