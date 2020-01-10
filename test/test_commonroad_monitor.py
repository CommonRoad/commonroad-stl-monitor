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
        self.ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), self.simulation_param)
        self.traffic_rule_sets = config.get("traffic_rule_monitoring").get("traffic_rule_sets")
        self.traffic_rules = config.get("traffic_rule_monitoring").get("traffic_rules")
        self.traffic_rules_param = config.get("traffic_rule_monitoring").get("traffic_rules_param")
        self.activated_traffic_rule_sets = config.get("traffic_rule_monitoring").get("activated_traffic_rule_sets")
        self.vehicle_dependent_rules = config.get("traffic_rule_monitoring").get("vehicle_dependent_rules")
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
        # trajectory which always violates speed limit
        # two trajectories which never violate speed limit
        # trajectory which violates speed limit partially
        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_max_speed_limit.xml").open()
        self.activated_traffic_rule_sets = [1]
        exp_result = [(100, {'max_speed_limit': False}), (101, {'max_speed_limit': True}),
                      (102, {'max_speed_limit': False}), (103, {'max_speed_limit': True})]
        result = self.execute_test(scenario)
        print("Max Lane Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_fov_speed_limit(self):
        # two trajectories which always violate speed limit
        # trajectory which never violates speed limit
        # trajectory which violates speed limit partially
        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_max_speed_limit.xml").open()
        self.activated_traffic_rule_sets = [1]
        exp_result = [(100, {'max_speed_limit': False}), (101, {'max_speed_limit': False}),
                      (102, {'max_speed_limit': False}), (103, {'max_speed_limit': True})]
        self.ego_vehicle_param["fov"] = 100
        self.ego_vehicle_param = create_ego_vehicle_param(self.ego_vehicle_param, self.simulation_param)
        result = self.execute_test(scenario)
        print("Max FOV Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    def test_keeps_min_speed_limit(self):
        # three vehicles without leading vehicle
        # one vehicle which drives to slow compared to second leading vehicle -> todo
        # one vehicle which drives to slow compared to directly leading vehicle -> todo
        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "test_min_speed_limit.xml").open()
        self.activated_traffic_rule_sets = [2]
        exp_result = [(100, {'min_speed_limit': False}), (101, {'min_speed_limit': True}),
                      (102, {'min_speed_limit': False}), (103, {'min_speed_limit': True}),
                      (104, {'min_speed_limit': True}), (105, {'min_speed_limit': True}),
                      (106, {'min_speed_limit': True})]
        result = self.execute_test(scenario)
        print("Min Speed Limit Test:")
        print(result)
        self.assertEqual(exp_result, result)

    # def test_keeps_safe_distance(self):
    #     scenario, planning_problem_set = \
    #         CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
    #                              "/" + "DEU_A9-3_1_T-1.xml").open()
    #     self.activated_traffic_rule_sets = [3]
    #     exp_result = [(200, {'safe_distance': True}), (201, {'safe_distance': True}),
    #                   (202, {'safe_distance_veh_201': True}), (203, {'safe_distance': True}),
    #                   (204, {'safe_distance': True}), (205, {'safe_distance': True})]
    #     result = self.execute_test(scenario)
    #     print("Safe Distance Test:")
    #     print(result)
    #     self.assertEqual(exp_result, result)

    # def test_brakes_abruptly(self):
    #     scenario, planning_problem_set = \
    #         CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
    #                              "/" + "DEU_A9-1_3_T-1.xml").open()
    #     self.activated_traffic_rule_sets = [4]
    #     exp_result = [(200, {'no_abrupt_braking': True}), (201, {'no_abrupt_braking': True}),
    #                   (202, {'no_abrupt_braking': True}), (203, {'no_abrupt_braking': True}),
    #                   (204, {'no_abrupt_braking': True}), (205, {'no_abrupt_braking': True})]
    #     result = self.execute_test(scenario)
    #     print("Brakes Abruptly Test:")
    #     print(result)
    #     self.assertEqual(exp_result, result)

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
        self.road_network = RoadNetwork(scenario.lanelet_network)
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
