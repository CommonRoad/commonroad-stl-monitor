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
        self.simulation_param = create_simulation_param(config.get("simulation_param"), 0.5, 'DEU')
        self.other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self.ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), self.simulation_param)
        self.traffic_rule_sets = config.get("traffic_rule_sets")
        self.traffic_rules = config.get("traffic_rules")
        self.traffic_rules_param = config.get("traffic_rules_param")
        self.road_network = None  # updated in each test case

    def create_vehicle(self, obstacle: DynamicObstacle) -> Vehicle:
        lane = self.road_network.find_lane(obstacle.obstacle_id, obstacle.initial_state.time_step)
        state_lon, state_lat = lane.create_curvilinear_states(obstacle.initial_state)
        vehicle = Vehicle(state_lon, state_lat, obstacle.obstacle_shape,
                          obstacle.initial_state, obstacle.obstacle_id, obstacle.obstacle_type,
                          obstacle.initial_lanelet_ids, obstacle.initial_signal_state)

        for state in obstacle.prediction.trajectory.state_list:
            lane = self.road_network.find_lane(obstacle.obstacle_id, state.time_step)
            vehicle.append_state_cr(state, state.time_step)
            state_lon, state_lat = lane.create_curvilinear_states(state)
            vehicle.append_state_lon(state_lon, state.time_step)
            vehicle.append_state_lat(state_lat, state.time_step)
            vehicle.append_lanelet_assignment(obstacle.prediction.lanelet_assignment[state.time_step],
                                              state.time_step)
            vehicle.append_signal_state(obstacle.signal_state_at_time_step(state.time_step), state.time_step)

        return vehicle

    def test_keeps_max_speed_limit(self):
        scenario, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "DEU_A9-1_3_T-1.xml").open()
        self.simulation_param["activated_traffic_rule_sets"] = [1]
        exp_result = [(200, {'max_speed_limit': True}), (201, {'max_speed_limit': True}),
                      (202, {'max_speed_limit': True}), (203, {'max_speed_limit': True}),
                      (204, {'max_speed_limit': False}), (205, {'max_speed_limit': True})]
        result = self.execute__test(scenario)
        print(result)
        self.assertEqual(exp_result, result)

    def execute__test(self, scenario) -> List[Tuple[int, Dict[str, bool]]]:
        dispatcher = TrafficRuleDispatcher(self.traffic_rules, self.traffic_rule_sets, scenario,
                                           self.simulation_param, self.ego_vehicle_param, self.other_vehicles_param,
                                           self.traffic_rules_param)
        vehicle_evaluation = []
        self.road_network = RoadNetwork(scenario.lanelet_network)
        vehicles = {}
        for obs in scenario.dynamic_obstacles:
            vehicles[obs.obstacle_id] = self.create_vehicle(obs)

        for veh in vehicles.values():
            other_vehicles = copy.deepcopy(vehicles)
            other_vehicles.pop(veh.id)
            vehicle_evaluation.append((veh.id, dispatcher.evaluate_trajectory(veh, other_vehicles)))

        return vehicle_evaluation


if __name__ == '__main__':
    unittest.main()
