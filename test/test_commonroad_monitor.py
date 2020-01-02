import unittest
from common.configuration import *
from traffic_rule_dispatcher import TrafficRuleDispatcher
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import DynamicObstacle
from common.vehicle import Vehicle
from common.road_network import RoadNetwork
from typing import List, Dict


class TestCommonRoadMonitor(unittest.TestCase):
    def setUp(self):
        config = load_yaml("./../config.yaml")
        self.simulation_param = create_simulation_param(config.get("simulation_param"), 0.5, 'DEU')
        self.other_vehicles_param = config.get("other_vehicles_param")
        self.ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), self.simulation_param)
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

    def test_keeps_speed_limit(self):
        scenario1, planning_problem_set = \
            CommonRoadFileReader("./../" + self.simulation_param.get("commonroad_scenario_folder") +
                                 "/" + "DEU_A9-1_3_T-1.xml").open()

        exp_result1 = [{'speed_limit': True}, {'speed_limit': True}, {'speed_limit': True}, {'speed_limit': True},
                       {'speed_limit': False}, {'speed_limit': True}]
        result1 = self.execute_velocity_test(scenario1)
        self.assertEqual(exp_result1, result1)


    def execute_velocity_test(self, scenario) -> List[Dict[str, bool]]:
        vehicle_evaluation = []
        self.road_network = RoadNetwork(scenario.lanelet_network)
        vehicles = []
        for obs in scenario.dynamic_obstacles:
            vehicles.append(self.create_vehicle(obs))
        dispatcher = TrafficRuleDispatcher(self.traffic_rules, scenario, self.simulation_param, self.ego_vehicle_param,
                                           self.other_vehicles_param, self.traffic_rules_param)
        for veh in vehicles:
            vehicle_evaluation.append(dispatcher.evaluate_trajectory(veh))

        return vehicle_evaluation


if __name__ == '__main__':
    unittest.main()
