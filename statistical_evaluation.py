from common.configuration import *
from monitor.traffic_rule_dispatcher import TrafficRuleDispatcher
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import DynamicObstacle
from common.vehicle import Vehicle
from common.road_network import RoadNetwork
from commonroad.scenario.scenario import Scenario
from typing import List, Dict, Tuple
import copy
import os


class CommonRoadObstacleEvaluation:
    def __init__(self):
        config = load_yaml("./config.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 0.1, 'DEU')
        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._traffic_rules_param = config.get("traffic_rule_monitoring").get("traffic_rules_param")
        self._ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), self._simulation_param,
                                                          self._traffic_rules_param)
        self._traffic_rule_sets = config.get("traffic_rule_monitoring").get("traffic_rule_sets")
        self._traffic_rules = config.get("traffic_rule_monitoring").get("traffic_rules")
        self._activated_traffic_rule_sets = config.get("traffic_rule_monitoring").get("activated_traffic_rule_sets")
        self._vehicle_dependent_rules = config.get("traffic_rule_monitoring").get("vehicle_dependent_rules")
        self._road_network_param = config.get("road_network_param")
        self._road_network = None  # updated in each test case

        self.max_speed_limit_satisfaction = 0
        self.min_speed_limit_satisfaction = 0
        self.safe_distance_satisfaction = 0
        self.no_unnecessary_braking_satisfaction = 0
        self.num_vehicles = 0
        self.num_scenarios = 0
        self.num_veh_all_correct = 0

    def create_vehicle(self, obstacle: DynamicObstacle) -> Vehicle:
        lane = self._road_network.find_lane_by_obstacle(list(obstacle.initial_center_lanelet_ids),
                                                        list(obstacle.initial_shape_lanelet_ids))
        state_lon, state_lat = lane.create_curvilinear_states(obstacle.initial_state)
        vehicle = None
        if state_lon is not None or state_lat is not None:
            vehicle = Vehicle(state_lon, state_lat, obstacle.obstacle_shape,
                              obstacle.initial_state, obstacle.obstacle_id, obstacle.obstacle_type,
                              obstacle.initial_shape_lanelet_ids, obstacle.initial_signal_state)

        for state in obstacle.prediction.trajectory.state_list:
            lane = self._road_network.find_lane_by_obstacle(
                list(obstacle.prediction.center_lanelet_assignment[state.time_step]),
                list(obstacle.prediction.shape_lanelet_assignment[state.time_step]))
            state_lon, state_lat = lane.create_curvilinear_states(state)
            if state_lon is None or state_lat is None:
                continue
            if vehicle is not None:
                vehicle.append_state_cr(state, state.time_step)
                vehicle.append_state_lon(state_lon, state.time_step)
                vehicle.append_state_lat(state_lat, state.time_step)
                vehicle.append_lanelet_assignment(obstacle.prediction.shape_lanelet_assignment[state.time_step],
                                              state.time_step)
                vehicle.append_signal_state(obstacle.signal_state_at_time_step(state.time_step), state.time_step)
            else:
                vehicle = Vehicle(state_lon, state_lat, obstacle.obstacle_shape,
                                  state, obstacle.obstacle_id, obstacle.obstacle_type,
                                  obstacle.prediction.shape_lanelet_assignment[state.time_step],
                                  obstacle.signal_state_at_time_step(state.time_step))

        return vehicle

    def _execute_evaluation(self, scenario) -> List[Tuple[int, Dict[str, bool]]]:
        self._road_network = RoadNetwork(scenario.lanelet_network, self._road_network_param)
        dispatcher = TrafficRuleDispatcher(self._traffic_rules, self._traffic_rule_sets, self._road_network,
                                           self._simulation_param, self._ego_vehicle_param, self._other_vehicles_param,
                                           self._traffic_rules_param, self._activated_traffic_rule_sets,
                                           self._vehicle_dependent_rules)
        vehicle_evaluation = []

        vehicles = []
        for obs in scenario.dynamic_obstacles:
            if obs.prediction is not None:
                vehicles.append(self.create_vehicle(obs))

        for idx, ego_veh in enumerate(vehicles):
            other_vehicles = copy.deepcopy(vehicles)
            other_vehicles.pop(idx)
            self.add_acceleration(ego_veh, self._simulation_param.get("dt"))
            self.add_jerk(ego_veh, self._simulation_param.get("dt"))
            vehicle_evaluation.append((ego_veh.id, dispatcher.evaluate_trajectory(ego_veh, other_vehicles)))

        return vehicle_evaluation

    @staticmethod
    def add_jerk(vehicle: Vehicle, dt: float):
        for idx, state in enumerate(vehicle.state_list_cr):
            if vehicle.jerk_profile.get(state.time_step) is None:
                if idx + 1 < len(vehicle.state_list_cr):
                    jerk = (vehicle.state_list_cr[idx + 1].acceleration - state.acceleration) / dt
                else:
                    jerk = 0
                vehicle.append_jerk(jerk, state.time_step)
        return vehicle

    @staticmethod
    def add_acceleration(vehicle: Vehicle, dt: float):
        for idx, state in enumerate(vehicle.state_list_cr):
            if hasattr(state, "acceleration") is False:
                if idx + 1 < len(vehicle.state_list_cr):
                    acceleration = (vehicle.state_list_cr[idx + 1].velocity - state.velocity) / dt
                else:
                    acceleration = 0
                vehicle.state_list_cr[idx].acceleration = acceleration
                vehicle.states_lon[state.time_step].a = acceleration
        return vehicle

    def evaluate_scenario(self, scenario: Scenario, activated_traffic_rule_set: List[int]):
        self._activated_traffic_rule_sets = activated_traffic_rule_set
        try:
            result = self._execute_evaluation(scenario)
        except RuntimeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Runtime Error")
            return
        except AttributeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Attribute Error")
            return
 #       except KeyError:
#            print("scenario ", scenario.benchmark_id, " could not be evaluated: Key Error")
 #           return
        except ValueError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Value Error")
            return
        self.evaluate_result(result)

    def evaluate_result(self, result):
        self.num_vehicles += len(result)
        self.num_scenarios += 1
        num_correct_rules = 0
        for vehicle in result:
            safe_distance_complete = True
            for rule_name, eval_result in vehicle[1].items():
                if rule_name == 'max_speed_limit':
                    if eval_result is True:
                        self.max_speed_limit_satisfaction += 1
                        num_correct_rules += 1
                elif rule_name == 'min_speed_limit':
                    if eval_result is True:
                        self.min_speed_limit_satisfaction += 1
                        num_correct_rules += 1
                elif rule_name == 'no_unnecessary_braking':
                    if eval_result is True:
                        self.no_unnecessary_braking_satisfaction += 1
                        num_correct_rules += 1
                elif 'safe_distance' in rule_name :
                    if eval_result is False:
                        safe_distance_complete = False
            if safe_distance_complete is True:
                self.safe_distance_satisfaction += 1
                num_correct_rules += 1
            if num_correct_rules == 4:
                self.num_veh_all_correct += 1
            num_correct_rules = 0


def main():
    cr_eval = CommonRoadObstacleEvaluation()
    scenarios = []
    root_dir = "./../../../commonroad/scenarios/tum_cps/scenarios"

    for subdir, dirs, files in os.walk(root_dir):
        for directory in dirs:
            if directory == "cooperative":
                continue
            for filename in os.listdir(subdir + "/" + directory):
                if not "DEU" in filename:
                    continue
                if "Stu" in filename:
                    continue
                if not filename.endswith('.xml') or "_S-" in filename:
                    continue
                fullname = os.path.join(subdir + "/" + directory, filename)
                scenario, planning_problem_set = \
                    CommonRoadFileReader(fullname).open()
                if "highway" in scenario.tags:
                    scenarios.append(scenario)

    for sc in scenarios:
        cr_eval.evaluate_scenario(sc, [0])

    print("number scenarios:" + str(cr_eval.num_scenarios))
    print("number vehicles: " + str(cr_eval.num_vehicles))
    print("max. speed limit compliance: " + str(cr_eval.max_speed_limit_satisfaction))
    print("min. speed limit compliance: " + str(cr_eval.min_speed_limit_satisfaction))
    print("no. unnecessary braking compliance: " + str(cr_eval.no_unnecessary_braking_satisfaction))
    print("safe distance compliance: " + str(cr_eval.safe_distance_satisfaction))
    print("perfect vehicles: " + str(cr_eval.num_veh_all_correct))


if __name__ == "__main__":
    main()
