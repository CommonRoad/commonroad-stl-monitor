import copy

from src.common.configuration import *
from src.monitor.traffic_rule_dispatcher import TrafficRuleDispatcher
from src.common.vehicle import Vehicle
from src.common.road_network import RoadNetwork

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle


class CommonRoadObstacleEvaluation:
    def __init__(self, config_path: str):
        config = load_yaml(config_path + "config.yaml")
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

        self.num_vehicles = 0
        self.num_scenarios = 0
        self.num_veh_all_correct = 0
        self.vehicles_dict = {}
        self.eval_dict = {}

    @property
    def simulation_param(self) -> Dict:
        return self._simulation_param

    @property
    def ego_vehicle_param(self) -> Dict:
        return self._ego_vehicle_param

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
                new_vehicle = self.create_vehicle(obs)
                self.add_acceleration(new_vehicle)
                self.add_jerk(new_vehicle)
                vehicles.append(new_vehicle)
                self.vehicles_dict[obs.obstacle_id] = new_vehicle

        for idx, ego_veh in enumerate(vehicles):
            other_vehicles = copy.deepcopy(vehicles)
            other_vehicles.pop(idx)
            vehicle_evaluation.append((ego_veh.id, dispatcher.evaluate_trajectory(ego_veh, other_vehicles)))

        return vehicle_evaluation

    def add_jerk(self, vehicle: Vehicle):
        for idx, state in enumerate(vehicle.state_list_cr):
            if vehicle.states_lon[state.time_step].j is None:
                if idx + 1 < len(vehicle.state_list_cr):
                    jerk = (state.acceleration - vehicle.state_list_cr[idx - 1].acceleration) / \
                           self.simulation_param.get("dt")
                else:
                    jerk = 0
                vehicle.states_lon[state.time_step].j = jerk
        return vehicle

    def add_acceleration(self, vehicle: Vehicle):
        for idx, state in enumerate(vehicle.state_list_cr):
            if vehicle.states_lon[state.time_step].a is None:
                if idx + 1 < len(vehicle.state_list_cr):
                    acceleration = (vehicle.state_list_cr[idx + 1].velocity - state.velocity) / \
                           self.simulation_param.get("dt")
                else:
                    acceleration = 0
                vehicle.state_list_cr[idx].acceleration = acceleration
                vehicle.states_lon[state.time_step].a = acceleration
        return vehicle

    def evaluate_scenario(self, scenario: Scenario, activated_traffic_rule_set: List[str]):
        self._activated_traffic_rule_sets = activated_traffic_rule_set
        self._simulation_param["dt"] = scenario.dt
        try:
            result = self._execute_evaluation(scenario)
        except RuntimeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Runtime Error")
            return
        except AttributeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Attribute Error")
            return
        except KeyError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Key Error")
            return
        except ValueError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Value Error")
            return
        self.evaluate_result(result, scenario.benchmark_id)

        return result

    def _init_eval_dict(self, vehicle):
        eval_dict = {}
        eval_vehicle_dependent_rules = {}
        for rule_name, eval_result in vehicle[1].items():
            if "_".join(rule_name.split("_", 2)[:2]) in self._vehicle_dependent_rules \
                    and eval_vehicle_dependent_rules.get(rule_name) is None:
                eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = True
                eval_dict["_".join(rule_name.split("_", 2)[:2])] = 0
            elif eval_dict.get(rule_name) is None and "_".join(rule_name.split("_", 2)[:2]):
                eval_dict[rule_name] = 0

        return eval_dict, eval_vehicle_dependent_rules

    def evaluate_result(self, result, scenario_name):
        self.num_vehicles += len(result)
        self.num_scenarios += 1
        num_correct_rules = 0
        self.eval_dict, eval_vehicle_dependent_rules = self._init_eval_dict(result[0])
        for vehicle in result:
            out_string = "scenario: " + scenario_name + " - evaluated obs-id: " + str(vehicle[0])
            for rule_name, eval_result in vehicle[1].items():
                if "_".join(rule_name.split("_", 2)[:2]) in self._vehicle_dependent_rules:
                    if eval_result is False:
                        eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = False
                elif eval_result is True:
                    self.eval_dict[rule_name] += 1
                    num_correct_rules += 1
                    out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
                elif eval_result is False:
                    out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
            for rule_name, eval_result in eval_vehicle_dependent_rules.items():
                if eval_result is True:
                    self.eval_dict["_".join(rule_name.split("_", 2)[:2])] += 1
                    num_correct_rules += 1
                out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
            if num_correct_rules == len(self.eval_dict.keys()):
                self.num_veh_all_correct += 1
            num_correct_rules = 0
            print(out_string)