import traceback

from src.common.helper import *
from src.monitor.traffic_rule_dispatcher import TrafficRuleDispatcher
from src.common.vehicle import Vehicle, VehicleClassification
from src.common.road_network import RoadNetwork, Lane
from src.common.vehicle import StateLongitudinal, StateLateral

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.lanelet import Lanelet, LaneletType


class CommonRoadObstacleEvaluation:
    """Class for the traffic rule evaluation of CommonRoad scenarios"""
    def __init__(self, config_path: str):
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        self._simulation_param = create_simulation_param(config.get("simulation_param"), 0.1, 'DEU')
        self._other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        self._traffic_rules_param = traffic_rules.get("traffic_rules_param")
        self._ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), self._simulation_param,
                                                           self._traffic_rules_param)
        self._traffic_rule_sets = traffic_rules.get("traffic_rule_sets")
        self._traffic_rules_forward = traffic_rules.get("traffic_rules_forward")
        self._traffic_rules_backward = traffic_rules.get("traffic_rules_backward")
        self._activated_traffic_rule_sets = traffic_rules.get("activated_traffic_rule_sets")
        self._vehicle_dependent_rules = traffic_rules.get("vehicle_dependent_rules")
        self._road_network_param = config.get("road_network_param")
        self._road_network: RoadNetwork  # updated in each test case

        self.num_vehicles = 0
        self.num_scenarios = 0
        self.num_veh_all_correct = 0
        self.vehicles_dict = {}
        self.eval_dict, self.eval_vehicle_dependent_rules = self._init_eval_dict()

    @property
    def simulation_param(self) -> Dict:
        return self._simulation_param

    @property
    def ego_vehicle_param(self) -> Dict:
        return self._ego_vehicle_param

    @property
    def activated_traffic_rule_sets(self) -> List[str]:
        return self._activated_traffic_rule_sets

    @activated_traffic_rule_sets.setter
    def activated_traffic_rule_sets(self, activated_traffic_rule_sets: List[str]):
        self._activated_traffic_rule_sets = activated_traffic_rule_sets

    def create_vehicle(self, obstacle: DynamicObstacle, vehicle_param: Dict, ego_vehicle: Vehicle = None) -> Vehicle:
        """
        Transforms a CommonRoad obstacle to a vehicle object

        :param obstacle: CommonRoad obstacle
        :param vehicle_param: dictionary with vehicle parameters
        :param ego_vehicle: ego vehicle object (if it exist already) for reference generation
        :return: vehicle object
        """
        acceleration = self._compute_acceleration(obstacle.initial_state.velocity,
                                                  obstacle.prediction.trajectory.state_list[0].velocity)
        jerk = self._compute_jerk(acceleration, 0)
        if ego_vehicle is None:
            vehicle_classification = VehicleClassification.EGO_VEHICLE
            initial_lanelets = [self._road_network.lanelet_network.find_lanelet_by_id(lanelet_id)
                                for lanelet_id in obstacle.initial_shape_lanelet_ids]
            if LaneletType.ACCESS_RAMP in initial_lanelets[0].lanelet_type:
                main_carriage_way_lanelet_id = self._find_main_carriage_way_lanelet_id(initial_lanelets[0])
                lane = self._road_network.find_lane_by_obstacle([main_carriage_way_lanelet_id], [])
            else:
                lane = self._road_network.find_lane_by_obstacle(list(obstacle.initial_center_lanelet_ids),
                                                                list(obstacle.initial_shape_lanelet_ids))
            reference_lane = lane
        elif self._adjacent_to_ego(list(ego_vehicle.lanelet_assignment[ego_vehicle.state_list_cr[0].time_step])[0],
                                   list(obstacle.initial_shape_lanelet_ids)[0]):
            vehicle_classification = VehicleClassification.ADJACENT_VEHICLE
            lane = self._road_network.find_lane_by_obstacle(list(obstacle.initial_center_lanelet_ids),
                                                            list(obstacle.initial_shape_lanelet_ids))
            reference_lane = ego_vehicle.lane
        else:
            vehicle_classification = VehicleClassification.CROSSING_VEHICLE
            lane = self._road_network.find_lane_by_obstacle(list(obstacle.initial_center_lanelet_ids),
                                                            list(obstacle.initial_shape_lanelet_ids))
            reference_lane = ego_vehicle.lane
        state_lon, state_lat = \
            CommonRoadObstacleEvaluation.create_curvilinear_states(obstacle.initial_state.position,
                                                                   obstacle.initial_state.velocity, acceleration, jerk,
                                                                   obstacle.initial_state.orientation,
                                                                   reference_lane)
        vehicle = None
        if state_lon is not None or state_lat is not None:
            vehicle = Vehicle(state_lon, state_lat, obstacle.obstacle_shape,
                              obstacle.initial_state, obstacle.obstacle_id, obstacle.obstacle_type,
                              obstacle.initial_shape_lanelet_ids, obstacle.initial_signal_state, vehicle_classification,
                              lane, vehicle_param)

        for state in obstacle.prediction.trajectory.state_list:
            acceleration = self._compute_acceleration(state_lon.v, state.velocity)
            jerk = self._compute_jerk(acceleration, 0)

            state_lon, state_lat = CommonRoadObstacleEvaluation.create_curvilinear_states(state.position,
                                                                                          state.velocity,
                                                                                          acceleration, jerk,
                                                                                          state.orientation,
                                                                                          reference_lane)
            if state_lon is None or state_lat is None:
                continue
            vehicle.append_time_step(state.time_step, state_lon, state_lat, state,
                                     obstacle.prediction.shape_lanelet_assignment[state.time_step],
                                     obstacle.signal_state_at_time_step(state.time_step))
        return vehicle

    def _adjacent_to_ego(self, ego_lanelet_id: int, obs_lanelet_id: int) -> bool:
        """
        Evaluates if a vehicle is in a to the ego vehicle adjacent lane

        :param ego_lanelet_id: IDs of lanelets the ego vehicle is on
        :param obs_lanelet_id: IDs of lanelets the other vehicle is on
        :return: boolean indicating if the vehicle is in an adjacent lane
        """
        adjacent_lanelet_ids = {ego_lanelet_id}
        ego_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
        current_lanelet = ego_lanelet
        while current_lanelet.adj_left_same_direction is not None:
            current_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(current_lanelet.adj_left)
            adjacent_lanelet_ids.add(current_lanelet.lanelet_id)
        while current_lanelet.adj_right_same_direction is not None:
            current_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(current_lanelet.adj_right)
            adjacent_lanelet_ids.add(current_lanelet.lanelet_id)
        for lanelet_id in list(adjacent_lanelet_ids):
            lane = self._road_network.find_lane_by_lanelet(lanelet_id)
            if obs_lanelet_id in lane.contained_lanelets:
                return True
        return False

    def _find_main_carriage_way_lanelet_id(self, lanelet: Lanelet) -> int:
        """
        Searches for an adjacent lanelet part of the main carriageway

        :param lanelet: start lanelet
        :return: ID of a lanelet which is part of the main carriageway
        """
        current_lanelet = lanelet
        if LaneletType.MAIN_CARRIAGE_WAY in current_lanelet.lanelet_type:
            return current_lanelet.lanelet_id
        while current_lanelet.adj_left_same_direction is not None:
            current_lanelet = self._road_network.lanelet_network.find_lanelet_by_id(current_lanelet.adj_left)
            if LaneletType.MAIN_CARRIAGE_WAY in current_lanelet.lanelet_type:
                return current_lanelet.lanelet_id

    def _execute_evaluation(self, scenario: Scenario) -> List[Tuple[int, Dict[str, bool]]]:
        """
        Traffic rule evaluation of each vehicle in a CommonRoad scenario

        :param scenario: CommonRoad scenario
        :return: evaluation results for each vehicle
        """
        self._road_network = RoadNetwork(scenario.lanelet_network, self._road_network_param)
        dispatcher = TrafficRuleDispatcher(self._traffic_rules_forward, self._traffic_rules_backward,
                                           self._traffic_rule_sets, self._road_network,
                                           self._simulation_param, self._traffic_rules_param,
                                           self._activated_traffic_rule_sets, self._vehicle_dependent_rules)
        vehicle_evaluation = []
        for ego in scenario.dynamic_obstacles:
            other_vehicles = []
            if not (self.simulation_param.get("operating_mode") == "test"
                    or self.simulation_param.get("operating_mode") == "evaluation"
                    or self.simulation_param.get("operating_mode") == "single_scenario"
                    or (self.simulation_param.get("operating_mode") == "single_vehicle"
                        and self.simulation_param.get("ego_vehicle_id") != ego.obstacle_id)):
                continue
            if self.simulation_param.get("operating_mode") == "evaluation" \
                    and self.simulation_param.get("num_vehicles") <= self.num_vehicles + len(vehicle_evaluation):
                break
            if ego.prediction is not None:
                ego_vehicle = self.create_vehicle(ego, self.ego_vehicle_param)
                for obs in scenario.dynamic_obstacles:
                    if obs.obstacle_id == ego.obstacle_id or obs.prediction is None:
                        continue
                    vehicle = self.create_vehicle(obs, self._other_vehicles_param, ego_vehicle)
                    other_vehicles.append(vehicle)
                vehicle_evaluation.append((ego_vehicle.id, dispatcher.evaluate_trajectory(ego_vehicle, other_vehicles)))

        return vehicle_evaluation

    def _compute_jerk(self, current_acceleration: float, previous_acceleration: float) -> float:
        """
        Computes jerk given acceleration

        :param current_acceleration: acceleration of current time step
        :param previous_acceleration: acceleration of previous time step
        :return: jerk
        """
        jerk = (current_acceleration - previous_acceleration) / self.simulation_param.get("dt")
        return jerk

    def _compute_acceleration(self, previous_velocity: float, current_velocity: float):
        """
        Computes acceleration given velocity

        :param current_velocity: velocity of current time step
        :param previous_velocity: velocity of previous time step
        :return: acceleration
        """
        acceleration = (current_velocity - previous_velocity) / self.simulation_param.get("dt")
        return acceleration

    def evaluate_scenario(self, scenario: Scenario) \
            -> Union[List[Tuple[int, Dict[str, bool]]], None]:
        """
        Evaluates CommonRoad scenario

        :param scenario: CommonRoad scenario
        :return: evaluation results
        """
        self._simulation_param["dt"] = scenario.dt
        try:
            result = self._execute_evaluation(scenario)
        except RuntimeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Runtime Error")
            traceback.print_exc()
            return
        except AttributeError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Attribute Error")
            traceback.print_exc()
            return
        except KeyError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Key Error")
            traceback.print_exc()
            return
        except ValueError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Value Error")
            traceback.print_exc()
            return
        except IndexError:
            print("scenario ", scenario.benchmark_id, " could not be evaluated: Index Error")
            traceback.print_exc()
            return
        self.evaluate_result(result, scenario.benchmark_id)

        return result

    def update_eval_dict(self):
        self.eval_dict, self.eval_vehicle_dependent_rules = self._init_eval_dict()

    def _init_eval_dict(self) -> Tuple[Dict[str, int], Dict[str, bool]]:
        """
        Prepares and creates evaluation dictionaries

        :param vehicle_result: result of single vehicle
        :return: evaluation results
        """
        eval_dict = {}
        eval_vehicle_dependent_rules = {}
        for traffic_rule_set_id in self._activated_traffic_rule_sets:
            for rule_name in self._traffic_rule_sets.get(traffic_rule_set_id):
                if "_".join(rule_name.split("_", 2)[:2]) in self._vehicle_dependent_rules \
                        and eval_vehicle_dependent_rules.get(rule_name) is None:
                    eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = True
                    eval_dict["_".join(rule_name.split("_", 2)[:2])] = 0
                elif eval_dict.get(rule_name) is None and "_".join(rule_name.split("_", 2)[:2]):
                    eval_dict[rule_name] = 0

        return eval_dict, eval_vehicle_dependent_rules

    def evaluate_result(self, result, scenario_name):
        """
        Statistical evaluation of results

        :param result: evaluation results
        :param scenario_name: CommonRoad scenario name
        """
        self.num_vehicles += len(result)
        self.num_scenarios += 1
        num_correct_rules = 0
        for vehicle in result:
            out_string = "scenario: " + scenario_name + " - evaluated obs-id: " + str(vehicle[0])
            for rule_name, eval_result in vehicle[1].items():
                if "_".join(rule_name.split("_", 2)[:2]) in self._vehicle_dependent_rules:
                    if eval_result is False:
                        self.eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = False
                elif eval_result is True:
                    self.eval_dict[rule_name] += 1
                    num_correct_rules += 1
                    out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
                elif eval_result is False:
                    out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
            for rule_name, eval_result in self.eval_vehicle_dependent_rules.items():
                if eval_result is True:
                    self.eval_dict["_".join(rule_name.split("_", 2)[:2])] += 1
                    num_correct_rules += 1
                out_string += " - evaluation of rule " + rule_name + ": " + str(eval_result)
                self.eval_vehicle_dependent_rules["_".join(rule_name.split("_", 2)[:2])] = True
            if num_correct_rules == len(self.eval_dict.keys()):
                self.num_veh_all_correct += 1
            num_correct_rules = 0
            print(out_string)

    @staticmethod
    def create_curvilinear_states(position: List[float], velocity: float,
                                  acceleration: float, jerk: float, orientation: float, lane: Lane) \
            -> Union[Tuple[StateLongitudinal, StateLateral], Tuple[None, None]]:
        """
        Computes initial state of ego vehicle

        :param position: position of vehicle in cartesian coordinates
        :param velocity: velocity of vehicle
        :param acceleration: acceleration of vehicle
        :param jerk: jerk of vehicle
        :param orientation: orientation of vehicle
        :param lane: reference lane of the vehicle
        :return: lateral and longitudinal state of vehicle
        """
        try:
            s, d = lane.clcs.convert_to_curvilinear_coords(position[0], position[1])
        except ValueError:
            print("Vehicle out of projection domain: State will not be considered")
            return None, None
        theta_cl = lane.orientation(s)
        if acceleration is not None and jerk is not None:
            x_lon = StateLongitudinal(s=s, v=velocity, a=acceleration, j=jerk)
        elif acceleration is not None:
            x_lon = StateLongitudinal(s=s, v=velocity, a=acceleration)
        else:
            x_lon = StateLongitudinal(s=s, v=velocity)
        x_lat = StateLateral(d=d, theta=(theta_cl - orientation))

        return x_lon, x_lat
