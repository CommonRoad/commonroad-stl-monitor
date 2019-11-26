import mtl
from predicates import *
from predicates import Predicate


class Monitor:
    def __init__(self, traffic_rules: Dict[str, str], predicates_per_mtl: Dict[str, List[str]], simulation_param: Dict,
                 ego_vehicle_param: Dict, other_vehicles_param: Dict, traffic_rule_param: Dict,
                 scenario: Scenario, initial_state: State):
        """
        :param traffic_rules: dictionary with MTL formulas of traffic rules
        :param predicates_per_mtl: dictionary with predicates for each MTL formula
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param traffic_rule_param: dictionary with parameters of traffic rule parameters
        :param scenario: CommonRoad scenario
        """
        self.predicate = Predicate(simulation_param, ego_vehicle_param, other_vehicles_param, scenario, initial_state)
        self._rules = self.create_monitors(traffic_rules)
        self._predicates_per_mtl = predicates_per_mtl
        self._simulation_param = simulation_param
        self._ego_vehicle_param = ego_vehicle_param
        self._other_vehicles_param = other_vehicles_param
        self._traffic_rule_param = traffic_rule_param
        self._scenario = scenario
        self._ego_lanelet_id, self._ego_lanelet, self._ego_lane, self._curvilinear_cosy_ego_lane, self._left_lane, \
        self._right_lane = update_ego_lane_info(scenario, initial_state, self._ego_vehicle_param)

    @staticmethod
    def create_monitors(mtl_rules: Dict[str, str]) -> Dict[str, str]:
        """
        Initialization of monitor for each MTL rule

        :param mtl_rules: dictionary with MTL formulas
        :returns list of monitors
        """
        monitors = {}
        for key, value in mtl_rules.items():
            monitor = mtl.parse(value)
            monitors[key] = monitor
        return monitors

    def evaluate_predicates(self, predicate: str, state: State) -> bool:
        if predicate == "keeps_speed_limit":
            return self.predicate.keeps_speed_limit(state.velocity, self._ego_lane.speed_limit)
        if predicate == "keeps_safe_distance":
            return self.predicate.keeps_safe_distance(state,
                                                      self._ego_lane.dynamic_obstacles_on_lanelet[state.time_step])
        if predicate == "brakes_abruptly":
            return self.predicate.brakes_abruptly(state, self._ego_lane.dynamic_obstacles_on_lanelet[state.time_step],
                                                  self._traffic_rule_param.get("j_min_abrupt"),
                                                  self._traffic_rule_param.get("delta_a_abrupt"))

    @staticmethod
    def convert_to_curvilinear(obstacle_states_cr: List[State], curvilinear_coord_system):
        clc_state_list = []
        for state in obstacle_states_cr:
            s, d = curvilinear_coord_system.convert_to_curvilinear_coords(state.position[0], state.position[1])
            clc_state_list.append([s, d])

        return clc_state_list

    @staticmethod
    def add_jerk(state_list: List[State]):
        for idx, state in enumerate(state_list):
            if idx == 0:
                state.jerk = 0
            else:
                state.jerk = state_list[idx].acceleration - state_list[idx - 1].acceleration

        return state_list

    def evaluate_trajectory(self, trajectory: List[State]):
        """
        :param trajectory: ego vehicle trajectory
        """
        trajectory = self.add_jerk(trajectory)

        data = {}
        for state in trajectory:
            for key, value in self._predicates_per_mtl.items():
                for predicate in value:
                    if data.get(predicate) is None:
                        data[predicate] = []
                    data[predicate].append((state.time_step, self.evaluate_predicates(predicate, state)))

        for key, rule in self._rules.items():
            predicate_list = self._predicates_per_mtl.get(key)
            formula_predicates = {}
            for predicate in predicate_list:
                formula_predicates[predicate] = data[predicate]
            print(rule(formula_predicates, quantitative=False))
