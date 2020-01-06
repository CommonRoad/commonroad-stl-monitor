from common.configuration import *
from traffic_rule_dispatcher import TrafficRuleDispatcher
from typing import List
from commonroad.scenario.trajectory import State
from commonroad.common.file_reader import CommonRoadFileReader


def main():
    # Initialization of variables for simulation
    config = load_yaml("config.yaml")
    simulation_param = config.get("simulation_param")
    other_vehicles_param = config.get("other_vehicles_param")
    ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), simulation_param)
    traffic_rules = config.get("traffic_rules")
    traffic_rules_param = config.get("traffic_rules_param")
    visualization_param = config.get("visualization")

    # Initialization of CommonRoad related variables
    scenario, planning_problem_set = \
        CommonRoadFileReader(simulation_param.get("commonroad_scenario_folder") + "/" +
                             simulation_param.get("commonroad_benchmark_id") + ".xml").open()
    #planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    #initial_ego_state = planning_problem.initial_state

    dispatcher = TrafficRuleDispatcher(traffic_rules, scenario, simulation_param, ego_vehicle_param,
                                       other_vehicles_param, traffic_rules_param)


if __name__ == "__main__":
    main()
