from configuration import *
from monitor import Monitor
from commonroad.common.file_reader import CommonRoadFileReader
import cProfile


def main():

    # Initialization of variables for simulation
    config = load_yaml("config.yaml")
    simulation_param = config.get("simulation_param")
    other_vehicles_param = config.get("other_vehicles_param")
    ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), simulation_param)
    traffic_rules = config.get("traffic_rules")
    traffic_rule_param = config.get("traffic_rule_param")
    predicates = config.get("predicates")

    scenario, planning_problem_set = \
        CommonRoadFileReader("./scenarios/" + simulation_param.get("commonroad_benchmark_id") + ".xml").open()

    # Create ego vehicle trajectory
    ego_obstacle = scenario.obstacle_by_id(ego_vehicle_param.get("vehicle_id"))
    scenario.remove_obstacle(ego_obstacle)

    if ego_obstacle.initial_state.time_step != ego_obstacle.prediction.trajectory.state_list[0].time_step:
        trajectory = [ego_obstacle.initial_state] + ego_obstacle.prediction.trajectory.state_list
    else:
        trajectory = ego_obstacle.prediction.trajectory.state_list
    for lanelet in scenario.lanelet_network.lanelets:
        lanelet.convert_to_polygon()

    # Monitor ego vehicle trajectory
    monitor = Monitor(traffic_rules, predicates, simulation_param, ego_vehicle_param, other_vehicles_param,
                      traffic_rule_param, scenario, ego_obstacle.initial_state)
    print("length of traj.: " + str(len(trajectory)))
    if simulation_param.get("time_measuring"):
        pr = cProfile.Profile()
        pr.enable()
    monitor.evaluate_trajectory(trajectory)
    if simulation_param.get("time_measuring"):
        pr.disable()
        pr.print_stats(sort='time')


if __name__ == "__main__":
    main()

