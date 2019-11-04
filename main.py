from configuration import *
from simple_monitor import SimpleMonitor
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from vehicle import Vehicle
# from util import create_curvilinear_states


def main():
    # Initialization of variables for simulation
    config = load_yaml("config.yaml")
    simulation_param = config.get("simulation_param")
    other_vehicles_param = config.get("other_vehicles_param")
    ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), simulation_param)
    traffic_rules = config.get("traffic_rules")
    predicates = config.get("predicates")

    monitor = SimpleMonitor(traffic_rules, predicates, simulation_param, other_vehicles_param, ego_vehicle_param)
    scenario, planning_problem_set = \
        CommonRoadFileReader("./scenarios/" + simulation_param.get("commonroad_benchmark_id") + ".xml").open()

    # Create ego vehicle trajectory
    ego_obstacle = scenario.obstacle_by_id(ego_vehicle_param.get("vehicle_id"))
    scenario.remove_obstacle(ego_obstacle)
    trajectory = [ego_obstacle.initial_state] + ego_obstacle.prediction.trajectory.state_list

    # Monitor ego vehicle trajectory
    monitor.evaluate_trajectory(scenario, trajectory)


if __name__ == "__main__":
    main()
