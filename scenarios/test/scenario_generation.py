from output.visualization import create_scenario_video
import numpy as np
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.planning.goal import GoalRegion, Interval, AngleInterval
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet, LineMarking, LaneletType, RoadUser
from commonroad.scenario.traffic_sign import TrafficSign, TrafficSignElement, TrafficSignIDGermany
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State, Trajectory
from common.configuration import *


def create_scenario(commonroad_benchmark_id: str, num_straight_lanes: int, num_lanelets_per_lane: int,
                    road_length: int, obstacles: List[DynamicObstacle], lanelet_types, speed_limit):
    # setting parameters
    author = "Sebastian Maierhofer"
    affiliation = 'Technical University of Munich, Germany'
    source = 'CommonRoad Monitor'
    tags = "highway multiple_lanes no_oncoming_traffic parallel_lanes"
    dt = 0.1

    # desired number of lanes and parameters
    lane_width = 3.5
    lanelet_length = int(road_length/num_lanelets_per_lane)

    # initializing scenario
    scenario = Scenario(dt, commonroad_benchmark_id)
    # create straight lanelets
    lanelet_id_list = range(1, num_straight_lanes * num_lanelets_per_lane + 1)
    traffic_sign_elem = TrafficSignElement(TrafficSignIDGermany.MAXSPEED.value, [str(speed_limit)])
    traffic_sign = TrafficSign(201, [traffic_sign_elem])
    scenario.lanelet_network.add_traffic_sign(traffic_sign, set())
    lanelet_id_idx = 0
    for lane in range(num_straight_lanes):
        predecessor = None
        successor = [lanelet_id_list[lanelet_id_idx + 1]]
        for lanelet_idx in range(1, num_lanelets_per_lane + 1):
            left_vertices_point_list = []
            center_vertices_point_list = []
            right_vertices_point_list = []

            for i in range(lanelet_length + 1):
                left_vertices_point_list.append(np.array([i + lanelet_length * (lanelet_idx - 1),
                                                          (lane + 1) * lane_width]))
                center_vertices_point_list.append(np.array([i + lanelet_length * (lanelet_idx - 1),
                                                            (lane + 0.5) * lane_width]))
                right_vertices_point_list.append(np.array([i + lanelet_length * (lanelet_idx - 1), lane * lane_width]))

            left_vertices = np.array(left_vertices_point_list)
            center_vertices = np.array(center_vertices_point_list)
            right_vertices = np.array(right_vertices_point_list)

            # setting adjecent lanes to correct object ID
            # first lane: no adjecent right lane
            if lane == 0:
                lanelet = Lanelet(left_vertices, center_vertices, right_vertices, lanelet_id_list[lanelet_id_idx],
                                  predecessor=predecessor, successor=successor,
                                  adjacent_left=lanelet_id_list[lanelet_id_idx] + num_lanelets_per_lane,
                                  adjacent_left_same_direction=True,
                                  line_marking_left_vertices=LineMarking.DASHED,
                                  line_marking_right_vertices=LineMarking.SOLID,
                                  lanelet_type=lanelet_types, user_one_way={RoadUser.VEHICLE}, traffic_signs={201})
            # last lane: no adjecent left lane
            elif lane == num_straight_lanes - 1:
                lanelet = Lanelet(left_vertices, center_vertices, right_vertices, lanelet_id_list[lanelet_id_idx],
                                  predecessor=predecessor, successor=successor,
                                  adjacent_right=lanelet_id_list[lanelet_id_idx] - num_lanelets_per_lane,
                                  adjacent_right_same_direction=True,
                                  line_marking_left_vertices=LineMarking.SOLID,
                                  line_marking_right_vertices=LineMarking.DASHED,
                                  lanelet_type=lanelet_types, user_one_way={RoadUser.VEHICLE}, traffic_signs={201})

            else:
                lanelet = Lanelet(left_vertices, center_vertices, right_vertices, lanelet_id_list[lanelet_id_idx],
                                  predecessor=predecessor, successor=successor,
                                  adjacent_left=lanelet_id_list[lanelet_id_idx] + num_lanelets_per_lane,
                                  adjacent_left_same_direction=True,
                                  adjacent_right=lanelet_id_list[lanelet_id_idx] - num_lanelets_per_lane,
                                  adjacent_right_same_direction=True,
                                  line_marking_left_vertices=LineMarking.DASHED,
                                  line_marking_right_vertices=LineMarking.DASHED,
                                  lanelet_type=lanelet_types, user_one_way={RoadUser.VEHICLE}, traffic_signs={201})
            predecessor = [lanelet_id_list[lanelet_id_idx]]
            lanelet_id_idx += 1
            if lanelet_id_idx + 1 < len(lanelet_id_list) and (lanelet_id_idx + 1) % num_lanelets_per_lane != 0.0:
                successor = [lanelet_id_list[lanelet_id_idx + 1]]
            else:
                successor = None
            scenario.lanelet_network.add_lanelet(lanelet)

    for obs in obstacles:
        scenario.add_objects(obs)

    # create planing problem set (goal state is abitrarily chosen, since it is not needed for this purpose)
    goal_position_shape = Rectangle(10, 3.5, np.array([50, 1.75]))
    goal_state = State(position=goal_position_shape, velocity=Interval(0, 50),
                       orientation=AngleInterval(-0.01, 0.01), time_step=Interval(0, 100))
    goal_region = GoalRegion([goal_state])
    init_state = State(position=np.array([0.0, 1.75]), orientation=0, velocity=0, yaw_rate=0, slip_angle=0, time_step=0)
    planning_problem = PlanningProblem(1, init_state, goal_region)
    planning_problem_set = PlanningProblemSet([planning_problem])

    # write new scenario
    fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)
    filename = "./" + commonroad_benchmark_id + ".xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)

    return scenario


def create_obstacle_by_acceleration(acceleration_profile, v_init, p_init, obs_id):
    initial_state = State(position=p_init, velocity=v_init, orientation=0, yaw_rate=0, slip_angle=0, time_step=0)
    y = p_init[1]
    x = p_init[0]
    v = v_init
    v_max = 50
    v_min = 0
    dt = 0.1
    state_list = []
    time_step = 1
    for a in acceleration_profile:
        x, v = vehicle_dynamics_acc(x, v, a, v_min, v_max, dt)
        state_list.append(State(position=np.array([x, y]), velocity=v, orientation=0, yaw_rate=0, slip_angle=0,
                                time_step=time_step))
        time_step += 1
    obstacle_lenght = 4.508
    obstacle_width = 1.610
    shape = Rectangle(obstacle_lenght, obstacle_width)
    trajectory = Trajectory(initial_time_step=0, state_list=state_list)
    prediction = TrajectoryPrediction(trajectory, shape)

    return DynamicObstacle(obs_id, ObstacleType.CAR, shape, initial_state, prediction)


def create_max_speed_limit_scenario():
    obstacles = []
    obs1 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                            -5, -5, -5, -5, -5, -5, -5, -5, -5, -5], 34, np.array([3.0, 1.75]), 100)
    obs2 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35, np.array([10.0, 5.25]), 101)
    obs3 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0 ,0 ,0 ,0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 36, np.array([3.0, 8.75]), 102)
    obs4 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30, np.array([3.0, 5.25]), 103)
    obstacles.append(obs1)
    obstacles.append(obs2)
    obstacles.append(obs3)
    obstacles.append(obs4)
    scenario = create_scenario("test_max_speed_limit", 3, 5, 200, obstacles,
                               {LaneletType.HIGHWAY, LaneletType.MAIN_CARRIAGE_WAY}, 35)

    return scenario


def create_min_speed_limit_scenario():
    obstacles = []
    obs1 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5, np.array([3.0, 1.75]), 100)
    obs2 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([13.0, 1.75]), 101)
    obs3 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 10, np.array([3.0, 5.25]), 102)
    obs4 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 15, np.array([13.0, 5.25]), 103)
    obs5 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 25, np.array([22.0, 5.25]), 104)
    obs6 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 22, np.array([3.0, 8.75]), 105)
    obs7 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 22, np.array([13.0, 8.75]), 106)
    obstacles.append(obs1)
    obstacles.append(obs2)
    obstacles.append(obs3)
    obstacles.append(obs4)
    obstacles.append(obs5)
    obstacles.append(obs6)
    obstacles.append(obs7)
    scenario = create_scenario("test_min_speed_limit", 3, 5, 200, obstacles,
                               {LaneletType.HIGHWAY, LaneletType.MAIN_CARRIAGE_WAY}, 22.22)

    return scenario


def create_abrupt_braking_scenario():
    obstacles = []
    obs1 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([13.0, 1.75]), 100)
    obs2 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([3.0, 1.75]), 101)
    obs3 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 22, np.array([3.0, 5.25]), 102)
    obs4 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5, np.array([3.0, 8.75]), 103)
    obs5 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 22, np.array([22.0, 8.75]), 104)
    obstacles.append(obs1)
    obstacles.append(obs2)
    obstacles.append(obs3)
    obstacles.append(obs4)
    obstacles.append(obs5)
    scenario = create_scenario("test_min_speed_limit", 3, 5, 150, obstacles,
                               {LaneletType.HIGHWAY, LaneletType.MAIN_CARRIAGE_WAY}, 22.22)
    return scenario


def main():
    config = load_yaml("./../../config.yaml")
    simulation_param = config.get("simulation_param")
    visualization_param = config.get("visualization").get("video")

    scenario = create_max_speed_limit_scenario()
    create_scenario_video("./../../videos", scenario, visualization_param, 50)

    scenario = create_min_speed_limit_scenario()
    create_scenario_video("./../../videos", scenario, visualization_param, 50)

if __name__ == "__main__":
    main()
