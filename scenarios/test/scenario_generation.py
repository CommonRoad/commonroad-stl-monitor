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
import bezier


def create_access_ramp_start(l_id: int) -> Lanelet:
    right = np.asfortranarray([[0.0, 2.5, 5, 7.5, 9.0, 10.0],
                               [-8.25, -6.75, -5.5, -4.5, -3.75, -3.5]])
    right_curve = bezier.Curve(right, degree=5)
    ax = right_curve.plot(num_pts=50)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    left = np.asfortranarray([[0.0, 2.5, 5, 7.5, 9.0, 10.0],
                              [-4.0, -3.0, -2.0, -1.0, -0.25, 0.0]])
    left_curve = bezier.Curve(left, degree=5)
    ax = left_curve.plot(num_pts=50)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    center = np.asfortranarray([[0.0, 2.5, 5, 7.5, 9, 10.0],
                               [-6.5, -5.0, -3.75, -2.75, -2.0, -1.75]])
    center_curve = bezier.Curve(center, degree=5)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    lanelet = Lanelet(left_vertices=left_vertices, center_vertices=center_vertices,
                      right_vertices=right_vertices, lanelet_id=l_id, predecessor=[1], successor=[l_id+1],
                      adjacent_left=None, adjacent_left_same_direction=None,
                      line_marking_left_vertices=LineMarking.SOLID,
                      line_marking_right_vertices=LineMarking.SOLID,
                      lanelet_type= {LaneletType.HIGHWAY, LaneletType.ACCESS_RAMP}, user_one_way={RoadUser.VEHICLE})
    return lanelet


def create_access_ramp_end(x_start, l_id: int) -> Lanelet:
    right = np.asfortranarray([[x_start, x_start+2.5, x_start+5, x_start+7.5, x_start+9.0, x_start+10.0],
                               [-3.5, -3.5, -3.25, -1, 0, 0]])
    right_curve = bezier.Curve(right, degree=5)
    ax = right_curve.plot(num_pts=20)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    left = np.asfortranarray([[x_start, x_start+2.5, x_start+5, x_start+7.5, x_start+9.0, x_start+10.0],
                               [0, 0, 0.25, 2.5, 3.5, 3.5]])
    left_curve = bezier.Curve(left, degree=5)
    ax = left_curve.plot(num_pts=20)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    center = np.asfortranarray([[x_start, x_start+2.5, x_start+5, x_start+7.5, x_start+9.0, x_start+10.0],
                               [-1.75, -1.75, -1.5, -1.25, 1.75, 1.75]])

    center_curve = bezier.Curve(center, degree=5)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    lanelet = Lanelet(left_vertices=left_vertices, center_vertices=center_vertices,
                      right_vertices=right_vertices, lanelet_id=l_id, predecessor=[l_id-1], successor=None,
                      adjacent_left=None, adjacent_left_same_direction=None,
                      line_marking_left_vertices=None,
                      line_marking_right_vertices=LineMarking.SOLID,
                      lanelet_type= {LaneletType.HIGHWAY, LaneletType.ACCESS_RAMP}, user_one_way={RoadUser.VEHICLE})
    return lanelet


def add_access_ramp(scenario: Scenario, road_length: int, l_id_start: int):
    lanelet_length = 10
    num_lanelets_per_lane = road_length // lanelet_length
    lane_width = 3.5

    # create straight lanelets
    lanelet = create_access_ramp_start(l_id_start)
    scenario.lanelet_network.add_lanelet(lanelet)

    predecessor = [l_id_start]
    successor = [l_id_start+2]
    adj_left = 2
    adj_right = None
    for lanelet_idx in range(l_id_start+1, l_id_start+num_lanelets_per_lane-2):
        left_vertices_point_list = []
        center_vertices_point_list = []
        right_vertices_point_list = []

        for i in range(lanelet_length + 1):
            left_vertices_point_list.append(np.array([i + lanelet_length * (lanelet_idx - l_id_start), 0]))
            center_vertices_point_list.append(np.array([i + lanelet_length *
                                                        (lanelet_idx - l_id_start),  -0.5 * lane_width]))
            right_vertices_point_list.append(np.array([i + lanelet_length *
                                                       (lanelet_idx - l_id_start), -lane_width]))

        left_vertices = np.array(left_vertices_point_list)
        center_vertices = np.array(center_vertices_point_list)
        right_vertices = np.array(right_vertices_point_list)

        lanelet = Lanelet(left_vertices=left_vertices, center_vertices=center_vertices,
                          right_vertices=right_vertices, lanelet_id=lanelet_idx, predecessor=predecessor,
                          successor=successor, adjacent_left=adj_left, adjacent_left_same_direction=True,
                          adjacent_right=adj_right, adjacent_right_same_direction=False,
                          line_marking_left_vertices=LineMarking.DASHED,
                          line_marking_right_vertices=LineMarking.SOLID,
                          lanelet_type={LaneletType.HIGHWAY, LaneletType.ACCESS_RAMP}, user_one_way={RoadUser.VEHICLE})
        scenario.lanelet_network.add_lanelet(lanelet)
        adj_left += 1
        predecessor = [lanelet_idx]
        if lanelet_idx  % num_lanelets_per_lane != 0.0:
            successor = [lanelet_idx + 1]
        else:
            successor = None
    lanelet = create_access_ramp_end(road_length-2*lanelet_length, l_id_start+num_lanelets_per_lane-2)
    scenario.lanelet_network.add_lanelet(lanelet)

    return scenario


def create_exit_ramp_start(l_id: int) -> Lanelet:
    right = np.asfortranarray([[10.0, 12.5, 15, 17.5, 19.0, 20.0],
                               [0.0, -1, -3, -3.25, -3.5, -3.5]])
    right_curve = bezier.Curve(right, degree=5)
    ax = right_curve.plot(num_pts=50)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    left = np.asfortranarray([[10.0, 12.5, 15, 17.5, 19.0, 20.0],
                               [3.5, 2.5, 0.5, 0.25, 0, 0]])
    left_curve = bezier.Curve(left, degree=5)
    ax = left_curve.plot(num_pts=50)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    center = np.asfortranarray([[10.0, 12.5, 15, 17.5, 19, 20.0],
                               [1.75, 0.75, -1.25, -1.5, -1.75, -1.75]])
    center_curve = bezier.Curve(center, degree=5)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    lanelet = Lanelet(left_vertices=left_vertices, center_vertices=center_vertices,
                      right_vertices=right_vertices, lanelet_id=l_id, predecessor=[1], successor=[l_id+1],
                      adjacent_left=None, adjacent_left_same_direction=None,
                      line_marking_left_vertices=None,
                      line_marking_right_vertices=LineMarking.SOLID,
                      lanelet_type= {LaneletType.HIGHWAY, LaneletType.EXIT_RAMP}, user_one_way={RoadUser.VEHICLE})
    return lanelet


def create_exit_ramp_end(x_start, l_id: int) -> Lanelet:
    right = np.asfortranarray([[x_start, x_start+2.5, x_start+5, x_start+7.5, x_start+9.0, x_start+10.0],
                               [-3.5, -3.75, -4.5, -5.5, -6.75, -8.25]])
    right_curve = bezier.Curve(right, degree=5)
    ax = right_curve.plot(num_pts=20)
    x_right, y_right = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_right[idx], y_right[idx]])
        point_list.append(point)
    left_vertices = np.array(point_list)

    left = np.asfortranarray([[x_start, x_start+2.5, x_start+5, x_start+7.5, x_start+9.0, x_start+10.0],
                               [0.0, -0.25, -1.0, -2.0, -3.0, -4.0]])
    left_curve = bezier.Curve(left, degree=5)
    ax = left_curve.plot(num_pts=20)
    x_left, y_left = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_left[idx], y_left[idx]])
        point_list.append(point)
    right_vertices = np.array(point_list)

    center = np.asfortranarray([[x_start, x_start+2.5, x_start+5, x_start+7.5, x_start+9.0, x_start+10.0],
                               [-1.75, -2.0, -2.75, -3.75, -5.0, -6.5]])

    center_curve = bezier.Curve(center, degree=5)
    ax = center_curve.plot(num_pts=50)
    x_center, y_center = ax.lines[0].get_data()
    point_list = []
    for idx in range(len(x_right)):
        point = np.array([x_center[idx], y_center[idx]])
        point_list.append(point)
    center_vertices = np.array(point_list)

    lanelet = Lanelet(left_vertices=left_vertices, center_vertices=center_vertices,
                      right_vertices=right_vertices, lanelet_id=l_id, predecessor=[l_id-1], successor=None,
                      adjacent_left=None, adjacent_left_same_direction=None,
                      line_marking_left_vertices=LineMarking.SOLID,
                      line_marking_right_vertices=LineMarking.SOLID,
                      lanelet_type= {LaneletType.HIGHWAY, LaneletType.EXIT_RAMP}, user_one_way={RoadUser.VEHICLE})
    return lanelet


def add_exit_ramp(scenario: Scenario, road_length: int, l_id_start: int):
    lanelet_length = 10
    num_lanelets_per_lane = road_length // lanelet_length
    lane_width = 3.5

    # create straight lanelets
    lanelet = create_exit_ramp_start(l_id_start)
    scenario.lanelet_network.add_lanelet(lanelet)

    predecessor = [l_id_start]
    successor = [l_id_start+2]
    adj_left = 2
    adj_right = None
    for lanelet_idx in range(l_id_start+1, l_id_start+num_lanelets_per_lane-2):
        left_vertices_point_list = []
        center_vertices_point_list = []
        right_vertices_point_list = []

        for i in range(lanelet_length + 1):
            left_vertices_point_list.append(np.array([i + lanelet_length * (lanelet_idx - l_id_start + 1), 0]))
            center_vertices_point_list.append(np.array([i + lanelet_length *
                                                        (lanelet_idx - l_id_start + 1),  -0.5 * lane_width]))
            right_vertices_point_list.append(np.array([i + lanelet_length *
                                                       (lanelet_idx - l_id_start + 1), -lane_width]))

        left_vertices = np.array(left_vertices_point_list)
        center_vertices = np.array(center_vertices_point_list)
        right_vertices = np.array(right_vertices_point_list)

        lanelet = Lanelet(left_vertices=left_vertices, center_vertices=center_vertices,
                          right_vertices=right_vertices, lanelet_id=lanelet_idx, predecessor=predecessor,
                          successor=successor, adjacent_left=adj_left, adjacent_left_same_direction=True,
                          adjacent_right=adj_right, adjacent_right_same_direction=False,
                          line_marking_left_vertices=LineMarking.DASHED,
                          line_marking_right_vertices=LineMarking.SOLID,
                          lanelet_type={LaneletType.HIGHWAY, LaneletType.EXIT_RAMP}, user_one_way={RoadUser.VEHICLE})
        scenario.lanelet_network.add_lanelet(lanelet)
        adj_left += 1
        predecessor = [lanelet_idx]
        if lanelet_idx  % num_lanelets_per_lane != 0.0:
            successor = [lanelet_idx + 1]
        else:
            successor = None
    lanelet = create_exit_ramp_end(road_length-lanelet_length, l_id_start+num_lanelets_per_lane-2)
    scenario.lanelet_network.add_lanelet(lanelet)

    return scenario


def create_straight_scenario(commonroad_benchmark_id: str, dt: float, num_straight_lanes: int,
                             num_lanelets_per_lane: int, road_length: int, obstacles: List[DynamicObstacle]):

    # desired number of lanes and parameters
    lane_width = 3.5
    lanelet_types = {LaneletType.HIGHWAY, LaneletType.MAIN_CARRIAGE_WAY}
    lanelet_length = int(road_length/num_lanelets_per_lane)

    # initializing scenario
    scenario = Scenario(dt, commonroad_benchmark_id)
    # create straight lanelets
    lanelet_id_list = range(1, num_straight_lanes * num_lanelets_per_lane + 1)
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
                                  lanelet_type=lanelet_types, user_one_way={RoadUser.VEHICLE})
            # last lane: no adjecent left lane
            elif lane == num_straight_lanes - 1:
                lanelet = Lanelet(left_vertices, center_vertices, right_vertices, lanelet_id_list[lanelet_id_idx],
                                  predecessor=predecessor, successor=successor,
                                  adjacent_right=lanelet_id_list[lanelet_id_idx] - num_lanelets_per_lane,
                                  adjacent_right_same_direction=True,
                                  line_marking_left_vertices=LineMarking.SOLID,
                                  line_marking_right_vertices=LineMarking.DASHED,
                                  lanelet_type=lanelet_types, user_one_way={RoadUser.VEHICLE})

            else:
                lanelet = Lanelet(left_vertices, center_vertices, right_vertices, lanelet_id_list[lanelet_id_idx],
                                  predecessor=predecessor, successor=successor,
                                  adjacent_left=lanelet_id_list[lanelet_id_idx] + num_lanelets_per_lane,
                                  adjacent_left_same_direction=True,
                                  adjacent_right=lanelet_id_list[lanelet_id_idx] - num_lanelets_per_lane,
                                  adjacent_right_same_direction=True,
                                  line_marking_left_vertices=LineMarking.DASHED,
                                  line_marking_right_vertices=LineMarking.DASHED,
                                  lanelet_type=lanelet_types, user_one_way={RoadUser.VEHICLE})
            predecessor = [lanelet_id_list[lanelet_id_idx]]
            lanelet_id_idx += 1
            if lanelet_id_idx + 1 < len(lanelet_id_list) and (lanelet_id_idx + 1) % num_lanelets_per_lane != 0.0:
                successor = [lanelet_id_list[lanelet_id_idx + 1]]
            else:
                successor = None
            scenario.lanelet_network.add_lanelet(lanelet)
    for obs in obstacles:
        scenario.add_objects(obs)

    return scenario


def write_to_file( scenario):
    author = "Sebastian Maierhofer"
    affiliation = 'Technical University of Munich, Germany'
    source = 'CommonRoad Monitor'
    tags = "highway multiple_lanes no_oncoming_traffic parallel_lanes"

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
    filename = "./" + scenario.benchmark_id + ".xml"
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)


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
    obs1 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
                                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                            1, 1, 1, 0, 0, 0, 0, 0, -5, -5,
                                            -5, -5, -5, -5, -5, -5, -5, -5, -5, -5], 31.5, np.array([3.0, 1.75]), 1000)
    obs2 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35, np.array([10.0, 5.25]), 1001)
    obs3 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0 ,0 ,0 ,0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 36, np.array([3.0, 8.75]), 1002)
    obs4 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30, np.array([3.0, 5.25]), 1003)
    obstacles.append(obs1)
    obstacles.append(obs2)
    obstacles.append(obs3)
    obstacles.append(obs4)
    scenario = create_straight_scenario("test_max_speed_limit", 0.1, 3, 5, 200, obstacles)
    traffic_sign_elem = TrafficSignElement(TrafficSignIDGermany.MAXSPEED.value, [str(35)])
    traffic_sign = TrafficSign(201, [traffic_sign_elem])
    scenario.lanelet_network.add_traffic_sign(traffic_sign, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 13, 14, 15})

    return scenario


def create_min_speed_limit_scenario():
    obstacles = []
    obs1 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35, np.array([3.0, 1.75]), 1000)
    obs2 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35, np.array([13.0, 1.75]), 1001)
    obs3 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 10, np.array([3.0, 5.25]), 1002)
    obs4 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 10, np.array([13.0, 5.25]), 1003)
    obs5 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35, np.array([22.0, 5.25]), 1004)
    obs6 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5, np.array([100.0, 8.75]), 1005)
    obs7 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 22, np.array([3.0, 12.25]), 1006)
    obs8 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 35, np.array([3.0, 15.75]), 1007)

    obstacles.append(obs1)
    obstacles.append(obs2)
    obstacles.append(obs3)
    obstacles.append(obs4)
    obstacles.append(obs5)
    obstacles.append(obs6)
    obstacles.append(obs7)
    obstacles.append(obs8)
    num_lanes = 5
    num_lanelets = 5
    road_length = 200
    scenario = create_straight_scenario("test_min_speed_limit", 0.1, num_lanes, num_lanelets, road_length, obstacles)
    traffic_sign_elem_1 = TrafficSignElement(TrafficSignIDGermany.MAXSPEED.value, [str(40)])
    traffic_sign_1 = TrafficSign(201, [traffic_sign_elem_1])
    traffic_sign_elem_2 = TrafficSignElement(TrafficSignIDGermany.MINSPEED.value, [str(30)])
    traffic_sign_2 = TrafficSign(202, [traffic_sign_elem_2])
    scenario.lanelet_network.add_traffic_sign(traffic_sign_1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
    scenario.lanelet_network.add_traffic_sign(traffic_sign_2, {16, 17, 18, 19, 20, 21, 22, 23, 24, 25})

    return scenario


def create_safe_distance_scenario():
    obstacles = []
    obs1 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 40, np.array([3.0, 1.75]), 1000)
    obs2 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([125.0, 1.75]), 1001)
    obs3 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30, np.array([3.0, 5.25]), 1002)
    obs4 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30, np.array([13.0, 5.25]), 1003)
    obs5 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([75.0, 5.25]), 1004)
    obs6 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([3.0, 8.75]), 1005)
    obs7 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([40.0, 8.75]), 1006)
    obstacles.append(obs1)
    obstacles.append(obs2)
    obstacles.append(obs3)
    obstacles.append(obs4)
    obstacles.append(obs5)
    obstacles.append(obs6)
    obstacles.append(obs7)
    num_lanes = 3
    num_lanelets = 10
    road_length = 250
    scenario = create_straight_scenario("test_safe_distance", 0.1, num_lanes, num_lanelets, road_length, obstacles)
    traffic_sign_elem = TrafficSignElement(TrafficSignIDGermany.MAXSPEED.value, [str(22.22)])
    traffic_sign = TrafficSign(201, [traffic_sign_elem])
    scenario.lanelet_network.add_traffic_sign(traffic_sign, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})

    return scenario


def create_unnecessary_braking_scenario_1():
    obstacles = []
    obs1 = create_obstacle_by_acceleration([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 20, np.array([3.0, 1.75]), 1000)
    obs2 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30, np.array([3.0, 5.25]), 1001)
    obs3 = create_obstacle_by_acceleration([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0,
                                            -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0,
                                            -2.1, -2.2, -2.3, -2.4, -2.5, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30, np.array([75.0, 8.75]), 1002)
    obs4 = create_obstacle_by_acceleration([0, 0, 0, 0, -1, -1, -1, -2, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, -1, -2, -2, -2, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([3.0, 12.25]), 1003)
    obs5 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, -5, -5, -5, -5, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([3.0, 15.75]), 1004)
    obs6 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([40.0, 15.75]), 1005)
    obs7 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9,
                                            -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([3.0, 19.25]), 1006)
    obs8 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20, np.array([40.0, 19.25]), 1007)
    obstacles.append(obs1)
    obstacles.append(obs2)
    obstacles.append(obs3)
    obstacles.append(obs4)
    obstacles.append(obs5)
    obstacles.append(obs6)
    obstacles.append(obs7)
    obstacles.append(obs8)
    num_lanes = 6
    num_lanelets = 10
    road_length = 250
    scenario = create_straight_scenario("test_unnecessary_braking_1", 0.1, num_lanes, num_lanelets, road_length,
                                        obstacles)

    return scenario


def create_unnecessary_braking_scenario_2():
    obstacles = []
    obs1 = create_obstacle_by_acceleration([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, -5, -5, -5, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 30, np.array([3.0, 1.75]), 1000)
    obstacles.append(obs1)
    num_lanes = 1
    num_lanelets = 10
    road_length = 250
    scenario = create_straight_scenario("test_unnecessary_braking_2", 0.1, num_lanes, num_lanelets, road_length,
                                        obstacles)

    return scenario

def main():
    config = load_yaml("./../../config.yaml")
    visualization_param = config.get("visualization").get("video")

    # scenario = create_max_speed_limit_scenario()
    # write_to_file(scenario)
    # create_scenario_video("./../../videos", scenario, visualization_param, 50)
    #
    # scenario = create_min_speed_limit_scenario()
    # write_to_file(scenario)
    # create_scenario_video("./../../videos", scenario, visualization_param, 50)
    #
    # scenario = create_safe_distance_scenario()
    # write_to_file(scenario)
    # create_scenario_video("./../../videos", scenario, visualization_param, 50)

    scenario = create_unnecessary_braking_scenario_1()
    write_to_file(scenario)
    create_scenario_video("./../../videos", scenario, visualization_param, 50)

    scenario = create_unnecessary_braking_scenario_2()
    write_to_file(scenario)
    create_scenario_video("./../../videos", scenario, visualization_param, 50)

if __name__ == "__main__":
    main()
