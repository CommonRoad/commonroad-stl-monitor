import random
from commonroad.scenario.trajectory import State
from commonroad.common.util import Interval, AngleInterval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.goal import GoalRegion


def get_planning_problem(scenario, 
                         orientation_half_range=0.2, 
                         velocity_half_range=10., 
                         time_step_half_range=50,
                         shifting_to_zero=False,
                         store_ego = False,
                         ego_list = None
):    
    # random choose obstacle as ego vehicle
    random.seed(0)
    # dynamic_obstacle = random.choice(scenario.dynamic_obstacles)
    dynamic_obstacle_selected = None
    for dynamic_obstacle in scenario.dynamic_obstacles:
        dynamic_obstacle_final_state = dynamic_obstacle.prediction.trajectory.final_state
        if dynamic_obstacle_final_state.position[0] > 350. and dynamic_obstacle.initial_state.position[0] < 30.0:
            for state in dynamic_obstacle.prediction.trajectory.state_list:
                if state.position[0] > 100:
                    dynamic_obstacle_selected = dynamic_obstacle
                    dynamic_obstacle_initial_state = state
                    break
            else:
                continue
            break
    if dynamic_obstacle_selected is None:
        print('No obstacle with trajectory which starts around 100 m and ends after 250m found')

        for dynamic_obstacle in scenario.dynamic_obstacles:
            dynamic_obstacle_final_state = dynamic_obstacle.prediction.trajectory.final_state
            if dynamic_obstacle.initial_state.position[0] > 100.0 and dynamic_obstacle_selected is None:
                dynamic_obstacle_selected = dynamic_obstacle
                dynamic_obstacle_initial_state = dynamic_obstacle.initial_state
                break

            else:
                continue

    dynamic_obstacle_shape = dynamic_obstacle_selected.obstacle_shape


    if store_ego:
        ego_list.append([scenario.benchmark_id, dynamic_obstacle_selected, dynamic_obstacle_initial_state])
    scenario.remove_obstacle(dynamic_obstacle_selected)

    final_timestep = dynamic_obstacle_final_state.time_step + time_step_half_range
    initial_timestep = dynamic_obstacle_initial_state.time_step
    
    # define orientation, velocity and time step intervals as goal region
    orientation_interval = AngleInterval(
        dynamic_obstacle_final_state.orientation - orientation_half_range,
        dynamic_obstacle_final_state.orientation + orientation_half_range
    )
    velocity_interval = Interval(
        dynamic_obstacle_final_state.velocity - velocity_half_range,
        dynamic_obstacle_final_state.velocity + velocity_half_range
    )
    if shifting_to_zero:
        time_step_interval = Interval(
            dynamic_obstacle_final_state.time_step - time_step_half_range - initial_timestep,
            dynamic_obstacle_final_state.time_step + time_step_half_range - initial_timestep
        )
    else:
        time_step_interval = Interval(
            dynamic_obstacle_final_state.time_step - time_step_half_range,
            dynamic_obstacle_final_state.time_step + time_step_half_range
        )
    goal_position = Rectangle(dynamic_obstacle_shape.length,dynamic_obstacle_shape.width,
                              center=dynamic_obstacle_final_state.position,
                              orientation=dynamic_obstacle_final_state.orientation)
    goal_region = GoalRegion([State(
        position=goal_position,
        orientation=orientation_interval, 
        velocity=velocity_interval, 
        time_step=time_step_interval
    )])

    if shifting_to_zero:
        dynamic_obstacle_initial_state.time_step = 0
    
    dynamic_obstacle_initial_state.yaw_rate = 0.
    dynamic_obstacle_initial_state.slip_angle = 0.
    
    return PlanningProblem(dynamic_obstacle_selected.obstacle_id, dynamic_obstacle_initial_state, goal_region), final_timestep, initial_timestep, ego_list