import matplotlib.pyplot as plt
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
import os
import ntpath
#matplotlib.use("Agg")
import matplotlib.animation as animation
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.obstacle import DynamicObstacle, Obstacle
from typing import Union


basic_shape_parameters_static = {'opacity': 1.0,
                                 'facecolor': '#0f55a3',
                                 'edgecolor': '#0f55a3',
                                 'zorder': 20}

basic_shape_parameters_dynamic = {'opacity': 1.0,
                                  'facecolor': '#ff4000',
                                  'edgecolor': '#ff4000',
                                  'zorder': 100}

draw_params_scenario = {'scenario': {
    'dynamic_obstacle': {
        'draw_shape': True,
        'draw_icon': False,
        'draw_bounding_box': True,
        'show_label': True,
        'trajectory_steps': 25,
        'zorder': 100,
        'occupancy': {
            'draw_occupancies': 1,  # -1= never, 0= if prediction of vehicle is set-based, 1=always
            'shape': {
                'polygon': {
                    'opacity': 1.0,
                    'facecolor': '#0f55a3',
                    'edgecolor': '#0f55a3',
                    'zorder': 100,
                },
                'rectangle': {
                    'opacity': 1,
                    'facecolor': '#0f55a3',
                    'edgecolor': '#0f55a3',
                    'zorder': 18,
                },
                'circle': {
                    'opacity': 0.2,
                    'facecolor': '#1d7eea',
                    'edgecolor': '#0066cc',
                    'zorder': 18,
                }
            },
        },
        'shape': {
            'polygon': basic_shape_parameters_dynamic,
            'rectangle': basic_shape_parameters_dynamic,
            'circle': basic_shape_parameters_dynamic
        },
        'trajectory': {'facecolor': '#0f55a3'}
    },
    'static_obstacle': {
        'shape': {
            'polygon': basic_shape_parameters_static,
            'rectangle': basic_shape_parameters_static,
            'circle': basic_shape_parameters_static,
        }
    },
    'lanelet_network': {
        'lanelet': {'left_bound_color': '#555555',
                    'right_bound_color': '#555555',
                    'center_bound_color': '#dddddd',
                    'draw_left_bound': True,
                    'draw_right_bound': True,
                    'draw_center_bound': True,
                    'draw_border_vertices': False,
                    'draw_start_and_direction': True,
                    'show_label': False,
                    'draw_linewidth': 0.5,
                    'fill_lanelet': True,
                    'facecolor': '#e8e8e8'}},
},
}


def plot_scenario_at_time_idx(time_idx: int, scenario: Scenario):
    """
    Plots a scenario at a specific point in time.
    :param time_idx: The time point for which the scenario will be plotted
    :param scenario: The scenario to be plotted
    :return:
    """
    draw_params_scenario['time_begin'] = time_idx
    draw_params_scenario['time_end'] = time_idx
    draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['facecolor'] ='#ff4000'
    draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['edgecolor'] ='#cc3300'
    draw_params_scenario['scenario']['dynamic_obstacle']['occupancy']['shape']['polygon']['opacity'] = .1
    draw_object(scenario, draw_params=draw_params_scenario)


def plot_vehicle_at_time_idx(time_idx: int, ego_obstacle: DynamicObstacle):
    """
    Plots the occupancy of a vehicle at a specific time point given its trajectory.
    :param time_idx: The time point for which the occupancy will be plotted
    :param trajectory: The trajectory of the vehicle whose occupancy will be plotted
    :return:
    """
    #draw_params_ego['time_begin'] = time_idx
    #draw_params_ego['time_end'] = time_idx
    draw_object(ego_obstacle, draw_params=draw_params_scenario)


def create_video(out_path: str, scenario: Scenario, ego_obstacle: Union[DynamicObstacle, Obstacle],
                 planning_problem_set: PlanningProblemSet=None):
    """
    Creates a video of the solution for a specific planning problem.
    :param out_path: The path where the video will be saved.
    :param scenario: The scenario of the planning problem that was solved.
    :param ego_trajectory: The trajectory of the ego vehicle that solves the planning problem
    :param planning_problem_set: The planning problem set in which the planning problem belongs.
    :return:
    """
    assert(os.path.isdir(os.path.dirname(os.path.abspath(out_path)))), \
        'Directory %s does not exist' % os.path.dirname(os.path.abspath(out_path))
    filename = ntpath.basename(out_path)

    ffmpeg_writer = animation.writers['ffmpeg']
    metadata = dict(title=filename, artist='Matplotlib')
    writer = ffmpeg_writer(fps=10, metadata=metadata)

    fig = plt.figure(figsize=(25, 10))
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.title(filename)

    # find figure size
    x = [x for lanelet in scenario.lanelet_network.lanelets for x in lanelet.center_vertices[:, 0]]
    y = [y for lanelet in scenario.lanelet_network.lanelets for y in lanelet.center_vertices[:, 1]]
    x_min = min(x) - 10
    y_min = min(y) - 10
    x_max = max(x) + 5
    y_max = max(y) + 5

    if os.path.isfile(out_path):
        os.remove(out_path)
    with writer.saving(fig, out_path + "/test.mp4", dpi=150):
        for t in [state.time_step for state in ego_obstacle.prediction.trajectory.state_list]:
            plt.cla()
            plot_scenario_at_time_idx(t, scenario)
            draw_object(planning_problem_set)
            plot_vehicle_at_time_idx(t, ego_obstacle)
            plt.gca().set_aspect('equal')
            plt.gca().set_xlim([x_min, x_max])
            plt.gca().set_ylim([y_min, y_max])
            plt.text(2, 6, str(t), fontsize=15)
            writer.grab_frame()
