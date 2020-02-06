import math
import matplotlib.pyplot as plt
import matplotlib as mp
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
import os
import ntpath
import matplotlib.animation as animation
from matplotlib.animation import MovieWriter
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.obstacle import DynamicObstacle
from common.vehicle import Vehicle
from typing import List, Dict, Tuple
from datetime import datetime
import warnings


# CommonRoad Visualization Parameters:
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
        'show_label': False,
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


def plot_scenario_at_time_idx(time_idx: int, scenario: Scenario, obstacle_label: bool):
    """
    Plots a scenario at a specific point in time.
    :param time_idx: the time point for which the scenario will be plotted
    :param scenario: the scenario to be plotted
    :param obstacle_label: boolean indicating if obstacle label (ID) should be visualized
    """
    draw_params_scenario['time_begin'] = time_idx
    draw_params_scenario['time_end'] = time_idx
    if obstacle_label:
        draw_params_scenario['scenario']['dynamic_obstacle']['show_label'] = True
    draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['facecolor'] = '#ff4000'
    draw_params_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['edgecolor'] = '#cc3300'
    draw_params_scenario['scenario']['dynamic_obstacle']['occupancy']['shape']['polygon']['opacity'] = .1
    draw_object(scenario, draw_params=draw_params_scenario)


def plot_vehicle_at_time_idx(ego_obstacle: DynamicObstacle):
    """
    Plots the occupancy of a vehicle at a specific time point given its trajectory
    :param ego_obstacle: ego vehicle as CommonRoad object
    """
    draw_object(ego_obstacle, draw_params=draw_params_scenario)


def create_scenario_video(out_path: str, scenario: Scenario, visualization_param: Dict, video_length: int,
                          ego_obstacle: DynamicObstacle = None, planning_problem_set: PlanningProblemSet = None):
    """
    Creates a video of the solution for a specific planning problem
    :param out_path: The path where the video will be saved
    :param scenario: The scenario of the planning problem that was solved
    :param ego_obstacle: ego vehicle as CommonRoad object
    :param visualization_param: dictionary with parameters for plotting of profiles
    :param planning_problem_set: The planning problem set in which the planning problem belongs
    """
    filename = ntpath.basename(out_path)
    obstacle_label = visualization_param.get("obstacle_label")
    ffmpeg_writer = animation.writers['ffmpeg']
    metadata = dict("", artist='TUM CPS GROUP')
    writer = ffmpeg_writer(fps=10, metadata=metadata)

    fig = plt.figure(figsize=(25, 10))
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.title(filename)

    # find figure size
    x = [x for lanelet in scenario.lanelet_network.lanelets for x in lanelet.center_vertices[:, 0]]
    y = [y for lanelet in scenario.lanelet_network.lanelets for y in lanelet.center_vertices[:, 1]]
    x_min = min(x) - 5
    y_min = min(y) - 5
    x_max = max(x) + 5
    y_max = max(y) + 5

    if os.path.isfile(out_path):
        os.remove(out_path)
    with writer.saving(fig, out_path + "/" + scenario.benchmark_id + ".mp4", dpi=150):
        for t in range(video_length):
            plt.cla()
            plot_scenario_at_time_idx(t, scenario, obstacle_label)
            if planning_problem_set is not None:
                draw_object(planning_problem_set)
            if ego_obstacle is not None:
                plot_vehicle_at_time_idx(ego_obstacle)
            plt.gca().set_aspect('equal')
            plt.gca().set_xlim([x_min, x_max])
            plt.gca().set_ylim([y_min, y_max])
            writer.grab_frame()


def create_ego_profiles(ego_vehicle: Vehicle) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Creates acceleration, velocity and time profiles for ego vehicle

    :param ego_vehicle: ego vehicle object
    :return: lists with ego vehicle profiles
    """
    jerk_list = []
    acceleration_list = []
    velocity_list = []
    time = []
    for time_step, state in ego_vehicle.states_lon.items():
        acceleration_list.append(state.a)
        velocity_list.append(state.v)
        time.append(time_step)
    for jerk in ego_vehicle.jerk_profile.values():
        jerk_list.append(jerk)
    jerk_list = [jerk_list[0]] + jerk_list

    return time, jerk_list, acceleration_list, velocity_list


def create_lead_profiles(ego_vehicle: Vehicle, vehicles: Dict, time: List[int], relevant_obs_ids: List[int]) -> \
        Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Creates acceleration, velocity and distance profiles for leading vehicles

    :param ego_vehicle: ego vehicle object
    :param vehicles: list with leading vehicle objects
    :param time: list with time steps where ego vehicle exists
    :param relevant_obs_ids: list with IDs of relevant obstacles
    :returns lists with leading vehicle profiles
    """
    acceleration_plots = {}
    velocity_plots = {}
    distance_plots = {}
    safe_distance_plots = {}
    for vehicle in vehicles.values():
        if vehicle.id in relevant_obs_ids:
            veh_acceleration_profile = []
            veh_velocity_profile = []
            veh_distance_profile = []
            veh_safe_distance_profile = []
            for time_step in time:
                if vehicle.states_lon.get(time_step) is not None:
                    veh_acceleration_profile.append(vehicle.states_lon.get(time_step).a)
                else:
                    veh_acceleration_profile.append(0)
                if vehicle.states_lon.get(time_step) is not None:
                    veh_velocity_profile.append(vehicle.states_lon.get(time_step).v)
                else:
                    veh_velocity_profile.append(0)
                if vehicle.states_lon.get(time_step) is not None:
                    veh_distance_profile.append(vehicle.rear_position(time_step) - ego_vehicle.front_position(time_step))
                else:
                    veh_distance_profile.append(0)
                if vehicle.states_lon.get(time_step) is not None:
                    veh_safe_distance_profile.append(vehicle.safe_distance_list.get(time_step))
                else:
                    veh_safe_distance_profile.append(0)

            acceleration_plots[vehicle.id] = veh_acceleration_profile
            velocity_plots[vehicle.id] = veh_velocity_profile
            distance_plots[vehicle.id] = veh_distance_profile
            safe_distance_plots[vehicle.id] = veh_safe_distance_profile
    return acceleration_plots, velocity_plots, distance_plots, safe_distance_plots


def num_vehicle_profiles(number_vehicles: List[Tuple[int]]) -> Tuple[List[int], List[int], List[int]]:
    """
    Creates profiles with number of vehicles in same lane, in reduced same lane, and of cut-in vehicles

    :param number_vehicles: list of tuples for each time step
    :returns lists with number of vehicles for each category
    """
    num_veh_same, num_veh_same_reduced, num_veh_cutin = [], [], []
    for num_veh_time_step in number_vehicles:
        num_veh_same.append(num_veh_time_step[0])
        num_veh_same_reduced.append(num_veh_time_step[1])
        num_veh_cutin.append(num_veh_time_step[2])

    return num_veh_same, num_veh_same_reduced, num_veh_cutin


def get_date_and_time() -> str:
    """
    Returns current data and time

    :return: Current date and time as string
    """
    current_time = datetime.now().time()
    current_time = str(current_time)
    current_time = current_time.replace(':', ' _')
    current_time = current_time.replace('.', ' _')
    current_date = str(datetime.now().day) + "_" + str(datetime.now().month) + "_" + str(datetime.now().year)

    return current_date + "_" + current_time


def plot_figures(ego_vehicle: Vehicle, vehicles: Dict, emergency_maneuver_activity: List[bool],
                 ego_vehicle_param: Dict, simulation_param: Dict, plot_param: Dict, comp_time: List[float],
                 number_vehicles: List[Tuple[int]]):
    """
    Plotting of positions, acceleration, velocity, and distance of ACC and preceding vehicles, respectively

    :param ego_vehicle: ego vehicle object
    :param vehicles: dictionary with vehicle objects of surrounding vehicles
    :param emergency_maneuver_activity: emergency maneuver acitivity per time step
    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param plot_param: dictionary with parameters for plotting of profiles
    :param comp_time: list with computation time at each simulation step
    :param strategy_list: list with strategy at each simulation step
    :param number_vehicles: list of tuples with number of vehicles at (reduced) same lane and cutting in
    """

    # Storage related configuration
    width = plot_param.get("width")
    height = width * (math.sqrt(5) - 1.0) / 2.0
    figsize = [width, height]
    linewidth_plot = plot_param.get("line_width")
    mp.rcParams.update({'font.size': plot_param.get("font_size")})
    mp.rcParams.update({'axes.linewidth': plot_param.get("axes_line_width")})
    mp.rcParams.update({'figure.autolayout': True})
    mp.rcParams.update({'legend.frameon': False})

    # Directory creation if plots should be stored
    if simulation_param.get("store_plots"):
        date_time = get_date_and_time()
        if not os.path.exists("figures/"):
            os.mkdir("figures/")
        path = "figures/" + date_time
        os.mkdir(path)

    relevant_obs_ids = simulation_param.get("other_vehicle_plots")

    # Create profiles
    time_steps, jerk_profile_ego, acceleration_profile_ego, velocity_profile_ego = create_ego_profiles(ego_vehicle)
    num_veh_same, num_veh_same_reduced, num_veh_cutin = num_vehicle_profiles(number_vehicles)
    acceleration_profiles_lead, velocity_profiles_lead, distance_profiles_lead, safe_distance_profiles_lead = \
        create_lead_profiles(ego_vehicle, vehicles, time_steps, relevant_obs_ids)
    time = [time_step * simulation_param.get("dt") for time_step in time_steps]

    # Distance plot
    if len(relevant_obs_ids) > 0:
        plt.figure(1, figsize=figsize)
        plt.ylabel(r'$s~[m]$')
        plt.xlabel(r'$t~[s]$')
        for obs_id in relevant_obs_ids:
            plt.plot(time, distance_profiles_lead.get(obs_id), color=(0.0, 0.0, 0.5, 1), label=r'$s_{lead}$',
                     linewidth=linewidth_plot)
            plt.plot(time, safe_distance_profiles_lead.get(obs_id), "-", color=(0.3, 0.3, 0.3, 0.35),
                     label=r'$s_{safe}$', linewidth=linewidth_plot)
        plt.legend(loc='best')
        if simulation_param.get("store_plots"):
            plt.savefig(path + "/distance" + ".svg", format="svg")

    # Velocity plot
    plt.figure(2, figsize=figsize)
    plt.ylabel(r'$v~[m/s]$')
    plt.xlabel(r'$t~[s]$')
    for obs_id in relevant_obs_ids:
        plt.plot(time, velocity_profiles_lead.get(obs_id), "-", color=(0.0, 0.0, 0.5, 1), label=r'$v_{lead}$',
                 linewidth=linewidth_plot)
    plt.plot(time, velocity_profile_ego, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$v_{acc}$', linewidth=linewidth_plot)
    plt.legend(loc='best')
    if simulation_param.get("store_plots"):
        plt.savefig(path + "/velocity" + ".svg", format="svg")

    # Acceleration plot
    plt.figure(3, figsize=figsize)
    plt.ylabel(r'$a~[m/s^2]$')
    plt.xlabel(r'$t~[s]$')
    plt.ylim([ego_vehicle_param.get("a_min") - 0.5, ego_vehicle_param.get("a_max") + 0.5])
    for obs_id in relevant_obs_ids:
        plt.step(time, acceleration_profiles_lead.get(obs_id), color=(0.0, 0.0, 0.5, 1), label=r'$a_{lead}$',
                 linewidth=linewidth_plot)
    plt.step(time, acceleration_profile_ego, color=(0.3, 0.3, 0.3, 0.35), label=r'$a_{acc}$', linewidth=linewidth_plot)
    plt.legend(loc='best')
    if simulation_param.get("store_plots"):
        plt.savefig(path + "/acceleration" + ".svg", format="svg")

    # Jerk plot
    plt.figure(4, figsize=figsize)
    plt.ylabel(r'$j~[m/s^3]$')
    plt.xlabel(r'$t~[s]$')
    plt.step(time, jerk_profile_ego, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$j_{acc}$', linewidth=linewidth_plot)
    plt.legend(loc='best')
    if simulation_param.get("store_plots"):
        plt.savefig(path + "/jerk" + ".svg", format="svg")

    # Emergency maneuver activity plot
    plt.figure(5, figsize=figsize)
    plt.xlabel(r'$t~[s]$')
    plt.step(time, emergency_maneuver_activity, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$emg$',
             linewidth=linewidth_plot)
    plt.legend(loc='best')
    if simulation_param.get("store_plots"):
        plt.savefig(path + "/emg" + ".svg", format="svg")

    # Computation time plot
    if simulation_param.get("time_measuring"):
        plt.figure(6, figsize=figsize)
        plt.ylabel(r'$t~[s]$')
        plt.xlabel(r'$t~[s]$')
        plt.step(time[0:-1], comp_time, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$t_{comp}$', linewidth=linewidth_plot)
        plt.legend(loc='best')
        if simulation_param.get("store_plots"):
            plt.savefig(path + "/comp_time" + ".svg", format="svg")

    # Number vehicles plot
    plt.figure(8, figsize=figsize)
    plt.xlabel(r'$t~[s]$')
    plt.step(time[0:-1], num_veh_same, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$same~lane$', linewidth=linewidth_plot)
    plt.step(time[0:-1], num_veh_same_reduced, "-", color=(0.0, 0.0, 0.5, 1),
             label=r'$same~lane~reduced$', linewidth=linewidth_plot)
    plt.step(time[0:-1], num_veh_cutin, "-", color=(0.75, 0.2, 0.6, 1), label=r'$cut-in$', linewidth=linewidth_plot)
    plt.legend(loc='best')
    if simulation_param.get("store_plots"):
        plt.savefig(path + "/strategy" + ".svg", format="svg")
    plt.show()


def create_profile_videos(out_path: str, ego_vehicle: Vehicle, vehicles: Dict,
                          ego_vehicle_param: Dict, other_vehicle_param: Dict,
                          simulation_param: Dict, number_vehicles: List[Tuple[int]]):
    """
    Plotting of positions, acceleration, velocity, and distance of ACC and preceding vehicles, respectively

    :param out_path: the path where the video will be saved
    :param ego_vehicle: ego vehicle object
    :param vehicles: dictionary with vehicle objects of surrounding vehicles
    :param emergency_maneuver_activity: emergency maneuver activity per time step
    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param other_vehicle_param: dictionary with physical parameters of other vehicles
    :param simulation_param: dictionary with parameters of the simulation environment
    :param plot_param: dictionary with parameters for plotting of profiles
    :param comp_time: list with computation time at each simulation step
    :param strategy_list: list with strategy at each simulation step
    :param number_vehicles: list of tuples with number of vehicles at (reduced) same lane and cutting in
    """
    relevant_obs_ids = simulation_param.get("other_vehicle_plots")
    if len(relevant_obs_ids) > 1:
        warnings.warn('Only the first leading vehicle will be plotted in the profile videos.')

    # Create profiles
    time_steps, jerk_profile_ego, acceleration_profile_ego, velocity_profile_ego = create_ego_profiles(ego_vehicle)
    num_veh_same, num_veh_same_reduced, num_veh_cutin = num_vehicle_profiles(number_vehicles)
    acceleration_profiles_lead, velocity_profiles_lead, distance_profiles_lead, safe_distance_profiles_lead = \
        create_lead_profiles(ego_vehicle, vehicles, time_steps, relevant_obs_ids)
    time = [time_step * simulation_param.get("dt") for time_step in time_steps]

    ffmpeg_writer = animation.writers['ffmpeg']
    metadata = dict(title="", artist='TUM CPS GROUP')
    writer = ffmpeg_writer(fps=10, metadata=metadata)

    line_width_plot = 1.75
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'legend.frameon': False})
    x_lim = [0, time[-1]]

    # Create velocity profiles
    create_animated_profiles(out_path, time, line_width_plot, writer, x_lim,
                             [min(ego_vehicle_param.get("v_min"), other_vehicle_param.get("v_min")) - 0.5,
                              max(ego_vehicle_param.get("v_max"), other_vehicle_param.get("v_max")) + 0.5],
                             '$v~[m/s]$', '$t~[s]$', "best",
                             simulation_param.get("commonroad_benchmark_id") + "_velocity",
                             'ego vehicle', 'preceding vehicle', velocity_profile_ego,
                             velocity_profiles_lead[relevant_obs_ids[0]])

    # Create acceleration profiles
    create_animated_profiles(out_path, time, line_width_plot, writer, x_lim,
                             [min(ego_vehicle_param.get("a_min"), other_vehicle_param.get("a_min")) - 0.5,
                              max(ego_vehicle_param.get("a_max"), other_vehicle_param.get("a_max")) + 0.5],
                             '$a~[m/s^2]$', '$t~[s]$', "best",
                             simulation_param.get("commonroad_benchmark_id") + "_acceleration",
                             'ego vehicle', 'preceding vehicle', acceleration_profile_ego,
                             acceleration_profiles_lead[relevant_obs_ids[0]])


def create_animated_profiles(out_path: str, x_axis: List[float], line_width_plot: float, writer: MovieWriter,
                             x_lim: List[float], y_lim: List[float], x_label: str, y_label: str, legend: str,
                             video_name: str, label_profile_1: str, label_profile_2: str, profile_1: List[float],
                             profile_2: List[float] = None):
    """
    Creation of animated plot of provided profiles

    :param out_path: the path where the video will be saved
    :param x_axis: values for x axis
    :param line_width_plot: line width of profiles
    :param writer: Matplotlib writer object for creation of animation
    :param x_lim: limits of x-axis
    :param y_lim: limits of y-axis
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param legend: placement strategy for legend
    :param video_name: name of the video
    :param label_profile_1: label for profile 1
    :param label_profile_2: label fro profile 2
    :param profile_1: first profile to plot
    :param profile_2: second profile to plot (optional)
    """
    fig = plt.figure(figsize=(5, 5), dpi=300)
    plot_1, = plt.plot([], [], '-', color=(0.0, 0.0, 0.5, 1), label=label_profile_1, linewidth=line_width_plot)
    plot_2, = plt.plot([], [], '-', color=(0.3, 0.3, 0.3, 0.35), label=label_profile_2, linewidth=line_width_plot)
    plt.ylim(y_lim[0], y_lim[1])
    plt.ylabel(y_label)
    plt.xlim(x_lim[0], x_lim[1])
    plt.xlabel(x_label)
    plt.legend(loc=legend)
    if os.path.isfile(out_path):
        os.remove(out_path)
    with writer.saving(fig, out_path + "/" + video_name + ".mp4", dpi=300):
        for k in range(len(x_axis) + 1):
            plot_1.set_data(x_axis[0:k], profile_1[0:k])
            if profile_2 is not None:
                plot_2.set_data(x_axis[0:k], profile_2[0:k])
            writer.grab_frame()
