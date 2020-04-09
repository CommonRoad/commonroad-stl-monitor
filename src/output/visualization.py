import matplotlib.pyplot as plt
from typing import List, Tuple


from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.visualization.scenario import create_default_draw_params as create_default_draw_params_scenario
from commonroad.visualization.planning import create_default_draw_params as create_default_draw_params_planning
from commonroad.scenario.obstacle import Obstacle


class Visualization:
    """Visualization class as interface to CommonRoad visualization"""
    def __init__(self, ego_vehicle_ids: List[int] = None, ego_vehicle_color: str = '#1d7eea',
                 figsize: Tuple[float, float] = (8, 4.5)):
        """
        Constructor

        :param ego_vehicle_ids: list of IDs of ego vehicles
        :param ego_vehicle_color: color for ego vehicles
        :param figsize: matplotlib figure size
        """
        self._default_parameters_scenario = create_default_draw_params_scenario()
        self._default_parameters_planning = create_default_draw_params_planning()
        self._ego_vehicle_ids = ego_vehicle_ids
        self._ego_vehicle_color = ego_vehicle_color
        self._figsize = figsize
        plt.figure(figsize=self._figsize)

    def plot_scenario(self, scenario: Scenario, planning_problem_set: PlanningProblemSet = None, time_begin: int = 0,
                      obstacle_label: bool = False, draw_trajectory: bool = False, lanelet_label: bool = False):
        """
        Management of visualization for complete CommonRoad scenario

        :param scenario: CommonRoad scenario
        :param planning_problem_set: CommonRoad planning problem set
        :param time_begin: time step which should be visualized
        :param obstacle_label: boolean indicating if obstacle label should be shown
        :param draw_trajectory: boolean indicating if trajectory should be drawn
        :param lanelet_label: boolean indicating if lanelet label should be drawn
        """
        x_lanelet_left = [point[0] for lanelet in scenario.lanelet_network.lanelets for point in lanelet.left_vertices]
        y_lanelet_left = [point[1] for lanelet in scenario.lanelet_network.lanelets for point in lanelet.left_vertices]
        x_lanelet_right = [point[0] for lanelet in scenario.lanelet_network.lanelets
                           for point in lanelet.right_vertices]
        y_lanelet_right = [point[1] for lanelet in scenario.lanelet_network.lanelets
                           for point in lanelet.right_vertices]

        x_min = min(x_lanelet_left + x_lanelet_right) - 5
        y_min = min(y_lanelet_left + y_lanelet_right) - 5
        x_max = max(x_lanelet_left + x_lanelet_right) + 5
        y_max = max(y_lanelet_left + y_lanelet_right) + 5
        plot_limits = [x_min, x_max, y_min, y_max]

        plt.clf()
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.margins(0, 0.1)

        self._draw_lanelet_network(scenario.lanelet_network, lanelet_label)
        if planning_problem_set is not None:
            draw_object(planning_problem_set, draw_params=self._default_parameters_planning, plot_limits=plot_limits)
        for obs in scenario.obstacles:
            if self._ego_vehicle_ids is not None and obs.obstacle_id in self._ego_vehicle_ids:
                self._draw_ego_obstacle(obs, time_begin, obstacle_label, draw_trajectory)
            else:
                self._draw_standard_obstacle(obs, time_begin, obstacle_label, draw_trajectory)
        plt.axis('off')
        plt.show()

    def _draw_ego_obstacle(self, obstacle: Obstacle, time_begin: int = 0, obstacle_label: bool = False,
                           draw_trajectory: bool = False):
        """
        Visualization of ego vehicle obstacle

        :param obstacle: CommonRoad obstacle
        :param time_begin: time step which should be visualized
        :param obstacle_label: boolean indicating if obstacle label should be shown
        :param draw_trajectory: boolean indicating if trajectory should be drawn
        """
        self._default_parameters_scenario['time_begin'] = time_begin
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['facecolor'] = \
            self._ego_vehicle_color
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['edgecolor'] = \
            self._ego_vehicle_color
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['show_label'] = obstacle_label
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['trajectory']['draw_trajectory'] = \
            draw_trajectory
        draw_object(obstacle, draw_params=self._default_parameters_scenario)

    def _draw_standard_obstacle(self, obstacle: Obstacle, time_begin: int = 0, obstacle_label: bool = False,
                                draw_trajectory: bool = False):
        """
        Visualization of non-ego vehicle obstacle

        :param obstacle: CommonRoad obstacle
        :param time_begin: time step which should be visualized
        :param obstacle_label: boolean indicating if obstacle label should be shown
        :param draw_trajectory: boolean indicating if trajectory should be drawn
        """
        self._default_parameters_scenario['time_begin'] = time_begin
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['shape']['rectangle']['facecolor'] = '#1d7eea'
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['show_label'] = obstacle_label
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['trajectory']['draw_trajectory'] = \
            draw_trajectory
        self._default_parameters_scenario['scenario']['dynamic_obstacle']['show_label'] = obstacle_label
        draw_object(obstacle, draw_params=self._default_parameters_scenario)

    def _draw_lanelet_network(self, lanelet_network: LaneletNetwork, lanelet_label: bool = False):
        """
        Visualization of lanelet network

        :param lanelet_network: CommonRoad lanelet network
        :param lanelet_label: boolean indicating if lanelet label should be shown
        """
        self._default_parameters_scenario['lanelet_network']['lanelet']['show_label'] = lanelet_label
        draw_object(lanelet_network, draw_params=self._default_parameters_scenario)

    # def _plot_scenario_at_time_idx(self, time_idx: int, scenario: Scenario, obstacle_label: bool):
    #     """
    #     Plots a scenario at a specific point in time.
    #     :param time_idx: the time point for which the scenario will be plotted
    #     :param scenario: the scenario to be plotted
    #     :param obstacle_label: boolean indicating if obstacle label (ID) should be visualized
    #     """
    #     self._default_parameters['time_begin'] = time_idx
    #     self._default_parameters['time_end'] = time_idx
    #     if obstacle_label:
    #         self._default_parameters['scenario']['dynamic_obstacle']['show_label'] = True
    #     self._default_parameters['scenario']['dynamic_obstacle']['shape']['rectangle']['facecolor'] = '#ff4000'
    #     self._default_parameters['scenario']['dynamic_obstacle']['shape']['rectangle']['edgecolor'] = '#cc3300'
    #     self._default_parameters['scenario']['dynamic_obstacle']['occupancy']['shape']['polygon']['opacity'] = .1
    #     draw_object(scenario, draw_params=self._default_parameters)

    # def _plot_vehicle_at_time_idx(self, ego_obstacle: DynamicObstacle):
    #     """
    #     Plots the occupancy of a vehicle at a specific time point given its trajectory
    #     :param ego_obstacle: ego vehicle as CommonRoad object
    #     """
    #     draw_object(ego_obstacle, draw_params=self._default_parameters)
    #
    # def create_video(self, out_path: str, scenario: Scenario, planning_problem_set: PlanningProblemSet= None):
    #     if planning_problem_set is None:
    #         create_scenario_video(scenario, )
    #
    # def create_scenario_video(self, out_path: str, scenario: Scenario, visualization_param: Dict, video_length: int,
    #                           ego_obstacle: DynamicObstacle = None, planning_problem_set: PlanningProblemSet = None):
    #     """
    #     Creates a video of the solution for a specific planning problem
    #     :param out_path: The path where the video will be saved
    #     :param scenario: The scenario of the planning problem that was solved
    #     :param ego_obstacle: ego vehicle as CommonRoad object
    #     :param visualization_param: dictionary with parameters for plotting of profiles
    #     :param planning_problem_set: The planning problem set in which the planning problem belongs
    #     """
    #     filename = ntpath.basename(out_path)
    #     obstacle_label = visualization_param.get("obstacle_label")
    #     ffmpeg_writer = animation.writers['ffmpeg']
    #     metadata = dict("", artist='TUM CPS GROUP')
    #     writer = ffmpeg_writer(fps=10, metadata=metadata)
    #
    #     fig = plt.figure(figsize=(25, 10))
    #     plt.xlabel('[m]')
    #     plt.ylabel('[m]')
    #     plt.title(filename)
    #
    #     # find figure size
    #     x = [x for lanelet in scenario.lanelet_network.lanelets for x in lanelet.center_vertices[:, 0]]
    #     y = [y for lanelet in scenario.lanelet_network.lanelets for y in lanelet.center_vertices[:, 1]]
    #     x_min = min(x) - 5
    #     y_min = min(y) - 5
    #     x_max = max(x) + 5
    #     y_max = max(y) + 5
    #
    #     if os.path.isfile(out_path):
    #         os.remove(out_path)
    #     with writer.saving(fig, out_path + "/" + scenario.benchmark_id + ".mp4", dpi=150):
    #         for t in range(video_length):
    #             plt.cla()
    #             self._plot_scenario_at_time_idx(t, scenario, obstacle_label)
    #             if planning_problem_set is not None:
    #                 draw_object(planning_problem_set)
    #             if ego_obstacle is not None:
    #                 self._plot_vehicle_at_time_idx(ego_obstacle)
    #             plt.gca().set_aspect('equal')
    #             plt.gca().set_xlim([x_min, x_max])
    #             plt.gca().set_ylim([y_min, y_max])
    #             writer.grab_frame()

    #
    # def create_ego_profiles(ego_vehicle: Vehicle) -> Tuple[List[int], List[float], List[float], List[float]]:
    #     """
    #     Creates acceleration, velocity and time profiles for ego vehicle
    #
    #     :param ego_vehicle: ego vehicle object
    #     :return: lists with ego vehicle profiles
    #     """
    #     jerk_list = []
    #     acceleration_list = []
    #     velocity_list = []
    #     time = []
    #     for time_step, state in ego_vehicle.states_lon.items():
    #         acceleration_list.append(state.a)
    #         velocity_list.append(state.v)
    #         time.append(time_step)
    #     for jerk in ego_vehicle.jerk_profile.values():
    #         jerk_list.append(jerk)
    #     jerk_list = [jerk_list[0]] + jerk_list
    #
    #     return time, jerk_list, acceleration_list, velocity_list
    #
    #
    # def create_lead_profiles(ego_vehicle: Vehicle, vehicles: Dict, time: List[int], relevant_obs_ids: List[int]) -> \
    #         Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]]]:
    #     """
    #     Creates acceleration, velocity and distance profiles for leading vehicles
    #
    #     :param ego_vehicle: ego vehicle object
    #     :param vehicles: list with leading vehicle objects
    #     :param time: list with time steps where ego vehicle exists
    #     :param relevant_obs_ids: list with IDs of relevant obstacles
    #     :returns lists with leading vehicle profiles
    #     """
    #     acceleration_plots = {}
    #     velocity_plots = {}
    #     distance_plots = {}
    #     safe_distance_plots = {}
    #     for vehicle in vehicles.values():
    #         if vehicle.id in relevant_obs_ids:
    #             veh_acceleration_profile = []
    #             veh_velocity_profile = []
    #             veh_distance_profile = []
    #             veh_safe_distance_profile = []
    #             for time_step in time:
    #                 if vehicle.states_lon.get(time_step) is not None:
    #                     veh_acceleration_profile.append(vehicle.states_lon.get(time_step).a)
    #                 else:
    #                     veh_acceleration_profile.append(0)
    #                 if vehicle.states_lon.get(time_step) is not None:
    #                     veh_velocity_profile.append(vehicle.states_lon.get(time_step).v)
    #                 else:
    #                     veh_velocity_profile.append(0)
    #                 if vehicle.states_lon.get(time_step) is not None:
    #                     veh_distance_profile.append(vehicle.rear_s(time_step) - ego_vehicle.front_s(time_step))
    #                 else:
    #                     veh_distance_profile.append(0)
    #                 if vehicle.states_lon.get(time_step) is not None:
    #                     veh_safe_distance_profile.append(vehicle.safe_distance_list.get(time_step))
    #                 else:
    #                     veh_safe_distance_profile.append(0)
    #
    #             acceleration_plots[vehicle.id] = veh_acceleration_profile
    #             velocity_plots[vehicle.id] = veh_velocity_profile
    #             distance_plots[vehicle.id] = veh_distance_profile
    #             safe_distance_plots[vehicle.id] = veh_safe_distance_profile
    #     return acceleration_plots, velocity_plots, distance_plots, safe_distance_plots
    #
    #
    # def num_vehicle_profiles(number_vehicles: List[Tuple[int]]) -> Tuple[List[int], List[int], List[int]]:
    #     """
    #     Creates profiles with number of vehicles in same lane, in reduced same lane, and of cut-in vehicles
    #
    #     :param number_vehicles: list of tuples for each time step
    #     :returns lists with number of vehicles for each category
    #     """
    #     num_veh_same, num_veh_same_reduced, num_veh_cutin = [], [], []
    #     for num_veh_time_step in number_vehicles:
    #         num_veh_same.append(num_veh_time_step[0])
    #         num_veh_same_reduced.append(num_veh_time_step[1])
    #         num_veh_cutin.append(num_veh_time_step[2])
    #
    #     return num_veh_same, num_veh_same_reduced, num_veh_cutin
    #
    #
    # def get_date_and_time() -> str:
    #     """
    #     Returns current data and time
    #
    #     :return: Current date and time as string
    #     """
    #     current_time = datetime.now().time()
    #     current_time = str(current_time)
    #     current_time = current_time.replace(':', ' _')
    #     current_time = current_time.replace('.', ' _')
    #     current_date = str(datetime.now().day) + "_" + str(datetime.now().month) + "_" + str(datetime.now().year)
    #
    #     return current_date + "_" + current_time
    #
    #
    # def plot_figures(ego_vehicle: Vehicle, vehicles: Dict, emergency_maneuver_activity: List[bool],
    #                  ego_vehicle_param: Dict, simulation_param: Dict, plot_param: Dict, comp_time: List[float],
    #                  number_vehicles: List[Tuple[int]]):
    #     """
    #     Plotting of positions, acceleration, velocity, and distance of ACC and preceding vehicles, respectively
    #
    #     :param ego_vehicle: ego vehicle object
    #     :param vehicles: dictionary with vehicle objects of surrounding vehicles
    #     :param emergency_maneuver_activity: emergency maneuver acitivity per time step
    #     :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    #     :param simulation_param: dictionary with parameters of the simulation environment
    #     :param plot_param: dictionary with parameters for plotting of profiles
    #     :param comp_time: list with computation time at each simulation step
    #     :param strategy_list: list with strategy at each simulation step
    #     :param number_vehicles: list of tuples with number of vehicles at (reduced) same lane and cutting in
    #     """
    #
    #     # Storage related configuration
    #     width = plot_param.get("width")
    #     height = width * (math.sqrt(5) - 1.0) / 2.0
    #     figsize = [width, height]
    #     linewidth_plot = plot_param.get("line_width")
    #     mp.rcParams.update({'font.size': plot_param.get("font_size")})
    #     mp.rcParams.update({'axes.linewidth': plot_param.get("axes_line_width")})
    #     mp.rcParams.update({'figure.autolayout': True})
    #     mp.rcParams.update({'legend.frameon': False})
    #
    #     # Directory creation if plots should be stored
    #     if simulation_param.get("store_plots"):
    #         date_time = get_date_and_time()
    #         if not os.path.exists("figures/"):
    #             os.mkdir("figures/")
    #         path = "figures/" + date_time
    #         os.mkdir(path)
    #
    #     relevant_obs_ids = simulation_param.get("other_vehicle_plots")
    #
    #     # Create profiles
    #     time_steps, jerk_profile_ego, acceleration_profile_ego, velocity_profile_ego = create_ego_profiles(ego_vehicle)
    #     num_veh_same, num_veh_same_reduced, num_veh_cutin = num_vehicle_profiles(number_vehicles)
    #     acceleration_profiles_lead, velocity_profiles_lead, distance_profiles_lead, safe_distance_profiles_lead = \
    #         create_lead_profiles(ego_vehicle, vehicles, time_steps, relevant_obs_ids)
    #     time = [time_step * simulation_param.get("dt") for time_step in time_steps]
    #
    #     # Distance plot
    #     if len(relevant_obs_ids) > 0:
    #         plt.figure(1, figsize=figsize)
    #         plt.ylabel(r'$s~[m]$')
    #         plt.xlabel(r'$t~[s]$')
    #         for obs_id in relevant_obs_ids:
    #             plt.plot(time, distance_profiles_lead.get(obs_id), color=(0.0, 0.0, 0.5, 1), label=r'$s_{lead}$',
    #                      linewidth=linewidth_plot)
    #             plt.plot(time, safe_distance_profiles_lead.get(obs_id), "-", color=(0.3, 0.3, 0.3, 0.35),
    #                      label=r'$s_{safe}$', linewidth=linewidth_plot)
    #         plt.legend(loc='best')
    #         if simulation_param.get("store_plots"):
    #             plt.savefig(path + "/distance" + ".svg", format="svg")
    #
    #     # Velocity plot
    #     plt.figure(2, figsize=figsize)
    #     plt.ylabel(r'$v~[m/s]$')
    #     plt.xlabel(r'$t~[s]$')
    #     for obs_id in relevant_obs_ids:
    #         plt.plot(time, velocity_profiles_lead.get(obs_id), "-", color=(0.0, 0.0, 0.5, 1), label=r'$v_{lead}$',
    #                  linewidth=linewidth_plot)
    #     plt.plot(time, velocity_profile_ego, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$v_{acc}$', linewidth=linewidth_plot)
    #     plt.legend(loc='best')
    #     if simulation_param.get("store_plots"):
    #         plt.savefig(path + "/velocity" + ".svg", format="svg")
    #
    #     # Acceleration plot
    #     plt.figure(3, figsize=figsize)
    #     plt.ylabel(r'$a~[m/s^2]$')
    #     plt.xlabel(r'$t~[s]$')
    #     plt.ylim([ego_vehicle_param.get("a_min") - 0.5, ego_vehicle_param.get("a_max") + 0.5])
    #     for obs_id in relevant_obs_ids:
    #         plt.step(time, acceleration_profiles_lead.get(obs_id), color=(0.0, 0.0, 0.5, 1), label=r'$a_{lead}$',
    #                  linewidth=linewidth_plot)
    #     plt.step(time, acceleration_profile_ego, color=(0.3, 0.3, 0.3, 0.35), label=r'$a_{acc}$', linewidth=linewidth_plot)
    #     plt.legend(loc='best')
    #     if simulation_param.get("store_plots"):
    #         plt.savefig(path + "/acceleration" + ".svg", format="svg")
    #
    #     # Jerk plot
    #     plt.figure(4, figsize=figsize)
    #     plt.ylabel(r'$j~[m/s^3]$')
    #     plt.xlabel(r'$t~[s]$')
    #     plt.step(time, jerk_profile_ego, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$j_{acc}$', linewidth=linewidth_plot)
    #     plt.legend(loc='best')
    #     if simulation_param.get("store_plots"):
    #         plt.savefig(path + "/jerk" + ".svg", format="svg")
    #
    #     # Emergency maneuver activity plot
    #     plt.figure(5, figsize=figsize)
    #     plt.xlabel(r'$t~[s]$')
    #     plt.step(time, emergency_maneuver_activity, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$emg$',
    #              linewidth=linewidth_plot)
    #     plt.legend(loc='best')
    #     if simulation_param.get("store_plots"):
    #         plt.savefig(path + "/emg" + ".svg", format="svg")
    #
    #     # Computation time plot
    #     if simulation_param.get("time_measuring"):
    #         plt.figure(6, figsize=figsize)
    #         plt.ylabel(r'$t~[s]$')
    #         plt.xlabel(r'$t~[s]$')
    #         plt.step(time[0:-1], comp_time, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$t_{comp}$', linewidth=linewidth_plot)
    #         plt.legend(loc='best')
    #         if simulation_param.get("store_plots"):
    #             plt.savefig(path + "/comp_time" + ".svg", format="svg")
    #
    #     # Number vehicles plot
    #     plt.figure(8, figsize=figsize)
    #     plt.xlabel(r'$t~[s]$')
    #     plt.step(time[0:-1], num_veh_same, "-", color=(0.3, 0.3, 0.3, 0.35), label=r'$same~lane$', linewidth=linewidth_plot)
    #     plt.step(time[0:-1], num_veh_same_reduced, "-", color=(0.0, 0.0, 0.5, 1),
    #              label=r'$same~lane~reduced$', linewidth=linewidth_plot)
    #     plt.step(time[0:-1], num_veh_cutin, "-", color=(0.75, 0.2, 0.6, 1), label=r'$cut-in$', linewidth=linewidth_plot)
    #     plt.legend(loc='best')
    #     if simulation_param.get("store_plots"):
    #         plt.savefig(path + "/strategy" + ".svg", format="svg")
    #     plt.show()
    #
    #
    # def create_profile_videos(out_path: str, ego_vehicle: Vehicle, vehicles: Dict,
    #                           ego_vehicle_param: Dict, other_vehicle_param: Dict,
    #                           simulation_param: Dict, number_vehicles: List[Tuple[int]]):
    #     """
    #     Plotting of positions, acceleration, velocity, and distance of ACC and preceding vehicles, respectively
    #
    #     :param out_path: the path where the video will be saved
    #     :param ego_vehicle: ego vehicle object
    #     :param vehicles: dictionary with vehicle objects of surrounding vehicles
    #     :param emergency_maneuver_activity: emergency maneuver activity per time step
    #     :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    #     :param other_vehicle_param: dictionary with physical parameters of other vehicles
    #     :param simulation_param: dictionary with parameters of the simulation environment
    #     :param plot_param: dictionary with parameters for plotting of profiles
    #     :param comp_time: list with computation time at each simulation step
    #     :param strategy_list: list with strategy at each simulation step
    #     :param number_vehicles: list of tuples with number of vehicles at (reduced) same lane and cutting in
    #     """
    #     relevant_obs_ids = simulation_param.get("other_vehicle_plots")
    #     if len(relevant_obs_ids) > 1:
    #         warnings.warn('Only the first leading vehicle will be plotted in the profile videos.')
    #
    #     # Create profiles
    #     time_steps, jerk_profile_ego, acceleration_profile_ego, velocity_profile_ego = create_ego_profiles(ego_vehicle)
    #     num_veh_same, num_veh_same_reduced, num_veh_cutin = num_vehicle_profiles(number_vehicles)
    #     acceleration_profiles_lead, velocity_profiles_lead, distance_profiles_lead, safe_distance_profiles_lead = \
    #         create_lead_profiles(ego_vehicle, vehicles, time_steps, relevant_obs_ids)
    #     time = [time_step * simulation_param.get("dt") for time_step in time_steps]
    #
    #     ffmpeg_writer = animation.writers['ffmpeg']
    #     metadata = dict(title="", artist='TUM CPS GROUP')
    #     writer = ffmpeg_writer(fps=10, metadata=metadata)
    #
    #     line_width_plot = 1.75
    #     plt.rcParams.update({'font.size': 14})
    #     plt.rcParams.update({'legend.frameon': False})
    #     x_lim = [0, time[-1]]
    #
    #     # Create velocity profiles
    #     create_animated_profiles(out_path, time, line_width_plot, writer, x_lim,
    #                              [min(ego_vehicle_param.get("v_min"), other_vehicle_param.get("v_min")) - 0.5,
    #                               max(ego_vehicle_param.get("v_max"), other_vehicle_param.get("v_max")) + 0.5],
    #                              '$v~[m/s]$', '$t~[s]$', "best",
    #                              simulation_param.get("commonroad_benchmark_id") + "_velocity",
    #                              'ego vehicle', 'preceding vehicle', velocity_profile_ego,
    #                              velocity_profiles_lead[relevant_obs_ids[0]])
    #
    #     # Create acceleration profiles
    #     create_animated_profiles(out_path, time, line_width_plot, writer, x_lim,
    #                              [min(ego_vehicle_param.get("a_min"), other_vehicle_param.get("a_min")) - 0.5,
    #                               max(ego_vehicle_param.get("a_max"), other_vehicle_param.get("a_max")) + 0.5],
    #                              '$a~[m/s^2]$', '$t~[s]$', "best",
    #                              simulation_param.get("commonroad_benchmark_id") + "_acceleration",
    #                              'ego vehicle', 'preceding vehicle', acceleration_profile_ego,
    #                              acceleration_profiles_lead[relevant_obs_ids[0]])
    #
    #
    # def create_animated_profiles(out_path: str, x_axis: List[float], line_width_plot: float, writer: MovieWriter,
    #                              x_lim: List[float], y_lim: List[float], x_label: str, y_label: str, legend: str,
    #                              video_name: str, label_profile_1: str, label_profile_2: str, profile_1: List[float],
    #                              profile_2: List[float] = None):
    #     """
    #     Creation of animated plot of provided profiles
    #
    #     :param out_path: the path where the video will be saved
    #     :param x_axis: values for x axis
    #     :param line_width_plot: line width of profiles
    #     :param writer: Matplotlib writer object for creation of animation
    #     :param x_lim: limits of x-axis
    #     :param y_lim: limits of y-axis
    #     :param x_label: label for x-axis
    #     :param y_label: label for y-axis
    #     :param legend: placement strategy for legend
    #     :param video_name: name of the video
    #     :param label_profile_1: label for profile 1
    #     :param label_profile_2: label fro profile 2
    #     :param profile_1: first profile to plot
    #     :param profile_2: second profile to plot (optional)
    #     """
    #     fig = plt.figure(figsize=(5, 5), dpi=300)
    #     plot_1, = plt.plot([], [], '-', color=(0.0, 0.0, 0.5, 1), label=label_profile_1, linewidth=line_width_plot)
    #     plot_2, = plt.plot([], [], '-', color=(0.3, 0.3, 0.3, 0.35), label=label_profile_2, linewidth=line_width_plot)
    #     plt.ylim(y_lim[0], y_lim[1])
    #     plt.ylabel(y_label)
    #     plt.xlim(x_lim[0], x_lim[1])
    #     plt.xlabel(x_label)
    #     plt.legend(loc=legend)
    #     if os.path.isfile(out_path):
    #         os.remove(out_path)
    #     with writer.saving(fig, out_path + "/" + video_name + ".mp4", dpi=300):
    #         for k in range(len(x_axis) + 1):
    #             plot_1.set_data(x_axis[0:k], profile_1[0:k])
    #             if profile_2 is not None:
    #                 plot_2.set_data(x_axis[0:k], profile_2[0:k])
    #             writer.grab_frame()
