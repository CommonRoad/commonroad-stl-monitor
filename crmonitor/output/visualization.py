import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.visualization.scenario import (
    create_default_draw_params as create_default_draw_params_scenario,
)
from commonroad.visualization.planning import (
    create_default_draw_params as create_default_draw_params_planning,
)
from commonroad.scenario.obstacle import Obstacle


class Visualization:
    """Visualization class as interface to CommonRoad visualization"""

    def __init__(
        self,
        ego_vehicle_color: Dict[int, str] = None,
        figsize: Tuple[float, float] = (8, 4.5),
    ):
        """
        Constructor

        :param ego_vehicle_color: color for ego vehicles
        :param figsize: matplotlib figure size
        """
        self._default_parameters_scenario = create_default_draw_params_scenario()
        self._default_parameters_planning = create_default_draw_params_planning()
        self._ego_vehicle_color = ego_vehicle_color
        self._figsize = figsize
        plt.figure(figsize=self._figsize)

    def plot_scenario(
        self,
        scenario: Scenario,
        planning_problem_set: PlanningProblemSet = None,
        time_begin: int = 0,
        obstacle_label: bool = False,
        draw_trajectory: bool = False,
        lanelet_label: bool = False,
        draw_planning_problem: bool = True,
        draw_traffic_signs: bool = False,
    ):
        """
        Management of visualization for complete CommonRoad scenario

        :param scenario: CommonRoad scenario
        :param planning_problem_set: CommonRoad planning problem set
        :param time_begin: time step which should be visualized
        :param obstacle_label: boolean indicating if obstacle label should be shown
        :param draw_trajectory: boolean indicating if trajectory should be drawn
        :param lanelet_label: boolean indicating if lanelet label should be drawn
        :param draw_planning_problem: boolean indicating if planning problem should be drawn
        :param draw_traffic_signs: boolean indicating if traffic signs should be drawn
        """
        plot_limits = get_plot_limits(scenario)
        plt.clf()
        plt.gca().set_aspect("equal")
        plt.gca().set_axis_off()
        plt.margins(0, 0.1)

        if planning_problem_set is not None and draw_planning_problem is not False:
            draw_object(
                planning_problem_set,
                draw_params=self._default_parameters_planning,
                plot_limits=plot_limits,
            )
        for obs in scenario.obstacles:
            if (
                self._ego_vehicle_color is not None
                and obs.obstacle_id in self._ego_vehicle_color.keys()
            ):
                self._draw_ego_obstacle(
                    obs, plot_limits, time_begin, obstacle_label, draw_trajectory
                )
            else:
                self._draw_standard_obstacle(
                    obs, plot_limits, time_begin, obstacle_label, draw_trajectory
                )
        self._draw_lanelet_network(
            plot_limits, scenario.lanelet_network, lanelet_label, draw_traffic_signs
        )
        plt.axis("off")
        plt.show()

    def _draw_ego_obstacle(
        self,
        obstacle: Obstacle,
        plot_limits: List[float],
        time_begin: int = 0,
        obstacle_label: bool = False,
        draw_trajectory: bool = False,
    ):
        """
        Visualization of ego vehicle obstacle

        :param obstacle: CommonRoad obstacle
        :param time_begin: time step which should be visualized
        :param obstacle_label: boolean indicating if obstacle label should be shown
        :param draw_trajectory: boolean indicating if trajectory should be drawn
        """
        self._default_parameters_scenario["time_begin"] = time_begin
        self._default_parameters_scenario["scenario"]["dynamic_obstacle"]["shape"][
            "rectangle"
        ]["facecolor"] = self._ego_vehicle_color[obstacle.obstacle_id]
        self._default_parameters_scenario["scenario"]["dynamic_obstacle"]["shape"][
            "rectangle"
        ]["edgecolor"] = self._ego_vehicle_color[obstacle.obstacle_id]
        self._default_parameters_scenario["scenario"]["dynamic_obstacle"][
            "show_label"
        ] = obstacle_label
        self._default_parameters_scenario["scenario"]["dynamic_obstacle"]["trajectory"][
            "draw_trajectory"
        ] = draw_trajectory
        draw_object(
            obstacle,
            draw_params=self._default_parameters_scenario,
            plot_limits=plot_limits,
        )

    def _draw_standard_obstacle(
        self,
        obstacle: Obstacle,
        plot_limits: List[float],
        time_begin: int = 0,
        obstacle_label: bool = False,
        draw_trajectory: bool = False,
    ):
        """
        Visualization of non-ego vehicle obstacle

        :param obstacle: CommonRoad obstacle
        :param time_begin: time step which should be visualized
        :param obstacle_label: boolean indicating if obstacle label should be shown
        :param draw_trajectory: boolean indicating if trajectory should be drawn
        """
        self._default_parameters_scenario["time_begin"] = time_begin
        self._default_parameters_scenario["scenario"]["dynamic_obstacle"][
            "show_label"
        ] = obstacle_label
        self._default_parameters_scenario["scenario"]["dynamic_obstacle"]["trajectory"][
            "draw_trajectory"
        ] = draw_trajectory
        self._default_parameters_scenario["scenario"]["dynamic_obstacle"][
            "show_label"
        ] = obstacle_label
        draw_object(
            obstacle,
            draw_params=self._default_parameters_scenario,
            plot_limits=plot_limits,
        )

    def _draw_lanelet_network(
        self,
        plot_limits: List[float],
        lanelet_network: LaneletNetwork,
        lanelet_label: bool = False,
        draw_traffic_signs: bool = False,
    ):
        """
        Visualization of lanelet network

        :param lanelet_network: CommonRoad lanelet network
        :param lanelet_label: boolean indicating if lanelet label should be shown
        :param draw_traffic_signs: boolean indicating if traffic signs should be drawn
        """
        self._default_parameters_scenario["lanelet_network"]["lanelet"][
            "show_label"
        ] = lanelet_label
        self._default_parameters_scenario["lanelet_network"]["lanelet"][
            "draw_start_and_direction"
        ] = False
        self._default_parameters_scenario["lanelet_network"]["lanelet"][
            "show_label"
        ] = lanelet_label
        draw_object(
            lanelet_network,
            draw_params=self._default_parameters_scenario,
            plot_limits=plot_limits,
        )


def get_plot_limits(scenario: Scenario) -> List[float]:
    """
    Extracts plot limits of a scenario by searching for smalles/largest x-, y-value of lanelets
    and adds small margin to limits

    :param scenario: CommonRoad scenario
    :returns x/y min/max plot limits
    """
    x_lanelet_left = [
        point[0]
        for lanelet in scenario.lanelet_network.lanelets
        for point in lanelet.left_vertices
    ]
    y_lanelet_left = [
        point[1]
        for lanelet in scenario.lanelet_network.lanelets
        for point in lanelet.left_vertices
    ]
    x_lanelet_right = [
        point[0]
        for lanelet in scenario.lanelet_network.lanelets
        for point in lanelet.right_vertices
    ]
    y_lanelet_right = [
        point[1]
        for lanelet in scenario.lanelet_network.lanelets
        for point in lanelet.right_vertices
    ]

    x_min = min(x_lanelet_left + x_lanelet_right) - 5
    y_min = min(y_lanelet_left + y_lanelet_right) - 5
    x_max = max(x_lanelet_left + x_lanelet_right) + 5
    y_max = max(y_lanelet_left + y_lanelet_right) + 5
    plot_limits = [x_min, x_max, y_min, y_max]

    return plot_limits


def get_scenario_duration(scenario: Scenario) -> int:
    """
    Extracts maximum time step of all obstacles

    :param scenario: CommonRoad scenario
    :returns maximum time step of all obstacles
    """
    duration = 0
    for obs in scenario.dynamic_obstacles:
        if duration < obs.prediction.trajectory.state_list[-1].time_step:
            duration = obs.prediction.trajectory.state_list[-1].time_step
    return duration
