from collections import defaultdict
from copy import deepcopy
from enum import Enum
import textwrap
from functools import singledispatchmethod
from itertools import groupby
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from rtamt.syntax.node.abstract_node import AbstractNode as RtamtAbstractNode
from rtamt.syntax.node.binary_node import BinaryNode as RtamtBinaryNode
from rtamt.syntax.node.ltl.variable import Variable as RtamtVariableNode
from rtamt.syntax.node.unary_node import UnaryNode as RtamtUnaryNode

from crmonitor.evaluation.visitor import MonitorToStringVisitor
from crmonitor.monitor.monitor_node import (
    MonitorNode,
    MonitorVisitorInterface,
    PredicateMonitorNode,
    QuantMonitorNode,
    RuleMonitorNode,
    UnaryMonitorNode,
)
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.predicates.scaling import RobustnessScaler

EGO_VEHICLE_DRAW_PARAMS = {
    "dynamic_obstacle": {
        "vehicle_shape": {"occupancy": {"shape": {"rectangle": {"facecolor": "yellow"}}}}
    }
}


class TUMcolor(Enum):
    TUMblue = [0, 101 / 255, 189 / 255]
    TUMgreen = [162 / 255, 173 / 255, 0]
    TUMgray = [156 / 255, 157 / 255, 159 / 255]
    TUMdarkgray = [88 / 255, 88 / 255, 99 / 255]
    TUMorange = [227 / 255, 114 / 255, 34 / 255]
    TUMdarkblue = [0, 82 / 255, 147 / 255]
    TUMwhite = [1, 1, 1]
    TUMblack = [0, 0, 0]
    TUMlightgray = [217 / 255, 218 / 255, 219 / 255]


def plot_rule_robustness_course(
    rule_robustness_course: List[Tuple[int, float]],
    ax,
    plot_limits: Tuple[float, float],
    rules: List[str],
):
    np_rule_robustness_course = np.array(rule_robustness_course)
    rob_values = np_rule_robustness_course[:, 1]
    times = np_rule_robustness_course[:, 0]
    ax.plot(rob_values, "b-")
    ax.plot(times, np.where(rob_values < 0.0, rob_values, np.nan), "rx")
    ax.plot(times, np.where(rob_values >= 0.0, rob_values, np.nan), "g.")
    ax.set_xlim([np_rule_robustness_course[0, 0], np_rule_robustness_course[-1, 0]])
    ax.set_ylim(plot_limits)
    ax.grid(True)
    ax.set_ylabel(f"robustness of rule {','.join(rules)}")
    ax.set_xlabel("time step")


def plot_predicate_bar_chart(
    predicate_vehicle_values: Dict[str, Dict[Tuple[int, ...], float]],
    ax,
    bar_chart_plot_limits: Tuple[float, float],
):
    df = pd.DataFrame.from_dict(
        {
            predicate_name: {
                str(vehicle_ids): values for vehicle_ids, values in vehicle_ids2values.items()
            }
            for predicate_name, vehicle_ids2values in predicate_vehicle_values.items()
        }
    )

    # we use a different color map, as default one produces non-distinguishable
    # colors for different bars
    cmap = plt.get_cmap("turbo")
    numbers_for_bars = np.linspace(0, 1, num=len(df.columns), endpoint=False)
    ax = df.plot.barh(
        rot=0,
        ax=ax,
        width=1.0,
        edgecolor="black",
        linewidth=0.5,
        color=cmap(numbers_for_bars),
        xlim=bar_chart_plot_limits,
    )
    ax.set_ylabel("vehicle ids")
    ax.set_xlabel("predicate robustness")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))  # place legend to the right


def _create_axes(
    scenario_fig_size: Tuple[float, float],
    nr_rules: int,
    flag_plot_predicate_bar_chart: bool,
    flag_plot_rule_robustness_course: bool,
    flag_rule_conjunction: bool,
):
    bar_plots = []
    rob_plots = []

    if flag_plot_predicate_bar_chart or flag_plot_rule_robustness_course:
        fig = plt.figure(
            constrained_layout=True,
            figsize=(scenario_fig_size[0], scenario_fig_size[1] * (1 + nr_rules)),
        )
        if flag_rule_conjunction:
            n_rows = 2
        else:
            n_rows = nr_rules + 1
        n_cols = sum([flag_plot_predicate_bar_chart, flag_plot_rule_robustness_course])
        gs = GridSpec(nrows=n_rows, ncols=n_cols, figure=fig)
        scenario_ax = fig.add_subplot(gs[0, :])
        rob_plot_index = 0
        for r in range(n_rows - 1):
            if flag_plot_predicate_bar_chart:
                bar_plots.append(fig.add_subplot(gs[r + 1, 0]))
                rob_plot_index = 1
            if flag_plot_rule_robustness_course:
                rob_plots.append(fig.add_subplot(gs[r + 1, rob_plot_index]))
    else:
        plt.figure(figsize=scenario_fig_size)
        scenario_ax = plt.gca()

    return scenario_ax, bar_plots, rob_plots


def _plot_scenario_legend(
    predicate_name2predicate_evaluator: Dict[str, BasePredicateEvaluator],
    scenario_fig_size: Tuple[float, float],
):
    width, _ = scenario_fig_size
    figsize = (width, width / 4)
    num_predicates = len(predicate_name2predicate_evaluator)
    fig, (axes_row_1, axes_row_2) = plt.subplots(figsize=figsize, nrows=2, ncols=num_predicates)
    fig.suptitle("Legend: predicate visualization in scenario", fontsize=14)
    for ax1, ax2, (pred_name, pred_evaluator) in zip(
        axes_row_1, axes_row_2, predicate_name2predicate_evaluator.items()
    ):
        ax1.text(0.1, 0.5, pred_name, fontsize=12)
        ax1.axis("off")
        pred_evaluator.plot_predicate_visualization_legend(ax2)


def plot_rule_visualization(
    scenario: Scenario,
    ego_vehicle_id: int,
    time_step: int,
    rule_evaluator_list,
    visualization_config: Dict[str, any],
    scenario_fig_size: Tuple[float, float] = (10.0, 2.0),
    bar_chart_plot_limits: Tuple[float, float] = (-1.0, 1.0),
    rule_robustness_course_plot_limits: Tuple[float, float] = (-1.0, 1.0),
    flag_plot_predicate_bar_chart: bool = True,
    flat_plot_rule_robustness_course: bool = True,
    scenario_plot_limits: Union[List[Union[int, float]], None] = None,
    flag_rule_conjunction: bool = False,
    plot_scenario_legend: Optional[bool] = None,
):
    """
    Plotting the rule evaluation result

    :param scenario: the CommonRoad scenario to be visualized
    :param ego_vehicle_id: id of ego vehicle (the vehicle to be controlled)
    :param time_step: the time step of the current scenario
    :param rule_evaluator_list: precreated list of rule evaluators
    :param visualization_config: user-defined configuration of visualization
    :param scenario_fig_size: size of scenario plot
    :param bar_chart_plot_limits: the plot limits of x-axis
    :param rule_robustness_course_plot_limits: the plot limits of x-axis
    :param flag_plot_predicate_bar_chart: flag of whether the bar chart needs to be
        plotted
    :param flat_plot_rule_robustness_course: flag of whether the robustness curve
        needs to be plotted
    :param scenario_plot_limits: the plot limits of scenario,
    :param flag_rule_conjunction: whether consider the conjunction of rules or
        separately calculate them
    :param plot_scenario_legend: whether the legend for the scenario visualization
        should be plotted. If None, it is plotted for the first time-step only
    """

    nr_rules = len(rule_evaluator_list)

    general_draw_params = {
        "time_begin": time_step,
        "dynamic_obstacle": {
            "show_label": True,
            "vehicle_shape": {"occupancy": {"shape": {"rectangle": {"facecolor": "#90ee90"}}}},
        },
    }

    vehicle2draw_params = {}
    pred_result_dict = {}
    rule_result_dict = {}
    rule_name_list = []
    all_predicate_name2predicate_evaluator = {}
    # Hint: plotting further stuff on the scenario only works after renderer.render()
    # was called; therefore, the predicates need to return functions instead of directly
    # plotting
    all_draw_functions = []
    for i in range(nr_rules):
        rule_evaluator_list[i].update()
        (
            predicate_name2predicate_evaluator,
            pred_result,
            rule_result,
            draw_functions,
        ) = rule_evaluator_list[i].visualize_predicates(vehicle2draw_params, visualization_config)
        all_predicate_name2predicate_evaluator.update(predicate_name2predicate_evaluator)
        all_draw_functions += draw_functions
        pred_result_dict[rule_evaluator_list[i]._rule.name] = pred_result  # merge the dict
        rule_result_dict[rule_evaluator_list[i]._rule.name] = rule_result
        rule_name_list.append(rule_evaluator_list[i]._rule.name)

    # if plot_scenario_legend or plot_scenario_legend is None and time_step == 0:
    #     _plot_scenario_legend(all_predicate_name2predicate_evaluator, scenario_fig_size)

    scenario_ax, bar_chart_axs, robustness_course_axs = _create_axes(
        scenario_fig_size,
        nr_rules,
        flag_plot_predicate_bar_chart,
        flat_plot_rule_robustness_course,
        flag_rule_conjunction,
    )

    rnd = MPRenderer(ax=scenario_ax, plot_limits=scenario_plot_limits)

    if flag_rule_conjunction:
        if flag_plot_predicate_bar_chart:
            pred_conjunct_dict = defaultdict(dict)
            for _, pred_result_sep in pred_result_dict.items():
                for veh_ids, rob_pairs in pred_result_sep.items():
                    pred_conjunct_dict[veh_ids].update(rob_pairs)
            plot_predicate_bar_chart(pred_conjunct_dict, bar_chart_axs[0], bar_chart_plot_limits)

        if flat_plot_rule_robustness_course:
            # conjunction of all rules, i.e., the min of the robustness is calculated
            rule_rob_list = [r for _, rule_rob in rule_result_dict.items() for r in rule_rob]
            rule_conjunct_list = [
                min(time_rob[1])
                for time_rob in groupby(rule_rob_list, lambda rule_rob_list: rule_rob_list[0])
            ]
            plot_rule_robustness_course(
                rule_conjunct_list,
                robustness_course_axs[0],
                rule_robustness_course_plot_limits,
                rule_name_list,
            )

    else:
        i = 0
        for rule in rule_name_list:
            if flag_plot_predicate_bar_chart:
                plot_predicate_bar_chart(
                    pred_result_dict[rule], bar_chart_axs[i], bar_chart_plot_limits
                )

            if flat_plot_rule_robustness_course:
                plot_rule_robustness_course(
                    rule_result_dict[rule],
                    robustness_course_axs[i],
                    rule_robustness_course_plot_limits,
                    [rule],
                )
            i += 1
    # after vehicle2draw_params is determined, draw the scenarios
    ego_initial = scenario.obstacle_by_id(ego_vehicle_id)

    rnd.draw_params.time_begin = time_step
    rnd.draw_params.trajectory.draw_trajectory = False
    rnd.draw_params.lanelet_network.lanelet.fill_lanelet = False
    rnd.draw_params.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.draw_occupancies = False
    rnd.draw_params.dynamic_obstacle.occupancy.draw_occupancies = False
    # rnd.draw_params.dynamic_obstacle.draw_shape = False
    rnd.draw_params.dynamic_obstacle["show_label"] = True
    scenario.draw(rnd)

    rnd.draw_params.dynamic_obstacle.draw_shape = True
    np_rule_robustness_course = np.array(rule_conjunct_list)
    rob_values = np_rule_robustness_course[:, 1]
    if rob_values[-1] >= 0:
        ego_color = TUMcolor.TUMblue.value
    else:
        ego_color = TUMcolor.TUMorange.value
    ego_mark = "x"
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = ego_color
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = ego_color
    ego_initial.draw(rnd)

    # render scenario and ego vehicle
    rnd.render()

    pos_x_initial = [ego_initial.initial_state.position[0]]
    pos_y_initial = [ego_initial.initial_state.position[1]]

    for state in ego_initial.prediction.trajectory.state_list:
        pos_x_initial.append(state.position[0])
        pos_y_initial.append(state.position[1])

    rnd.ax.plot(
        pos_x_initial[time_step:],
        pos_y_initial[time_step:],
        color=ego_color,
        marker=ego_mark,
        markersize=7.5,
        zorder=10000,
        linewidth=1.5,
        label="initial trajectory",
    )
    # scenario.lanelet_network.draw(renderer, draw_params=general_draw_params)
    #
    # # plotting scenario and obstacles
    # if scenario_plot_limits:
    #     plot_veh_ids = [
    #         obs.obstacle_id
    #         for obs in scenario.obstacles_by_position_intervals(
    #             [
    #                 Interval(scenario_plot_limits[0], scenario_plot_limits[1]),
    #                 Interval(scenario_plot_limits[2], scenario_plot_limits[3]),
    #             ],
    #             time_step=time_step,
    #         )
    #     ]
    # else:
    #     plot_veh_ids = [obs.obstacle_id for obs in scenario.obstacles]
    # for i in plot_veh_ids:
    #     if i != ego_vehicle_id:
    #         draw_params = vehicle2draw_params.get(i, {})
    #         scenario.obstacle_by_id(i).draw(
    #             renderer,
    #             draw_params=merge_dicts_recursively(general_draw_params, draw_params),
    #         )
    #
    # scenario.obstacle_by_id(ego_vehicle_id).draw(
    #     renderer,
    #     draw_params=merge_dicts_recursively(
    #         general_draw_params, EGO_VEHICLE_DRAW_PARAMS
    #     ),
    # )
    # renderer.render()
    #
    # for f in all_draw_functions:
    #     f(renderer)


class VariableCollectionVisitor(MonitorVisitorInterface[Dict[str, MonitorNode]]):
    def collect_variables(self, node: MonitorNode) -> Dict[str, MonitorNode]:
        return self.visit(node, {})

    @singledispatchmethod
    def visit(self, node: MonitorNode, state: Dict[str, MonitorNode]) -> Dict[str, MonitorNode]:
        state[node.name] = node
        return state

    @visit.register
    def _(self, node: UnaryMonitorNode, state: Dict[str, MonitorNode]) -> Dict[str, MonitorNode]:
        self.visit(node.child, state)
        state[node.name] = node
        return state

    @visit.register
    def _(self, node: RuleMonitorNode, state: Dict[str, MonitorNode]) -> Dict[str, MonitorNode]:
        [self.visit(child, state) for child in node.children]
        state[node.name] = node
        return state


class AstVisualizier:
    """
    Visualize the AST of a monitor rule. Visualizes the whole tree, including the rtamt nodes.
    """

    def __init__(self, ax: Optional[Axes] = None):
        # Record the matplotlib artists (=texts) here, so that we can map clicks on texts to their respective nodes.
        self._artist_to_node = {}

        if ax is None:
            self._fig, self._ax = plt.subplots()
        else:
            self._ax = ax
            self._fig = self._ax.figure

        self._ax.axis("off")

        self._vars = {}

        # To achieve an efficient layout networkx + graphiz is used.
        self._graph = nx.DiGraph()

    def build_graph(self, node: Union[MonitorNode, RtamtAbstractNode]) -> str:
        """
        Construct a networkx graph from the AST to make the layouting easier.

        :param node: The node from which on the graph is created.

        :returns: The node id. Can be used by the caller to establish a relationship to it's canonical child in the AST.
        """
        if isinstance(node, RuleMonitorNode):
            # Skip rule nodes and only visualize their constituents aka. the rtamt rule.
            ast = node.monitor._spec.offline_interpreter.ast
            return self.build_graph(ast.specs[0])
        elif isinstance(node, RtamtVariableNode):
            # During parsing the predicates and custom operators are replaced with variables.
            # To correctly visualizes them in the tree, they are resolved here and the variable itself is not visualized.
            rep_node = self._vars[node.var]
            return self.build_graph(rep_node)

        # We do not realy care about the node_id, so the only requirement is that it is unique for all our nodes (which the node name might not!).
        node_id = str(id(node))
        label = str(node)
        # By default, all nodes are inactive (greyed out).
        self._graph.add_node(node_id, label=label, ast_node=node, active=False)

        if isinstance(node, UnaryMonitorNode):
            child_node = self.build_graph(node.child)
            self._graph.add_edge(node_id, child_node)
        elif isinstance(node, RtamtBinaryNode):
            left_child = self.build_graph(node.children[0])
            right_child = self.build_graph(node.children[1])
            self._graph.add_edge(node_id, left_child)
            self._graph.add_edge(node_id, right_child)
        elif isinstance(node, RtamtUnaryNode):
            child_node = self.build_graph(node.children[0])
            self._graph.add_edge(node_id, child_node)

        # The node_id is important to establish the parent -> child relationship, because RuleMonitorNode and RtamtVariableNode are not recorded in the graph.
        # Therefore, they should passthrough the id of their children, so that their parent establishes the relationship with the correct node.
        return node_id

    def visualize(self, root_node: MonitorNode, interactive: bool = True):
        self._vars = VariableCollectionVisitor().collect_variables(root_node)
        self.build_graph(root_node)
        self.redraw()
        if interactive:
            self._fig.canvas.mpl_connect("pick_event", self.on_pick)

    def redraw(self) -> None:
        # Use graphiz to layout the tree, as this is more efficient than doing our own layouting.
        pos = nx.nx_agraph.graphviz_layout(self._graph, prog="dot")

        # Get node attributes
        labels = nx.get_node_attributes(self._graph, "label")
        active_states = nx.get_node_attributes(self._graph, "active")

        # Clear previous drawing
        self._ax.clear()

        edge_colors = []
        for u, v in self._graph.edges():
            # If either connected node is active, color the edge black, otherwise gray
            if active_states.get(u, False) or active_states.get(v, False):
                edge_colors.append("black")
            else:
                edge_colors.append("gray")
        nx.draw_networkx_edges(
            self._graph,
            pos,
            ax=self._ax,
            arrows=False,
            arrowsize=10,
            width=1.0,
            alpha=1.0,
            edge_color=edge_colors,
        )

        # Custom draw logic for the labels, to make interactivity easier.
        # This way, we can store the artists which are associated in the event and thus achieve a direct mapping between text element and AST node.
        # Otherwise, we would need to do our own position based matching.
        for node, (x, y) in pos.items():
            is_active = active_states[node]
            label = labels[node]

            # Choose colors based on active state
            color = "black" if is_active else "gray"
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=1.0,
            )

            # Draw the text with appropriate styling
            text = self._ax.text(
                x,
                y,
                label,
                fontsize=8,
                ha="center",
                va="center",
                color=color,
                bbox=bbox_props,
                picker=True,
            )
            self._artist_to_node[text] = node

        self._ax.figure.canvas.draw()

    def on_pick(self, event) -> Optional[Union[MonitorNode, RtamtAbstractNode]]:
        """Handle pick events on the graph nodes (text objects)"""
        # Check if the picked artist is in our mapping
        if event.artist not in self._artist_to_node:
            return None

        node = self._artist_to_node[event.artist]
        self._graph.nodes[node]["active"] = not self._graph.nodes[node]["active"]

        self.redraw()

        return self._graph.nodes[node]["ast_node"]


class TraceVisualizationVisitor(MonitorVisitorInterface[None]):
    def __init__(
        self,
        scale_rob: bool = True,
        trace_ax: Optional[Axes] = None,
        legend_ax: Optional[Axes] = None,
    ):
        self._rob_scaler = RobustnessScaler(scale=scale_rob)

        if trace_ax is None:
            self._trace_fig, self._trace_ax = plt.subplots()
        else:
            self._trace_ax = trace_ax
            trace_fig = self._trace_ax.figure
            assert trace_fig is not None
            self._trace_fig = trace_fig

        if legend_ax is None:
            self._legend_fig, self._legend_ax = plt.subplots()
        else:
            self._legend_ax = legend_ax
            # Make the type checker happy...
            legend_fig = self._legend_ax.figure
            assert legend_fig is not None
            self._legend_fig = legend_fig
        self._legend_ax.axis("off")

        self._legend = None
        self._map_legend_to_line = {}
        self._active_nodes = []
        self._lines = {}

        self._to_string_visitor = MonitorToStringVisitor()

    def visualize(
        self,
        monitor: MonitorNode,
        plot_limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.visit(monitor, {})
        self._trace_ax.grid(True)

        if plot_limits is not None:
            self._trace_ax.set_ylim(plot_limits)
        elif self._rob_scaler.scale:
            # If no explict plot limit is given, but robustness scaling is active, we have some other lower and upper bounds.
            # From those we can set the limits with a 5% margin.
            self._trace_ax.set_ylim(self._rob_scaler.min * 1.05, self._rob_scaler.max * 1.05)

        self._redraw()

    def _redraw(
        self, new_active_node: Optional[Union[MonitorNode, RtamtAbstractNode]] = None
    ) -> None:
        handles = []
        labels = []
        lines = []
        fig_width = self._legend_fig.get_figwidth()
        # Calculate approximate characters per inch (adjust the divisor as needed)
        chars_per_line = int(fig_width * 10)  # Roughly 10 chars per inch
        for ax_line, node in self._lines.items():
            if not self._is_active(node):
                ax_line.set_visible(False)
                continue
            elif node == new_active_node:
                # If a node was not previously shown, make it visible by default.
                ax_line.set_visible(True)

            color = ax_line.get_color()
            linestyle = ax_line.get_linestyle()
            marker = ax_line.get_marker()
            visible = ax_line.get_visible()

            # The line that is shown in the legend.
            handle = Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                marker=marker,
                alpha=1.0 if visible else 0.2,
            )
            handles.append(handle)
            original_label = ax_line.get_label()
            wrapped_label = "\n".join(textwrap.wrap(original_label, width=chars_per_line))
            labels.append(wrapped_label)
            lines.append(ax_line)

        # Clear the old legend, to make room for the new one.
        if self._legend is not None:
            self._legend.remove()

        self._legend = self._legend_ax.legend(
            handles, labels, loc="center", ncols=2, fontsize=8, framealpha=1, fancybox=True
        )
        self._legend.set_draggable(True)

        pickradius = 5
        for legend_line, ax_line in zip(self._legend.get_lines(), lines):
            legend_line.set_picker(pickradius)
            self._map_legend_to_line[legend_line] = ax_line

        self._trace_fig.canvas.draw()
        self._legend_fig.canvas.draw()

    def on_pick(self, event) -> None:
        legend_line = event.artist
        if legend_line not in self._map_legend_to_line:
            return

        ax_line = self._map_legend_to_line[legend_line]
        visible = not ax_line.get_visible()
        ax_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)
        self._trace_fig.canvas.draw()
        self._legend_fig.canvas.draw()

    def _plot_node(self, node: MonitorNode, label: str) -> None:
        # If the line plot is created with `visible=False` it will get no color by default.
        # To make sure a color is assigned, the private implementation from matplotlib is used here.
        # TODO: Is there a better way?
        color = self._trace_ax._get_lines.get_next_color()
        (line,) = self._trace_ax.plot(node.values, "x-", label=label, visible=False, color=color)
        self._lines[line] = node

    def _is_active(self, node: Union[MonitorNode, RtamtAbstractNode]) -> bool:
        return node in self._active_nodes

    def toggle_active(self, node: Union[MonitorNode, RtamtAbstractNode]) -> None:
        if self._is_active(node):
            self._active_nodes.remove(node)
        else:
            self._active_nodes.append(node)

        self._redraw(node)

    @singledispatchmethod
    def visit(self, node: MonitorNode, vehicle_ids: Dict[int, int]) -> None: ...

    @visit.register
    def _(self, node: RuleMonitorNode, vehicle_ids: Dict[int, int]) -> None:
        [self.visit(child, vehicle_ids) for child in node.children]
        label = self._to_string_visitor.to_string(node, vehicle_ids)
        self._plot_node(node, label)
        # Custom operators are replaced by 'g{i}' identifiers in rtamt rules.
        # To enhance the visualization, those placeholders are replaced by their computed label.
        name_replacements = {}
        for child in node.children:
            child_label = self._to_string_visitor.to_string(child, vehicle_ids)
            name_replacements[child.name] = child_label

        values = node.monitor._spec.offline_interpreter.ast_node_values
        for rtamt_ast_node, trace in values.items():
            name = rtamt_ast_node.name
            for target_name, name_replacement in name_replacements.items():
                if target_name in name:
                    name = name.replace(target_name, name_replacement)

            # rtamt operators might return traces with +-inf. As +-inf cannot be shown
            # in a plot, the lines will be missing from the plot. For the case, where
            # robustness scaling is enabled, we can normalize the intermediate traces, such that they are displayed in the plot.
            scaled_trace = np.clip(trace, self._rob_scaler.min, self._rob_scaler.max)
            color = self._trace_ax._get_lines.get_next_color()
            (line,) = self._trace_ax.plot(
                scaled_trace, "x-", label=name, visible=False, color=color
            )
            self._lines[line] = rtamt_ast_node

    @visit.register
    def _(self, node: UnaryMonitorNode, vehicle_ids: Dict[int, int]) -> None:
        self.visit(node.child, vehicle_ids)
        label = self._to_string_visitor.to_string(node, vehicle_ids)
        self._plot_node(node, label)

    @visit.register
    def _(self, node: QuantMonitorNode, vehicle_ids: Dict[int, int]) -> None:
        for vehicle_id, monitor in node.monitors.items():
            new_vehicle_ids = deepcopy(vehicle_ids)
            new_vehicle_ids[node.quantified_vehicle] = vehicle_id
            self.visit(monitor, new_vehicle_ids)
        label = self._to_string_visitor.to_string(node, vehicle_ids)
        self._plot_node(node, label)

    @visit.register
    def _(self, node: PredicateMonitorNode, vehicle_ids: Dict[int, int]) -> None:
        label = self._to_string_visitor.to_string(node, vehicle_ids)
        self._plot_node(node, label)


class VisualizationController:
    def __init__(self) -> None:
        self._fig = plt.figure()
        self._fig.canvas.mpl_connect("pick_event", lambda e: self._on_pick(e))

        gs = self._fig.add_gridspec(2, 2)
        self._trace_ax = self._fig.add_subplot(gs[0, 0])
        self._ast_ax = self._fig.add_subplot(gs[0, 1])
        self._leg_ax = self._fig.add_subplot(gs[1, :])
        self._fig.tight_layout(pad=0.0)
        self._fig.subplots_adjust(
            wspace=0.01, hspace=0.01, left=0.03, right=0.99, top=0.95, bottom=0.05
        )

        self._trace_visualization_visitor = TraceVisualizationVisitor(
            trace_ax=self._trace_ax, legend_ax=self._leg_ax
        )
        self._ast_visualizier = AstVisualizier(ax=self._ast_ax)
        self._to_string_visitor = MonitorToStringVisitor()

    def visualize(self, node: MonitorNode) -> None:
        self._trace_visualization_visitor.visualize(node)
        self._ast_visualizier.visualize(node, interactive=False)

    def _on_pick(self, event) -> None:
        artist = event.artist
        if artist is None:
            return

        # Correctly dispatch the pick event.
        if artist.axes == self._trace_ax or artist.axes == self._leg_ax:
            self._trace_visualization_visitor.on_pick(event)
        elif artist.axes == self._ast_ax:
            node = self._ast_visualizier.on_pick(event)
            if node:
                self._trace_visualization_visitor.toggle_active(node)
