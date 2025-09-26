import textwrap
from collections.abc import Iterable
from copy import deepcopy
from functools import singledispatchmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from rtamt.syntax.node.abstract_node import AbstractNode as RtamtAbstractNode
from rtamt.syntax.node.ltl.constant import Constant as RtamtContantNode

from crmonitor.monitor.monitor_node import (
    MonitorNode,
    MonitorVisitorInterface,
    QuantMonitorNode,
    RtamtRuleMonitorNode,
    UnaryMonitorNode,
)
from crmonitor.monitor.visitors import MonitorToStringVisitor
from crmonitor.predicates.scaling import RobustnessScaler


class TraceVisualizationVisitor(MonitorVisitorInterface[None]):
    """
    A visitor that visualizes the output of STL monitor nodes as robustness signal traces.

    This class traverses monitor node trees and renders corresponding signal traces using matplotlib,
    with optional robustness scaling and interactive legend toggling.

    Parameters
    ----------
    scale_rob : bool, optional
        Whether to scale the robustness values for normalized plotting (default is True).
    trace_ax : Optional[Axes], optional
        Optional matplotlib Axes to plot traces. If not provided, a new one is created.
    legend_ax : Optional[Axes], optional
        Optional matplotlib Axes to display the legend. If not provided, a new one is created.
    """

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

        # Track different constants to only plot each constant once.
        self._tracked_constants = set()
        self._tracked_rtamt_expressions = set()

        self._to_string_visitor = MonitorToStringVisitor()

    def visualize(
        self,
        monitor: MonitorNode,
        plot_limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Generate a plot of robustness traces for a monitor node.

        Parameters
        ----------
        monitor : MonitorNode
            The root node of the monitor tree to visualize.
        plot_limits : Optional[Tuple[float, float]], optional
            Manual y-axis limits for robustness values. If None, limits are auto-scaled.
        """
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
        # Calculate approximate characters per inch.
        chars_per_line = int(fig_width * 10)  # Roughly 10 chars per inch
        for ax_line, node in self._lines.items():
            if not self._is_active(node):
                ax_line.set_visible(False)
                continue
            elif new_active_node is not None and node.name == new_active_node.name:
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
        """
        Callback triggered when a legend item is clicked. Toggles the visibility
        of the corresponding trace line.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The event object passed from matplotlib's pick handler.
        """
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
        return node.name in self._active_nodes

    def toggle_active(self, node: Union[MonitorNode, RtamtAbstractNode]) -> None:
        """
        Toggle visibility of a signal trace corresponding to a given monitor or AST node.

        Parameters
        ----------
        node : Union[MonitorNode, RtamtAbstractNode]
            Node whose trace should be toggled on/off in the visualization.
        """
        if self._is_active(node):
            self._active_nodes.remove(node.name)
            self._redraw()
        else:
            self._active_nodes.append(node.name)
            self._redraw(node)

    @singledispatchmethod
    def visit(self, node: MonitorNode, vehicle_ids: Dict[int, int]) -> None: ...

    @visit.register
    def _(self, node: RtamtRuleMonitorNode, vehicle_ids: Dict[int, int]) -> None:
        [self.visit(child, vehicle_ids) for child in node.children]
        label = self._to_string_visitor.to_string(node, vehicle_ids)
        self._plot_node(node, label)
        # Custom operators are replaced by 'g{i}' identifiers in rtamt rules.
        # To enhance the visualization, those placeholders are replaced by their computed label.
        name_replacements = {}
        for child in node.children:
            child_label = self._to_string_visitor.to_string(child, vehicle_ids)
            name_replacements[child.name] = child_label

        values = node.monitor.ast_node_values
        for rtamt_ast_node in self._filter_unique_rtamt_nodes(values.keys()):
            name = rtamt_ast_node.name
            for target_name, name_replacement in name_replacements.items():
                if target_name in name:
                    name = name.replace(target_name, name_replacement)

            if name in self._tracked_rtamt_expressions:
                continue
            self._tracked_rtamt_expressions.add(name)

            trace = values[rtamt_ast_node]
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
            new_vehicle_ids[node.quantified_agent] = vehicle_id
            self.visit(monitor, new_vehicle_ids)
        label = self._to_string_visitor.to_string(node, vehicle_ids)
        self._plot_node(node, label)

    def _filter_unique_rtamt_nodes(self, nodes: Iterable) -> Iterable:
        for node in nodes:
            if isinstance(node, RtamtContantNode):
                if node.val in self._tracked_constants:
                    continue
                else:
                    self._tracked_constants.add(node.val)

            yield node
