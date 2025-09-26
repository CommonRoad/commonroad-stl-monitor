from matplotlib import pyplot as plt

from crmonitor.monitor.monitor_node import (
    MonitorNode,
)
from crmonitor.monitor.visitors import MonitorToStringVisitor

from .ast_visualizer import AstVisualizer
from .trace_visualization_visitor import TraceVisualizationVisitor


class VisualizationController:
    def __init__(self) -> None:
        self._fig = plt.figure()
        self._fig.canvas.mpl_connect("pick_event", lambda e: self._on_pick(e))

        gs = self._fig.add_gridspec(2, 2)
        self._trace_ax = self._fig.add_subplot(gs[0, 0])
        self._ast_ax = self._fig.add_subplot(gs[1, :])
        self._leg_ax = self._fig.add_subplot(gs[0, 1])
        self._fig.tight_layout(pad=0.0)
        self._fig.subplots_adjust(
            wspace=0.01, hspace=0.01, left=0.03, right=0.99, top=0.95, bottom=0.05
        )

        self._trace_visualization_visitor = TraceVisualizationVisitor(
            trace_ax=self._trace_ax, legend_ax=self._leg_ax
        )
        self._ast_visualizier = AstVisualizer(ax=self._ast_ax)
        self._to_string_visitor = MonitorToStringVisitor()

    def visualize(self, node: MonitorNode) -> None:
        self._trace_visualization_visitor.visualize(node)
        # Disable interactivity, because the controller overrides the pick event.
        # If interactivity for the AstVisualizer would be enabled,
        # this would override the pick event and picking in the trace visualization would no longer work.
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
