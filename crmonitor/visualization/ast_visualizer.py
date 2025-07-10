from typing import Optional, Union

try:
    import networkx as nx
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

from rtamt.syntax.node.abstract_node import AbstractNode as RtamtAbstractNode
from rtamt.syntax.node.binary_node import BinaryNode as RtamtBinaryNode
from rtamt.syntax.node.ltl.variable import Variable as RtamtVariableNode
from rtamt.syntax.node.unary_node import UnaryNode as RtamtUnaryNode

from crmonitor.monitor.monitor_node import (
    MonitorNode,
    RtamtRuleMonitorNode,
    UnaryMonitorNode,
)
from crmonitor.monitor.visitors import VariableCollectionVisitor


class AstVisualizer:
    """
    Visualize the Abstract Syntax Tree (AST) of a monitor rule using NetworkX and Graphviz.

    This tool shows the complete structure of a monitor rule including both custom monitor nodes
    and their corresponding rtamt nodes. Nodes can be interactively toggled via matplotlib.
    """

    def __init__(self, ax: Axes | None = None):
        """
        Initialize the AST visualizer.

        Parameters
        ----------
        ax : Optional[Axes]
            An optional matplotlib Axes to draw the tree on. If None, a new figure and axes are created.
        """
        if not _VISUALIZATION_AVAILABLE:
            raise RuntimeError(
                "Visualization is not available because dependencies are missing. Please install the 'visualization' extra."
            )
        if ax is None:
            self._fig, self._ax = plt.subplots()
        else:
            self._ax = ax
            self._fig = self._ax.figure

        self._ax.axis("off")

        self._vars = {}
        # Record the matplotlib artists (=texts) here, so that we can map clicks on texts to their respective nodes.
        self._artist_to_node = {}
        # To achieve an efficient layout networkx + graphiz is used.
        self._graph = nx.DiGraph()
        self._depth = 0

    def build_graph(self, node: MonitorNode | RtamtAbstractNode) -> str:
        """
        Construct a networkx graph from the AST to make the layouting easier.

        :param node: The node from which on the graph is created.

        :returns: The node id. Can be used by the caller to establish a relationship to it's canonical child in the AST.
        """
        if isinstance(node, RtamtRuleMonitorNode):
            # Skip rule nodes and only visualize their constituents aka. the rtamt rule.
            ast = node.monitor._spec.offline_interpreter.ast
            return self.build_graph(ast.specs[0])
        elif isinstance(node, RtamtVariableNode):
            # During parsing the predicates and custom operators are replaced with variables.
            # To correctly visualizes them in the tree, they are resolved here and the variable itself is not visualized.
            rep_node = self._vars[node.var]
            return self.build_graph(rep_node)

        # We do not really care about the node_id, so the only requirement is that it is unique for all our nodes (which the node name might not!).
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
        """
        Visualize the AST rooted at the given monitor node.

        Parameters
        ----------
        root_node : MonitorNode
            The root of the monitor rule's AST.
        interactive : bool, optional
            If True, enables clicking to toggle node activity (default is True).
        """
        self._vars = VariableCollectionVisitor().collect_variables(root_node)
        self.build_graph(root_node)
        self.redraw()
        if interactive:
            self._fig.canvas.mpl_connect("pick_event", self.on_pick)

    def redraw(self) -> None:
        """
        Redraw the entire AST graph based on the current state of nodes and layout.
        """
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
        """
        Handle pick events triggered by clicking on graph nodes.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event from matplotlib.

        Returns
        -------
        Optional[Union[MonitorNode, RtamtAbstractNode]]
            The AST node associated with the clicked artist, or None if not found.
        """
        """Handle pick events on the graph nodes (text objects)"""
        # Check if the picked artist is in our mapping
        if event.artist not in self._artist_to_node:
            return None

        node = self._artist_to_node[event.artist]
        self._graph.nodes[node]["active"] = not self._graph.nodes[node]["active"]

        self.redraw()

        return self._graph.nodes[node]["ast_node"]
