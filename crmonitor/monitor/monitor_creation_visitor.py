from functools import singledispatchmethod

from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    CompareToThresholdScaledMonitorNode,
    ExistMonitorNode,
    ExistsMultipleMonitorNode,
    HistoricallyDurationMonitorNode,
    HistoricallyDurationSeverityMonitorNode,
    MonitorNode,
    PredicateMonitorNode,
    RtamtRuleMonitorNode,
    SigmoidMonitorNode,
    SumIfPositiveMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import (
    OfflineRtamtStlMonitor,
    OnlineRtamtStlMonitor,
    OutputType,
)
from crmonitor.rule import (
    AllNode,
    CompareToThresholdScaledNode,
    ExistNode,
    ExistsMultipleNode,
    HistoricallyDurationNode,
    HistoricallyDurationSeverityNode,
    PredicateNode,
    RtamtRuleNode,
    RuleAstNode,
    RuleTreeVisitorInterface,
    SigmoidNode,
    SumIfPositiveNode,
)


class MonitorCreationRuleTreeVisitor(RuleTreeVisitorInterface[MonitorNode]):
    """
    This visitor is used to transform a rule tree to a monitor tree.
    """

    def create_monitors(
        self,
        rule_node: RuleAstNode,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        online: bool = False,
    ) -> MonitorNode:
        """
        :param dt: The dt is required for RTAMT monitors, since they need to be pre-configured with the sampling frequency.
        """
        return self.visit(rule_node, dt, output_type, online)

    @singledispatchmethod
    def visit(self, node: RuleAstNode, *args, **kwargs) -> MonitorNode:
        raise NotImplementedError(
            f"Failed to create monitor for node '{node}': Transformation for this node is currently not implemented!"
        )

    @visit.register
    def _(
        self, node: RtamtRuleNode, dt: float, output_type: OutputType, online: bool
    ) -> MonitorNode:
        children = [self.visit(child, dt, output_type, online) for child in node.children]

        if online:
            rtamt_monitor_node_type = OnlineRtamtStlMonitor
        else:
            rtamt_monitor_node_type = OfflineRtamtStlMonitor

        rtamt_monitor = rtamt_monitor_node_type.create_from_rule_node(node, dt, output_type)

        return RtamtRuleMonitorNode(node.name, children, rtamt_monitor)

    @visit.register
    def _(self, node: AllNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return AllMonitorNode(node.name, child_monitor, node.quantified_vehicle)

    @visit.register
    def _(self, node: ExistNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return ExistMonitorNode(node.name, child_monitor, node.quantified_vehicle)

    @visit.register
    def _(self, node: SigmoidNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return SigmoidMonitorNode(node.name, child_monitor)

    @visit.register
    def _(self, node: HistoricallyDurationNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return HistoricallyDurationMonitorNode(node.name, child_monitor, node.interval)

    @visit.register
    def _(self, node: HistoricallyDurationSeverityNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return HistoricallyDurationSeverityMonitorNode(node.name, child_monitor, node.interval)

    @visit.register
    def _(self, node: SumIfPositiveNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return SumIfPositiveMonitorNode(node.name, child_monitor, node.quantified_vehicle)

    @visit.register
    def _(self, node: CompareToThresholdScaledNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return CompareToThresholdScaledMonitorNode(node.name, child_monitor, node.threshold)

    @visit.register
    def _(self, node: ExistsMultipleNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return ExistsMultipleMonitorNode(
            node.name, child_monitor, node.quantified_vehicle, node.threshold
        )

    @visit.register
    def _(self, node: PredicateNode, *args, **kwargs) -> MonitorNode:
        if node.io_type is None:
            raise RuntimeError(f"I/O type of predicate {node.base_name} is not set!")

        return PredicateMonitorNode(
            node.name,
            node.base_name,
            node.agent_placeholders,
            node.io_type,
        )
