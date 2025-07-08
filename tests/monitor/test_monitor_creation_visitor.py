import pytest

from crmonitor.monitor import (
    MonitorCreationRuleTreeVisitor,
    PredicateMonitorNode,
    RtamtRuleMonitorNode,
)
from crmonitor.monitor.monitor_node import AllMonitorNode, ExistMonitorNode
from crmonitor.rule import (
    PredicateNode,
    IOType,
    RuleAstNode,
    RtamtRuleNode,
    AllNode,
    ExistNode,
    SigmoidNode,
    HistoricallyDurationNode,
    HistoricallyDurationSeverityNode,
    CompareToThresholdScaledNode,
    SumIfPositiveNode,
    ExistsMultipleNode,
)


PREDICATE_MONITOR_CREATION_TEST_DATA = [
    PredicateNode(
        name="g1", base_name="in_same_lane", agent_placeholders=(45, 3), io_type=IOType.OUTPUT
    ),
    PredicateNode(
        name="xyz",
        base_name="keeps_lane_speed_limit",
        agent_placeholders=(96,),
        io_type=IOType.INPUT,
    ),
]
RTAMT_NODE_TEST_DATA = [
    RtamtRuleNode(
        name="rtamt1",
        children=(PredicateNode("p1", "in_front_of", (1, 2), io_type=IOType.OUTPUT),),
        rule_str="p1",
    ),
    RtamtRuleNode(
        name="rtamt2",
        children=(
            PredicateNode("p2", "in_same_lane", (3, 4), io_type=IOType.INPUT),
            PredicateNode("p3", "keeps_lane_speed_limit", (5,), io_type=IOType.OUTPUT),
        ),
        rule_str="p2 and p3",
    ),
]

ALL_NODE_TEST_DATA = [
    AllNode(
        name="all1",
        child=PredicateNode("p0", "slow_as_leading_vehicle", (6,), io_type=IOType.OUTPUT),
        quantified_vehicle=10,
    ),
    AllNode(
        name="all2",
        child=PredicateNode("p1", "close_to_right_bound", (20,), io_type=IOType.INPUT),
        quantified_vehicle=20,
    ),
]

EXIST_NODE_TEST_DATA = [
    ExistNode(
        name="exist1",
        child=PredicateNode("p0", "close_to_left_bound", (8, 9), io_type=IOType.OUTPUT),
        quantified_vehicle=30,
    ),
    ExistNode(
        name="exist2",
        child=PredicateNode("p1", "heading_right", (10,), io_type=IOType.INPUT),
        quantified_vehicle=40,
    ),
]

SIGMOID_NODE_TEST_DATA = [
    SigmoidNode(
        name="sigmoid1", child=PredicateNode("p8", "comfort_level", (11,), io_type=IOType.OUTPUT)
    ),
    SigmoidNode(
        name="sigmoid2", child=PredicateNode("p9", "responsiveness", (12,), io_type=IOType.INPUT)
    ),
]

HISTORICALLY_DURATION_NODE_TEST_DATA = [
    HistoricallyDurationNode(
        name="hist_dur1",
        child=PredicateNode("p10", "lane_stability", (13,), io_type=IOType.OUTPUT),
        interval=5.0,
    ),
    HistoricallyDurationNode(
        name="hist_dur2",
        child=PredicateNode("p11", "consistent_speed", (14,), io_type=IOType.INPUT),
        interval=10.0,
    ),
]

HISTORICALLY_DURATION_SEVERITY_NODE_TEST_DATA = [
    HistoricallyDurationSeverityNode(
        name="hist_dur_sev1",
        child=PredicateNode("p12", "collision_risk", (15,), io_type=IOType.OUTPUT),
        interval=7.5,
    ),
    HistoricallyDurationSeverityNode(
        name="hist_dur_sev2",
        child=PredicateNode("p13", "path_deviation", (16,), io_type=IOType.INPUT),
        interval=12.5,
    ),
]

SUM_IF_POSITIVE_NODE_TEST_DATA = [
    SumIfPositiveNode(
        name="sum_pos1",
        child=PredicateNode("p14", "risk_factor", (17,), io_type=IOType.OUTPUT),
        quantified_vehicle=50,
    ),
    SumIfPositiveNode(
        name="sum_pos2",
        child=PredicateNode("p15", "safety_margin", (18,), io_type=IOType.INPUT),
        quantified_vehicle=60,
    ),
]

COMPARE_TO_THRESHOLD_SCALED_NODE_TEST_DATA = [
    CompareToThresholdScaledNode(
        name="comp_thresh1",
        child=PredicateNode("p16", "distance_ratio", (19,), io_type=IOType.OUTPUT),
        threshold=0.75,
    ),
    CompareToThresholdScaledNode(
        name="comp_thresh2",
        child=PredicateNode("p17", "speed_ratio", (20,), io_type=IOType.INPUT),
        threshold=0.85,
    ),
]

EXISTS_MULTIPLE_NODE_TEST_DATA = [
    ExistsMultipleNode(
        name="exists_multi1",
        child=PredicateNode("p18", "nearby_vehicles", (21,), io_type=IOType.OUTPUT),
        quantified_vehicle=70,
        threshold=3,
    ),
    ExistsMultipleNode(
        name="exists_multi2",
        child=PredicateNode("p19", "lane_vehicles", (22,), io_type=IOType.INPUT),
        quantified_vehicle=80,
        threshold=5,
    ),
]


class TestMonitorCreationVisitor:
    @pytest.mark.parametrize("predicate_node", PREDICATE_MONITOR_CREATION_TEST_DATA)
    def test_predicate_monitor_creation(
        self,
        predicate_node: PredicateNode,
    ) -> None:
        monitor_node = MonitorCreationRuleTreeVisitor().create_monitors(predicate_node, dt=0.1)
        assert isinstance(monitor_node, PredicateMonitorNode)
        assert monitor_node.agent_placeholders == predicate_node.agent_placeholders
        assert monitor_node.io_type == predicate_node.io_type
        assert monitor_node.name == predicate_node.name

    def test_predicate_monitor_creation_fails_if_io_type_is_not_set(self) -> None:
        predicate_node = PredicateNode("foo1", "foo", (1,), io_type=None)
        with pytest.raises(RuntimeError):
            MonitorCreationRuleTreeVisitor().create_monitors(predicate_node, dt=0.1)

    def test_visit_unimplemented_node(self) -> None:
        """Test that visiting an unimplemented node type raises NotImplementedError."""

        class UnimplementedNode(RuleAstNode):
            pass

        unimplemented_node = UnimplementedNode("unimplemented")

        with pytest.raises(NotImplementedError):
            MonitorCreationRuleTreeVisitor().create_monitors(unimplemented_node, dt=0.1)

    @pytest.mark.parametrize("node", ALL_NODE_TEST_DATA)
    def test_all_node_visit(self, node: AllNode):
        monitor_node = MonitorCreationRuleTreeVisitor().create_monitors(node, dt=0.1)
        assert isinstance(monitor_node, AllMonitorNode)
        assert monitor_node.name == node.name
        assert monitor_node.quantified_agent == node.quantified_vehicle

    @pytest.mark.parametrize("node", EXIST_NODE_TEST_DATA)
    def test_exist_node_visit(self, node: ExistNode):
        """Test that RtamtRuleNode is correctly handled."""
        monitor_node = MonitorCreationRuleTreeVisitor().create_monitors(node, dt=0.1)
        assert isinstance(monitor_node, ExistMonitorNode)
        assert monitor_node.name == node.name
        assert monitor_node.quantified_agent == node.quantified_vehicle
