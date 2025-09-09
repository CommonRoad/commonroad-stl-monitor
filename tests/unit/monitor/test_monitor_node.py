from copy import deepcopy
from unittest.mock import Mock

from crmonitor.monitor import (
    AllMonitorNode,
    CompareToThresholdScaledMonitorNode,
    ConstantTraceMonitorNode,
    ExistMonitorNode,
    ExistsMultipleMonitorNode,
    HistoricallyDurationMonitorNode,
    HistoricallyDurationSeverityMonitorNode,
    MonitorNode,
    PredicateMonitorNode,
    QuantMonitorNode,
    RtamtRuleMonitorNode,
    SelectiveQuantMonitorNode,
    SigmoidMonitorNode,
    SumIfPositiveMonitorNode,
    UnaryMonitorNode,
    VaradicMonitorNode,
    ZeroArityMonitorNode,
)
from crmonitor.rule.rule_node import IOType
from rtamt.semantics.interval.interval import Interval


class TestMonitorNodeBase:
    def test_copy_and_equality(self):
        node = MonitorNode("base")
        copy = deepcopy(node)
        assert node == copy
        assert node is not copy

    def test_value_tracking(self):
        node = MonitorNode("base")
        node.last_value = 1.0
        node.last_value = 2.0
        assert node.values == [1.0, 2.0]
        assert node.last_value == 2.0

    def test_reset(self):
        node = MonitorNode("base")
        node.values = [1.0, 2.0]
        node.reset()
        assert node.values == []

    def test_values_setter(self):
        """Test the values property setter."""
        node = MonitorNode("base")
        test_values = [1.0, 2.0, 3.0]
        node.values = test_values
        assert node.values == test_values
        # Ensure it's a copy, not reference
        test_values.append(4.0)
        assert node.values == [1.0, 2.0, 3.0]

    def test_hash(self):
        """Test hash functionality based on name."""
        node1 = MonitorNode("test")
        node2 = MonitorNode("test")
        node3 = MonitorNode("different")
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)

    def test_equality_with_different_types(self):
        """Test equality with non-MonitorNode objects."""
        node = MonitorNode("test")
        assert node != "test"
        assert node != 123
        assert node is not None

    def test_name_equality(self):
        """Test equality based on name."""
        node1 = MonitorNode("same")
        node2 = MonitorNode("same")
        node3 = MonitorNode("different")
        assert node1 == node2
        assert node1 != node3


class TestUnaryMonitorNode:
    def test_child_copy(self):
        child = MonitorNode("child")
        node = UnaryMonitorNode("parent", child)
        copy = deepcopy(node)
        assert node == copy
        assert node is not copy
        assert node.child == copy.child
        assert node.child is not copy.child

    def test_hash_includes_child(self):
        """Test that hash includes child node."""
        child1 = MonitorNode("child1")
        child2 = MonitorNode("child2")
        node1 = UnaryMonitorNode("parent", child1)
        node2 = UnaryMonitorNode("parent", child2)
        assert hash(node1) != hash(node2)

    def test_equality_with_different_children(self):
        """Test equality considers child nodes."""
        child1 = MonitorNode("child1")
        child2 = MonitorNode("child2")
        node1 = UnaryMonitorNode("parent", child1)
        node2 = UnaryMonitorNode("parent", child2)
        node3 = UnaryMonitorNode("parent", child1)
        assert node1 != node2
        assert node1 == node3

    def test_equality_with_wrong_type(self):
        """Test equality with non-UnaryMonitorNode objects."""
        child = MonitorNode("child")
        node = UnaryMonitorNode("parent", child)
        base_node = MonitorNode("parent")
        assert node != base_node


class TestVaradicMonitorNode:
    def test_multiple_children(self):
        children = (MonitorNode("c1"), MonitorNode("c2"))
        node = VaradicMonitorNode("multi", children)
        assert node.children == children

    def test_empty_children(self):
        """Test with no children."""
        node = VaradicMonitorNode("empty", ())
        assert node.children == ()

    def test_single_child(self):
        """Test with single child."""
        child = MonitorNode("single")
        node = VaradicMonitorNode("parent", (child,))
        assert node.children == (child,)

    def test_copy_with_children(self):
        """Test deep copy functionality."""
        children = (MonitorNode("c1"), MonitorNode("c2"))
        node = VaradicMonitorNode("multi", children)
        copy = deepcopy(node)
        assert node == copy
        assert node is not copy
        assert node.children == copy.children
        assert node.children is not copy.children
        assert node.children[0] is not copy.children[0]

    def test_hash_includes_children(self):
        """Test that hash includes children."""
        children1 = (MonitorNode("c1"), MonitorNode("c2"))
        children2 = (MonitorNode("c1"), MonitorNode("c3"))
        node1 = VaradicMonitorNode("multi", children1)
        node2 = VaradicMonitorNode("multi", children2)
        # Note: This test might be implementation dependent due to deepcopy in hash
        assert hash(node1) != hash(node2)

    def test_equality_with_different_children(self):
        """Test equality considers all children."""
        children1 = (MonitorNode("c1"), MonitorNode("c2"))
        children2 = (MonitorNode("c1"), MonitorNode("c3"))
        node1 = VaradicMonitorNode("multi", children1)
        node2 = VaradicMonitorNode("multi", children2)
        assert node1 != node2

    def test_equality_with_wrong_type(self):
        """Test equality with non-VaradicMonitorNode objects."""
        children = (MonitorNode("c1"),)
        node = VaradicMonitorNode("multi", children)
        base_node = MonitorNode("multi")
        assert node != base_node


class TestQuantMonitorNode:
    def test_copy(self):
        node = QuantMonitorNode("quant", MonitorNode("c"), 1)
        copy = deepcopy(node)
        assert node == copy
        assert node is not copy

    def test_initialization(self):
        """Test proper initialization."""
        child = MonitorNode("child")
        node = QuantMonitorNode("quant", child, 2)
        assert node.child == child
        assert node.quantified_agent == 2
        assert len(node.monitors) == 0

    def test_monitors_defaultdict(self):
        """Test that monitors is a defaultdict."""
        child = MonitorNode("child")
        node = QuantMonitorNode("quant", child, 1)
        # Accessing a key should create a new monitor
        monitor = node.monitors["key1"]
        assert monitor is not None
        assert isinstance(monitor, MonitorNode)

    def test_reset_clears_monitors(self):
        """Test that reset clears the monitors dict."""
        child = MonitorNode("child")
        node = QuantMonitorNode("quant", child, 1)
        node.monitors["key1"]  # Create an entry
        node.values = [1.0, 2.0]
        node.reset()
        assert len(node.monitors) == 0
        assert node.values == []


class TestSelectiveQuantMonitorNode:
    def test_selected_tracking(self):
        node = SelectiveQuantMonitorNode("sel", MonitorNode("c"), 1)
        node.last_selected = MonitorNode("sel1")
        assert node.last_selected.name == "sel1"
        node.reset()
        assert node.selected == []

    def test_selected_property_setter(self):
        """Test selected property setter."""
        node = SelectiveQuantMonitorNode("sel", MonitorNode("c"), 1)
        monitors = [MonitorNode("m1"), MonitorNode("m2")]
        node.selected = monitors
        assert node.selected == monitors

    def test_multiple_selected_tracking(self):
        """Test tracking multiple selected monitors."""
        node = SelectiveQuantMonitorNode("sel", MonitorNode("c"), 1)
        m1 = MonitorNode("m1")
        m2 = MonitorNode("m2")
        node.last_selected = m1
        node.last_selected = m2
        assert len(node.selected) == 2
        assert node.selected[0] == m1
        assert node.selected[1] == m2
        assert node.last_selected == m2

    def test_selected_with_none(self):
        """Test selected tracking with None values."""
        node = SelectiveQuantMonitorNode("sel", MonitorNode("c"), 1)
        node.last_selected = None
        assert node.last_selected is None
        assert len(node.selected) == 1


class TestZeroArityMonitorNode:
    def test_creation(self):
        """Test basic creation of zero arity node."""
        node = ZeroArityMonitorNode("zero")
        assert node.name == "zero"
        assert isinstance(node, MonitorNode)


class TestRtamtRuleMonitorNode:
    def test_initialization(self):
        """Test initialization with mock monitor."""
        children = [MonitorNode("c1"), MonitorNode("c2")]
        mock_monitor = Mock()
        mock_monitor._rule = "test_rule"
        node = RtamtRuleMonitorNode("rule", children, mock_monitor)
        assert node.children == tuple(children)
        assert node.monitor == mock_monitor

    def test_update(self):
        """Test update method calls monitor."""
        mock_monitor = Mock()
        mock_monitor.evaluate_monitor_online.return_value = 0.5
        node = RtamtRuleMonitorNode("rule", [], mock_monitor)
        result = node.update(10, [("var", 1.0)])
        mock_monitor.evaluate_monitor_online.assert_called_once_with(10, [("var", 1.0)])
        assert result == 0.5

    def test_evaluate(self):
        """Test evaluate method calls monitor."""
        mock_monitor = Mock()
        mock_monitor.evaluate_monitor_offline.return_value = [0.1, 0.2]
        node = RtamtRuleMonitorNode("rule", [], mock_monitor)
        result = node.evaluate([("var", [1.0, 2.0])])
        mock_monitor.evaluate_monitor_offline.assert_called_once_with([("var", [1.0, 2.0])])
        assert result == [0.1, 0.2]

    def test_reset(self):
        """Test reset calls both parent and monitor reset."""
        mock_monitor = Mock()
        node = RtamtRuleMonitorNode("rule", [], mock_monitor)
        node.values = [1.0, 2.0]
        node.reset()
        assert node.values == []
        mock_monitor.reset.assert_called_once()

    def test_str(self):
        """Test string representation."""
        mock_monitor = Mock()
        mock_monitor._rule = "test_rule_string"
        node = RtamtRuleMonitorNode("rule", [], mock_monitor)
        assert str(node) == "test_rule_string"

    def test_deepcopy(self):
        """Test deep copy functionality."""
        children = [MonitorNode("c1")]
        mock_monitor = Mock()
        node = RtamtRuleMonitorNode("rule", children, mock_monitor)
        copy = deepcopy(node)
        assert copy.name == node.name
        assert len(copy.children) == len(node.children)
        assert copy.children[0] is not node.children[0]


class TestAllMonitorNode:
    def test_str_representation(self):
        """Test string representation of AllMonitorNode."""
        child = MonitorNode("child")
        node = AllMonitorNode("all", child, 2)
        assert str(node) == "A a2:"


class TestExistMonitorNode:
    def test_str_representation(self):
        """Test string representation of ExistMonitorNode."""
        child = MonitorNode("child")
        node = ExistMonitorNode("exist", child, 3)
        assert str(node) == "E a3:"


class TestSigmoidMonitorNode:
    def test_str_representation(self):
        """Test string representation of SigmoidMonitorNode."""
        child = MonitorNode("child")
        node = SigmoidMonitorNode("sigmoid", child)
        assert str(node) == "sigmoid"


class TestHistoricallyDurationMonitorNode:
    def test_initialization_without_interval(self):
        """Test initialization without interval."""
        child = MonitorNode("child")
        node = HistoricallyDurationMonitorNode("hist", child)
        assert node.child == child
        assert node.interval is None

    def test_initialization_with_interval(self):
        """Test initialization with interval."""
        child = MonitorNode("child")
        interval = Interval(begin=5, end=15, begin_unit="s", end_unit="s")
        node = HistoricallyDurationMonitorNode("hist", child, interval)
        assert node.interval == interval

    def test_str_without_interval(self):
        """Test string representation without interval."""
        child = MonitorNode("child")
        node = HistoricallyDurationMonitorNode("hist", child)
        assert str(node) == "historically_duration"

    def test_str_with_interval(self):
        """Test string representation with interval."""
        child = MonitorNode("child")
        interval = Interval(begin=0, end=10, begin_unit="s", end_unit="s")
        node = HistoricallyDurationMonitorNode("hist", child, interval)
        assert str(node) == "historically_duration[0s, 10s]"

    def test_deepcopy(self):
        """Test deep copy functionality."""
        child = MonitorNode("child")
        interval = Interval(begin=5, end=15, begin_unit="s", end_unit="s")
        node = HistoricallyDurationMonitorNode("hist", child, interval)
        copy = deepcopy(node)
        assert copy.name == node.name
        assert copy.child is not node.child
        assert copy.interval == node.interval


class TestHistoricallyDurationSeverityMonitorNode:
    def test_str_without_interval(self):
        """Test string representation without interval."""
        child = MonitorNode("child")
        node = HistoricallyDurationSeverityMonitorNode("hist_sev", child)
        assert str(node) == "historically_duration_severity"

    def test_str_with_interval(self):
        """Test string representation with interval."""
        child = MonitorNode("child")
        interval = Interval(begin=5, end=15, begin_unit="s", end_unit="s")
        node = HistoricallyDurationSeverityMonitorNode("hist_sev", child, interval)
        assert str(node) == "historically_duration_severity[5s, 15s]"


class TestSumIfPositiveMonitorNode:
    def test_str_representation(self):
        """Test string representation of SumIfPositiveMonitorNode."""
        child = MonitorNode("child")
        node = SumIfPositiveMonitorNode("sum", child, 1)
        assert str(node) == "sum_if_positive a1:"


class TestCompareToThresholdScaledMonitorNode:
    def test_initialization(self):
        """Test initialization with threshold."""
        child = MonitorNode("child")
        node = CompareToThresholdScaledMonitorNode("compare", child, 0.5)
        assert node.child == child
        assert node.threshold == 0.5

    def test_str_representation(self):
        """Test string representation."""
        child = MonitorNode("child")
        node = CompareToThresholdScaledMonitorNode("compare", child, 0.75)
        assert str(node) == "compare_to_threshold_scaled[>=0.75]"

    def test_deepcopy(self):
        """Test deep copy functionality."""
        child = MonitorNode("child")
        node = CompareToThresholdScaledMonitorNode("compare", child, 0.5)
        copy = deepcopy(node)
        assert copy.name == node.name
        assert copy.threshold == node.threshold
        assert copy.child is not node.child


class TestExistsMultipleMonitorNode:
    def test_initialization(self):
        """Test initialization with threshold."""
        child = MonitorNode("child")
        node = ExistsMultipleMonitorNode("exists_multi", child, 2, 3)
        assert node.child == child
        assert node.quantified_agent == 2
        assert node.threshold == 3

    def test_str_representation(self):
        """Test string representation."""
        child = MonitorNode("child")
        node = ExistsMultipleMonitorNode("exists_multi", child, 1, 5)
        assert str(node) == "exists_multiple[5]"

    def test_deepcopy(self):
        """Test deep copy functionality."""
        child = MonitorNode("child")
        node = ExistsMultipleMonitorNode("exists_multi", child, 1, 3)
        copy = deepcopy(node)
        assert copy.name == node.name
        assert copy.threshold == node.threshold
        assert copy.quantified_agent == node.quantified_agent
        assert copy.child is not node.child


class TestPredicateMonitorNode:
    def test_initialization(self):
        """Test initialization with all parameters."""
        predicate_name = "keeps_safe_distance"

        agent_placeholders = (1, 2)
        io_type = IOType.OUTPUT

        node = PredicateMonitorNode("pred", predicate_name, agent_placeholders, io_type)
        assert node.predicate_name == predicate_name
        assert node.agent_placeholders == agent_placeholders
        assert node.io_type == io_type

    def test_str_representation(self):
        """Test string representation."""
        agent_placeholders = (1, 2)

        node = PredicateMonitorNode("pred", "in_same_lane", agent_placeholders, IOType.OUTPUT)
        assert str(node) == "in_same_lane(a1, a2)"

    def test_format_with_vehicle_ids_partial(self):
        """Test formatting with partial vehicle ID mapping."""
        predicate_name = "in_front_of"
        agent_placeholders = (1, 2, 3)

        node = PredicateMonitorNode("pred", predicate_name, agent_placeholders, IOType.INPUT)
        vehicle_ids = {1: 101, 3: 103}
        result = node.format_with_vehicle_ids(vehicle_ids)
        assert result == "in_front_of(101, a2, 103)_i"

        node = PredicateMonitorNode("pred", predicate_name, agent_placeholders, IOType.OUTPUT)
        vehicle_ids = {1: 101, 4: 10999}
        result = node.format_with_vehicle_ids(vehicle_ids)
        assert result == "in_front_of(101, a2, a3)"

    def test_format_with_vehicle_ids_complete(self):
        """Test formatting with complete vehicle ID mapping."""
        predicate_name = "keeps_lane_speed_limit"
        agent_placeholders = (1,)

        node = PredicateMonitorNode("pred", predicate_name, agent_placeholders, IOType.INPUT)
        vehicle_ids = {1: 42}
        result = node.format_with_vehicle_ids(vehicle_ids)
        assert result == "keeps_lane_speed_limit(42)_i"


class TestConstantTraceMonitorNode:
    def test_initialization(self):
        """Test initialization with trace."""
        trace = [1.0, 2.0, 3.0]
        node = ConstantTraceMonitorNode("const", trace)
        assert node.trace == trace

    def test_str_representation(self):
        """Test string representation returns name."""
        trace = [1.0, 2.0]
        node = ConstantTraceMonitorNode("constant_trace", trace)
        assert str(node) == "constant_trace"

    def test_deepcopy(self):
        """Test deep copy functionality."""
        trace = [1.0, 2.0, 3.0]
        node = ConstantTraceMonitorNode("const", trace)
        copy = deepcopy(node)
        assert copy.name == node.name
        assert copy.trace == node.trace
        assert copy.trace is not node.trace  # Ensure deep copy of trace
