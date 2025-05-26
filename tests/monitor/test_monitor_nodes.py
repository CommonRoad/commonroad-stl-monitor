from crmonitor.monitor import (
    MonitorNode,
    UnaryMonitorNode,
    VaradicMonitorNode,
    QuantMonitorNode,
    SelectiveQuantMonitorNode,
)


class TestMonitorNodeBase:
    def test_copy_and_equality(self):
        node = MonitorNode("base")
        copy = node.copy()
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


class TestUnaryMonitorNode:
    def test_child_copy(self):
        child = MonitorNode("child")
        node = UnaryMonitorNode("parent", child)
        copy = node.copy()
        assert node == copy
        assert node is not copy
        assert node.child == copy.child


class TestVaradicMonitorNode:
    def test_multiple_children(self):
        children = (MonitorNode("c1"), MonitorNode("c2"))
        node = VaradicMonitorNode("multi", children)
        assert node.children == children


class TestQuantMonitorNode:
    def test_copy(self):
        node = QuantMonitorNode("quant", MonitorNode("c"), 1)
        copy = node.copy()
        assert node == copy
        assert node is not copy


class TestSelectiveQuantMonitorNode:
    def test_selected_tracking(self):
        node = SelectiveQuantMonitorNode("sel", MonitorNode("c"), 1)
        node.last_selected = MonitorNode("sel1")
        assert node.last_selected.name == "sel1"
        node.reset()
        assert node.selected == []
