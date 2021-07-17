from abc import abstractmethod, ABC
from collections import defaultdict


class MonitorNode(ABC):
    def __init__(self, name, children=None):
        self.name = name
        self.children = children

    @abstractmethod
    def visit(self, *ctx):
        pass

    @classmethod
    def _copy_cls(cls, o):
        return cls(o.name, [c.copy() for c in o.children])

    def copy(self):
        return self._copy_cls(self)

    def reset(self):
        for c in self.children:
            c.reset()


class RuleMonitorNode(MonitorNode):
    def __init__(self, name, children, monitor):
        super().__init__(name, children)
        self.monitor = monitor

    def visit(self, visitor, *ctx):
        return visitor.visit_rule_node(self, *ctx)

    def evaluate_incremental(self, time, values):
        return self.monitor.evaluate_monitor_online(time, values)

    def copy(self):
        return RuleMonitorNode(
            self.name, [c.copy() for c in self.children], self.monitor.copy()
        )

    def reset(self):
        self.monitor.reset()


class AllMonitorNode(MonitorNode):
    def __init__(self, name, children):
        assert len(children) == 1
        super().__init__(name, children)
        self.monitors = defaultdict(children[0].copy)
        self.last_selected = None

    def visit(self, visitor, *ctx):
        return visitor.visit_all_node(self, *ctx)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()


class ExistMonitorNode(MonitorNode):
    def __init__(self, name, children):
        assert len(children) == 1
        super().__init__(name, children)
        self.monitors = defaultdict(children[0].copy)
        self.last_selected = None

    def visit(self, visitor, *ctx):
        return visitor.visit_exist_node(self, *ctx)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()