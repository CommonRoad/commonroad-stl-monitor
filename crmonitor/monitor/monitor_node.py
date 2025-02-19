from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, Sequence

from rtamt.semantics.interval.interval import Interval


class MonitorNode(ABC):
    def __init__(self, name, children=None, **kwargs):
        self.name = name
        self.children = children

    @abstractmethod
    def visit(self, *ctx):
        pass

    @classmethod
    def _copy_cls(cls, o):
        child_copy = [c.copy() for c in o.children] if o.children is not None else None
        return cls(o.name, child_copy)

    def copy(self):
        return self._copy_cls(self)

    def reset(self):
        if self.children is not None:
            for c in self.children:
                c.reset()


class RuleMonitorNode(MonitorNode):
    def __init__(self, name: str, children: Sequence[Any], monitor):
        super().__init__(name, children)
        self.monitor = monitor

    def visit(self, visitor, *ctx):
        return visitor.visit_rule_node(self, *ctx)

    def update(self, time, values):
        return self.monitor.evaluate_monitor_online(time, values)

    def evaluate(self, values):
        return self.monitor.evaluate_monitor_offline(values)

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


class AndsmoothMonitorNode(MonitorNode):
    def __init__(self, name, children):
        assert len(children) == 2
        super().__init__(name, children)
        self.monitors = defaultdict(children[0].copy)
        self.last_selected = None

    def visit(self, visitor, *ctx):
        return visitor.visit_andsmooth_node(self, *ctx)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()


class HistoricallyDurationMonitorNode(MonitorNode):
    def __init__(self, name, children, interval: Optional[Interval]):
        assert len(children) == 1
        super().__init__(name, children)
        self.interval = interval
        self.monitors = defaultdict(children[0].copy)
        self.last_selected = None

    def visit(self, visitor, *ctx):
        return visitor.visit_historicallyduration_node(self, *ctx)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()


class HistoricallyDurationSeverityMonitorNode(MonitorNode):
    def __init__(self, name, children, interval: Optional[Interval]):
        assert len(children) == 1
        super().__init__(name, children)
        self.interval = interval
        self.monitors = defaultdict(children[0].copy)
        self.last_selected = None

    def visit(self, visitor, *ctx):
        return visitor.visit_historicallydurationseverity_node(self, *ctx)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()
