from abc import ABC, abstractmethod
from collections import defaultdict
from functools import singledispatchmethod
from typing import Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

from rtamt.semantics.interval.interval import Interval

from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.rule.rule_node import VisitorNode


class MonitorNode(VisitorNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        # For recording values of this monitor.
        # The values are populated by other visitors with `record_values`.
        self._values = []

    @property
    def values(self) -> List[float]:
        return self._values

    def record_values(self, values: Union[Iterable[float], float]) -> None:
        """
        Save the values of this monitor after a computation.

        :param values: Either a single value or an iterable of values.

        :returns: Nothing.
        """
        if isinstance(values, Iterable):
            self._values.extend(values)
        else:
            self._values.append(values)

    @classmethod
    def _copy_cls(cls, node: "MonitorNode") -> "MonitorNode":
        return cls(node.name)

    def copy(self) -> "MonitorNode":
        """
        Create a copy of this monitor, without including any runtime attributes like its recorded values.

        :returns: A copy of the monitor.
        """
        return self._copy_cls(self)

    def reset(self):
        self._values = []


class ZeroArityMonitorNode(MonitorNode): ...


class OneArityMonitorNode(MonitorNode):
    def __init__(self, name: str, child: MonitorNode) -> None:
        super().__init__(name)
        self.child = child

    @classmethod
    def _copy_cls(cls, node: "OneArityMonitorNode") -> "OneArityMonitorNode":
        return cls(node.name, node.child.copy())


class TwoArityMonitorNode(MonitorNode):
    def __init__(self, name: str, left_child: MonitorNode, right_child: MonitorNode) -> None:
        super().__init__(name)
        self.left_child = left_child
        self.right_child = right_child

    @classmethod
    def _copy_cls(cls, node: "TwoArityMonitorNode") -> "TwoArityMonitorNode":
        return cls(node.name, node.left_child.copy(), node.right_child.copy())


class NArityMonitorNode(MonitorNode):
    def __init__(self, name: str, children: Sequence[MonitorNode]) -> None:
        super().__init__(name)
        self.children = children


class RuleMonitorNode(NArityMonitorNode):
    def __init__(
        self, name: str, children: Sequence[MonitorNode], monitor: RtamtStlMonitor
    ) -> None:
        super().__init__(name, children)
        self.monitor = monitor

    def update(self, time, values) -> float:
        return self.monitor.evaluate_monitor_online(time, values)

    def evaluate(self, values) -> List[float]:
        return self.monitor.evaluate_monitor_offline(values)

    def copy(self):
        return RuleMonitorNode(self.name, [c.copy() for c in self.children], self.monitor.copy())

    def reset(self):
        super().reset()
        self.monitor.reset()


class QuantMonitorNode(OneArityMonitorNode):
    def __init__(self, name: str, child: MonitorNode, quantified_vehicle: int) -> None:
        super().__init__(name, child)
        self.quantified_vehicle = quantified_vehicle
        self.monitors = defaultdict(child.copy)
        self.last_selected = None

    @classmethod
    def _copy_cls(cls, node: "QuantMonitorNode") -> "QuantMonitorNode":
        return cls(node.name, node.child.copy(), node.quantified_vehicle)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()


class AllMonitorNode(QuantMonitorNode): ...


class ExistMonitorNode(QuantMonitorNode): ...


class AndSmoothMonitorNode(TwoArityMonitorNode):
    def __init__(self, name: str, child_left: MonitorNode, child_right: MonitorNode) -> None:
        super().__init__(name, child_left, child_right)


class HistoricallyDurationMonitorNode(OneArityMonitorNode):
    def __init__(self, name: str, child: MonitorNode, interval: Optional[Interval]) -> None:
        super().__init__(name, child)
        self.interval = interval


class HistoricallyDurationSeverityMonitorNode(OneArityMonitorNode):
    def __init__(self, name: str, child: MonitorNode, interval: Optional[Interval]) -> None:
        super().__init__(name, child)
        self.interval = interval


class SumIfPositiveMonitorNode(QuantMonitorNode): ...


class CompareToThresholdScaledMonitorNode(OneArityMonitorNode):
    def __init__(self, name: str, child: MonitorNode, threshold: float) -> None:
        super().__init__(name, child)
        self.threshold = threshold


class PredicateMonitorNode(ZeroArityMonitorNode):
    def __init__(
        self, name: str, evaluator: BasePredicateEvaluator, agent_placeholders: Tuple[int, ...]
    ) -> None:
        super().__init__(name)
        self.evaluator = evaluator
        self.agent_placeholders = agent_placeholders

    @classmethod
    def _copy_cls(cls, node: "PredicateMonitorNode") -> "PredicateMonitorNode":
        return cls(node.name, node.evaluator, node.agent_placeholders)

    def evaluate_boolean(self, world, time_step, vehicle_ids):
        value = self.evaluator.evaluate_boolean(world, time_step, vehicle_ids)
        self.latest_value = 1.0 if value else -1.0
        self.latest_vehicle_ids = tuple(vehicle_ids)
        return value

    def evaluate_robustness(self, world, mpr_world, time_step, vehicle_ids):
        value = self.evaluator.evaluate_robustness_with_cache(
            world, mpr_world, time_step, vehicle_ids
        )
        self.latest_vehicle_ids = tuple(vehicle_ids)
        if mpr_world is not None:
            self.mpr_gradient = self.evaluator.gradient_mpr()
        return value


T = TypeVar("T")


class MonitorVisitorInterface(Generic[T], ABC):
    @singledispatchmethod
    @abstractmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: RuleMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: AllMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: ExistMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: SumIfPositiveMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: AndSmoothMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: HistoricallyDurationMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: HistoricallyDurationSeverityMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: CompareToThresholdScaledMonitorNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: PredicateMonitorNode, *args, **kwargs) -> T: ...
