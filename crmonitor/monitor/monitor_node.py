from abc import ABC, abstractmethod
from collections import defaultdict
from functools import singledispatchmethod
from typing import Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar

from rtamt.semantics.interval.interval import Interval as RtamtInterval

from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.rule.rule_node import VisitorNode


class MonitorNode(VisitorNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self._values = []

    @property
    def values(self) -> List[float]:
        """
        Retrive all values for the evaluation of this monitor.
        """
        return self._values

    @values.setter
    def values(self, values: Iterable[float]) -> None:
        """
        Set the values for the evaluation of this monitor.
        """
        self._values.extend(values)

    @property
    def last_value(self) -> float:
        """
        Get the value of the last evaluation of this monitor.
        """
        return self._values[-1]

    @last_value.setter
    def last_value(self, value: float) -> None:
        """
        Set the value of the last evaluation of this monitor.
        """
        self._values.append(value)

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


class UnaryMonitorNode(MonitorNode):
    def __init__(self, name: str, child: MonitorNode) -> None:
        super().__init__(name)
        self.child = child

    @classmethod
    def _copy_cls(cls, node: "UnaryMonitorNode") -> "UnaryMonitorNode":
        return cls(node.name, node.child.copy())


class BinaryMonitorNode(MonitorNode):
    def __init__(self, name: str, left_child: MonitorNode, right_child: MonitorNode) -> None:
        super().__init__(name)
        self.left_child = left_child
        self.right_child = right_child

    @classmethod
    def _copy_cls(cls, node: "BinaryMonitorNode") -> "BinaryMonitorNode":
        return cls(node.name, node.left_child.copy(), node.right_child.copy())


class VaradicMonitorNode(MonitorNode):
    def __init__(self, name: str, children: Sequence[MonitorNode]) -> None:
        super().__init__(name)
        self.children = children


class RuleMonitorNode(VaradicMonitorNode):
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

    def __str__(self) -> str:
        return self.monitor._rule


class QuantMonitorNode(UnaryMonitorNode):
    def __init__(self, name: str, child: MonitorNode, quantified_vehicle: int) -> None:
        super().__init__(name, child)
        self.quantified_vehicle = quantified_vehicle
        self.monitors = defaultdict(child.copy)

    @classmethod
    def _copy_cls(cls, node: "QuantMonitorNode") -> "QuantMonitorNode":
        return cls(node.name, node.child.copy(), node.quantified_vehicle)

    def reset(self):
        super().reset()
        self.monitors.clear()


class SelectiveQuantMonitorNode(QuantMonitorNode):
    def __init__(self, name: str, child: MonitorNode, quantified_vehicle: int) -> None:
        super().__init__(name, child, quantified_vehicle)

        self._selected: List[Optional[MonitorNode]] = []

    @property
    def selected(self) -> List[Optional[MonitorNode]]:
        return self._selected

    @selected.setter
    def selected(self, monitors: List[Optional[MonitorNode]]) -> None:
        self._selected = monitors

    @property
    def last_selected(self) -> Optional[MonitorNode]:
        return self._selected[-1]

    @last_selected.setter
    def last_selected(self, monitor: Optional[MonitorNode]) -> None:
        self._selected.append(monitor)

    def reset(self):
        super().reset()
        self._selected = []


class AllMonitorNode(SelectiveQuantMonitorNode):
    def __str__(self) -> str:
        return f"A a{self.quantified_vehicle}:"


class ExistMonitorNode(SelectiveQuantMonitorNode):
    def __str__(self) -> str:
        return f"E a{self.quantified_vehicle}:"


class SigmoidMonitorNode(UnaryMonitorNode):
    def __init__(self, name: str, child: MonitorNode) -> None:
        super().__init__(name, child)

    def __str__(self) -> str:
        return "sigmoid"


class HistoricallyDurationMonitorNode(UnaryMonitorNode):
    def __init__(
        self, name: str, child: MonitorNode, interval: Optional[RtamtInterval] = None
    ) -> None:
        super().__init__(name, child)
        self.interval = interval

    @classmethod
    def _copy_cls(
        cls, node: "HistoricallyDurationMonitorNode"
    ) -> "HistoricallyDurationMonitorNode":
        return cls(node.name, node.child.copy(), node.interval)

    def __str__(self) -> str:
        if self.interval is not None:
            return f"historically_duration[{self.interval.begin}{self.interval.begin_unit}, {self.interval.end}{self.interval.end_unit}]"
        else:
            return "historically_duration"


class HistoricallyDurationSeverityMonitorNode(UnaryMonitorNode):
    def __init__(
        self, name: str, child: MonitorNode, interval: Optional[RtamtInterval] = None
    ) -> None:
        super().__init__(name, child)
        self.interval = interval

    @classmethod
    def _copy_cls(
        cls, node: "HistoricallyDurationSeverityMonitorNode"
    ) -> "HistoricallyDurationSeverityMonitorNode":
        return cls(node.name, node.child.copy(), node.interval)

    def __str__(self) -> str:
        if self.interval is not None:
            return f"historically_duration_severity[{self.interval.begin}{self.interval.begin_unit}, {self.interval.end}{self.interval.end_unit}]"
        else:
            return "historically_duration_severity"


class SumIfPositiveMonitorNode(QuantMonitorNode):
    def __str__(self) -> str:
        return f"sum_if_positive a{self.quantified_vehicle}:"


class CompareToThresholdScaledMonitorNode(UnaryMonitorNode):
    def __init__(self, name: str, child: MonitorNode, threshold: float) -> None:
        super().__init__(name, child)
        self.threshold = threshold

    @classmethod
    def _copy_cls(
        cls, node: "CompareToThresholdScaledMonitorNode"
    ) -> "CompareToThresholdScaledMonitorNode":
        return cls(node.name, node.child.copy(), node.threshold)

    def __str__(self) -> str:
        return f"compare_to_threshold_scaled[>={self.threshold}]"


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
        self.latest_vehicle_ids = tuple(vehicle_ids)
        return value

    def evaluate_robustness(self, world, mpr_world, time_step, vehicle_ids):
        value = self.evaluator.evaluate_robustness_with_cache(
            world, mpr_world, time_step, vehicle_ids
        )
        self.latest_vehicle_ids = tuple(vehicle_ids)
        if (
            self.evaluator.config.mpr.enabled
            and self.evaluator.config.mpr.ml
            and self.evaluator.config.mpr.extract_gradient
        ):
            self.mpr_gradient = self.evaluator.last_gradient
        return value

    def __str__(self) -> str:
        placeholders = ", ".join(f"a{placeholder}" for placeholder in self.agent_placeholders)
        return f"{self.evaluator.predicate_name.value}({placeholders})"

    def format_with_vehicle_ids(self, vehicle_ids: Dict[int, int] = {}) -> str:
        optionally_filled_placeholders = map(
            lambda placeholder_id: str(vehicle_ids[placeholder_id])
            if placeholder_id in vehicle_ids
            else f"a{placeholder_id}",
            self.agent_placeholders,
        )
        argument_str = ", ".join(optionally_filled_placeholders)
        return f"{self.evaluator.predicate_name.value}({argument_str})"


class ConstantTraceMonitorNode(ZeroArityMonitorNode):
    """
    Helper node for artificial monitor tree constructions.

    Use this node to inject a constant trace into the tree.
    """

    def __init__(self, name: str, trace: List[float]) -> None:
        super().__init__(name)
        self.trace = trace

    def __str__(self) -> str:
        return self.name


T = TypeVar("T")


class MonitorVisitorInterface(Generic[T], ABC):
    @singledispatchmethod
    @abstractmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> T: ...
