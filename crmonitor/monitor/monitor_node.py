from abc import ABC, abstractmethod
from collections import defaultdict
from functools import singledispatchmethod
from typing import Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

from rtamt.semantics.interval.interval import Interval as RtamtInterval

from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.predicates.base import BasePredicateEvaluator
from crmonitor.rule.rule_node import IOType


class MonitorNode:
    """
    Base class for all monitor nodes.

    :param name: Unique name for the monitor node.
    """

    def __init__(self, name: str) -> None:
        self.name = name

        self._values = []

    @property
    def values(self) -> List[float]:
        """
        Retrive all values for the evaluation of this monitor.
        """
        return self._values

    @values.setter
    def values(self, values: List[float]) -> None:
        """
        Set the values for the evaluation of this monitor.
        """
        self._values = list(values)

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

    def copy(self) -> "MonitorNode":
        """
        Create a copy of this monitor, without including any runtime attributes like its recorded values.

        :returns: A copy of the monitor.
        """
        return type(self)(self.name)

    def reset(self):
        self._values = []

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MonitorNode):
            return False
        return self.name == other.name


class ZeroArityMonitorNode(MonitorNode):
    """
    Monitor node with no children.
    """

    ...


class UnaryMonitorNode(MonitorNode):
    """
    Monitor node with a single child.

    :param name: Unique name for the monitor node.
    :param child: Child monitor node.
    """

    def __init__(self, name: str, child: MonitorNode) -> None:
        super().__init__(name)
        self.child = child

    def copy(self) -> "UnaryMonitorNode":
        return type(self)(self.name, self.child.copy())

    def __hash__(self) -> int:
        return hash((self.name, self.child))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnaryMonitorNode):
            return False
        return super().__eq__(other) and self.child == other.child


class VaradicMonitorNode(MonitorNode):
    """
    Monitor node with a variable number of children.

    :param name: Unique name for the monitor node.
    :param children: Tuple of child monitor nodes.
    """

    def __init__(self, name: str, children: Tuple[MonitorNode, ...]) -> None:
        super().__init__(name)
        self.children = children

    def __hash__(self) -> int:
        return hash((self.name, tuple(child.copy() for child in self.children)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VaradicMonitorNode):
            return False
        return super().__eq__(other) and self.children == other.children


class RtamtRuleMonitorNode(VaradicMonitorNode):
    """
    Monitor node wrapping an RTAMT STL monitor.

    :param name: Unique name for the monitor node.
    :param children: Iterable of child monitor nodes.
    :param monitor: RtamtStlMonitor instance for evaluation.
    """

    def __init__(
        self, name: str, children: Iterable[MonitorNode], monitor: RtamtStlMonitor
    ) -> None:
        super().__init__(name, tuple(children))
        self.monitor = monitor

    def update(self, time: int, values: List[Tuple[str, float]]) -> float:
        return self.monitor.evaluate_monitor_online(time, values)

    def evaluate(
        self, values: List[Tuple[str, List[float]]], marker: Optional[str] = None
    ) -> List[float]:
        return self.monitor.evaluate_monitor_offline(values, marker)

    def copy(self):
        return type(self)(self.name, [c.copy() for c in self.children], self.monitor.copy())

    def reset(self):
        super().reset()
        self.monitor.reset()

    def __str__(self) -> str:
        return self.monitor._rule


class QuantMonitorNode(UnaryMonitorNode):
    """
    Monitor node for quantified expressions over agents.

    :param name: Unique name.
    :param child: Child monitor node.
    :param quantified_agent: Index of the quantified agent.
    """

    def __init__(self, name: str, child: MonitorNode, quantified_agent: int) -> None:
        super().__init__(name, child)
        self.quantified_agent = quantified_agent
        self.monitors = defaultdict(child.copy)

    def copy(self) -> "QuantMonitorNode":
        return type(self)(self.name, self.child.copy(), self.quantified_agent)

    def reset(self):
        super().reset()
        self.monitors.clear()


class SelectiveQuantMonitorNode(QuantMonitorNode):
    """
    Quantified monitor node that tracks monitors which were selected during evaluation (e.g. pivotal monitor for 'ALL').

    :param name: Unique name.
    :param child: Child monitor node.
    :param quantified_agent: Index of the quantified agent.
    """

    def __init__(self, name: str, child: MonitorNode, quantified_agent: int) -> None:
        super().__init__(name, child, quantified_agent)

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
    """
    Quantified monitor node representing a universal quantifier (∀).
    """

    def __str__(self) -> str:
        return f"A a{self.quantified_agent}:"


class ExistMonitorNode(SelectiveQuantMonitorNode):
    """
    Quantified monitor node representing an existential quantifier (∃).
    """

    def __str__(self) -> str:
        return f"E a{self.quantified_agent}:"


class SigmoidMonitorNode(UnaryMonitorNode):
    """
    Monitor node applying a sigmoid transformation to its child.
    """

    def __init__(self, name: str, child: MonitorNode) -> None:
        super().__init__(name, child)

    def __str__(self) -> str:
        return "sigmoid"


class HistoricallyDurationMonitorNode(UnaryMonitorNode):
    """
    Monitor node representing a historically-duration temporal operator.

    :param name: Unique name.
    :param child: Child monitor node.
    :param interval: Optional RtamtInterval specifying duration bounds.
    """

    def __init__(
        self, name: str, child: MonitorNode, interval: Optional[RtamtInterval] = None
    ) -> None:
        super().__init__(name, child)
        self.interval = interval

    def copy(self) -> "HistoricallyDurationMonitorNode":
        return type(self)(self.name, self.child.copy(), self.interval)

    def __str__(self) -> str:
        if self.interval is not None:
            return f"historically_duration[{self.interval.begin}{self.interval.begin_unit}, {self.interval.end}{self.interval.end_unit}]"
        else:
            return "historically_duration"


class HistoricallyDurationSeverityMonitorNode(UnaryMonitorNode):
    """
    Monitor node representing a historically-duration operator with severity evaluation.

    :param name: Unique name.
    :param child: Child monitor node.
    :param interval: Optional RtamtInterval specifying duration bounds.
    """

    def __init__(
        self, name: str, child: MonitorNode, interval: Optional[RtamtInterval] = None
    ) -> None:
        super().__init__(name, child)
        self.interval = interval

    def copy(self) -> "HistoricallyDurationSeverityMonitorNode":
        return type(self)(self.name, self.child.copy(), self.interval)

    def __str__(self) -> str:
        if self.interval is not None:
            return f"historically_duration_severity[{self.interval.begin}{self.interval.begin_unit}, {self.interval.end}{self.interval.end_unit}]"
        else:
            return "historically_duration_severity"


class SumIfPositiveMonitorNode(QuantMonitorNode):
    """
    Quantified monitor node that sums values if they are positive.
    """

    def __str__(self) -> str:
        return f"sum_if_positive a{self.quantified_agent}:"


class CompareToThresholdScaledMonitorNode(UnaryMonitorNode):
    """
    Monitor node that compares child output to a scaled threshold.

    :param name: Unique name.
    :param child: Child monitor node.
    :param threshold: Threshold value for comparison.
    """

    def __init__(self, name: str, child: MonitorNode, threshold: float) -> None:
        super().__init__(name, child)
        self.threshold = threshold

    def copy(self) -> "CompareToThresholdScaledMonitorNode":
        return type(self)(self.name, self.child.copy(), self.threshold)

    def __str__(self) -> str:
        return f"compare_to_threshold_scaled[>={self.threshold}]"


class ExistsMultipleMonitorNode(QuantMonitorNode):
    """
    Quantified monitor that checks for the existence of multiple agents satisfying a condition.

    :param name: Unique name.
    :param child: Child monitor node.
    :param quantified_vehicle: Quantified agent index.
    :param threshold: Minimum number of agents that must satisfy the condition.
    """

    def __init__(
        self, name: str, child: MonitorNode, quantified_vehicle: int, threshold: int
    ) -> None:
        super().__init__(name, child, quantified_vehicle)
        self.threshold = threshold

    def copy(self) -> "ExistsMultipleMonitorNode":
        return type(self)(self.name, self.child.copy(), self.quantified_agent, self.threshold)

    def __str__(self) -> str:
        return f"exists_multiple[{self.threshold}]"


class PredicateMonitorNode(ZeroArityMonitorNode):
    """
    Monitor node for evaluating a predicate.

    :param name: Unique name.
    :param evaluator: BasePredicateEvaluator instance.
    :param agent_placeholders: Tuple of agent placeholder indices.
    :param io_type: IOType representing the input/output type.
    """

    def __init__(
        self,
        name: str,
        evaluator: BasePredicateEvaluator,
        agent_placeholders: Tuple[int, ...],
        io_type: IOType,
    ) -> None:
        super().__init__(name)
        self.evaluator = evaluator
        self.agent_placeholders = agent_placeholders
        self.io_type = io_type

    def copy(self) -> "PredicateMonitorNode":
        return type(self)(self.name, self.evaluator, self.agent_placeholders, self.io_type)

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
    Helper node that injects a constant trace into the monitor tree.

    :param name: Unique name.
    :param trace: List of float values representing the constant trace.
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
