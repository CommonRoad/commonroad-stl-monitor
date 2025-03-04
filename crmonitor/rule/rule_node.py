from abc import ABC, abstractmethod
from enum import Enum
from functools import singledispatchmethod
from typing import Generic, Optional, Tuple, TypeVar

from rtamt.semantics.interval.interval import Interval


class IOType(Enum):
    OUTPUT = "output"
    INPUT = "input"


class VisitorNode:
    def __init__(self, name: str) -> None:
        self.name = name


class ZeroArityNode(VisitorNode): ...


class OneArityNode(VisitorNode):
    def __init__(self, name: str, child: VisitorNode) -> None:
        super().__init__(name)
        self.child = child


class TwoArityNode(VisitorNode):
    def __init__(self, name: str, left_child: VisitorNode, right_child: VisitorNode) -> None:
        super().__init__(name)
        self.left_child = left_child
        self.right_child = right_child


class RuleNode(VisitorNode):
    def __init__(self, children, rule_str, name):
        self.children = children
        self.name = name
        self.rule_str = rule_str

    def visit(self, visitor, *ctx):
        return visitor.visit_rule_node(self, *ctx)


class QuantNode(OneArityNode):
    """
    Base node for all quantifiers.

    :param name: The name in the RTAMT rule.
    :param child: The rule for this quantifier.
    :param quantified_vehicle: The id of the quantification placeholder. E.g. the placeholder 'a0' results in the id 0.
    """

    def __init__(self, name: str, child: VisitorNode, quantified_vehicle: int):
        super().__init__(name, child)
        self.quantified_vehicle = quantified_vehicle


class AllNode(QuantNode): ...


class ExistNode(QuantNode): ...


class SumIfPositiveNode(QuantNode): ...


class AndsmoothNode(TwoArityNode): ...


class HistoricallyDurationNode(OneArityNode):
    def __init__(self, name: str, child: VisitorNode, interval: Optional[Interval]):
        super().__init__(name, child)
        self.interval = interval


class HistoricallyDurationSeverityNode(OneArityNode):
    def __init__(self, name: str, child: VisitorNode, interval: Optional[Interval]):
        super().__init__(name, child)
        self.interval = interval


class CompareToThresholdScaledNode(OneArityNode):
    def __init__(self, name: str, child: VisitorNode, threshold: float):
        super().__init__(name, child)
        self.threshold = threshold


class PredicateNode(ZeroArityNode):
    def __init__(
        self,
        node_name: str,
        base_name: str,
        agent_placeholders: Tuple[int, ...],
        io_type: IOType = IOType.OUTPUT,
    ) -> None:
        super().__init__(node_name)
        self.base_name = base_name
        self.agent_placeholders = agent_placeholders
        self.io_type = io_type


T = TypeVar("T")


class RuleTreeVisitorInterface(Generic[T], ABC):
    """
    Abstract visitor for rule trees.
    """

    @singledispatchmethod
    @abstractmethod
    def visit(self, node: VisitorNode, *args, **kwargs) -> T:
        raise RuntimeError

    @visit.register
    def _(self, node: RuleNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: AllNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: ExistNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: HistoricallyDurationNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: HistoricallyDurationSeverityNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: SumIfPositiveNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: CompareToThresholdScaledNode, *args, **kwargs) -> T: ...

    @visit.register
    def _(self, node: PredicateNode, *args, **kwargs) -> T: ...
