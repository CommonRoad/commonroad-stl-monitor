from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import singledispatchmethod
from typing import Generic, List, Optional, Tuple, TypeVar

from rtamt.semantics.interval.interval import Interval


class IOType(Enum):
    OUTPUT = "output"
    INPUT = "input"


@dataclass
class VisitorNode:
    """
    Base class for nodes that can be processed by a visitor.
    """

    name: str
    """The unique name of this node."""


@dataclass
class ZeroArityNode(VisitorNode):
    """Rule nodes that do not have any children."""

    ...


@dataclass
class OneArityNode(VisitorNode):
    """Rule nodes that only have one child. This is used for unary operators."""

    child: VisitorNode


@dataclass
class TwoArityNode(VisitorNode):
    """Rule nodes that have two children. This is used for binary operators"""

    left_child: VisitorNode
    right_child: VisitorNode


@dataclass
class RuleNode(VisitorNode):
    """A node to contain RTAMT rules, which do not contain any further custom operators."""

    children: List[VisitorNode]
    """Children that are referenced in the RTAMT rule."""

    rule_str: str
    """The RTAMT rule."""


@dataclass
class QuantNode(OneArityNode):
    """
    A quantifier node fixes a vehicle placeholder and evaluates its child for each vehicle in the scenario.
    """

    quantified_vehicle: int
    """The ID of the vehicle placeholder. If the placeholder in the rule was `a0` the id will be `0`."""


@dataclass
class AllNode(QuantNode): ...


@dataclass
class ExistNode(QuantNode): ...


@dataclass
class SumIfPositiveNode(QuantNode): ...


@dataclass
class AndsmoothNode(TwoArityNode): ...


@dataclass
class HistoricallyDurationNode(OneArityNode):
    interval: Optional[Interval]


@dataclass
class HistoricallyDurationSeverityNode(OneArityNode):
    interval: Optional[Interval]


@dataclass
class CompareToThresholdScaledNode(OneArityNode):
    threshold: float


@dataclass
class PredicateNode(ZeroArityNode):
    base_name: str
    """The name of the predicate, which can be resolved to an predicate  evaluator."""

    agent_placeholders: Tuple[int, ...]
    """The agent placeholder IDs (`a0`, `a1`, ...) which were passed to this predicate."""

    io_type: IOType = IOType.OUTPUT
    """Specifies whether this predicate is an input or output predicate."""

    def __hash__(self) -> int:
        return hash((self.name, self.agent_placeholders))


T = TypeVar("T")


class RuleTreeVisitorInterface(Generic[T], ABC):
    """
    Abstract visitor for rule trees.
    """

    @singledispatchmethod
    @abstractmethod
    def visit(self, node: VisitorNode, *args, **kwargs) -> T:
        """
        Dispatch method for visiting different types of nodes.
        Must be implemented in subclasses.

        :param node: The node to visit.
        :return: Generic type T representing the result of the visit.
        """
        ...
