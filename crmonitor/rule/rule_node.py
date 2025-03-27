from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import singledispatchmethod
from typing import Generic, Optional, Tuple, TypeVar

from rtamt.semantics.interval.interval import Interval


class IOType(Enum):
    OUTPUT = "output"
    INPUT = "input"


@dataclass(unsafe_hash=True)
class VisitorNode:
    """
    Base class for nodes that can be processed by a visitor.
    """

    name: str
    """The unique name of this node."""


@dataclass(unsafe_hash=True)
class NullaryNode(VisitorNode):
    """Rule nodes that do not have any children."""

    ...


@dataclass(unsafe_hash=True)
class UnaryNode(VisitorNode):
    """Rule nodes that only have one child. This is used for unary operators."""

    child: VisitorNode


@dataclass(unsafe_hash=True)
class BinaryNode(VisitorNode):
    """Rule nodes that have two children. This is used for binary operators"""

    left_child: VisitorNode
    right_child: VisitorNode


@dataclass(unsafe_hash=True)
class RuleNode(VisitorNode):
    """A node to contain RTAMT rules, which do not contain any further custom operators."""

    # children are a tuple because they are immutable. This helps with hashing.
    children: Tuple[VisitorNode, ...]
    """Children that are referenced in the RTAMT rule."""

    rule_str: str
    """The RTAMT rule."""


@dataclass(unsafe_hash=True)
class QuantNode(UnaryNode):
    """
    A quantifier node fixes a vehicle placeholder and evaluates its child for each vehicle in the scenario.
    """

    quantified_vehicle: int
    """The ID of the vehicle placeholder. If the placeholder in the rule was `a0` the id will be `0`."""


@dataclass(unsafe_hash=True)
class AllNode(QuantNode): ...


@dataclass(unsafe_hash=True)
class ExistNode(QuantNode): ...


@dataclass(unsafe_hash=True)
class SumIfPositiveNode(QuantNode): ...


@dataclass(unsafe_hash=True)
class SigmoidNode(UnaryNode): ...


@dataclass(unsafe_hash=True)
class HistoricallyDurationNode(UnaryNode):
    interval: Optional[Interval]


@dataclass(unsafe_hash=True)
class HistoricallyDurationSeverityNode(UnaryNode):
    interval: Optional[Interval]


@dataclass(unsafe_hash=True)
class CompareToThresholdScaledNode(UnaryNode):
    threshold: float


@dataclass(unsafe_hash=True)
class PredicateNode(NullaryNode):
    base_name: str
    """The name of the predicate, which can be resolved to an predicate  evaluator."""

    agent_placeholders: Tuple[int, ...]
    """The agent placeholder IDs (`a0`, `a1`, ...) which were passed to this predicate."""

    io_type: Optional[IOType] = IOType.OUTPUT
    """Specifies whether this predicate is an input or output predicate."""


@dataclass(unsafe_hash=True)
class MetaPredicateNode(VisitorNode):
    metapredicate_name: str

    quantified_agents: Tuple[int, ...]

    io_type: Optional[IOType] = IOType.OUTPUT


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
