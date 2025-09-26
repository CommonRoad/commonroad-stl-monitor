from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from typing import Callable, Iterable, TypeVar

import rtamt
from rtamt.spec.abstract_specification import (
    AbstractOfflineSpecification,
    AbstractOnlineSpecification,
    AbstractSpecification,
)
from rtamt.syntax.ast.parser.abstract_ast_parser import AbstractAst
from rtamt.syntax.ast.parser.stl.specification_parser import StlAst
from typing_extensions import Self, override

from crmonitor.rule.rule_node import IOType, PredicateNode, RtamtRuleNode

from .specification_dict import (
    stl_discrete_time_offline_specification_factory,
    stl_discrete_time_online_specification_factory,
)


class OutputType(Enum):
    """Specifies output semantics for STL monitoring."""

    STANDARD = rtamt.Semantics.STANDARD
    OUTPUT_ROBUSTNESS = rtamt.Semantics.OUTPUT_ROBUSTNESS


def _declare_variables_in_ast(ast: AbstractAst, predicates: Iterable[tuple[str, IOType]]) -> None:
    """Declares and sets input/output types for predicates."""
    for pred_name, io_type in predicates:
        ast.declare_var(pred_name, "float")
        ast.set_var_io_type(pred_name, "input" if io_type == IOType.INPUT else "output")


def _wrap_predicates_with_nonneg_check(
    formula: str, predicates: Iterable[tuple[str, IOType]]
) -> str:
    """Ensures predicates are robustness-safe by wrapping them with >= 0."""
    for pred_name, _ in predicates:
        formula = formula.replace(pred_name, f"({pred_name} >= 0)")
    return formula


@lru_cache(None)
def _parse_rtamt_formula(formula: str, predicates: Iterable[tuple[str, IOType]]) -> AbstractAst:
    ast = StlAst()
    ast.spec = f"out = {formula}"

    _declare_variables_in_ast(ast, predicates)
    ast.declare_var("out", "float")

    ast.parse()

    return ast


_T = TypeVar("_T", bound=AbstractSpecification)


def _create_rtamt_spec(
    formula: str,
    output_type: OutputType,
    predicates: Iterable[tuple[str, IOType]],
    dt: float,
    spec_factory: Callable[
        [rtamt.Semantics, AbstractAst], _T
    ] = stl_discrete_time_online_specification_factory,
) -> _T:
    """Creates a fresh STL spec with a unique online/offline interpreter.

    Custom spec factories can be provided to change what kind of spec is created.
    This is useful to create online and offline specs through a similar interface.

    :param formula: The formula for which this spec is created.
    :param output_type: Output type for the spec.
    :param predicates: Collection of predicates alongside their I/O type.
    :param dt: The sampling dt.
    :param spec_factory: Provide a factory method, to construct the spec based on the semantics.
    """
    if output_type == OutputType.OUTPUT_ROBUSTNESS:
        # Force robustness-safe predicate expression
        formula = _wrap_predicates_with_nonneg_check(formula, predicates)

    ast = _parse_rtamt_formula(formula, tuple(predicates))

    spec = spec_factory(output_type.value, ast)

    spec.set_sampling_period(dt, "s")

    return spec


class AbstractRtamtStlMonitor(ABC):
    """
    A wrapper around a RTAMT spec.

    Can be implemented to provide specific evaluation logic for a RTAMT spec, e.g., online/offline evaluation.
    """

    _rule: str
    _predicates: list[tuple[str, IOType]]
    _dt: float
    _output_type: OutputType

    @classmethod
    def create_from_rule_node(
        cls, rule_node: RtamtRuleNode, dt: float, output_type: OutputType = OutputType.STANDARD
    ) -> Self:
        """
        Create a new monitor from a given RTAMT node.
        """
        predicates = [
            (
                c.name,
                c.io_type
                if isinstance(c, PredicateNode) and c.io_type is not None
                else IOType.OUTPUT,
            )
            for c in rule_node.children
        ]
        return cls(rule_node.rule_str, predicates, dt, output_type)

    def __init__(
        self,
        rule_str: str,
        predicates: list[tuple[str, IOType]],
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
    ):
        self._rule = rule_str
        self._predicates = predicates
        self._dt = dt
        self._output_type = output_type

    @property
    def dt(self) -> float:
        return self._dt

    @property
    @abstractmethod
    def ast_node_values(self) -> dict[str, list[float]]:
        """
        The robustness trace of all internal RTAMT nodes over of the last evaluation time steps.
        """
        ...

    @abstractmethod
    def evaluate_monitor_online(
        self, time_step: int, predicates: list[tuple[str, float]]
    ) -> float: ...

    @abstractmethod
    def evaluate_monitor_offline(
        self, predicates: list[tuple[str, list[float]]]
    ) -> list[float]: ...

    def __deepcopy__(self, memo):
        """
        Copy the monitor, but recreate the spec.
        This ensures that we always start from a clean spec, and do not accidently copy
        old values during online evaluation.
        """
        return type(self)(self._rule, self._predicates, self.dt, self._output_type)

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the monitor for re-evaluation.
        """
        ...


class OnlineRtamtStlMonitor(AbstractRtamtStlMonitor):
    """
    A monitor for RTAMT STL formulas which supports online evaluation.
    """

    _spec: AbstractOnlineSpecification

    def __init__(
        self,
        rule_str: str,
        predicates: list[tuple[str, IOType]],
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
    ) -> None:
        super().__init__(rule_str, predicates, dt, output_type)

        self._spec = _create_rtamt_spec(
            rule_str, output_type, predicates, dt, stl_discrete_time_online_specification_factory
        )
        self._ast_node_values: dict[str, list[float]] = defaultdict(list)

    @property
    @override
    def ast_node_values(self) -> dict[str, list[float]]:
        return dict(self._ast_node_values)

    @override
    def evaluate_monitor_online(self, time_step: int, predicates: list[tuple[str, float]]) -> float:
        time = time_step * self.dt
        rob: float = self._spec.update(time, predicates)

        for (
            node_name,
            node_rob,
        ) in self._spec.online_interpreter.updateVisitor.ast_node_values.items():
            self._ast_node_values[node_name].append(node_rob)

        return rob

    @override
    def evaluate_monitor_offline(self, predicates: list[tuple[str, list[float]]]) -> list[float]:
        raise RuntimeError("Cannot evaluate `OnlineRtamtStlMonitor` in offline mode")

    @override
    def reset(self) -> None:
        self._spec.reset()

        del self._ast_node_values
        self._ast_node_values = defaultdict(list)


class OfflineRtamtStlMonitor(AbstractRtamtStlMonitor):
    """
    A monitor for RTAMT STL formulas which supports offline evaluation.
    """

    _spec: AbstractOfflineSpecification

    def __init__(
        self,
        rule_str: str,
        predicates: list[tuple[str, IOType]],
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
    ) -> None:
        super().__init__(rule_str, predicates, dt, output_type)

        self._spec = _create_rtamt_spec(
            rule_str, output_type, predicates, dt, stl_discrete_time_offline_specification_factory
        )

    @property
    @override
    def ast_node_values(self) -> dict[str, list[float]]:
        return self._spec.offline_interpreter.ast_node_values

    @override
    def evaluate_monitor_online(self, time_step: int, predicates: list[tuple[str, float]]) -> float:
        raise RuntimeError("Cannot evaluate `OfflineRtamtStlMonitor` in online mode")

    @override
    def evaluate_monitor_offline(self, predicates: list[tuple[str, list[float]]]) -> list[float]:
        max_time = 0
        dataset = {}
        for i, (predicate_name, values) in enumerate(predicates):
            dataset[predicate_name] = values
            max_time = max(max_time, len(values))

        # RTAMT requires a time column.
        dataset["time"] = []
        for i in range(0, max_time):
            dataset["time"].append(i)
        robustness_values = self._spec.evaluate(dataset)

        # The robustness values are of the form [[time_step, robustness_value], [time_step + 1, robustness_value]]
        return [entry[1] for entry in robustness_values]

    @override
    def reset(self) -> None:
        # For offline evaluation, reset is a noop.
        pass
