from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import rtamt
from rtamt.pastifier.stl.pastifier import StlPastifier
from rtamt.spec.abstract_specification import (
    AbstractOfflineOnlineSpecification,
)
from rtamt.syntax.ast.parser.abstract_ast_parser import AbstractAst
from rtamt.syntax.ast.parser.stl.specification_parser import StlAst

from crmonitor.rule.rule_node import IOType, PredicateNode, RtamtRuleNode

from .specification_dict import stl_discrete_time_online_specification_factory


class OutputType(Enum):
    """Specifies output semantics for STL monitoring."""

    STANDARD = rtamt.Semantics.STANDARD
    OUTPUT_ROBUSTNESS = rtamt.Semantics.OUTPUT_ROBUSTNESS


def _declare_variables_in_spec(
    spec: AbstractOfflineOnlineSpecification, predicates: Iterable[Tuple[str, IOType]]
) -> None:
    """Declares and sets input/output types for predicates."""
    for pred_name, io_type in predicates:
        spec.declare_var(pred_name, "float")
        spec.set_var_io_type(pred_name, "input" if io_type == IOType.INPUT else "output")


def _wrap_predicates_with_nonneg_check(
    formula: str, predicates: Iterable[Tuple[str, IOType]]
) -> str:
    """Ensures predicates are robustness-safe by wrapping them with >= 0."""
    for pred_name, _ in predicates:
        formula = formula.replace(pred_name, f"({pred_name} >= 0)")
    return formula


@lru_cache(None)
def _parse_rtamt_formula(formula: str) -> AbstractAst:
    ast = StlAst()
    ast.spec = formula

    ast.parse()

    pastifier = StlPastifier()
    pastified_ast = pastifier.pastify(ast)

    return pastified_ast


def _create_rtamt_spec(
    formula: str,
    output_type: OutputType,
    predicates: Iterable[Tuple[str, IOType]],
    dt: float,
    spec_factory: Callable[
        [rtamt.Semantics], AbstractOfflineOnlineSpecification
    ] = stl_discrete_time_online_specification_factory,
) -> AbstractOfflineOnlineSpecification:
    """Creates a fresh STL spec with a unique online interpreter.

    :param formula: The formula for which this spec is created.
    :param output_type: Output type for the spec.
    :param predicates: Collection of predicates alongside their I/O type.
    :param dt: The sampling dt.
    :param spec_factory: Provide a factory method, to construct the spec based on the semantics.
    """
    if output_type == OutputType.OUTPUT_ROBUSTNESS:
        # Force robustness-safe predicate expression
        formula = _wrap_predicates_with_nonneg_check(formula, predicates)

    spec = spec_factory(output_type.value)

    _declare_variables_in_spec(spec, predicates)
    spec.declare_var("out", "float")
    spec.spec = f"out = {formula}"

    spec.set_sampling_period(dt, "s")

    spec.parse()
    spec.pastify()

    return spec


class RtamtStlMonitor:
    @classmethod
    def create_from_rule_node(
        cls, rule_node: RtamtRuleNode, dt: float, output_type: OutputType = OutputType.STANDARD
    ):
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

    def __init__(self, rule_str, predicates, dt, output_type=OutputType.STANDARD):
        self._rule = rule_str
        self._predicates = predicates
        self._output_type = output_type
        self._dt = dt

        self._spec = _create_rtamt_spec(rule_str, output_type, predicates, dt)
        self._ast_node_values = {}

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def ast_node_values(self) -> Dict[str, float]:
        if len(self._spec.offline_interpreter.ast_node_values) > 0:
            return self._spec.offline_interpreter.ast_node_values
        else:
            return self._spec.online_interpreter.updateVisitor.ast_node_values

    def evaluate_monitor_online(self, time_step: int, predicates: List[Tuple[str, float]]) -> float:
        time = time_step * self.dt
        rob: float = self._spec.update(time, predicates)
        return rob

    def evaluate_monitor_offline(
        self, predicates: List[Tuple[str, List[float]]], marker: Optional[str] = None
    ) -> List[float]:
        dataset = {}
        max_time = 0
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

    def copy(self):
        return RtamtStlMonitor(self._rule, self._predicates, self.dt, self._output_type)

    def reset(self):
        self._spec.reset()
