import copy
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import rtamt
from rtamt.spec.abstract_specification import (
    AbstractOfflineOnlineSpecification,
    AbstractOnlineSpecification,
)

from crmonitor.rule.rule_node import IOType, RuleAstNode

from .specification_dict import stl_discrete_time_online_specification_factory


class OutputType(Enum):
    STANDARD = rtamt.Semantics.STANDARD
    OUTPUT_ROBUSTNESS = rtamt.Semantics.OUTPUT_ROBUSTNESS


@lru_cache(None)
def _template_spec(
    logic_formula: str,
    output_type: OutputType,
    predicates: Tuple[Tuple[str, IOType], ...],
    dt,
    spec_factory: Callable[
        [rtamt.Semantics], AbstractOnlineSpecification
    ] = stl_discrete_time_online_specification_factory,
) -> AbstractOfflineOnlineSpecification:
    if output_type != OutputType.STANDARD:
        # Workaround for rtamt when working with output-robustness and input vacuity
        for pred_name, _ in predicates:
            logic_formula = logic_formula.replace(pred_name, f"({pred_name} >= 0)")

    spec = spec_factory(output_type.value)
    for pred_name, io_type in predicates:
        spec.declare_var(pred_name, "float")
        if io_type == IOType.INPUT:
            spec.set_var_io_type(pred_name, "input")
        else:
            spec.set_var_io_type(pred_name, "output")
    spec.declare_var("out", "float")

    spec.iosem = output_type
    spec.unit = "s"
    spec.spec = f"out = {logic_formula}"
    spec.set_sampling_period(dt, "s")
    spec.parse()
    spec.pastify()
    # new ast of online interpreter is not set until the update method
    # of AbstractOnlineSpecification is called
    spec.online_interpreter.set_ast(spec.ast)

    return spec


def _create_spec(
    rule_str: str,
    output_type: OutputType,
    predicates: List[Tuple[str, Any]],
    dt: float,
    spec_factory: Callable[
        [rtamt.Semantics], AbstractOfflineOnlineSpecification
    ] = stl_discrete_time_online_specification_factory,
) -> AbstractOfflineOnlineSpecification:
    template_spec = _template_spec(rule_str, output_type, tuple(predicates), dt, spec_factory)
    # The dynamic part of the template spec has to be replaced.
    spec = copy.copy(template_spec)
    # Create a dummy spec to obtain a new interpreter
    dummy_spec = spec_factory(output_type.value)
    spec.online_interpreter = dummy_spec.online_interpreter
    # new ast of online interpreter is not set until the
    # update method of AbstractOnlineSpecification is called
    dummy_spec.online_interpreter.set_sampling_period(dt, "s")
    spec.online_interpreter.set_ast(spec.ast)
    spec.reset()
    return spec


class RtamtStlMonitor:
    """
    Represents single formalized STL rule
    """

    @classmethod
    def create_from_rule_node(
        cls, rule_node: RuleAstNode, dt: float, output_type=OutputType.STANDARD
    ):
        predicates = [
            (c.name, c.io_type if hasattr(c, "io_type") else IOType.OUTPUT)
            for c in rule_node.children
        ]
        return cls(rule_node.rule_str, predicates, dt, output_type)

    def __init__(self, rule_str, predicates, dt, output_type=OutputType.STANDARD):
        self._rule = rule_str
        self._predicates = predicates
        self._output_type = output_type
        self._dt = dt

        # Flat copy spec and only recreate the online evaluator to
        # avoid parsing the rule.
        self._spec = _create_spec(rule_str, output_type, predicates, dt)
        self._ast_node_values = {}

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def ast_node_values(self) -> Dict[str, Dict[str, float]]:
        return self._ast_node_values

    def evaluate_monitor_online(self, time_step: int, predicates: List[Tuple[str, float]]) -> float:
        time = time_step * self.dt
        rob = self._spec.update(time, predicates)
        return rob

    def evaluate_monitor_offline(
        self, predicates: List[Tuple[str, List[float]]], marker: Optional[str] = None
    ) -> List[float]:
        dataset = {}
        max_time = 0
        for i, (predicate_name, values) in enumerate(predicates):
            dataset[predicate_name] = values
            max_time = max(max_time, len(values))

        dataset["time"] = []
        for i in range(0, max_time):
            dataset["time"].append(i)
        robustness_values = self._spec.evaluate(dataset)

        # Extract the node values and save them with the marker.
        self._ast_node_values[marker] = copy.deepcopy(
            self._spec.offline_interpreter.ast_node_values
        )
        self._spec.offline_interpreter.ast_nodes_values = {}

        # The robustness values are of the form [[time_step, robustness_value], [time_step + 1, robustness_value]]
        return [entry[1] for entry in robustness_values]

    def copy(self):
        return RtamtStlMonitor(self._rule, self._predicates, self.dt, self._output_type)

    def reset(self):
        self._spec.reset()
