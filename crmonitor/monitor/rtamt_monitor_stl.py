import copy
from enum import Enum
from typing import List, Tuple

import rtamt

from crmonitor.monitor.rule import IOType, RuleNode
from .specification_dict import (
    stl_discrete_time_online_specification_factory,
)


class OutputType(Enum):
    STANDARD = rtamt.Semantics.STANDARD
    OUTPUT_ROBUSTNESS = rtamt.Semantics.OUTPUT_ROBUSTNESS


class RtamtStlMonitor:
    specs = {}
    """
    Represents single formalized STL rule
    """

    @staticmethod
    def construct_monitor(
        logic_formula: str, output_type: OutputType, predicates, dt
    ) -> rtamt.STLSpecification:
        # Workaround for rtamt when working with output-robustness and input vacuity
        for pred in predicates:
            logic_formula = logic_formula.replace(
                pred[0].name, f"({pred[0].name} >= 0)"
            )

        spec = stl_discrete_time_online_specification_factory(
            semantics=output_type.value
        )
        for var, io_type in predicates:
            spec.declare_var(var.name, "float")
            if io_type == IOType.INPUT:
                spec.set_var_io_type(var.name, "input")
            else:
                spec.set_var_io_type(var.name, "output")
        spec.declare_var("out", "float")

        spec.iosem = output_type
        spec.unit = "ms"
        spec.spec = f"out = {logic_formula}"
        spec.set_sampling_period(dt * 1000.0, "ms")
        spec.parse()
        spec.pastify()
        # new ast of online interpreter is not set until the update method
        # of AbstractOnlineSpecification is called
        spec.online_interpreter.set_ast(spec.ast)

        return spec

    @classmethod
    def create_from_rule_node(
        cls, rule_node: RuleNode, dt: float, output_type=OutputType.STANDARD
    ):
        predicates = [
            (c, c.io_type if hasattr(c, "io_type") else IOType.OUTPUT)
            for c in rule_node.children
        ]
        return cls(rule_node.rule_str, predicates, dt, output_type)

    def __init__(self, rule_str, predicates, dt, output_type=OutputType.STANDARD):
        self._rule = rule_str
        self._predicates = predicates
        self._output_type = output_type
        self.dt = dt
        spec = self.specs.get((rule_str, output_type, dt))
        if spec is None:
            spec = self.specs.setdefault(
                (rule_str, output_type, dt),
                self.construct_monitor(rule_str, output_type, predicates, dt),
            )
            self.specs[(rule_str, output_type, dt)] = spec
        # Flat copy spec and only recreate the online evaluator to
        # avoid parsing the rule.
        self._monitor = spec
        self._monitor = copy.copy(spec)
        # Create a dummy spec to obtain a new interpreter
        dummy_spec = stl_discrete_time_online_specification_factory(output_type.value)
        self._monitor.online_interpreter = dummy_spec.online_interpreter
        # new ast of online interpreter is not set until the
        # update method of AbstractOnlineSpecification is called
        self._monitor.online_interpreter.set_ast(self._monitor.ast)
        self._monitor.reset()

    def reset_monitor(self):
        self._monitor.reset()

    def evaluate_monitor_online(
        self, time_step: int, predicates: List[Tuple[str, float]]
    ):
        time = time_step * self.dt * 1000.0
        rob = self._monitor.update(time, predicates)
        return rob

    def copy(self):
        return RtamtStlMonitor(self._rule, self._predicates, self.dt, self._output_type)

    def reset(self):
        self._monitor.reset()
