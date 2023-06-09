import copy
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import rtamt

from crmonitor.monitor.rule import IOType, RuleNode
from rtamt.syntax.node.ltl.implies import Implies
from rtamt.syntax.node.ltl.conjunction import Conjunction
from rtamt.syntax.node.ltl.disjunction import Disjunction
from rtamt.syntax.node.stl.timed_historically import TimedHistorically
from rtamt.syntax.node.stl.timed_eventually import TimedEventually
from rtamt.syntax.node.stl.timed_always import TimedAlways
from rtamt.syntax.node.stl.timed_once import TimedOnce
from rtamt.syntax.node.ltl.previous import Previous
from rtamt.syntax.node.ltl.neg import Neg
from rtamt.syntax.node.ltl.predicate import Predicate
from rtamt.syntax.node.ltl.variable import Variable
from rtamt.syntax.node.unary_node import UnaryNode

from .specification_dict import stl_discrete_time_online_specification_factory


class OutputType(Enum):
    STANDARD = rtamt.Semantics.STANDARD
    OUTPUT_ROBUSTNESS = rtamt.Semantics.OUTPUT_ROBUSTNESS


@lru_cache(None)
def _template_spec(
    logic_formula: str, output_type: OutputType, predicates, dt
) -> rtamt.STLSpecification:
    if output_type != OutputType.STANDARD:
        # Workaround for rtamt when working with output-robustness and input vacuity
        for pred in predicates:
            logic_formula = logic_formula.replace(
                pred[0].name, f"({pred[0].name} >= 0)"
            )

    spec = stl_discrete_time_online_specification_factory(semantics=output_type.value)
    for var, io_type in predicates:
        spec.declare_var(var.name, "float")
        if io_type == IOType.INPUT:
            spec.set_var_io_type(var.name, "input")
        else:
            spec.set_var_io_type(var.name, "output")
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
    rule_str: str, output_type: OutputType, predicates: List[Tuple[str, Any]], dt: float
) -> rtamt.STLSpecification:
    template_spec = _template_spec(rule_str, output_type, tuple(predicates), dt)
    # The dynamic part of the template spec has to be replaced.
    spec = copy.copy(template_spec)
    # Create a dummy spec to obtain a new interpreter
    dummy_spec = stl_discrete_time_online_specification_factory(output_type.value)
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
        self._dt = dt

        # Flat copy spec and only recreate the online evaluator to
        # avoid parsing the rule.
        self._spec = _create_spec(rule_str, output_type, predicates, dt)

        # Flat copy spec and only recreate the online evaluator to avoid parsing the rule.
        # self._monitor = copy.copy(self._spec)
        self._propositions = {}

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def ast_node_values(self) -> Dict[str, float]:
        return self._spec.online_interpreter.updateVisitor.ast_node_values

    def evaluate_monitor_online(
        self, time_step: int, predicates: List[Tuple[str, float]]
    ):
        time = time_step * self.dt
        rob = self._spec.update(time, predicates)
        self.collect_prop_rob(self._spec.ast.specs[0], self._propositions)
        return rob

    def collect_prop_rob(self, specs_node=None, prop_list=None):
        """
        Collects the propositions (abstractions) recursively to pass them to the monitor wrapper.
        If a sub-formula is encapsulated by an LTL/STL indicator, it constitutes a proposition.
        If negations exist, the non-negated formula that follows the negation is considered.
        If a predicate is not encapsulated by an LTL/STL indicator, it constitutes a proposition alone.
        Formulas may contain only: Implications, Con/Disjunctions, Negations, LTL/STL indicators.
        The values are obtained directly from the Rtamt.

        Returns:
        None. Acts directly on the dict that was passed as an argument: dict{proposition, robustness_value}
        """
        if specs_node is None:
            specs_node = self._spec.ast.specs[0]
        if isinstance(specs_node, UnaryNode):
            if isinstance(specs_node, Neg):
                self.collect_prop_rob(specs_node.children[0], prop_list)
            if (
                isinstance(specs_node, TimedOnce)
                or isinstance(specs_node, Previous)
                or isinstance(specs_node, TimedAlways)
                or isinstance(specs_node, TimedHistorically)
                or isinstance(specs_node, TimedEventually)
            ):
                prop_list[specs_node.name] = self.ast_node_values[specs_node.name]
        elif isinstance(specs_node, Predicate) or isinstance(specs_node, Variable):
            prop_list[specs_node.name] = self.ast_node_values[specs_node.name]
        else:
            if isinstance(specs_node, Implies):
                if isinstance(specs_node.children[0], Predicate) or isinstance(
                    specs_node.children[0], Variable
                ):
                    prop_list[specs_node.children[0].name] = self.ast_node_values[
                        specs_node.children[0].name
                    ]
                    self.collect_prop_rob(specs_node.children[1], prop_list)
                elif isinstance(specs_node.children[1], Predicate) or isinstance(
                    specs_node.children[1], Variable
                ):
                    prop_list[specs_node.children[1].name] = self.ast_node_values[
                        specs_node.children[1].name
                    ]
                    self.collect_prop_rob(specs_node.children[0], prop_list)
            if isinstance(specs_node, Conjunction) or isinstance(
                specs_node, Disjunction
            ):
                self.collect_prop_rob(specs_node.children[0], prop_list)
                self.collect_prop_rob(specs_node.children[1], prop_list)

    def copy(self):
        return RtamtStlMonitor(self._rule, self._predicates, self.dt, self._output_type)

    def reset(self):
        self._spec.reset()
