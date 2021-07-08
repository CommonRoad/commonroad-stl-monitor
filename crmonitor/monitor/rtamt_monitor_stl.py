from typing import List, Tuple

import rtamt
from crmonitor.predicates.rule import IOType, RuleNode
from rtamt import Language


class RtamtStlMonitor:
    """
    Represents single formalized STL rule
    """
    @staticmethod
    def _reconstruct_logic_formula(logic_formula, predicates):
        replacements = {"~": "not"}
        for el in replacements.keys():
            logic_formula = logic_formula.replace(el, replacements[el])
        # Workaround for rtamt when working with output-robustness and input vacuity
        mod_formula = logic_formula
        # TODO: Only required for IA-STL
        # for pred in predicates:
        #     mod_formula.replace(pred, "({} >= 0)".format(pred))
        return mod_formula

    @staticmethod
    def construct_monitor(formula, output_type, predicates) -> rtamt.STLSpecification:
        logic_formula = RtamtStlMonitor._reconstruct_logic_formula(formula, predicates)
        monitor = rtamt.STLDiscreteTimeSpecification(
            semantics=output_type, language=Language.PYTHON
        )
        for var, io_type in predicates:
            monitor.declare_var(var.name, "float")
            if io_type == IOType.INPUT:
                monitor.set_var_io_type(var.name, "input")
            else:
                monitor.set_var_io_type(var.name, "output")
        monitor.declare_var("out", "float")

        monitor.iosem = output_type

        monitor.spec = f"out = {logic_formula}"
        monitor.parse()

        return monitor

    @classmethod
    def create_from_rule_node(cls, rule_node: RuleNode):
        predicates = [(c, c.io_type if hasattr(c, "io_type") else IOType.OUTPUT) for c in rule_node.children]
        return cls(rule_node.rule_str, predicates)

    def __init__(self, rule_str, predicates, output_type="standard"):
        self._rule = rule_str
        self._predicates = predicates
        self._output_type = output_type
        self._monitor = self.construct_monitor(rule_str, output_type, predicates)

    def reset_monitor(self):
        self._monitor.reset()

    def evaluate_monitor_online(self, time: float, predicates: List[Tuple[str, float]]):
        rob = self._monitor.update(time, predicates)
        return rob

    def copy(self):
        return RtamtStlMonitor(self._rule, self._predicates, self._output_type)