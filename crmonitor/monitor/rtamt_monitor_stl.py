import copy
from enum import Enum
from typing import List, Tuple

import rtamt
from rtamt import Language
from rtamt.evaluator.stl.online_evaluator import STLOnlineEvaluator

from crmonitor.monitor.rule import IOType, RuleNode
from rtamt.node.ltl.implies import Implies
from rtamt.node.ltl.conjunction import Conjunction
from rtamt.node.ltl.disjunction import Disjunction
from rtamt.node.stl.timed_once import TimedOnce
from rtamt.node.ltl.previous import Previous
from rtamt.node.ltl.neg import Neg
from rtamt.node.ltl.predicate import Predicate
from rtamt.node.unary_node import UnaryNode

class OutputType(Enum):
    STANDARD = rtamt.Semantics.STANDARD
    OUTPUT_ROBUSTNESS = rtamt.Semantics.OUTPUT_ROBUSTNESS


class RtamtStlMonitor:
    specs = {}
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
        for pred in predicates:
            mod_formula = mod_formula.replace(pred[0].name, f"({pred[0].name} >= 0)")
        return mod_formula

    @staticmethod
    def construct_monitor(formula, output_type: OutputType, predicates, dt) -> rtamt.STLSpecification:
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
        monitor.unit = "ms"
        monitor.spec = f"out = {logic_formula}"
        monitor.set_sampling_period(dt * 1000.0, 'ms')
        monitor.parse()
        monitor.pastify()

        return monitor

    @classmethod
    def create_from_rule_node(cls, rule_node: RuleNode, dt: float, output_type=OutputType.STANDARD):
        predicates = [(c, c.io_type if hasattr(c, "io_type") else IOType.OUTPUT) for c in rule_node.children]
        return cls(rule_node.rule_str, predicates, dt, output_type)

    def __init__(self, rule_str, predicates, dt, output_type=OutputType.STANDARD):
        self._rule = rule_str
        self._predicates = predicates
        self._output_type = output_type
        self.dt = dt
        spec = self.specs.get((rule_str, output_type, dt))
        if spec is None:
            spec = self.specs.setdefault((rule_str, output_type, dt), self.construct_monitor(rule_str, output_type, predicates, dt))
            self.specs[(rule_str, output_type, dt)] = spec
        # Flat copy spec and only recreate the online evaluator to avoid parsing the rule.
        self._monitor = copy.copy(spec)
        self._monitor.online_evaluator = STLOnlineEvaluator(self._monitor)
        self._monitor.top.accept(self._monitor.online_evaluator)
        self._monitor.reseter.node_monitor_dict = self._monitor.online_evaluator.node_monitor_dict
        self._monitor.reset()
        self._props = None

    def reset_monitor(self):
        self._monitor.reset()
        

    def collect_prop_rob(self,top_node = None, prop_list = None):
        """ 
        Collects the propositions (abstractions) recursively to pass them to the monitor wrapper of the repairer.
        If a sub-formula is encapsulated by an LTL/STL indicator, it constitutes a proposition.
        If negations exist, the non-negated formula that follows the negation is considered.
        If a predicate is not encapsulated by an LTL/STL indicator, it constitutes a proposition alone.
        Formulas may contain only: Implications, Con/Disjunctions, Negations, LTL/STL indicators.
        The values are obtained directly from the Rtamt.

        Returns:
        None. Acts directly on the dict that was passed as an argument: dict{proposition, robustness_value}
        """
        if top_node is None:
            top_node = self._monitor.top
        if isinstance(top_node, UnaryNode):
            if isinstance(top_node, Neg):
                self.collect_prop_rob(top_node.children[0], prop_list)
            if isinstance(top_node, TimedOnce) or isinstance(top_node, Previous):
                prop_list[top_node.name] = self._monitor.online_evaluator.evaluate(top_node, [])
        elif isinstance(top_node, Predicate):
            prop_list[top_node.name] = self._monitor.online_evaluator.evaluate(top_node, [])
        else:
            if isinstance(top_node, Implies):
                if isinstance(top_node.children[0], Predicate):
                    prop_list[top_node.children[0].name] = self._monitor.online_evaluator.evaluate(top_node.children[0], [])
                    self.collect_prop_rob(top_node.children[1], prop_list)
                elif isinstance(top_node.children[1], Predicate):
                    prop_list[top_node.children[1].name] = self._monitor.online_evaluator.evaluate(top_node.children[1], [])
                    self.collect_prop_rob(top_node.children[0], prop_list)
            if isinstance(top_node, Conjunction) or isinstance(top_node, Disjunction):
                self.collect_prop_rob(top_node.children[0], prop_list)
                self.collect_prop_rob(top_node.children[1], prop_list)
         
    def evaluate_monitor_online(self, time_step: int, predicates: List[Tuple[str, float]]):
        time = time_step * self.dt * 1000.0
        top_node = self._monitor.top
        prop_list = {}
        rob = self._monitor.update(time, predicates)
        self.collect_prop_rob(top_node, prop_list)
        self._props = prop_list
        return rob
    
    def copy(self):
        return RtamtStlMonitor(self._rule, self._predicates, self.dt, self._output_type)

    def reset(self):
        self._monitor.reset()
