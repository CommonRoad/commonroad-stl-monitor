from typing import List, Tuple, Union, Dict

import rtamt

from crmonitor.common.evaluation import PredicateValueCollection
from crmonitor.predicates.python.rule import Rule


class TrafficRuleMonitorForwardSTL:
    """
    Represents single formalized traffic rule
    """

    def __init__(self, rule: Rule,
                 output_type='standard'):  # , predicate_references):

        self._rule = rule
        self._output_type = output_type
        self._monitor = self.construct_monitor()

    def reset_monitor(self):
        self._monitor.reset()

    # TODO: Could be made static
    def _reconstruct_logic_formula(self):
        logic_formula = self._rule._rule_str
        replacements = {"~": "not"}
        for el in replacements.keys():
            logic_formula = logic_formula.replace(el, replacements[el])
        # Workaround for rtamt when working with output-robustness and input vacuity
        predicates = self._rule.predicate_names
        mod_formula = logic_formula
        for pred in predicates:
            mod_formula.replace(pred, "({} >= 0)".format(pred))
        return mod_formula

    # TODO: Could be made static
    def construct_monitor(self) -> rtamt.STLSpecification:
        logic_formula = self._reconstruct_logic_formula()
        monitor = rtamt.STLIOSpecification(0)  # 0: cpp; 1: Python
        monitor.name = "HandMadeMonitor"  # self.name
        for var in self._rule.predicate_names:
            monitor.declare_var(var, "float")
            if var.split("_")[-1] == "i":
                monitor.set_var_io_type(var, "input")
            else:
                monitor.set_var_io_type(var, "output")
        monitor.declare_var("out", "float")

        monitor.iosem = self._output_type

        monitor.spec = f"out = {logic_formula}"
        monitor.parse()

        return monitor

    def _prepare_predicates_online(self, predicates: PredicateValueCollection):
        res = []
        for pred_name in self._rule.predicate_names:
            pred = predicates.by_name(pred_name)
            res.append((pred.predicate_str, pred.value))
        return res

    def evaluate_monitor_offline(self, predicates: Dict[
        str, List[Tuple[float, bool]]], vehicle_ids) -> Union[bool, float]:
        pass

    def evaluate_monitor_online(self, predicates: PredicateValueCollection):
        time = predicates._predicate_values[0].time_step
        predicate_values = self._prepare_predicates_online(predicates)
        rob = self._monitor.update(time, predicate_values)
        return rob

    def evaluate_monitor_offline_stepwise(self,
                                          predicates: PredicateValueCollection):
        self.reset_monitor()
        rob_series = []
        for t in predicates.get_time_steps():
            time_pred = predicates.by_time_step(t)
            rob = self.evaluate_monitor_online(time_pred)
            rob_series.append((t, rob))
        return rob_series
