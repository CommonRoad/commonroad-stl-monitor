from collections import defaultdict
from typing import Tuple, Dict, List

from monitors import mtl

from crmonitor.predicates.rule import Rule


class TrafficRuleMonitorBackward:
    """
    Represents single formalized traffic rule
    """

    def __init__(self, rule: Rule):
        """
        :param logic_formula: temporal logic formula
        :param vehicle_dependency: boolean indicating if rule must be evaluated with respect to several vehicles
        """
        self.rule = rule
        self._monitor = mtl.monitor(self._reconstruct_logic_formula())
        self._safety_monitor_init = self._monitor.states.copy()

    def reset_monitor(self):
        self._monitor.states = self._safety_monitor_init.copy()

    def _reconstruct_logic_formula(self):
        rule_str = self.rule._rule_str
        replacements = [("implies", "->"), ("and", "&&"), ("prev", "pre")]
        for old, new in replacements:
            rule_str = rule_str.replace(old, new)
        return rule_str

    def evaluate_monitor_offline(self, predicates: Dict[
        float, List[Tuple[str, bool]]]) -> bool:
        """
        Evaluates monitor with provided trace of predicates

        :param predicates: trace for each predicate used in rule
        :returns boolean indicating if rule is fulfilled
        """
        self.reset_monitor()
        # pivot predicates
        pivot_pred = defaultdict(list)
        for t_values in predicates.values():
            d = dict(t_values)
            for p in self.rule.predicate_names:
                pivot_pred[p].append(d[p])
        output = self._monitor.update(**pivot_pred)
        return output

    def evaluate_monitor_offline_stepwise(self, predicates):
        self.reset_monitor()
        rob_series = []
        for t, t_values in predicates.items():
            rob = self._monitor.update(**dict(t_values))
            rob_series.append(rob)
        return rob_series
