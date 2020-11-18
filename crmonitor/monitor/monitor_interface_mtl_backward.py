from typing import Tuple, Set
from monitors import mtl


class TrafficRuleMonitorBackward:
    """
    Represents single formalized traffic rule
    """

    def __init__(
        self,
        logic_formula: Tuple[str, str],
        vehicle_dependency: bool,
        predicate_references,
    ):
        """
        :param logic_formula: temporal logic formula
        :param vehicle_dependency: boolean indicating if rule must be evaluated with respect to several vehicles
        """
        self._name = logic_formula[0]
        self._logic_formula = logic_formula[1]
        self._predicate_names = self._extract_predicate_names(logic_formula[1])
        self._predicate_functions = self._extract_predicate_functions(
            predicate_references
        )
        self._monitor = mtl.monitor(logic_formula[1], **self._predicate_functions)
        self._vehicle_dependency = vehicle_dependency
        self._safety_monitor_init = self._monitor.states.copy()

    @property
    def name(self) -> str:
        return self._name

    @property
    def predicates(self) -> Set[str]:
        return self._predicate_names

    def reset_monitor(self):
        self._monitor.states = self._safety_monitor_init.copy()

    @property
    def vehicle_dependency(self) -> bool:
        return self._vehicle_dependency

    def _extract_predicate_functions(self, predicate_references):
        """
        Extracts all predicate functions from predicate collections

        :param predicate_references: list of predicate collections
        :returns list of predicates
        """
        predicate_functions = {}
        for name in self._predicate_names:
            for collection in predicate_references:
                if hasattr(collection, name):
                    predicate_functions[name] = collection.__getattribute__(name)
        return predicate_functions

    @staticmethod
    def _extract_predicate_names(logic_formula: str) -> Set[str]:
        """
        Extracts all predicates from temporal logic formula given as string

        :param logic_formula: temporal logic formula
        :returns list of predicates
        """
        replacements = [
            "always",
            "since",
            "once",
            "pre(",
            "&&",
            "->",
            "(",
            ")",
            "!",
            "||",
            "ego_vehicle",
            "other_vehicles",
            "time_step",
            "other_vehicle",
            ",",
            "[",
            "]",
            "150",
            "0",
        ]  # , 'operating_mode']
        for el in replacements:
            logic_formula = logic_formula.replace(el, "")
        predicates_tmp = list(logic_formula.split(" "))
        predicates = [x for x in predicates_tmp if x != ""]
        return set(predicates)

    def evaluate_monitor(self, predicates) -> bool:
        """
        Evaluates monitor with provided trace of predicates

        :param predicates: trace for each predicate used in rule
        :returns boolean indicating if rule is fulfilled
        """
        output = self._monitor.update(**predicates)
        return output
