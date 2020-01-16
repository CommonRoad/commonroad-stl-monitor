from typing import List, Dict, Tuple
import mtl


class TrafficRuleMonitor:
    """
    Represents single formalized traffic rule
    """
    def __init__(self, logic_formula: Tuple[str, str], vehicle_dependency: bool):
        """
        :param logic_formula: temporal logic formula
        :param vehicle_dependency: boolean indicating if rule must be evaluated with respect to several vehicles
        """
        self._name = logic_formula[0]
        self._logic_formula = logic_formula[1]
        self._monitor = mtl.parse(logic_formula[1])
        self._predicates = self._extract_predicates(logic_formula[1])
        self._vehicle_dependency = vehicle_dependency

    @property
    def name(self) -> str:
        return self._name

    @property
    def predicates(self) -> List[str]:
        return self._predicates

    @property
    def vehicle_dependency(self) -> bool:
        return self._vehicle_dependency

    @staticmethod
    def _extract_predicates(logic_formula: str) -> List[str]:
        """
        Extracts all predicates from temporal logic formula given as string

        :param logic_formula: temporal logic formula
        :returns list of predicates
        """
        replacements = ['U', 'X', 'G', '&', '->', '(', ')', '~', '|']
        for el in replacements:
            logic_formula = logic_formula.replace(el, "")
        predicates_tmp = list(logic_formula.split(" "))
        predicates = [x for x in predicates_tmp if x != ""]
        return predicates

    def evaluate_monitor(self, predicates: Dict[str, List[Tuple[float, bool]]]) -> bool:
        """
        Evaluates monitor with provided trace of predicates

        :param predicates: trace for each predicate used in rule
        :returns boolean indicating if rule is fulfilled
        """
        return self._monitor(predicates, quantitative=False)
