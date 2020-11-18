import rtamt
from typing import List, Tuple, Set, Union, Dict

from crmonitor.common.helper import OperatingMode

class TrafficRuleMonitorForwardSTL:
    """
    Represents single formalized traffic rule
    """
    def __init__(self, logic_formula: Tuple[str, str], vehicle_dependency: bool): #, predicate_references):
        """
        :param logic_formula: temporal logic formula
        :param vehicle_dependency: boolean indicating if rule must be evaluated with respect to several vehicles
        """
        self._name = logic_formula[0]
        self._logic_formula = self._reconstruct_logic_formula(logic_formula[1])
        self._predicate_names = self._extract_predicate_names(self._logic_formula)
        # self._predicate_functions = self._extract_predicate_functions(predicate_references)
        self._monitor = {} # self.construct_monitor(logic_formula[1])
        self._vehicle_dependency = vehicle_dependency

    @property
    def name(self) -> str:
        return self._name

    @property
    def predicates(self) -> Set[str]:
        return self._predicate_names

    @property
    def vehicle_dependency(self) -> bool:
        return self._vehicle_dependency

    def reset_monitor(self):
        pass

    @staticmethod
    def _reconstruct_logic_formula(logic_formula: str) -> str:
        replacements = {'~': 'not'}
        for el in replacements.keys():
            logic_formula = logic_formula.replace(el, replacements[el])
        return logic_formula

    @staticmethod
    def _extract_predicate_names(logic_formula: str) -> Set[str]:
        """
        Extracts all predicates from temporal logic formula given as string

        :param logic_formula: temporal logic formula
        :returns list of predicates
        """
        # https://github.com/nickovic/rtamt/blob/master/rtamt/parser/stl/StlParser.tokens
        replacements = ['U', 'X', 'G', 'F', '&', '->', '(', ')', '~', '|', '[', ']',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.',
                        ' and', 'or', 'not', 'always', 'eventually', 'historically',
                        'comp', 'iff', 'implies', 'once', 'precedes', 'since', 'xor',
                        'until', 'prev', '!', '-', 'next']
        for el in replacements:
            logic_formula = logic_formula.replace(el, "")
        predicates_tmp = list(logic_formula.split(" "))
        predicates = [x for x in predicates_tmp if x != ""]
        return set(predicates)

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

    def construct_monitor(self, logic_formula: str) -> rtamt.STLSpecification:
        monitor = rtamt.STLSpecification(0) # 0: cpp; 1: Python
        monitor.name = "HandMadeMonitor" #self.name
        for var in self.predicates:
            monitor.declare_var(var, "float")
        monitor.declare_var("out", "float")
        monitor.spec = f"out = {logic_formula}"
        monitor.parse()

        return monitor

    def evaluate_monitor(self, predicates: Dict[str, List[Tuple[float, bool]]],
                         operating_mode: OperatingMode) -> Union[bool, float]:
        """
        Evaluates monitor with provided predicate trace (for all time steps)
        :param predicates:
        :param operating_mode:
        :return:
        """
        # TODO: offline and online evaluation of whole trace
        # offline
        # create monitor for vehicle (iterate over all vehicles in the traffic rule dispatcher)
        self._monitor = self.construct_monitor(self._logic_formula)

        robustness = self._monitor.offline(predicates)
        # online
        self._monitor = self.construct_monitor(self._logic_formula)
        time_steps = len(list(predicates.values())[0])
        rob = []
        for i in range(time_steps):
            inputs = [(key, value[i][1]) for key, value in predicates.items()]
            time = list(predicates.values())[0][i][0]
            rob.append(self._monitor.update(time, inputs))

        return robustness



    def evaluate_monitor_online(self, predicates: List[Union[float, Tuple[str, Union[float, bool]]]], vehicle_id: int):
        """
        Evaluates monitor with provided current time step of predicates

        :param predicates: current time step for each predicate used in rule
        :returns boolean indicating if rule is fulfilled
        """
        if vehicle_id not in self._monitor:
            # this vehicle appears for the first time, no monitor yet, construct a new monitor
            self._monitor[vehicle_id] = self.construct_monitor(self._logic_formula)
        rob = self._monitor[vehicle_id].update(predicates[0], predicates[1:])
        return rob
