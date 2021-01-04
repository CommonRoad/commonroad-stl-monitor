import inspect
import sys
import re

import enum


def get_all_predicate_evaluators():
    mod_name = "crmonitor.predicates.python.predicate"
    import crmonitor.predicates.python.predicate
    classes = inspect.getmembers(sys.modules[mod_name], inspect.isclass)
    classes = list(filter(lambda p: "Pred" in p[0], classes))
    d = {}
    for name, cls in classes:
        d[cls.predicate_name] = cls
    return d


class IOType(enum.Enum):
    OUTPUT = enum.auto()
    INPUT = enum.auto()


class Rule:
    full_predicate_pattern = r"(?P<pred>((?P<pred_name>[a-z]+(?:_[a-z]+)*?)(?P<io_type>_i)?_(?P<agents>(_a(\d)+)+)))"

    class PredicateAssignment:
        def __init__(self, full_name, agent_placeholders, evaluator,
                     io_type=IOType.OUTPUT):
            self.full_name = full_name
            self.agent_placeholders = tuple(agent_placeholders)
            self.evaluator = evaluator
            self.io_type = io_type

        @property
        def base_name(self):
            return self.evaluator.predicate_name

        @property
        def num_dependencies(self):
            return len(self.agent_placeholders)

        def __eq__(self, o) -> bool:
            return self.full_name == o.full_name and self.agent_placeholders == o.agent_placeholders

        def __hash__(self) -> int:
            return hash((self.full_name, self.agent_placeholders))

    def __init__(self, rule_str, config=None):
        self._rule_str = rule_str
        self.config = config
        self.predicate_assignment = set()
        self.num_dependent_vehicles: int
        self._extract_predicates()

    @property
    def is_vehicle_dependent(self):
        return self.num_dependent_vehicles > 0

    @property
    def predicate_names(self):
        return [pred.full_name for pred in self.predicate_assignment]

    def _extract_predicates(self):
        required_predicates = set()
        agent_placeholders = set()
        pred_matches = re.finditer(Rule.full_predicate_pattern, self._rule_str)
        pred_evaluators = get_all_predicate_evaluators()
        for m in pred_matches:
            pred_basename = m.group('pred_name')
            required_predicates.add(pred_basename)
            agent_string = m.group('agents')
            agents = re.split(r"_a", agent_string)
            predicate_agent_placeholders = []
            for a in agents:
                if a != '':
                    predicate_agent_placeholders.append(int(a))
                    agent_placeholders.add(int(a))
            evaluator = pred_evaluators[pred_basename]
            if m.group('io_type') is None:
                io_type = IOType.OUTPUT
            else:
                io_type = IOType.INPUT
            assert evaluator is not None
            p = Rule.PredicateAssignment(m.group("pred_name") + "_" + m.group("agents"),
                                         predicate_agent_placeholders,
                                         evaluator(self.config), io_type)
            self._rule_str = self._rule_str.replace(m.group(0), p.full_name)
            self.predicate_assignment.add(p)

        # Check increasing order
        a_ids = sorted(list(agent_placeholders))
        for i, aid in enumerate(a_ids):
            assert i == aid, f"Agent place holder IDs are not in increasing order. Missing {i}!"
        # List is sorted. Last item is largest.
        self.num_dependent_vehicles = a_ids[-1]
