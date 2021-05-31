import inspect
import logging
import re
import sys
from enum import auto, Enum

logger = logging.getLogger(__name__)


def get_all_predicate_evaluators():
    mod_name = "crmonitor.predicates.predicate"
    # noinspection PyUnresolvedReferences
    import crmonitor.predicates.predicate
    classes = inspect.getmembers(sys.modules[mod_name], inspect.isclass)
    classes = list(filter(lambda p: "Pred" in p[0], classes))
    d = {}
    for name, cls in classes:
        d[cls.predicate_name] = cls
    return d


class IOType(Enum):
    OUTPUT = auto()
    INPUT = auto()


class QuantificationType(Enum):
    ALL = auto()
    EXISTENTIAL = auto()


class Rule:
    full_predicate_pattern = re.compile(r"(?P<pred>((?P<pred_name>[a-z]+(?:_[a-z]+)*?)(?P<io_type>_i)?_(?P<agents>(_a(\d)+)+)))")
    rule_pattern = re.compile(r"^(?P<quant>[AE])\s(?P<rule>.*)")

    class PredicateAssignment:
        def __init__(self, full_name, agent_placeholders, evaluator,
                     io_type=IOType.OUTPUT):
            assert len(
                agent_placeholders) == evaluator.arity, f"The arity of the evaluator for {full_name} should be {len(agent_placeholders)}, but is {evaluator.arity}!"
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

    def __init__(self, full_rule_str, config, name=None):
        rule_str, config, name, num_dependent_vehicles, quantification, \
            predicate_assignment = self._from_string(full_rule_str, config, name)
        self._rule_str = rule_str
        self.config = config
        self.predicate_assignment = predicate_assignment
        self.num_dependent_vehicles = num_dependent_vehicles
        self.quantification = quantification
        self.name = name

    @property
    def is_vehicle_dependent(self):
        return self.num_dependent_vehicles > 0

    @property
    def predicate_names(self):
        return sorted([pred.full_name for pred in self.predicate_assignment])

    @classmethod
    def _from_string(cls, full_rule_str, config, name=None):
        required_predicates = set()
        agent_placeholders = set()
        predicate_assignment = set()
        match = Rule.rule_pattern.match(full_rule_str)

        assert match is not None, f"Could not find quantification type for rule {full_rule_str}!"
        if match.group("quant") == "E":
            quantification = QuantificationType.EXISTENTIAL
        else:
            quantification = QuantificationType.ALL
        rule_str = match.group("rule")
        pred_matches = Rule.full_predicate_pattern.finditer(rule_str)
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
            p = Rule.PredicateAssignment(
                m.group("pred_name") + "_" + m.group("agents"),
                predicate_agent_placeholders, evaluator(config["traffic_rules_param"]), io_type)
            rule_str = rule_str.replace(m.group(0), p.full_name)
            predicate_assignment.add(p)

        # Check increasing order
        a_ids = sorted(list(agent_placeholders))
        for i, aid in enumerate(a_ids):
            assert i == aid, f"Agent place holder IDs are not in increasing order. Missing {i}!"
        # List is sorted. Last item is largest.
        num_dependent_vehicles = a_ids[-1]
        if name is None:
            name = full_rule_str
        return rule_str, config, name, num_dependent_vehicles, quantification, predicate_assignment
