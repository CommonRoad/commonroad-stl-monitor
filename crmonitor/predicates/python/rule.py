import inspect
import sys
import re


def get_all_predicate_evaluators():
    mod_name = "crmonitor.predicates.python.predicate"
    classes = inspect.getmembers(sys.modules[mod_name], inspect.isclass)
    classes = list(filter(lambda p: "Pred" in p[0], classes))
    d = {}
    for name, cls in classes:
        d[name] = cls
    return d


class Rule:
    full_predicate_pattern = r"(\s|\A)(?P<pred>[a-z]+(?:_[a-z]+)*)_(?P<agents>(_a(\d)+)+)(\s|\Z)"

    class PredicateAssignment:
        def __init__(self, pred_str, agent_placeholders, evaluator):
            self.pred_str = pred_str
            self.agent_placeholders = agent_placeholders
            self.evaluator = evaluator

        def __eq__(self, o) -> bool:
            return self.pred_str == o.pred_str and self.agent_placeholders == o.agent_placeholders

    def __init__(self, rule_str, config):
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
        return [pred.pred_str for pred in self.predicate_assignment]

    def _extract_predicates(self):
        required_predicates = set()
        agent_placeholders = set()
        pred_matches = re.finditer(Rule.full_predicate_pattern, self._rule_str)
        pred_evaluators = get_all_predicate_evaluators()
        for m in pred_matches:
            pred_basename = m.group('pred')
            required_predicates.add(pred_basename)
            agent_string = m.group('agents')
            agents = re.split(r"_a", agent_string)
            predicate_agent_placeholders = []
            for a in agents:
                if a != '':
                    predicate_agent_placeholders.append(int(a))
                    agent_placeholders.add(int(a))
            evaluator = pred_evaluators[pred_basename]
            assert evaluator is not None
            p = Rule.PredicateAssignment(m.group(0),
                                         predicate_agent_placeholders,
                                         evaluator(self.config))
            self.predicate_assignment.add(p)

        # Check increasing order
        a_ids = sorted(list(agent_placeholders))
        for i, aid in enumerate(a_ids):
            assert i == aid, f"Agent place holder IDs are not in increasing order. Missing {i}!"
        # List is sorted. Last item is largest.
        self.num_dependent_vehicles = a_ids[-1]


