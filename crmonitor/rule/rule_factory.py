import re
from typing import Optional

from crmonitor.predicates.predicate_factory import PredicateFactory
from crmonitor.rule.rule_node import AllNode, ExistNode, IOType, PredicateNode, RuleNode


class RuleFactory:
    full_predicate_pattern = re.compile(
        r"(?P<pred>((?P<pred_name>[a-z]+(?:_[a-z]+)*?)(?P<io_type>_i)"
        r"?_(?P<agents>(_a(\d)+)+)))"
    )
    quantification_pattern = re.compile(
        r"^(?P<quant>[AE])\sa(?P<veh_id>\d+):\s\((?P<rule>.*)\)$"
    )
    subrule_pattern = re.compile(r"[AE]\sa\d+:\s\(.*\)")

    def __init__(self, predicate_factory: Optional[PredicateFactory] = None):
        self._predicate_factory = predicate_factory or PredicateFactory()

    def parse_rule(self, full_rule_str, name=None):
        if name is None:
            name = full_rule_str
        m = self.quantification_pattern.match(full_rule_str)
        if m is not None:
            # Quantification on top level
            sub_rule_str = m["rule"]
            children = self.parse_rule(sub_rule_str, "g0")
            quantified_vehicle = int(m.group("veh_id"))
            if m.group("quant") == "E":
                node = ExistNode([children], quantified_vehicle, name)
            elif m.group("quant") == "A":
                node = AllNode([children], quantified_vehicle, name)
            else:
                raise ValueError()
        else:
            mod_rule_str = full_rule_str
            predicate_assignment = set()
            sub_rules = []
            m = self.subrule_pattern.search(mod_rule_str)
            while m is not None:
                mod_rule_str = (
                    mod_rule_str[: m.start()]
                    + f"g{len(sub_rules)}"
                    + mod_rule_str[m.end() :]
                )
                sub_rule_str = m[0]
                sub_rules.append(self.parse_rule(sub_rule_str, f"g{len(sub_rules)}"))
                m = self.subrule_pattern.match(mod_rule_str)

            pred_matches = self.full_predicate_pattern.finditer(mod_rule_str)
            for m in pred_matches:
                pred_basename = m.group("pred_name")
                agent_string = m.group("agents")
                agents = re.split(r"_a", agent_string)
                predicate_agent_placeholders = []
                for a in agents:
                    if a != "":
                        predicate_agent_placeholders.append(int(a))
                if m.group("io_type") is None:
                    io_type = IOType.OUTPUT
                    full_name = m.group("pred_name") + "_" + m.group("agents")
                else:
                    io_type = IOType.INPUT
                    full_name = m.group("pred_name") + "_" + m.group("agents") + "_i"
                predicate_evaluator = self._predicate_factory.get_predicate(
                    pred_basename
                )
                p = PredicateNode(
                    full_name,
                    predicate_agent_placeholders,
                    predicate_evaluator,
                    io_type,
                )
                mod_rule_str = mod_rule_str.replace(m.group(0), p.name)
                predicate_assignment.add(p)
            node = RuleNode(sub_rules + list(predicate_assignment), mod_rule_str, name)
        return node
