from typing import Optional, Protocol

from crmonitor.rule.rule_node import VisitorNode


# Forward definition of parser interface to avoid cyclic dependency between real RuleParser and MetaPredicateReplacementVisitor
class RuleParserInterface(Protocol):
    def parse(self, rule: str, name: Optional[str] = None) -> VisitorNode: ...
