from typing import Optional, Protocol

from crmonitor.rule.rule_node import RuleAstNode


# Forward definition of parser interface to avoid cyclic dependency between real RuleParser and MetaPredicateReplacementVisitor
class RuleParserInterface(Protocol):
    def parse(
        self, rule: str, name: Optional[str] = None, replace_meta_predicates: bool = True
    ) -> RuleAstNode: ...
