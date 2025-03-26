from typing import Dict, Optional

from antlr4 import CommonTokenStream
from antlr4.InputStream import InputStream

from crmonitor.common.config import get_traffic_rule_config
from crmonitor.rule.fastl.FaStlLexer import FaStlLexer
from crmonitor.rule.fastl.FaStlParser import FaStlParser
from crmonitor.rule.meta_predicate_replacement_visitor import (
    MetaPredicateLookupTable,
    MetaPredicateReplacementVisitor,
)
from crmonitor.rule.parse_tree_visitor import TrafficRuleParseTreeVisitor
from crmonitor.rule.rule_node import (
    VisitorNode,
)
from crmonitor.rule.rule_parser_interface import RuleParserInterface


class RuleParser(RuleParserInterface):
    def __init__(self, meta_predicates: Optional[Dict[str, str]] = None):
        if meta_predicates is None:
            self._meta_predicates = get_traffic_rule_config()["meta_predicates"]
        else:
            self._meta_predicates = meta_predicates

        self._meta_predicate_lookup_table = MetaPredicateLookupTable.from_dict(
            self._meta_predicates
        )

        self._sub_rule_counter = 0

    def _new_unique_sub_rule_name(self) -> str:
        return f"g{self._sub_rule_counter}"

    def parse(self, rule: str, name: Optional[str] = None) -> VisitorNode:
        stream = InputStream(rule)
        lexer = FaStlLexer(stream)
        stream = CommonTokenStream(lexer)
        parser = FaStlParser(stream)
        tree = parser.compile_unit()
        visitor = TrafficRuleParseTreeVisitor(stream, self._new_unique_sub_rule_name)
        rule_node_tree = visitor.visit(tree)[0]
        if name is not None:
            rule_node_tree.name = name

        visitor = MetaPredicateReplacementVisitor(self._meta_predicate_lookup_table, self)
        rule_node_tree = visitor.visit(rule_node_tree)
        return rule_node_tree
