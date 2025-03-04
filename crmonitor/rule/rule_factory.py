from typing import Optional

from antlr4 import CommonTokenStream
from antlr4.InputStream import InputStream

from crmonitor.predicates.predicate_factory import PredicateFactory
from crmonitor.rule.fastl.FaStlLexer import FaStlLexer
from crmonitor.rule.fastl.FaStlParser import FaStlParser
from crmonitor.rule.meta_predicate_replacement_visitor import (
    MetaPredicateReplacementVisitor,
)
from crmonitor.rule.parse_tree_visitor import TrafficRuleParseTreeVisitor
from crmonitor.rule.rule_node import VisitorNode


class RuleFactory:
    def __init__(self, predicate_factory: Optional[PredicateFactory] = None):
        self._predicate_factory = predicate_factory or PredicateFactory()

    def _parse_rule_str_to_stream_and_tree(self, rule_str: str):
        stream = InputStream(rule_str)
        lexer = FaStlLexer(stream)
        stream = CommonTokenStream(lexer)
        parser = FaStlParser(stream)
        tree = parser.compile_unit()
        return stream, tree

    def _meta_predicate_replacement_pass(self, rule_str: str) -> str:
        stream, tree = self._parse_rule_str_to_stream_and_tree(rule_str)
        visitor = MetaPredicateReplacementVisitor(stream)
        return visitor.visit(tree)

    def _traffic_rule_parse_pass(self, rule_str: str) -> VisitorNode:
        stream, tree = self._parse_rule_str_to_stream_and_tree(rule_str)
        visitor = TrafficRuleParseTreeVisitor(stream)
        return visitor.visit(tree)[0]

    def parse_rule(self, full_rule_str, name=None) -> VisitorNode:
        if name is None:
            name = full_rule_str
        modified_rule_str = self._meta_predicate_replacement_pass(full_rule_str)
        node = self._traffic_rule_parse_pass(modified_rule_str)
        node.name = name
        return node
