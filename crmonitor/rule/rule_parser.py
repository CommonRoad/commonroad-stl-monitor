from typing import Optional

from antlr4 import CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener
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
    RuleAstNode,
)
from crmonitor.rule.rule_parser_context import RuleParserContext
from crmonitor.rule.rule_parser_interface import RuleParserInterface


class RuleParseError(Exception):
    def __init__(
        self, rule: str, reason: str, line: int, column: int, name: Optional[str] = None
    ) -> None:
        if name is not None:
            rule_description = f"{name} ({rule})"
        else:
            rule_description = rule
        super().__init__(f"Syntax error in {rule_description} at {line}:{column}: {reason}")


class PropagatingErrorListener(ErrorListener):
    def __init__(self, rule: str, rule_name: Optional[str] = None):
        super().__init__()
        self._rule = rule
        self._rule_name = rule_name

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise RuleParseError(self._rule, msg, line, column, self._rule_name)


class RuleParser(RuleParserInterface):
    def __init__(self, meta_predicates: dict[str, str] | None = None):
        if meta_predicates is None:
            self._meta_predicates = get_traffic_rule_config()["meta_predicates"]
        else:
            self._meta_predicates = meta_predicates

        self._meta_predicate_lookup_table = MetaPredicateLookupTable.from_dict(
            self._meta_predicates
        )
        self._parser_context = RuleParserContext()

    def parse(
        self, rule: str, name: str | None = None, replace_meta_predicates: bool = True
    ) -> RuleAstNode:
        stream = InputStream(rule)
        lexer = FaStlLexer(stream)
        error_listener = PropagatingErrorListener(rule, name)
        lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)

        stream = CommonTokenStream(lexer)
        parser = FaStlParser(stream)

        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)

        tree = parser.compile_unit()

        visitor = TrafficRuleParseTreeVisitor(stream, self._parser_context)
        rule_node_tree = visitor.visit(tree)[0]
        if name is not None:
            rule_node_tree.name = name

        if replace_meta_predicates:
            visitor = MetaPredicateReplacementVisitor(self._meta_predicate_lookup_table, self)
            rule_node_tree = visitor.visit(rule_node_tree)
        return rule_node_tree
