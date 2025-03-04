import re
from typing import Dict, List, Optional

from antlr4.BufferedTokenStream import BufferedTokenStream
from antlr4.TokenStreamRewriter import TokenStreamRewriter

from crmonitor.common.config import get_traffic_rule_config
from crmonitor.rule.fastl.FaStlParser import FaStlParser
from crmonitor.rule.fastl.FaStlParserVisitor import FaStlParserVisitor
from crmonitor.rule.rule_node import IOType


class ReplacementRule:
    def __init__(
        self,
        rule: str,
        quantified_agents_placeholders: List[str],
        io_type_placeholder: Optional[str] = None,
    ) -> None:
        self._rule = rule
        self._quantified_agents_placeholders = quantified_agents_placeholders
        self._io_type_placeholder = io_type_placeholder

    @classmethod
    def from_str(cls, signature: str, rule: str) -> "ReplacementRule":
        argument_string = re.search(r"\(([^)]+)\)(_.*)?", signature)
        if argument_string is None:
            raise RuntimeError()

        quantified_agent_placeholders = [
            placeholder.strip() for placeholder in argument_string.group(1).split(",")
        ]
        if argument_string.group(2) is not None:
            io_type_placeholder = argument_string.group(2)
        else:
            io_type_placeholder = None

        return cls(rule, quantified_agent_placeholders, io_type_placeholder)

    def get_interpolated_rule(self, quantified_agents: List[str], io_type: IOType) -> str:
        if len(quantified_agents) != len(self._quantified_agents_placeholders):
            raise RuntimeError()

        replacement = self._rule
        for placeholder, quantified_vehicle in zip(
            self._quantified_agents_placeholders, quantified_agents
        ):
            replacement = replacement.replace(placeholder, quantified_vehicle)

        if self._io_type_placeholder:
            if io_type == IOType.INPUT:
                replacement = replacement.replace(self._io_type_placeholder, "_i")
            else:
                replacement = replacement.replace(self._io_type_placeholder, "")

        return replacement


class ReplacementTable:
    def __init__(self, replacement_table: Dict[str, ReplacementRule]) -> None:
        self._replacement_table = replacement_table

    @classmethod
    def from_dict(cls, _dict: dict) -> "ReplacementTable":
        table = {}
        for meta_predicate_signature, rule in _dict.items():
            meta_predicate_name = meta_predicate_signature.split("(")[0]
            table[meta_predicate_name] = ReplacementRule.from_str(meta_predicate_signature, rule)

        return cls(table)

    def get_replacement_rule(self, meta_predicate: str) -> ReplacementRule:
        return self._replacement_table[meta_predicate]


class MetaPredicateReplacementVisitor(FaStlParserVisitor):
    """
    A visitor for traffic rules, which replaces meta rules of the from `$meta_predicate_name(a0, a1)` with their replacement rule.
    Performs only a simple string replacement and does not interpolate the arguments.
    """

    # The program name is used to uniquely identify our token stream.
    # It's value does not really matter, because we only apply one kind of rewrite.
    DEFAULT_PROGRAM_NAME = "meta-predicates"

    def __init__(
        self,
        tokens: BufferedTokenStream,
        replacement_table: Optional[ReplacementTable] = None,
    ) -> None:
        """
        Initialze a new `MetaPredicateReplacementVisitor` and optionally load the a replacement table.

        :param tokens: The token stream as produced by an ANTLR lexer.
        :param replacement_table: An optional replacement table, with meta predicates as keys and their replacement rule as values. If None, the default table will be loaded from `traffic_rules_rtamt.yaml`.
        """
        self._rewriter: TokenStreamRewriter = TokenStreamRewriter(tokens)
        if replacement_table is None:
            self._replacement_table = ReplacementTable.from_dict(
                get_traffic_rule_config()["meta_predicates"]
            )
        else:
            self._replacement_table = replacement_table

        self._used_quantifiers = []

    def visit(self, tree) -> str:
        """
        A custom visit method, that returns the rule source code with all meta predicate replaced.
        """
        tree.accept(self)

        modified_source_code = self._rewriter.getText(
            self.DEFAULT_PROGRAM_NAME, 0, len(self._rewriter.tokens.tokens) - 1
        )
        return modified_source_code

    # def visitSpecQuantSumIfPositive(self, ctx: FaStlParser.SpecQuantSumIfPositiveContext):
    #     self._used_quantifiers.append(ctx.vehicle().getText())

    # def visitSpecQuantExist(self, ctx: FaStlParser.SpecQuantExistContext):
    #     self._used_quantifiers.append(ctx.vehicle().getText())

    # def visitSpecQuantForall(self, ctx: FaStlParser.SpecQuantForallContext):
    #     self._used_quantifiers.append(ctx.vehicle().getText())

    def visitPredicate(self, ctx: FaStlParser.PredicateContext) -> None:
        """
        Visit a predicate node and perform a replacement if the predicate is a meta-predicate.
        """
        pred_name = ctx.Identifier().getText()
        if not pred_name.startswith("$"):
            return

        replacement_rule = self._replacement_table.get_replacement_rule(pred_name)
        if replacement_rule is None:
            raise RuntimeError(
                f"Failed to replace meta predicate '{pred_name}': No replacement rule found!"
            )

        if ctx.IO_TYPE_INPUT() is not None:
            io_type = IOType.INPUT
        else:
            io_type = IOType.OUTPUT

        quantified_vehicles = []
        i = 0
        while ctx.vehicle(i) is not None:
            quantified_vehicles.append(ctx.vehicle(i).getText())
            i += 1

        rule_str = replacement_rule.get_interpolated_rule(quantified_vehicles, io_type)

        # The replacement might mess up precedence, so we make sure this does not happen.
        rule_str = f"({rule_str})"

        self._rewriter.replace(
            self.DEFAULT_PROGRAM_NAME,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            rule_str,
        )

        return
