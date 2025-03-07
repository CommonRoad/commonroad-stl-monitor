import re
from typing import Dict, List, Optional

from antlr4.BufferedTokenStream import BufferedTokenStream
from antlr4.TokenStreamRewriter import TokenStreamRewriter

from crmonitor.common.config import get_traffic_rule_config
from crmonitor.rule.fastl.FaStlParser import FaStlParser
from crmonitor.rule.fastl.FaStlParserVisitor import FaStlParserVisitor
from crmonitor.rule.rule_node import IOType


class ReplacementRule:
    """
    Defines a rule that replaces meta-predicates with concrete expressions in traffic rules.

    This class handles both agent placeholders (variables that represent different agents in the rule)
    and input/output type placeholders, ensuring that the generated rule correctly interpolates these values.
    """

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
        """
        Parses a meta-predicate signature and extracts placeholders for agents and IO types.

        This function is crucial because meta-predicates follow a structured naming convention
        where agent variables and IO types are embedded in the predicate name.

        :param signature: The meta-predicate signature, e.g., "$meta_predicate(a0, a1)_x".
        :param rule: The rule string that replaces the meta-predicate.
        :return: A `ReplacementRule` object with extracted placeholders.
        """
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
        """
        Replaces placeholders in the rule string with actual agent names and IO types.

        Ensuring proper interpolation prevents syntax errors and guarantees correctness when translating
        high-level meta-rules into concrete STL formulas.

        :param quantified_agents: A list of agent names to replace placeholders.
        :param io_type: The input/output type for the predicate.
        :return: A fully interpolated rule string.
        """
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
    """
    Stores and manages a mapping of meta-predicate names to their corresponding replacement rules.
    """

    def __init__(self, replacement_table: Dict[str, ReplacementRule]) -> None:
        self._replacement_table = replacement_table

    @classmethod
    def from_dict(cls, _dict: dict) -> "ReplacementTable":
        """
        Constructs a `ReplacementTable` from a dictionary where the keys are meta-predicate signatures and the values are their replacement rules.

        :param _dict: A dictionary mapping predicate signatures to rule strings.
        :return: A `ReplacementTable` instance.
        """
        table = {}
        for meta_predicate_signature, rule in _dict.items():
            meta_predicate_name = meta_predicate_signature.split("(")[0]
            table[meta_predicate_name] = ReplacementRule.from_str(meta_predicate_signature, rule)

        return cls(table)

    def get_replacement_rule(self, meta_predicate: str) -> Optional[ReplacementRule]:
        """
        Retrieves the replacement rule for a given meta-predicate.

        :param meta_predicate: The name of the meta-predicate.
        :return: The corresponding `ReplacementRule`, or None if the meta-predicate is unknown.
        """
        return self._replacement_table.get(meta_predicate)


class MetaPredicateReplacementVisitor(FaStlParserVisitor):
    """
    A visitor that replaces meta-predicates in traffic rules with concrete expressions.

    This visitor operates on a parsed AST, performing replacements based on a predefined `ReplacementTable`.
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
        Processes the AST and replaces meta-predicates with their corresponding concrete expressions.

        This function ensures that the transformation maintains syntactic validity by handling replacements
        within the ANTLR-generated parse tree.

        :param tree: The AST to process.
        :return: The modified source code as a string.
        """
        tree.accept(self)

        modified_source_code = self._rewriter.getText(
            self.DEFAULT_PROGRAM_NAME, 0, len(self._rewriter.tokens.tokens) - 1
        )
        return modified_source_code

    def visitPredicate(self, ctx: FaStlParser.PredicateContext) -> None:
        """
        Visits predicate nodes and replaces meta-predicates with concrete rules.

        This ensures that STL formulas are correctly rewritten before evaluation.

        :param ctx: The predicate node context.
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
