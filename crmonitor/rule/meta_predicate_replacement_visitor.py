from typing import Dict, Optional

from antlr4.BufferedTokenStream import BufferedTokenStream
from antlr4.TokenStreamRewriter import TokenStreamRewriter

from crmonitor.common.config import get_traffic_rule_config
from crmonitor.rule.fastl.FaStlParser import FaStlParser
from crmonitor.rule.fastl.FaStlParserVisitor import FaStlParserVisitor


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
        replacement_table: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialze a new `MetaPredicateReplacementVisitor` and optionally load the a replacement table.

        :param tokens: The token stream as produced by an ANTLR lexer.
        :param replacement_table: An optional replacement table, with meta predicates as keys and their replacement rule as values. If None, the default table will be loaded from `traffic_rules_rtamt.yaml`.
        """
        self._rewriter: TokenStreamRewriter = TokenStreamRewriter(tokens)
        if replacement_table is None:
            self._replacement_table = get_traffic_rule_config()["meta_predicates"]
        else:
            self._replacement_table = replacement_table

    def visit(self, tree) -> str:
        """
        A custom visit method, that returns the rule source code with all meta predicate replaced.
        """
        tree.accept(self)

        modified_source_code = self._rewriter.getText(
            self.DEFAULT_PROGRAM_NAME, 0, len(self._rewriter.tokens.tokens) - 1
        )
        return modified_source_code

    def visitPredicate(self, ctx: FaStlParser.PredicateContext) -> None:
        """
        Visit a predicate node and perform a replacement if the predicate is a meta-predicate.
        """
        pred_name = ctx.Identifier().getText()
        if not pred_name.startswith("$"):
            return

        replacement = self._replacement_table.get(pred_name)
        if replacement is None:
            raise RuntimeError(
                f"Failed to replace meta predicate '{pred_name}': No replacement rule found!"
            )

        # The replacement might mess up precedence, so we make sure this does not happen.
        replacement = f"({replacement})"

        self._rewriter.replace(
            self.DEFAULT_PROGRAM_NAME,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            replacement,
        )

        return
