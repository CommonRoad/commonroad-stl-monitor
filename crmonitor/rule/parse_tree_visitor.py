from decimal import Decimal
from fractions import Fraction

from antlr4.TokenStreamRewriter import TokenStreamRewriter
from rtamt.semantics.interval.interval import Interval

from crmonitor.rule.fastl.FaStlParser import FaStlParser
from crmonitor.rule.fastl.FaStlParserVisitor import FaStlParserVisitor
from crmonitor.rule.rule_node import (
    AllNode,
    CompareToThresholdScaledNode,
    ExistNode,
    ExistsMultipleNode,
    HistoricallyDurationNode,
    HistoricallyDurationSeverityNode,
    IOType,
    MetaPredicateNode,
    PredicateNode,
    RtamtRuleNode,
    SigmoidNode,
    SumIfPositiveNode,
)
from crmonitor.rule.rule_parser_context import RuleParserContext


class TrafficRuleParseTreeVisitor(FaStlParserVisitor):
    """
    Build a modified tree from the original parse tree.
    """

    # The program name is used to uniquely identify our token stream.
    # It's value does not really matter, because we only apply one kind of rewrite.
    DEFAULT_TOKEN_REWRITER_PROGRAM = "predicate"

    def __init__(self, tokens, rule_parser_ctx: RuleParserContext):
        self._rewriter: TokenStreamRewriter = TokenStreamRewriter(tokens)
        self._rule_parser_ctx = rule_parser_ctx

    def defaultResult(self):
        return []

    def aggregateResult(self, aggregate, nextResult):
        aggregate.extend(nextResult)
        return aggregate

    def visitVehicle(self, ctx: FaStlParser.VehicleContext):
        # Get the placeholder id
        return [int(ctx.IntegerLiteral().getText())]

    def visitPredicate(self, ctx: FaStlParser.PredicateContext):
        # PredicateNode needs the vehicle ids as tuple, and not as list.
        vehicle_ids = tuple(self.visitChildren(ctx))
        pred_basename: str = ctx.Identifier().getText()
        if ctx.IO_TYPE() is not None:
            if ctx.IO_TYPE().getText() == "_i":
                io_type = IOType.INPUT
            else:
                io_type = None
            token_index = ctx.IO_TYPE().symbol.tokenIndex
            # Delete the input indicator, only keep the predicate name.
            self._rewriter.delete(self.DEFAULT_TOKEN_REWRITER_PROGRAM, token_index, token_index)
        else:
            io_type = IOType.OUTPUT
        # Rewrite the predicate name into RTAMT compliant syntax
        suffix = "__" + "_".join(str(i) for i in vehicle_ids)
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.LPAREN().symbol.tokenIndex,
            ctx.RPAREN().symbol.tokenIndex,
            suffix,
        )
        if pred_basename.startswith("$"):
            meta_predicate_name = pred_basename.lstrip("$")
            p = MetaPredicateNode(pred_basename + suffix, meta_predicate_name, vehicle_ids, io_type)
        else:
            p = PredicateNode(pred_basename + suffix, pred_basename, vehicle_ids, io_type)
        return [p]

    def visitSpecQuantForall(self, ctx: FaStlParser.SpecQuantForallContext):
        child = self.visit(ctx.spec())[0]
        quantified_vehicle = self.visitVehicle(ctx.vehicle())[0]
        node_name = self._get_new_unique_node_name()
        node = AllNode(node_name, child, quantified_vehicle)
        # Replace sub-formula inside the quantification by a "virtual" predicate g...
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    def visitSpecQuantExist(self, ctx: FaStlParser.SpecQuantExistContext):
        child = self.visit(ctx.spec())[0]
        quantified_vehicle = self.visitVehicle(ctx.vehicle())[0]
        node_name = self._get_new_unique_node_name()
        node = ExistNode(node_name, child, quantified_vehicle)
        # Replace sub-formula inside the quantification by a "virtual" predicate g...
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    def visitSpecNested(self, ctx: FaStlParser.SpecNestedContext):
        children = self.visitChildren(ctx)
        # De-duplicate
        children = tuple(dict.fromkeys(children))
        if not isinstance(ctx.parentCtx, FaStlParser.SpecNestedContext):
            # Flatten tree to evaluate with rtamt
            return [
                RtamtRuleNode(
                    self._get_new_unique_node_name(),
                    children,
                    self._rewriter.getText(
                        self.DEFAULT_TOKEN_REWRITER_PROGRAM,
                        ctx.start.tokenIndex,
                        ctx.stop.tokenIndex,
                    ),
                )
            ]
        else:
            return children

    def visitSpecSigmoid(self, ctx: FaStlParser.SpecSigmoidContext):
        child = self.visit(ctx.spec())[0]
        node_name = self._get_new_unique_node_name()
        node = SigmoidNode(node_name, child)
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    def visitSpecHistoricallyDuration(self, ctx: FaStlParser.SpecHistoricallyDurationContext):
        child = self.visit(ctx.spec())[0]
        if ctx.interval() is None:
            interval = None
        else:
            interval = self.process_interval(ctx.interval())
        node_name = self._get_new_unique_node_name()
        node = HistoricallyDurationNode(node_name, child, interval)
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    def visitSpecHistoricallyDurationSeverity(
        self, ctx: FaStlParser.SpecHistoricallyDurationSeverityContext
    ):
        child = self.visit(ctx.spec())[0]
        if ctx.interval() is None:
            interval = None
        else:
            interval = self.process_interval(ctx.interval())
        node_name = self._get_new_unique_node_name()
        node = HistoricallyDurationSeverityNode(node_name, child, interval)

        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    def visitSpecQuantSumIfPositive(self, ctx: FaStlParser.SpecQuantSumIfPositiveContext):
        child = self.visit(ctx.spec())[0]
        quantified_vehicle = self.visitVehicle(ctx.vehicle())[0]
        node_name = self._get_new_unique_node_name()
        node = SumIfPositiveNode(node_name, child, quantified_vehicle)
        # Replace sub-formula inside the quantification by a "virtual" predicate g...
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    def visitSpecCompareToThresholdScaled(
        self, ctx: FaStlParser.SpecCompareToThresholdScaledContext
    ):
        child = self.visitChildren(ctx)[0]
        threshold = float(ctx.threshold().literal().getText())
        node_name = self._get_new_unique_node_name()
        node = CompareToThresholdScaledNode(node_name, child, threshold)
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    def visitSpecExistsMultiple(self, ctx: FaStlParser.SpecExistsMultipleContext):
        child = self.visit(ctx.spec())[0]
        quantified_vehicle = self.visitVehicle(ctx.vehicle())[0]
        threshold = int(ctx.IntegerLiteral().getText())
        node_name = self._get_new_unique_node_name()
        node = ExistsMultipleNode(node_name, child, quantified_vehicle, threshold)
        self._rewriter.replace(
            self.DEFAULT_TOKEN_REWRITER_PROGRAM,
            ctx.start.tokenIndex,
            ctx.stop.tokenIndex,
            node_name,
        )
        return [node]

    # Not visitor methods, because this breaks the parsing when intervals occur outside of our custom operators
    def process_interval_time(self, ctx):
        time_bound = Fraction(Decimal(ctx.literal().getText()))
        if ctx.unit() is None:
            unit = ""
        else:
            unit = ctx.unit().getText()

        return time_bound, unit

    def process_interval(self, ctx):
        begin, begin_unit = self.process_interval_time(ctx.intervalTime(0))
        end, end_unit = self.process_interval_time(ctx.intervalTime(1))
        interval = Interval(begin, end, begin_unit, end_unit)
        return interval

    def _get_new_unique_node_name(self) -> str:
        return self._rule_parser_ctx.generate_new_unique_sub_rule_name()
