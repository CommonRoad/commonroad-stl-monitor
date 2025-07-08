from typing import Optional, Tuple, Type

import pytest
from rtamt.semantics.interval.interval import Interval

from crmonitor.rule import (
    IOType,
    PredicateNode,
    RtamtRuleNode,
    RuleParser,
    QuantNode,
    AllNode,
    ExistNode,
    ExistsMultipleNode,
    SumIfPositiveNode,
)
from crmonitor.rule.rule_node import CompareToThresholdScaledNode
from crmonitor.rule.rule_parser import RuleParseError

TEST_PARSE_PREDICATE_TEST_DATA = [
    ("in_same_lane(a0, a1)", "in_same_lane", (0, 1), IOType.OUTPUT),
    ("in_front_of(a57, a2)_i", "in_front_of", (57, 2), IOType.INPUT),
    ("keeps_lane_speed_limit(a123456789)", "keeps_lane_speed_limit", (123456789,), IOType.OUTPUT),
    ("reverses(a123456789)_i", "reverses", (123456789,), IOType.INPUT),
]

TEST_PARSE_QUANTIFIER_TEST_DATA = [
    ("A a1: (foo(a1))", AllNode, 1),
    ("E a99: (bar(a99))", ExistNode, 99),
    ("sum_if_positive a789: (foo_bar(a0, a789))", SumIfPositiveNode, 789),
    ("exists_multiple[2] a0: (foo(a0)_i)", ExistsMultipleNode, 0),
]

TEST_PARSE_COMPARE_TO_THRESHOLD_SCALED_NODE_TEST_DATA = [
    ("compare_to_threshold_scaled[>=3] (foo(a0))", 3),
    ("compare_to_threshold_scaled[>=98] (foo(a0))", 98),
    ("compare_to_threshold_scaled[>=0.34] (foo(a0))", 0.34),
    ("compare_to_threshold_scaled[>=-1] (foo(a0))", -1),
]

TEST_PARSE_EXISTS_MULTIPLE_TEST_DATA = [
    ("exists_multiple[3] a1: (foo(a0))", 3, None),
    ("exists_multiple[-1] a1: (foo(a0))", -1, RuleParseError),
]


class TestRuleParser:
    @pytest.mark.parametrize("rule_str,base_name,agents,io_type", TEST_PARSE_PREDICATE_TEST_DATA)
    def test_parse_predicate(
        self, rule_str: str, base_name: str, agents: Tuple[int, ...], io_type: IOType
    ) -> None:
        rule_node = RuleParser().parse(rule_str)
        assert isinstance(rule_node, RtamtRuleNode)
        assert len(rule_node.children) == 1
        predicate_node = rule_node.children[0]

        assert isinstance(predicate_node, PredicateNode)
        assert predicate_node.base_name == base_name
        assert len(predicate_node.agent_placeholders) == len(agents)
        assert predicate_node.agent_placeholders == agents
        assert predicate_node.io_type == io_type

    @pytest.mark.parametrize(
        "rule_str,quant_type,quantified_vehicle", TEST_PARSE_QUANTIFIER_TEST_DATA
    )
    def test_parse_quantifier(
        self, rule_str: str, quant_type: Type[QuantNode], quantified_vehicle: int
    ) -> None:
        quant_node = RuleParser().parse(rule_str)
        assert isinstance(quant_node, quant_type)
        assert quant_node.quantified_vehicle == quantified_vehicle

    @pytest.mark.parametrize(
        "rule_str,threshold", TEST_PARSE_COMPARE_TO_THRESHOLD_SCALED_NODE_TEST_DATA
    )
    def test_parse_compare_to_threshold_scaled(self, rule_str: str, threshold: float) -> None:
        rule_node = RuleParser().parse(rule_str)
        assert isinstance(rule_node, CompareToThresholdScaledNode)
        assert rule_node.threshold == threshold

    @pytest.mark.parametrize("rule_str,threshold,error", TEST_PARSE_EXISTS_MULTIPLE_TEST_DATA)
    def test_parse_exists_multiple(
        self, rule_str: str, threshold: int, error: Optional[Type[Exception]]
    ) -> None:
        if error is None:
            rule_node = RuleParser().parse(rule_str)
            assert isinstance(rule_node, ExistsMultipleNode)
            assert rule_node.threshold == threshold
        else:
            with pytest.raises(error):
                RuleParser().parse(rule_str)
