import copy
import re
from functools import singledispatchmethod
from typing import Dict, Optional, Tuple

from crmonitor.rule.rule_node import (
    BinaryNode,
    IOType,
    MetaPredicateNode,
    PredicateNode,
    RuleNode,
    RuleTreeVisitorInterface,
    UnaryNode,
    VisitorNode,
)
from crmonitor.rule.rule_parser_interface import RuleParserInterface


class EmbedingVisitor(RuleTreeVisitorInterface[None]):
    def embed(
        self,
        root: VisitorNode,
        io_type: IOType,
        agent_placeholder_replacements: Dict[int, int],
        namespace: str,
    ) -> None:
        return self.visit(root, io_type, agent_placeholder_replacements, namespace)

    @singledispatchmethod
    def visit(self, node: VisitorNode, io_type, agent_placeholder_replacements, namespace) -> None:
        return

    @visit.register(PredicateNode)
    def _(
        self, node: PredicateNode, io_type: IOType, agent_placeholder_replacements: Dict[int, int]
    ) -> None:
        node.io_type = io_type

        replacement_agent_placeholders = []
        for agent_placeholder in node.agent_placeholders:
            if agent_placeholder not in agent_placeholder_replacements:
                raise RuntimeError()
            replacement_agent_placeholders.append(agent_placeholder_replacements[agent_placeholder])

        node.agent_placeholders = tuple(replacement_agent_placeholders)


class MetaPredicateRule:
    """
    Defines a rule that replaces meta-predicates with concrete expressions in traffic rules.

    This class handles both agent placeholders (variables that represent different agents in the rule)
    and input/output type placeholders, ensuring that the generated rule correctly interpolates these values.
    """

    def __init__(
        self,
        rule: str,
        quantified_agents_placeholders: Tuple[int, ...],
        io_type_placeholder: Optional[str] = None,
    ) -> None:
        self._rule = rule
        self._quantified_agents_placeholders = quantified_agents_placeholders
        self._io_type_placeholder = io_type_placeholder

    @property
    def rule_str(self) -> str:
        return self._rule

    @property
    def agent_placeholders(self) -> Tuple[int, ...]:
        return self._quantified_agents_placeholders

    @classmethod
    def from_str(cls, signature: str, rule: str) -> "MetaPredicateRule":
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
            int(placeholder.strip().lstrip("a"))
            for placeholder in argument_string.group(1).split(",")
        ]
        if argument_string.group(2) is not None:
            io_type_placeholder = argument_string.group(2)
        else:
            io_type_placeholder = None

        return cls(rule, tuple(quantified_agent_placeholders), io_type_placeholder)


class MetaPredicateLookupTable:
    """
    Stores and manages a mapping of meta-predicate names to their corresponding replacement rules.
    """

    def __init__(self, meta_predicates: Dict[str, MetaPredicateRule]) -> None:
        self._meta_predicates = meta_predicates

    @classmethod
    def from_dict(cls, _dict: dict) -> "MetaPredicateLookupTable":
        """
        Constructs a `ReplacementTable` from a dictionary where the keys are meta-predicate signatures and the values are their replacement rules.

        :param _dict: A dictionary mapping predicate signatures to rule strings.
        :return: A `ReplacementTable` instance.
        """
        table = {}
        for meta_predicate_signature, rule in _dict.items():
            meta_predicate_name = meta_predicate_signature.split("(")[0].lstrip("$")
            meta_predicate_rule = MetaPredicateRule.from_str(meta_predicate_signature, rule)
            table[meta_predicate_name] = meta_predicate_rule

        return cls(table)

    def get_meta_predicate_rule(self, meta_predicate: str) -> Optional[MetaPredicateRule]:
        """
        Retrieves the replacement rule for a given meta-predicate.

        :param meta_predicate: The name of the meta-predicate.
        :return: The corresponding `ReplacementRule`, or None if the meta-predicate is unknown.
        """
        return self._meta_predicates.get(meta_predicate)


class MetaPredicateReplacementVisitor(RuleTreeVisitorInterface[VisitorNode]):
    def __init__(
        self, meta_predicate_loopkup_table: MetaPredicateLookupTable, parser: RuleParserInterface
    ):
        self._meta_predicate_lookup_table = meta_predicate_loopkup_table
        self._parser = parser

    @singledispatchmethod
    def visit(self, node: VisitorNode, *args, **kwargs) -> VisitorNode:
        return node

    @visit.register(UnaryNode)
    def _(self, node: UnaryNode) -> VisitorNode:
        new_child = self.visit(node.child)
        node.child = new_child
        return node

    @visit.register(BinaryNode)
    def _(self, node: BinaryNode) -> VisitorNode:
        new_left_child = self.visit(node.left_child)
        new_right_child = self.visit(node.right_child)
        node.left_child = new_left_child
        node.right_child = new_right_child
        return node

    @visit.register(RuleNode)
    def _(self, node: RuleNode) -> VisitorNode:
        new_children = [self.visit(child) for child in node.children]
        node.children = tuple(new_children)
        return node

    @visit.register(MetaPredicateNode)
    def _(self, node: MetaPredicateNode, *args, **kwargs) -> VisitorNode:
        meta_predicate_rule = self._meta_predicate_lookup_table.get_meta_predicate_rule(
            node.metapredicate_name
        )
        if meta_predicate_rule is None:
            raise RuntimeError(f"Unkown meta-predicate '{node.metapredicate_name}'!")
        print("replacing", node.metapredicate_name, meta_predicate_rule.rule_str)
        meta_predicate_tree = self._parser.parse(meta_predicate_rule.rule_str, name=node.name)

        agents_replacement_table = {
            x: y for (x, y) in zip(node.quantified_agents, meta_predicate_rule.agent_placeholders)
        }

        embeding_visior = EmbedingVisitor()
        embeding_visior.embed(
            copy.deepcopy(meta_predicate_tree), node.io_type, agents_replacement_table, node.name
        )

        return meta_predicate_tree
