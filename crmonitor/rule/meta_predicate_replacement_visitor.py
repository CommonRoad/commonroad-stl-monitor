import re
from functools import singledispatchmethod
from typing import Dict, Optional, Tuple

from crmonitor.rule.rule_node import (
    IOType,
    MetaPredicateNode,
    PredicateNode,
    RuleAstNode,
    RuleTreeVisitorInterface,
    UnaryNode,
    VaradicNode,
)
from crmonitor.rule.rule_parser_interface import RuleParserInterface


class MetaPredicateRule:
    """
    Defines a rule that replaces meta-predicates with concrete expressions in traffic rules.

    This class serves as a template for rule substitution, capturing both the pattern to
    be replaced (via meta-predicate signatures) and the replacement logic (rule string).
    It preserves critical contextual information such as agent placeholders and I/O types
    that must be carried over during substitution.

    :param rule: The rule string that will replace the meta-predicate
    :param quantified_agents_placeholders: Tuple of agent IDs referenced in the rule
    :param io_type_placeholder: Optional I/O type specifier for the rule context
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

    This registry acts as the central repository for meta-predicate definitions, allowing the
    system to look up the appropriate substitution rules when processing a rule tree. It provides
    a clean separation between rule definition and rule processing.

    :param meta_predicates: Dictionary mapping meta-predicate names to their replacement rules
    """

    def __init__(self, meta_predicates: Dict[str, MetaPredicateRule]) -> None:
        self._meta_predicates = meta_predicates

    @classmethod
    def from_dict(cls, _dict: dict) -> "MetaPredicateLookupTable":
        """
        Constructs a `MetaPredicateLookupTable` from a dictionary where the keys are meta-predicate signatures and the values are their replacement rules.

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


class MetaPredicateReplacementVisitor(RuleTreeVisitorInterface[RuleAstNode]):
    """
    Traverses a rule tree and replaces meta-predicate nodes with their concrete implementations.

    This visitor performs the core substitution logic, identifying meta-predicate nodes in the
    rule tree and replacing them with parsed subtrees based on their defined replacement rules.
    The visitor preserves all contextual information including agent bindings and I/O types.

    :param meta_predicate_loopkup_table: Registry of meta-predicates and their replacement rules
    :param parser: Rule parser for generating subtrees from replacement rule strings
    """

    def __init__(
        self, meta_predicate_loopkup_table: MetaPredicateLookupTable, parser: RuleParserInterface
    ):
        self._meta_predicate_lookup_table = meta_predicate_loopkup_table
        self._parser = parser

    @singledispatchmethod
    def visit(self, node: RuleAstNode, *args, **kwargs) -> RuleAstNode:
        return node

    @visit.register(UnaryNode)
    def _(self, node: UnaryNode) -> RuleAstNode:
        new_child = self.visit(node.child)
        node.child = new_child
        return node

    @visit.register(VaradicNode)
    def _(self, node: VaradicNode) -> RuleAstNode:
        new_children = [self.visit(child) for child in node.children]
        node.children = tuple(new_children)
        return node

    @visit.register(MetaPredicateNode)
    def _(self, node: MetaPredicateNode, *args, **kwargs) -> RuleAstNode:
        if node.io_type is None:
            raise RuntimeError(f"I/O type of meta-predicate {node.metapredicate_name} is not set!")

        meta_predicate_rule = self._meta_predicate_lookup_table.get_meta_predicate_rule(
            node.metapredicate_name
        )
        if meta_predicate_rule is None:
            raise RuntimeError(f"Unkown meta-predicate '{node.metapredicate_name}'!")

        # Parse the meta-predicate into a generic rule AST.
        meta_predicate_tree = self._parser.parse(
            meta_predicate_rule.rule_str, name=node.name, replace_meta_predicates=False
        )

        # Construct a mapping from the agent placeholders in the meta-predicate sub-tree to the agents in our current tree.
        # This allows the `EmbedingVisitor` to lookup which agent placeholderss it should replace.
        agents_replacement_table = {
            replacement_target: replacement
            for (replacement_target, replacement) in zip(
                meta_predicate_rule.agent_placeholders, node.agent_placeholders
            )
        }

        # Embed the replacement tree into our main tree.
        # This ensures that context information, such as the agent bindings and I/O types are correctly preserved.
        embeding_visior = EmbedingVisitor()
        embeding_visior.embed(
            meta_predicate_tree, node.io_type, agents_replacement_table, node.metapredicate_name
        )

        return self.visit(meta_predicate_tree)


class EmbedingVisitor(RuleTreeVisitorInterface[None]):
    """
    Applies context information to a subtree being embedded during meta-predicate replacement.

    When a meta-predicate is replaced with its concrete implementation, this visitor ensures
    that the replacement subtree inherits the appropriate context from its parent, including
    I/O type specifications and agent placeholder mappings. This maintains semantic consistency
    between the abstract meta-predicate and its concrete implementation.

    :param root: Root node of the subtree being embedded
    :param io_type: I/O type to apply throughout the subtree
    :param agent_placeholder_replacements: Mappings for agent placeholders
    """

    def embed(
        self,
        root: RuleAstNode,
        io_type: IOType,
        agent_placeholder_replacements: Dict[int, int],
        namespace: str,
    ) -> None:
        return self.visit(root, io_type, agent_placeholder_replacements, namespace)

    @singledispatchmethod
    def visit(self, node: RuleAstNode, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Embeding Visitor does not implement embeding for node {node}! This is a bug, and this node must be handled!"
        )

    @visit.register(UnaryNode)
    def _(self, node: UnaryNode, *args, **kwargs) -> None:
        self.visit(node.child, *args, **kwargs)

    @visit.register(VaradicNode)
    def _(self, node: VaradicNode, *args, **kwargs) -> None:
        for child in node.children:
            self.visit(child, *args, **kwargs)

    @visit.register(PredicateNode)
    def _(
        self,
        node: PredicateNode,
        io_type: IOType,
        agent_placeholder_replacements: Dict[int, int],
        namespace: str,
    ) -> None:
        node.io_type = io_type

        replacement_agent_placeholders = []
        for agent_placeholder in node.agent_placeholders:
            if agent_placeholder in agent_placeholder_replacements:
                replacement_agent_placeholders.append(
                    agent_placeholder_replacements[agent_placeholder]
                )
            else:
                replacement_agent_placeholders.append(agent_placeholder)

        node.agent_placeholders = tuple(replacement_agent_placeholders)
        # node.name = f"{namespace}_{node.name}"

    @visit.register(MetaPredicateNode)
    def _(
        self,
        node: MetaPredicateNode,
        io_type: IOType,
        agent_placeholder_replacements: Dict[int, int],
        namespace: str,
    ) -> None:
        node.io_type = io_type

        replacement_agent_placeholders = []
        for agent_placeholder in node.agent_placeholders:
            if agent_placeholder in agent_placeholder_replacements:
                replacement_agent_placeholders.append(
                    agent_placeholder_replacements[agent_placeholder]
                )
            else:
                replacement_agent_placeholders.append(agent_placeholder)

        node.agent_placeholders = tuple(replacement_agent_placeholders)
        # node.name = f"{namespace}_{node.name}"
