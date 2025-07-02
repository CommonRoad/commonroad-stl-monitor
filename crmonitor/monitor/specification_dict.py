from typing import Dict, List

import rtamt
from rtamt.semantics.abstract_discrete_time_offline_interpreter import (
    discrete_time_offline_interpreter_factory,
)
from rtamt.semantics.abstract_discrete_time_online_interpreter import (
    DiscreteTimeOnlineUpdateVisitor,
    discrete_time_online_interpreter_factory,
)
from rtamt.semantics.iastl.discrete_time.offline.ast_visitor import (
    IAStlOutputRobustnessDiscreteTimeOfflineAstVisitor,
)
from rtamt.semantics.iastl.discrete_time.online.ast_visitor import IAStlDiscreteTimeOnlineAstVisitor
from rtamt.semantics.stl.discrete_time.offline.ast_visitor import (
    StlDiscreteTimeOfflineAstVisitor,
)
from rtamt.semantics.stl.discrete_time.online.ast_visitor import StlDiscreteTimeOnlineAstVisitor
from rtamt.spec.abstract_specification import AbstractOfflineOnlineSpecification
from rtamt.syntax.ast.parser.abstract_ast_parser import AbstractAst
from rtamt.syntax.node.abstract_node import AbstractNode as RtamtAbstractNode


class DiscreteTimeOnlineUpdateVisitorDict(DiscreteTimeOnlineUpdateVisitor):
    def __init__(self) -> None:
        super().__init__()
        self._ast_node_values = dict()

    @property
    def ast_node_values(self) -> Dict[str, float]:
        return self._ast_node_values

    def visit(self, node, *args, **kwargs):
        result = super(DiscreteTimeOnlineUpdateVisitorDict, self).visit(node, *args, **kwargs)
        self._ast_node_values.update({node.name: result})
        return result


class DiscreteTimeOfflineEvaluationVisitorDict(StlDiscreteTimeOfflineAstVisitor):
    """
    Custom visitor to collect the traces of all nodes in the rtamt AST. This visitor is used for the standard STL semantics, for IA-STL semantics use `IAStlDiscreteTimeOfflineEvaluationVisitorDict`.

    Use with `discrete_time_offline_interpreter_factory` to create a new offline interpreter.
    """

    @property
    def ast_node_values(self) -> Dict[RtamtAbstractNode, List[float]]:
        """
        Retrive the mapping from node names to their traces.
        """
        return self._ast_node_values

    def visit(self, node, *args, **kwargs):
        # Usually, this should go into __init__, but `discrete_time_offline_interpreter_factory` does not call __init__ of this AST visitor.
        if not hasattr(self, "_ast_node_values"):
            self._ast_node_values = dict()
        result = super().visit(node, *args, **kwargs)
        self._ast_node_values.update({node: result})
        return result


class IAStlDiscreteTimeOfflineEvaluationVisitorDict(
    IAStlOutputRobustnessDiscreteTimeOfflineAstVisitor
):
    """
    Custom visitor to collect the traces of all nodes in the rtamt AST. This visitor is used for the IA-STL semantics, for standard STL semantics use `DiscreteTimeOfflineEvaluationVisitorDict`.

    Use with `discrete_time_offline_interpreter_factory` to create a new offline interpreter.
    """

    @property
    def ast_node_values(self) -> Dict[RtamtAbstractNode, List[float]]:
        """
        Retrive the mapping from node names to their traces.
        """
        return self._ast_node_values

    def visit(self, node, *args, **kwargs):
        # Usually, this should go into __init__, but `discrete_time_offline_interpreter_factory` does not call __init__ of this AST visitor.
        if not hasattr(self, "_ast_node_values"):
            self._ast_node_values = dict()
        result = super().visit(node, *args, **kwargs)
        self._ast_node_values.update({node: result})
        return result


def stl_discrete_time_online_specification_factory(
    semantics: rtamt.Semantics, ast: AbstractAst
) -> AbstractOfflineOnlineSpecification:
    """
    Creates a new rtamt specification with custom interpreters, that collect the values of each rtamt AST node.
    """
    # To collect the values of each AST node, we need to inject a custom visitor that intercepts the traces.
    if semantics == rtamt.Semantics.OUTPUT_ROBUSTNESS:
        offline_visitor = IAStlDiscreteTimeOfflineEvaluationVisitorDict
        online_visitor = IAStlDiscreteTimeOnlineAstVisitor
    elif semantics == rtamt.Semantics.STANDARD:
        offline_visitor = DiscreteTimeOfflineEvaluationVisitorDict
        online_visitor = StlDiscreteTimeOnlineAstVisitor

    else:
        raise ValueError(
            f"Cannot create spec for rtamt semantics {semantics}. Choose a valid semantic from {rtamt.Semantics.OUTPUT_ROBUSTNESS} and {rtamt.Semantics.STANDARD}."
        )
    offline_interpreter = discrete_time_offline_interpreter_factory(offline_visitor)()

    online_interpreter = discrete_time_online_interpreter_factory(online_visitor)()
    online_interpreter.updateVisitor = DiscreteTimeOnlineUpdateVisitorDict()

    return AbstractOfflineOnlineSpecification(
        ast, offlineInterpreter=offline_interpreter, onlineInterpreter=online_interpreter
    )
