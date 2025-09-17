"""
This module provides special visitors for RTAMT to enable recording of the robustness values during  evaluation.
"""

import rtamt
from rtamt.pastifier.stl.pastifier import StlPastifier
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
from rtamt.spec.abstract_specification import (
    AbstractOfflineSpecification,
    AbstractOnlineSpecification,
)
from rtamt.syntax.ast.parser.abstract_ast_parser import AbstractAst
from rtamt.syntax.node.abstract_node import AbstractNode as RtamtAbstractNode


class DiscreteTimeOnlineUpdateVisitorDict(DiscreteTimeOnlineUpdateVisitor):
    """
    Custom visitor to collect the traces of all nodes in the RTAMT AST during online evaluation.

    Use with `discrete_time_online_interpreter_factory` from RTAMT to create a new offline interpreter.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ast_node_values = dict()

    @property
    def ast_node_values(self) -> dict[str, float]:
        return self._ast_node_values

    def visit(self, node, *args, **kwargs):
        result = super(DiscreteTimeOnlineUpdateVisitorDict, self).visit(node, *args, **kwargs)
        self._ast_node_values.update({node.name: result})
        return result


class DiscreteTimeOfflineEvaluationVisitorDict(StlDiscreteTimeOfflineAstVisitor):
    """
    Custom visitor to collect the traces of all nodes in the RTAMT AST during offline evaluation.
    This visitor is used for the standard STL semantics, for IA-STL semantics use `IAStlDiscreteTimeOfflineEvaluationVisitorDict`.

    Use with `discrete_time_offline_interpreter_factory` from RTAMT to create a new offline interpreter.
    """

    @property
    def ast_node_values(self) -> dict[RtamtAbstractNode, list[float]]:
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
    Custom visitor to collect the traces of all nodes in the RTAMT AST. This visitor is used for the IA-STL semantics, for standard STL semantics use `DiscreteTimeOfflineEvaluationVisitorDict`.

    Use with `discrete_time_offline_interpreter_factory` to create a new offline interpreter.
    """

    @property
    def ast_node_values(self) -> dict[RtamtAbstractNode, list[float]]:
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
) -> AbstractOnlineSpecification:
    """
    Creates a new rtamt specification with custom interpreters, that collect the values of each rtamt AST node.
    """
    # To collect the values of each AST node, we need to inject a custom visitor that intercepts the traces.
    if semantics == rtamt.Semantics.OUTPUT_ROBUSTNESS:
        online_visitor = IAStlDiscreteTimeOnlineAstVisitor
    elif semantics == rtamt.Semantics.STANDARD:
        online_visitor = StlDiscreteTimeOnlineAstVisitor
    else:
        raise ValueError(
            f"Cannot create spec for rtamt semantics {semantics}. Choose a valid semantic from {rtamt.Semantics.OUTPUT_ROBUSTNESS} and {rtamt.Semantics.STANDARD}."
        )

    ast.semantics = semantics

    online_interpreter = discrete_time_online_interpreter_factory(online_visitor)()
    online_interpreter.updateVisitor = DiscreteTimeOnlineUpdateVisitorDict()

    pastifier = StlPastifier()
    pastified_ast = pastifier.pastify(ast)

    online_interpreter.set_ast(pastified_ast)

    return AbstractOnlineSpecification(pastified_ast, onlineInterpreter=online_interpreter)


def stl_discrete_time_offline_specification_factory(
    semantics: rtamt.Semantics, ast: AbstractAst
) -> AbstractOfflineSpecification:
    """
    Creates a new rtamt specification with custom interpreters, that collect the values of each rtamt AST node.
    """
    # To collect the values of each AST node, we need to inject a custom visitor that intercepts the traces.
    if semantics == rtamt.Semantics.OUTPUT_ROBUSTNESS:
        offline_visitor = IAStlDiscreteTimeOfflineEvaluationVisitorDict
    elif semantics == rtamt.Semantics.STANDARD:
        offline_visitor = DiscreteTimeOfflineEvaluationVisitorDict
    else:
        raise ValueError(
            f"Cannot create spec for rtamt semantics {semantics}. Choose a valid semantic from {rtamt.Semantics.OUTPUT_ROBUSTNESS} and {rtamt.Semantics.STANDARD}."
        )

    ast.semantics = semantics

    offline_interpreter = discrete_time_offline_interpreter_factory(offline_visitor)()

    offline_interpreter.set_ast(ast)

    return AbstractOfflineSpecification(ast, offlineInterpreter=offline_interpreter)
