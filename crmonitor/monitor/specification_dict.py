from typing import Dict, List

import rtamt
from rtamt import Language, StlDiscreteTimeSpecification
from rtamt.semantics.abstract_discrete_time_offline_interpreter import (
    discrete_time_offline_interpreter_factory,
)
from rtamt.semantics.abstract_discrete_time_online_interpreter import (
    DiscreteTimeOnlineUpdateVisitor,
)
from rtamt.semantics.iastl.discrete_time.offline.ast_visitor import (
    IAStlOutputRobustnessDiscreteTimeOfflineAstVisitor,
)
from rtamt.semantics.stl.discrete_time.offline.ast_visitor import (
    StlDiscreteTimeOfflineAstVisitor,
)
from rtamt.spec.abstract_specification import AbstractOfflineOnlineSpecification


class DiscreteTimeOnlineUpdateVisitorDict(DiscreteTimeOnlineUpdateVisitor):
    def __init__(self) -> None:
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
    def ast_node_values(self) -> Dict[str, List[float]]:
        """
        Retrive the mapping from node names to their traces.
        """
        return self._ast_node_values

    def visit(self, node, *args, **kwargs):
        # Usually, this should go into __init__, but `discrete_time_offline_interpreter_factory` does not call __init__ of this AST visitor.
        if not hasattr(self, "_ast_node_values"):
            self._ast_node_values = dict()
        result = super().visit(node, *args, **kwargs)
        self._ast_node_values.update({node.name: result})
        return result


class IAStlDiscreteTimeOfflineEvaluationVisitorDict(
    IAStlOutputRobustnessDiscreteTimeOfflineAstVisitor
):
    """
    Custom visitor to collect the traces of all nodes in the rtamt AST. This visitor is used for the IA-STL semantics, for standard STL semantics use `DiscreteTimeOfflineEvaluationVisitorDict`.

    Use with `discrete_time_offline_interpreter_factory` to create a new offline interpreter.
    """

    @property
    def ast_node_values(self) -> Dict[str, List[float]]:
        """
        Retrive the mapping from node names to their traces.
        """
        return self._ast_node_values

    def visit(self, node, *args, **kwargs):
        # Usually, this should go into __init__, but `discrete_time_offline_interpreter_factory` does not call __init__ of this AST visitor.
        if not hasattr(self, "_ast_node_values"):
            self._ast_node_values = dict()
        result = super().visit(node, *args, **kwargs)
        self._ast_node_values.update({node.name: result})
        return result


def stl_discrete_time_online_specification_factory(
    semantics: rtamt.Semantics,
) -> AbstractOfflineOnlineSpecification:
    """
    Creates a new rtamt specification with custom interpreters, that collect the values of each rtamt AST node.
    """
    # To collect the values of each AST node, we need to inject a custom visitor that intercepts the traces.
    if semantics == rtamt.Semantics.OUTPUT_ROBUSTNESS:
        visitor = IAStlDiscreteTimeOfflineEvaluationVisitorDict
    elif semantics == rtamt.Semantics.STANDARD:
        visitor = DiscreteTimeOfflineEvaluationVisitorDict
    else:
        raise ValueError(
            f"Cannot create spec for rtamt semantics {semantics}. Choose a valid semantic from {rtamt.Semantics.OUTPUT_ROBUSTNESS} and {rtamt.Semantics.STANDARD}."
        )
    offline_interpreter = discrete_time_offline_interpreter_factory(visitor)()

    spec = StlDiscreteTimeSpecification(semantics, Language.PYTHON)
    spec.online_interpreter.updateVisitor = DiscreteTimeOnlineUpdateVisitorDict()
    spec.offline_interpreter = offline_interpreter
    return spec
