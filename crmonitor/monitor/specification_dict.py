from typing import Dict

import rtamt
from rtamt import Language, StlDiscreteTimeSpecification
from rtamt.semantics.abstract_discrete_time_offline_interpreter import (
    discrete_time_offline_interpreter_factory,
)
from rtamt.semantics.abstract_discrete_time_online_interpreter import (
    DiscreteTimeOnlineUpdateVisitor,
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
        result = super(DiscreteTimeOnlineUpdateVisitorDict, self).visit(
            node, *args, **kwargs
        )
        self._ast_node_values.update({node.name: result})
        return result


class DiscreteTimeOfflineEvaluationVisitorDict(StlDiscreteTimeOfflineAstVisitor):
    @property
    def ast_node_values(self) -> Dict[str, float]:
        return self._ast_node_values

    def visit(self, node, *args, **kwargs):
        if not hasattr(self, "_ast_node_values"):
            self._ast_node_values = dict()
        result = super().visit(node, *args, **kwargs)
        self._ast_node_values.update({node.name: result})
        return result


def stl_discrete_time_online_specification_factory(
    semantics: rtamt.Semantics,
) -> AbstractOfflineOnlineSpecification:
    offline_interpreter = discrete_time_offline_interpreter_factory(
        DiscreteTimeOfflineEvaluationVisitorDict
    )()
    spec = StlDiscreteTimeSpecification(semantics, Language.PYTHON)
    spec.online_interpreter.updateVisitor = DiscreteTimeOnlineUpdateVisitorDict()
    spec.offline_interpreter = offline_interpreter
    return spec
