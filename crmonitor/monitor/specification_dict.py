from typing import Dict

from rtamt import Language, StlDiscreteTimeSpecification
import rtamt
from rtamt.semantics.abstract_discrete_time_online_interpreter import (
    DiscreteTimeOnlineUpdateVisitor,
)
from rtamt.spec.abstract_specification import (
    AbstractOfflineOnlineSpecification,
)


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


def stl_discrete_time_online_specification_factory(
    semantics: rtamt.Semantics,
) -> AbstractOfflineOnlineSpecification:
    spec = StlDiscreteTimeSpecification(semantics, Language.PYTHON)
    spec.online_interpreter.updateVisitor = DiscreteTimeOnlineUpdateVisitorDict()
    return spec
