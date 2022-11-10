from crmonitor.nodedict.semantics.offline.interpreter_dict import StlDiscreteTimeOfflineInterpreterDict
from crmonitor.nodedict.semantics.online.interpreter_dict import StlDiscreteTimeOnlineInterpreterDict
from rtamt.pastifier.stl.pastifier import StlPastifier
from rtamt.spec.abstract_specification import AbstractOfflineOnlineSpecification
from rtamt.syntax.ast.parser.stl.specification_parser import StlAst


def StlDiscreteTimeSpecificationDict():
    spec = AbstractOfflineOnlineSpecification(StlAst(), StlDiscreteTimeOfflineInterpreterDict(),
                                              StlDiscreteTimeOnlineInterpreterDict(), pastifier=StlPastifier())
    return spec