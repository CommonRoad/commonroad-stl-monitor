# -*- coding: utf-8 -*-
from rtamt.semantics.abstract_discrete_time_offline_interpreter import AbstractDiscreteTimeOfflineInterpreter
from rtamt.exception.exception import RTAMTException
from crmonitor.nodedict.syntax.abstract_ast_visitor_dict import AbstractAstVisitorDict


def discrete_time_offline_interpreter_factory(AstVisitorDict):
    if not issubclass(AstVisitorDict, AbstractAstVisitorDict):  # type check
        raise RTAMTException('{} is not rtamtnicko.rtamt AST visitor dict'.format(AstVisitorDict.__name__))

    class DiscreteTimeOfflineInterpreterDict(AbstractDiscreteTimeOfflineInterpreter, AstVisitorDict):
        def __init__(self, *args, **kwargs):
            super(DiscreteTimeOfflineInterpreterDict, self).__init__(*args, **kwargs)
    return DiscreteTimeOfflineInterpreterDict


