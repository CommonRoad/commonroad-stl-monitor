from rtamt.semantics.abstract_discrete_time_online_interpreter import DiscreteTimeOnlineUpdateVisitor, \
    AbstractDiscreteTimeOnlineInterpreter
from crmonitor.nodedict.semantics.abstract_online_interpreter_dict import AbstractOnlineUpdateVisitorDict
from rtamt.semantics.abstract_online_interpreter import AbstractOnlineResetVisitor

from rtamt.exception.exception import RTAMTException
from crmonitor.nodedict.syntax.abstract_ast_visitor_dict import AbstractAstVisitorDict



class AbstractDiscreteTimeOnlineInterpreterDict(AbstractDiscreteTimeOnlineInterpreter):

    def __init__(self):
        super(AbstractDiscreteTimeOnlineInterpreterDict, self).__init__()
        self.updateVisitor = DiscreteTimeOnlineUpdateVisitorDict()
        self.resetVisitor = AbstractOnlineResetVisitor()
        return



class DiscreteTimeOnlineUpdateVisitorDict(DiscreteTimeOnlineUpdateVisitor, AbstractOnlineUpdateVisitorDict):
    pass


def discrete_time_online_interpreter_factory(AstVisitor):
    if not issubclass(AstVisitor, AbstractAstVisitorDict):  # type check
        raise RTAMTException('{} is not nickovicrtamt.rtamt AST visitor dict'.format(AstVisitor.__name__))

    class DiscreteTimeOnlineInterpreterDict(AbstractDiscreteTimeOnlineInterpreterDict, AstVisitor):
        def __init__(self, *args, **kwargs):
            super(DiscreteTimeOnlineInterpreterDict, self).__init__(*args, **kwargs)

    return DiscreteTimeOnlineInterpreterDict
