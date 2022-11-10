from crmonitor.nodedict.syntax.abstract_ast_visitor_dict import AbstractAstVisitorDict

from rtamt.semantics.abstract_online_interpreter import AbstractOnlineUpdateVisitor


class AbstractOnlineUpdateVisitorDict(AbstractOnlineUpdateVisitor, AbstractAstVisitorDict):
    pass
