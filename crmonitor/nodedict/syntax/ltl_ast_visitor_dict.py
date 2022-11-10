from crmonitor.nodedict.syntax.abstract_ast_visitor_dict import AbstractAstVisitorDict
from rtamt.syntax.ast.visitor.ltl.ast_visitor import LtlAstVisitor

class LtlAstVisitorDict(LtlAstVisitor, AbstractAstVisitorDict):
    def visit(self, node, *args, **kwargs):
        result = super(LtlAstVisitorDict, self).visit(node, *args, **kwargs)
        self.nodeDict.update({node.name: result})
        return result

