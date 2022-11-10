from crmonitor.nodedict.syntax.ltl_ast_visitor_dict import LtlAstVisitorDict
from rtamt.syntax.ast.visitor.stl.ast_visitor import StlAstVisitor

class StlAstVisitorDict(StlAstVisitor, LtlAstVisitorDict):
    def visit(self, node, *args, **kwargs):
        result = super(StlAstVisitorDict, self).visit(node, *args, **kwargs)
        self.nodeDict.update({node.name: result})
        return result
