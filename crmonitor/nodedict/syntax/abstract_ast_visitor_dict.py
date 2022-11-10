from rtamt.syntax.ast.visitor.abstract_ast_visitor import AbstractAstVisitor

class AbstractAstVisitorDict(AbstractAstVisitor):
    nodeDict = dict()

    def visit(self, node, *args, **kwargs):
        result = super(AbstractAstVisitorDict, self).visit(node, *args, **kwargs)
        self.nodeDict.update({node.name: result})
        return result
