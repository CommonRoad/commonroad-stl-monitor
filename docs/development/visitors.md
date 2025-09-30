# Concepts

## Visitors

We view rules as abstract syntax trees (AST) which can be traversed by visitor objects. The AST has the following node types:

* Predicate node: A single predicate in a rule. This node does not have any descendants (leaf).
* Rule node: A simple temporal logic rule containing temporal and boolean operators. Its descendants are the predicates
it contains.
* For-all node: A temporal logic rule `A o: r(o)` where an object is bound by a for-all quantifier. This is merely syntactic sugar for:
`r(o_1) and r(o_2) and ... r(o_n)` where `o` are objects from a finite set and `r` is the rule.
* Exists node: A temporal logic rule `E o: r(o)` where an object is bound by an existential quantifier. This is merely syntactic sugar for:
`r(o_1) or r(o_2) or ... r(o_n)` where `o` are objects from a finite set and `r` is the rule.

The rule strings are parsed into the AST. The nodes itself in the AST are not functional, but they encode the type of each node.
Functional copies of the AST can be created with other visitors e.g. `MonitorCreationRuleTreeVisitor`.
