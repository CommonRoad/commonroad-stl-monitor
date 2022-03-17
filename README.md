## Toolbox installation

The toolbox is installable as a regular pypi package. However, it is currently not available under the public index.

Available versions can be found [here](https://gitlab.lrz.de/ge69xek/stl_crmonitor/-/packages).

Installation by:

```bash
pip install --extra-index-url https://commonroad-dc-package:WRzXAy7oJ8S2atx8iit6@gitlab.lrz.de/api/v4/projects/62155/packages/pypi/simple \
--extra-index-url https://stl-crmonitor-package:4qhSxJuW2qo4q-A6dyMY@gitlab.lrz.de/api/v4/projects/70411/packages/pypi/simple \
stl-crmonitor
```

Both extra indicies are needed to obtain the internal versions of the commonroad-dc package and the stl-crmonitor package. 


## Development setup
- Install [anconda](https://www.anaconda.com/)
- Clone and enter the repository
- Create the environment and install the dependencies by
```
conda env create -f environment.yml
```
- Activate the created environment
```
conda activate stl-crmonitor
```
- Install package to development path
```
conda develop .
```

## Run the tests
- Run 
```
cd crmonitor/tests
python -m unittest
```

## Doing a version bump

Version numbers are automatically bumbed using `bump2version` tool. It autmatically updates the version number in the
setup.py file. Version numbering schema: `{major}.{minor}.{patch}-{release}{build}`. The version of each component can
be bumped by e.g. `bump2version major` or `bump2version minor`. Merges to the development branch are automatically packaged,
deployed and the build number is bumped.

## Getting Started
Checkout the [minimum working example](mwe.ipynb)

## Concepts

### Visitors

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