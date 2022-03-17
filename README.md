## Installation
- Install [anconda](https://www.anaconda.com/)
- Clone and enter the repository
- Create the environment and install the dependencies by
```
conda env create -f environment.yml
```
- Activate the created environment
```
conda activate py-cr37
```
- Checkout submodules
```
git submodule update --init --recursive
```
- Install commonroad-drivability-checker (press Ctrl-c if asked for root permission)
```
sudo apt install libboost-dev libboost-thread-dev libboost-test-dev libboost-filesystem-dev libeigen3-dev
cd external/commonroad-drivability-checker
bash build.sh -e /home/`whoami`/anaconda3/envs/cr-py37 -v 3.7 -i -j 4
cd ../..
```
- Install rtamt
```
pip install external/rtamt
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