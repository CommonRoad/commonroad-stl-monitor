## Toolbox installation

The toolbox is installable as a regular pypi package. However, it is currently not available under the public index.

Installation by:

```bash
pip install git+git@gitlab.lrz.de:ge69xek/stl_crmonitor.git@refactor_gnn
```


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
Checkout the [minimum working example](tutorials/monitor_scenario.py)

```python
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

scenario_path = "../scenarios/test_interstate/DEU_test_safe_distance.xml"

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario)

# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
rule_evaluator = RuleEvaluator.create_from_config(world, ego_vehicle)

# Either step through time steps sequentially
robustness = rule_evaluator.update()
current_time_step = rule_evaluator.current_time

# Also all predicate robustness values are available
predicate_robustness = rule_evaluator.get_predicates()

# Or evaluate for all time steps of the vehicle
robustness_array = rule_evaluator.evaluate()
```

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