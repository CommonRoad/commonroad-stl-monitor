# CommonRoad STL Monitor

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/commonroad-stl-monitor.svg)](https://pypi.python.org/pypi/commonroad-stl-monitor/)
[![PyPI version fury.io](https://badge.fury.io/py/commonroad-stl-monitor.svg)](https://pypi.python.org/pypi/commonroad-stl-monitor/)
[![PyPI download month](https://img.shields.io/pypi/dm/commonroad-stl-monitor.svg?label=PyPI%20downloads)](https://pypi.python.org/pypi/commonroad-stl-monitor/)
[![PyPI license](https://img.shields.io/pypi/l/commonroad-stl-monitor.svg)](https://pypi.python.org/pypi/commonroad-stl-monitor/)

CommonRoad STL Monitor is a toolbox to evaluate the robustness of model-free and model-predictive robustness on CommonRoad scenarios using signal temporal logic.

# Getting Started

## Installation

The toolbox is installable as a regular PyPI package:

```bash
pip install commonroad-stl-monitor
```

Additionally, you can install extras like `visualization` to enable advanced AST visualizations. The visualization requires a working `graphiz` installation, which you should be able to source from your distros package registry.

```bash
pip install commonroad-stl-monitor[visualization]
```

## Example Usage

```python
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common import World
from crmonitor.evaluation import OfflineRuleEvaluator

scenario_path = "../scenarios/test_interstate/DEU_test_safe_distance.xml"

# Open the scenario
scenario, _ = CommonRoadFileReader(scenario_path)

# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario)

# Create a rule evaluator for the traffic rule 'R_G1'
rule_evaluator = OfflineRuleEvaluator.create_for_rule("R_G1", dt=world.dt)


ego_vehicle = next(iter(world.vehicles))
# Evaluate the robustness across the whole time frame the ego vehicle is defined.
robustness = rule_evaluator.evaluate(world, ego_vehicle.vehicle_id)
```

## Documentation

The full documentation can be found at [cps.pages.gitlab.lrz.de/commonroad/commonroad-stl-monitor](https://cps.pages.gitlab.lrz.de/commonroad/commonroad-stl-monitor/).
