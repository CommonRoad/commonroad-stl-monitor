"""
Template script that can be used for STL monitoring with model predictive robustness and can be extended with custom aggregation logic.

To get started, you must provide the pre-trained models and put them into `/tmp/models` or change the MprConfig option to your preferred path.
You can either use your own models or download pre-trained ones from https://nextcloud.in.tum.de/index.php/s/bijGnSNZQB92GRz (see commonroad-model-predictive-robustness for more information).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.evaluation.predicate_interface import (
    PredicateEvaluationMode,
    PredicateInterfaceConfig,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.mpr import (
    MprPredicateEvaluatorConfig,
    MprGpPredicateEvaluatorConfig,
    MprPredicateEvaluator,
    MprGpPredicateEvaluator,
)
from crmonitor.predicates.base import PredicateEvaluatorConfig

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

scenarios_load_path = Path(__file__).parents[3] / "scenarios-for-semantic-aware-stl" / "highD"
scenario_id = "DEU_LocationELower15-1_1510041_T-10291"
predicate_evaluation_mode = PredicateEvaluationMode.MPR_GP
# If True (default), robustness values will be normalized to the interval [-1.0, 1.0]. If False, robustness values are not normalized and may lay in the interval [-inf, +inf].
# Disable with caution when use_mpr is also enabled, as mpr with gaussian processes does not perform any normalization on its own.
scale_rob = True  # not use_mpr

# Optionally provide a Path where pre-trained models can be found. If None is specified, the models from the mpr repo are used.
model_path = Path(__file__).parent.parent.joinpath("output/models")

# Specify the traffic rule you want to evaluate. For an overview of the available traffic rules, see `traffic_rules_rtamt.yaml`.
traffic_rule = "R_G3"

# Set to `OutputType.OUTPUT_ROBUSTNESS` for IA-STL, and to `OutputType.STANDARD` for standard STL.
output_type = OutputType.OUTPUT_ROBUSTNESS

logging.basicConfig(level=logging.INFO)

scenario_path = scenarios_load_path / scenario_id
# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path.with_suffix(".xml")).open(lanelet_assignment=True)


# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario)

# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
_LOGGER.info(
    f"Evaluating rule {traffic_rule} for ego vehicle {ego_vehicle.id} in scenario {scenario_id}"
)
predicate_interface_config = PredicateInterfaceConfig(
    mode=predicate_evaluation_mode,
    base=PredicateEvaluatorConfig(scale_rob=scale_rob),
    mpr=MprPredicateEvaluatorConfig(),
    mpr_gp=MprGpPredicateEvaluatorConfig(model_path=model_path),
)

rule_evaluator = OfflineRuleEvaluator.create_for_rule(
    traffic_rule,
    world,
    ego_vehicle.id,
    output_type=output_type,
    predicate_interface_config=predicate_interface_config,
)
# Either step through time steps sequentially
robustness = rule_evaluator.evaluate(end_time=10)
print(f"robustness is {robustness}")

rule_evaluator.visualize()
plt.show()
