"""
Template script that can be used for STL monitoring with model predictive robustness and can be extended with custom aggregation logic.

To get started, you must provide the pre-trained models and put them into `/tmp/models` or change the MprConfig option to your preferred path.
You can either use your own models or download pre-trained ones from https://nextcloud.in.tum.de/index.php/s/bijGnSNZQB92GRz (see commonroad-model-predictive-robustness for more information).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.evaluation.visualization import FormulaVisualizationVisitor
from crmonitor.monitor.rtamt_monitor_stl import OutputType

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

scenario_path = "./scenarios/test_interstate/DEU_test_unnecessary_braking.xml"
use_mpr = False
# If True (default), robustness values will be normalized to the interval [-1.0, 1.0]. If False, robustness values are not normalized and may lay in the interval [-inf, +inf].
# Disable with caution when use_mpr is also enabled, as mpr with gaussian processes does not perform any normalization on its own.
scale_rob = False

# Optionally provide a Path where pre-trained models can be found. If None is specified, the models from the mpr repo are used.
model_path = Path(__file__).parent.parent.joinpath("output/models")

# Specify the traffic rule you want to evaluate. For an overview of the available traffic rules, see `traffic_rules_rtamt.yaml`.
traffic_rule = "R_I1"

# Set to `OutputType.OUTPUT_ROBUSTNESS` for IA-STL, and to `OutputType.STANDARD` for standard STL.
output_type = OutputType.OUTPUT_ROBUSTNESS

logging.basicConfig(level=logging.INFO)

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

if use_mpr:
    MprCfg.build_configuration(
        config={
            "common": {
                "scenario": "interstate",
                "lane": {
                    # Increased the default parameters to work around projection limit issues in MPR
                    "lateral_projection_domain_limit": 500,
                    "extend_length": 500,
                    "large_resampling_step": 3.5,
                    "num_chankins_corner_cutting": 1,
                },
                "road_network": {
                    "interstate": {
                        "use_phantom_lane": False
                    }  # Must disable phantom lanes, because otherwise commonroad-dc segfaults...
                },
            },
            "path": {
                "path_models": "/home/beicekol/projects/work/uni/commonroad-model-predictive-robustness/output/models/"
            },  # point to the models, either the ones you have trained or the pre-trained ones.
        },
        # Path root must point to a local revision of commonroad-model-predictive-robustness.
        # This configuration, assumes that the repo is in the same directory as stl-monitor repo.
        # If this is not the case for your setup, adjust the path here accordingly.
        path_root=str(
            Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"
        ),
        folder_config="config_files",
        default_profile="default",
    )


# config used for the world creation
config = get_world_config()
# MPR must be explicitly enabled
config["use_mpr"] = use_mpr

# MPR must be explicitly enabled
rule_evaluator_config = get_traffic_rule_config()
rule_evaluator_config["traffic_rules_param"]["use_mpr"] = use_mpr
rule_evaluator_config["traffic_rules_param"]["scale_rob"] = scale_rob
rule_evaluator_config["traffic_rules_param"]["model_path"] = model_path


# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario, config=config)


# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
rule_evaluator = OfflineRuleEvaluator.create_from_config(
    world, ego_vehicle.id, rule_name=traffic_rule, output_type=output_type, use_mpr=use_mpr
)
# Either step through time steps sequentially
robustness = rule_evaluator.evaluate()
print(f"robustness is {robustness}")

# TODO: Expose the AST values over a public API, such that we no longer need to access the private attributes.
visualization_visitor = FormulaVisualizationVisitor(scale_rob)
visualization_visitor.visualize(rule_evaluator._monitor)
plt.show()
