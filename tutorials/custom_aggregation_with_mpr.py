"""
Template script that can be used for STL monitoring with model predictive robustness and can be extended with custom aggregation logic.

To get started, you must provide the pre-trained models and put them into `/tmp/models` or change the MprConfig option to your preferred path.
You can either use your own models or download pre-trained ones from https://nextcloud.in.tum.de/index.php/s/bijGnSNZQB92GRz (see commonroad-model-predictive-robustness for more information).
"""

from pathlib import Path
from commonroad.common.file_reader import CommonRoadFileReader
import rtamt
from rtamt.pastifier.stl.pastifier import StlPastifier
from rtamt.semantics.abstract_discrete_time_offline_interpreter import (
    discrete_time_offline_interpreter_factory,
)
from rtamt.semantics.abstract_discrete_time_online_interpreter import (
    discrete_time_online_interpreter_factory,
)
from rtamt.semantics.stl.discrete_time.offline.ast_visitor import (
    StlDiscreteTimeOfflineAstVisitor,
)
from rtamt.semantics.stl.discrete_time.online.ast_visitor import (
    StlDiscreteTimeOnlineAstVisitor,
)
from rtamt.spec.abstract_specification import (
    AbstractOfflineOnlineSpecification,
)
from rtamt.syntax.ast.parser.stl.specification_parser import StlAst

from crmonitor.common.helper import gather
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import RuleEvaluator
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
import commonroad_mpr
from crmonitor.evaluation.visitor import (
    EvaluationMonitorTreeVisitor,
    MonitorCreationRuleTreeVisitor,
)
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    ExistMonitorNode,
    RuleMonitorNode, AndsmoothMonitorNode,
)
from crmonitor.rule.rule_node import PredicateNode

scenario_path = "./scenarios/test_interstate/DEU_test_unnecessary_braking.xml"
use_mpr = False

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

if use_mpr:
    MprCfg.set_paths(
        path_root=str(Path(commonroad_mpr.__file__).parent.parent),
        folder_config="config_files",
    )
    MprCfg.update_with_config(
        {
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
                "path_models": "/tmp/models"
            },  # point to the models, either the ones you have trained or the pre-trained ones.
        }
    )


class OfflineEvaluationMonitorTreeVisitor(EvaluationMonitorTreeVisitor):
    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        print(rule_node)
        world = ctx[0]
        # Collect child_values
        assert rule_node.monitor.dt == world.dt, (
            f"Monitor constructed with dt="
            f"{rule_node.monitor.dt} but got "
            f"world state with dt={world.dt}!"
        )
        child_values = {c.name: c.visit(self, *ctx) for c in rule_node.children}
        val = rule_node.evaluate(list(child_values.items()))  # evaluate instead of update for offline usage
        return val

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        print(all_node)
        world, mpr_world, time_step, other_ids = ctx[:4]
        # TODO: Implement the All quantifier
        return 1.0

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        print(exist_node)
        world, mpr_world, time_step, other_ids = ctx[:4]
        # TODO: Implement the Exist quantifier
        return 1.0

    def visit_andsmooth_node(self, andsmooth_node: AndsmoothMonitorNode, *ctx):
        print(andsmooth_node)
        world, mpr_world, time_step, other_ids = ctx[:4]
        # TODO: Implement the Andsmooth operator
        return 2.0

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        print(predicate_node)
        world, mpr_world, time_step, other_ids = ctx[:4]
        predicate_ids = gather(other_ids, predicate_node.agent_placeholders)
        samples = []
        for i in range(0, time_step):
            samples.append(
                predicate_node.evaluate_robustness(world, mpr_world, i, predicate_ids)
            )

        return samples


# ==============================
# End custom aggregation logic
# ==============================

# config used for the world creation
config = get_world_config()
# MPR must be explicitly enabled
config["use_mpr"] = use_mpr

# MPR must be explicitly enabled
rule_evaluator_config = get_traffic_rule_config()
rule_evaluator_config["traffic_rules_param"]["use_mpr"] = use_mpr


# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario, config=config)


# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
rule_evaluator = RuleEvaluator.create_from_config(
    world,
    ego_vehicle.id,
    rule="R_G3",
    monitor_creation_visitor=MonitorCreationRuleTreeVisitor(dt=scenario.dt),
    monitor_evaluation_visitor=OfflineEvaluationMonitorTreeVisitor(),
)

# Either step through time steps sequentially
robustness = rule_evaluator.evaluate_offline()
print(f"robustness is {robustness}")
