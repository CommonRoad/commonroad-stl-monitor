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
from rtamt.semantics.stl.discrete_time.online.and_operation import AndOperation
from rtamt.semantics.stl.discrete_time.online.ast_visitor import (
    StlDiscreteTimeOnlineAstVisitor,
)
from rtamt.semantics.stl.discrete_time.online.once_timed_operation import (
    OnceTimedOperation,
)
from rtamt.spec.abstract_specification import (
    AbstractOfflineOnlineSpecification,
    AbstractOnlineSpecification,
)
from rtamt.syntax.ast.parser.stl.specification_parser import StlAst

from crmonitor.monitor.rtamt_monitor_stl import _create_spec
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
    RuleMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.monitor.specification_dict import DiscreteTimeOnlineUpdateVisitorDict
from crmonitor.rule.rule_node import RuleNode

scenario_path = "./scenarios/test_interstate/DEU_test_unnecessary_braking.xml"

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

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

# ==============================
# Start custom aggregation logic
# ==============================

# How to implement custom operator logic:
# * Find the relevant base class for your operation (all operations can be found in `external/rtamt/rtamt/semantics/stl/discrete_time/online/`)
# * Implement a new class which inherits from the base (It is important that you choose the right base, because otherwise the visitors will fail!)
# * Put your custom aggregation logic into the `update` method
# * Register your operator in the `CustomAstVisitor` and override the base `visit<Operator>` method


class CustomAndOperation(AndOperation):
    def update(self, sample_left, sample_right):
        # TODO: replace custom logic or use the default RTAMT implementation.
        return super().update(sample_left, sample_right)


class CustomOnceTimedOperation(OnceTimedOperation):
    def update(self, sample):
        # TODO: replace custom logic or use the default RTAMT implementation.
        return super().update(sample)


# Register your custom operators in this class. Go to the base class, to get an overview over the available `visit<Operation>` methods.
class CustomAstVisitor(StlDiscreteTimeOnlineAstVisitor):
    def visitAnd(self, node, *args, **kwargs):
        self.visitChildren(node, *args, **kwargs)
        self.online_operator_dict[node.name] = CustomAndOperation()

    def visitTimedOnce(self, node, *args, **kwargs):
        self.visitChildren(node, *args, **kwargs)
        begin, end = self.time_unit_transformer(node)
        self.online_operator_dict[node.name] = CustomOnceTimedOperation(begin, end)


# The `CustomUpdateVisitor` is responsible for calling the `update` methods on all operators.
# It also collects the input values (e.g. from subformulas) for each operator.
# In case context aware aggregations are required, those could be added here.
#
# Theoretically, the custom aggregation logic could also be implemented here. But it looks like
# the "more RTAMT way" is to use the custom operators.
# Nevertheless, it might make sense to implement the logic here, if a single operator relies on context information like its sub-formulars or sub-predicates.
class CustomUpdateVisitor(DiscreteTimeOnlineUpdateVisitorDict):
    def visit(self, node, *args, **kwargs):
        return super().visit(node, *args, **kwargs)


# The stlmonitor adds two new operators: all and exist (which are equivilant to always and future operators). To override their behaviour, this evaluation visitor has to be modified.
class EnhancedEvaluationMonitorTreeVisitor(EvaluationMonitorTreeVisitor):
    """
    An evaluation visitor, with custom aggregation logic for all and exist nodes.
    """

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        # TODO: Put custom aggregation logic here. See base class for usage details.
        return super().visit_all_node(all_node, *ctx)

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        # TODO: Put custom aggregation logic here. See base class for usage details.
        return super().visit_exist_node(exist_node, *ctx)


# The pasitifer applies transformations to the RTAMT AST.
# Here we could implement custom transformations to the rules, in case this is necessary.
class CustomPastifier(StlPastifier): ...


# ==============================
# End custom aggregation logic
# ==============================

# config used for the world creation
config = get_world_config()
# MPR must be explicitly enabled
config["use_mpr"] = True

# MPR must be explicitly enabled
rule_evaluator_config = get_traffic_rule_config()
rule_evaluator_config["traffic_rules_param"]["use_mpr"] = True


# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario, config=config)


# Plumbing code to inject our custom RTAMT interpreter and visitors into the STL monitor code
def custom_stl_discrete_time_online_specification_factory(
    semantics: rtamt.Semantics,
) -> AbstractOnlineSpecification:
    online_interpreter = discrete_time_online_interpreter_factory(CustomAstVisitor)()
    # TODO: The offline interpreter is also created with our custom visitor and custom online operators -> can this lead to uninted issues?
    offline_interpreter = discrete_time_offline_interpreter_factory(CustomAstVisitor)()

    spec = AbstractOfflineOnlineSpecification(
        StlAst(), offline_interpreter, online_interpreter, pastifier=CustomPastifier()
    )
    spec.online_interpreter.updateVisitor = CustomUpdateVisitor()

    return spec


class CustomRtamtStlMonitor(RtamtStlMonitor):
    def __init__(self, rule_str, predicates, dt, output_type):
        self._rule = rule_str
        self._predicates = predicates
        self._output_type = output_type
        self._dt = dt

        self._spec = _create_spec(
            rule_str,
            output_type,
            predicates,
            dt,
            custom_stl_discrete_time_online_specification_factory,
        )

    def copy(self):
        return CustomRtamtStlMonitor(
            self._rule, self._predicates, self.dt, self._output_type
        )


class EnhancedCreationMonitorTreeVisitor(MonitorCreationRuleTreeVisitor):
    def visit_rule_node(self, rule_node: RuleNode, *ctx):
        children = [c.visit(self, *ctx) for c in rule_node.children]
        monitor = CustomRtamtStlMonitor.create_from_rule_node(
            rule_node, self.dt, self.output_type
        )
        return RuleMonitorNode(rule_node.name, children, monitor)


# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
rule_evaluator = RuleEvaluator.create_from_config(
    world,
    ego_vehicle.id,
    monitor_creation_visitor=EnhancedCreationMonitorTreeVisitor(dt=scenario.dt),
    monitor_evaluation_visitor=EnhancedEvaluationMonitorTreeVisitor(),
)

# Either step through time steps sequentially
robustness = rule_evaluator.update()
current_time_step = rule_evaluator.current_time
print(f"robustness is {robustness} at time step {current_time_step}")

# Also all predicate robustness values are available
predicate_robustness = rule_evaluator.get_predicates()

# Or evaluate for all time steps of the vehicle
robustness_array = rule_evaluator.evaluate()

rule_evaluator.reset(ego_vehicle, world)
print(robustness_array)
