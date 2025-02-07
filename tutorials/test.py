from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as Cfg

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.visitor import RuleTreeVisitor
from crmonitor.monitor.monitor_node import MonitorNode
from crmonitor.rule.rule_node import AllNode, ExistNode, PredicateNode, RuleNode

scenario_path = "/tmp/USA_US101-9_1_T-1.xml"
# Cfg.set_paths(
#     path_root="./",
#     folder_config="../commonroad-model-predictive-robustness/config_files",
#     default_profile="default",
# )
# Cfg.construct_configuration("default")
# Cfg.update_with_config(
#     {
#         "path": {"path_models": "/tmp/pretrainedMPR/interstate/2023-07-04/"},
#         "common": {
#             "scenario": "interstate",
#             "road_network": {"interstate": {"use_phantom_lane": False}},
#         },
#     }
# )


# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario)

# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
rule_evaluator = RuleEvaluator.create_from_config(world, ego_vehicle, rule="R_G1")

# Either step through time steps sequentially
robustness = rule_evaluator.update()
current_time_step = rule_evaluator.current_time
exit()

# Also all predicate robustness values are available
predicate_robustness = rule_evaluator.get_predicates()

# Or evaluate for all time steps of the vehicle
robustness_array = rule_evaluator.evaluate()

rule_evaluator.reset(ego_vehicle, world)

# nodedicts = rule_evaluator.get_node_dicts()

# for n in nodedicts:
#     print(n)
