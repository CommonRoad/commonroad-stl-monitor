from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

scenario_path = "/home/adam/commonroad-stl-monitor/scenarios/test_interstate/CHN_SinDD-831_1_T-9.xml"

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario)

# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
#rule_evaluator = RuleEvaluator.create_from_config(world, ego_vehicle)
rule_evaluator = RuleEvaluator.create_from_config(world, ego_vehicle, rule="R_G3")

# Either step through time steps sequentially
robustness = rule_evaluator.update()
current_time_step = rule_evaluator.current_time

# Also all predicate robustness values are available
predicate_robustness = rule_evaluator.get_predicates()

# Or evaluate for all time steps of the vehicle
robustness_array = rule_evaluator.evaluate()

rule_evaluator.reset(ego_vehicle, world)
