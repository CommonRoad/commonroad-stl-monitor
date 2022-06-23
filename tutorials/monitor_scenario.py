from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world_state import World
from crmonitor.evaluation.evaluation import RuleEvaluator

scenario_path = "../scenarios/test_interstate/DEU_test_safe_distance.xml"

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world_state = World.create_from_scenario(scenario)

# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world_state.vehicles))
rule_evaluator = RuleEvaluator.create_from_config(world_state, ego_vehicle)

# Either step through time steps sequentially
robustness = rule_evaluator.update()
current_time_step = rule_evaluator.current_time

# Also all predicate robustness values are available
predicate_robustness = rule_evaluator.get_predicates()

# Or evaluate for all time steps of the vehicle
robustness_array = rule_evaluator.evaluate()
