import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator

scenario_path = "../scenarios/ZAM_Sandra-5838_1_T-1001.xml"

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario)

# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
for vehicle in world.vehicles:
    if np.array_equal(vehicle.state_list_cr[0].position, planning_problem.initial_state.position):
        ego_vehicle = vehicle
        break
# ego_vehicle = next(iter(world.vehicles))
for rule in ["R_G1", "R_G2", "R_G3"]:
    rule_evaluator = RuleEvaluator.create_from_config(world, ego_vehicle, rule)

    # Either step through time steps sequentially
    robustness = rule_evaluator.update()
    current_time_step = rule_evaluator.current_time

    # Also all predicate robustness values are available
    predicate_robustness = rule_evaluator.get_predicates()

    # Or evaluate for all time steps of the vehicle
    robustness_array = rule_evaluator.evaluate()
    print(robustness_array)
# rule_evaluator.reset(ego_vehicle, world)

# nodedicts = rule_evaluator.get_node_dicts()

# for n in nodedicts:
#     print(n)
