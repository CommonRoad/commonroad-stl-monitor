from pathlib import Path

import pandas as pd
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.visitor import MonitorCreationRuleTreeVisitor

input_scenarios = Path(__file__).parent.parent.parent / "scenarios-for-semantic-aware-stl" / "highD"
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"

results = []

scale_rob = True
model_path = "/path/to/mpr/models"

for scenario in input_scenarios.glob("*.xml"):
    for rule in (
        "R_G1",
        "R_G2",
        "R_G3",
        "R_G4",
        "R_I1",
        "R_I2",
        "R_I3",
        "R_I4",
        "R_I5",
    ):
        for use_mpr in (False,):
            for use_enhanced_aggregation in (False, True):
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
                rule_evaluator = RuleEvaluator.create_from_config(
                    world,
                    ego_vehicle.id,
                    rule=rule,
                    monitor_creation_visitor=MonitorCreationRuleTreeVisitor(
                        dt=scenario.dt, output_type=output_type
                    ),
                    monitor_evaluation_visitor=OfflineEvaluationMonitorTreeVisitor(),
                    output_type=output_type,
                )
                # Either step through time steps sequentially
                robustness = rule_evaluator.evaluate_offline()

                results.append(
                    {
                        "Scenario": scenario,
                        "Rule": rule,
                        "MPR": use_mpr,
                        "Enhanced Aggregation": use_enhanced_aggregation,
                    }
                )

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

###
input_scenarios = Path(__file__).parent.parent.parent / "scenarios-for-semantic-aware-stl" / "highD"
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"

results = []

scale_rob = True
model_path = "/path/to/mpr/models"

for scenario_path in sorted(input_scenarios.glob("*.xml"))[1:4]:
    for rule in (
        "R_G1",
        "R_G2",
        "R_G3",
        "R_G4",
        "R_I1",
        "R_I2",
        "R_I3",
        "R_I4",
        "R_I5",
    ):
        for use_mpr in (False,):
            for use_enhanced_aggregation in (False, True):
                # try:
                scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

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
                rule_evaluator = RuleEvaluator.create_from_config(
                    world,
                    ego_vehicle.id,
                    rule=rule,
                    monitor_creation_visitor=MonitorCreationRuleTreeVisitor(
                        dt=scenario.dt, output_type=output_type
                    ),
                    monitor_evaluation_visitor=OfflineEvaluationMonitorTreeVisitor(),
                    output_type=output_type,
                )
                # Either step through time steps sequentially
                robustness = rule_evaluator.evaluate_offline()

                results.append(
                    {
                        "Scenario": scenario.scenario_id,
                        "Rule": rule,
                        "MPR": use_mpr,
                        "Enhanced Aggregation": use_enhanced_aggregation,
                    }
                )
                # except Exception as e:
                #     print(f"Failed to process {scenario.scenario_id}, {rule}, {use_mpr}, {use_enhanced_aggregation}: {e}")


results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
