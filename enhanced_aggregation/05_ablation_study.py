import logging
from pathlib import Path

import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.predicates.base import PredicateEvaluatorConfig, PredicateMprConfig

_LOGGER = logging.getLogger(__name__)
_LOG

input_scenarios = Path(__file__).parent.parent.parent / "scenarios-for-semantic-aware-stl" / "highD"
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"

results = []

scale_rob = True
model_path = Path("/path/to/mpr/models")

for scenario_path in sorted(input_scenarios.glob("*.xml"))[1:1]:
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
        for use_mpr in (True, False):
            scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

            # Create a world state, which is a holder class for intermediate results produced by the monitoring.
            # Use the convenience class method to create with default configuration from a scenario.
            world = World.create_from_scenario(scenario)

            # Create a rule evaluator
            # Provide the vehicle to evaluate traffic rules for as ego vehicle
            predicate_evaluator_config = PredicateEvaluatorConfig(
                mpr=PredicateMprConfig(enabled=use_mpr, model_path=model_path),
                scale_rob=not use_mpr,
            )

            # Create a rule evaluator
            # Provide the vehicle to evaluate traffic rules for as ego vehicle
            ego_vehicle = next(iter(world.vehicles))
            rule_evaluator = OfflineRuleEvaluator.create_for_rule(
                world,
                ego_vehicle.id,
                rule,
                predicate_evaluator_config=predicate_evaluator_config,
            )
            # Either step through time steps sequentially
            robustness = rule_evaluator.evaluate()

            results.append(
                {
                    "Scenario": scenario.scenario_id,
                    "Rule": rule,
                    "MPR": use_mpr,
                    "Robustness": robustness,
                }
            )
            # except Exception as e:
            #     print(f"Failed to process {scenario.scenario_id}, {rule}, {use_mpr}, {use_enhanced_aggregation}: {e}")


results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
