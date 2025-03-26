import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from commonroad.scenario.scenario import Scenario
import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.predicates.base import PredicateEvaluatorConfig, PredicateMprConfig

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

input_scenarios = Path(__file__).parent.parent.parent / "highD-scenarios"
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"


scale_rob = True
model_path = Path("/path/to/mpr/models")

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
                    "use_phantom_lane": True
                }  # Must disable phantom lanes, because otherwise commonroad-dc segfaults...
            },
        },
    },
    # Path root must point to a local revision of commonroad-model-predictive-robustness.
    # This configuration, assumes that the repo is in the same directory as stl-monitor repo.
    # If this is not the case for your setup, adjust the path here accordingly.
    path_root=str(Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"),
    folder_config="config_files",
    default_profile="default",
)


def process_scenario_with_rule(scenario_path: Path, rule: str, use_mpr: bool) -> dict:
    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
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
    _LOGGER.info(
        f"Evaluating rule {rule} {'with mpr' if use_mpr else 'without mpr'} for vehicle {ego_vehicle.id} in scenario {scenario.scenario_id} from time step {ego_vehicle.start_time} to {ego_vehicle.end_time}"
    )
    rule_evaluator = OfflineRuleEvaluator.create_for_rule(
        world,
        ego_vehicle.id,
        rule,
        predicate_evaluator_config=predicate_evaluator_config,
    )
    # Either step through time steps sequentially
    robustness = rule_evaluator.evaluate()

    return {
        "scenario_id": str(scenario.scenario_id),
        "rule": rule,
        "mpr": use_mpr,
        "vehicle_id": ego_vehicle.id,
        "start_time_step": ego_vehicle.start_time,
        "end_time_step": ego_vehicle.end_time,
        "robustness": ", ".join(map(str, robustness)),
    }


rules = [
    "R_G1",
    "R_G2",
    "R_G3",
    "R_G4",
    "R_I1",
    "R_I2",
    "R_I3",
    "R_I4",
    "R_I5",
]

scenarios_paths = list(input_scenarios.glob("*.xml"))[0:1]
use_mpr = [True, False]


results = []
with ProcessPoolExecutor() as executor:
    for result in executor.map(
        process_scenario_with_rule,
        *zip(*itertools.product(scenarios_paths, rules, (True, False))),
    ):
        results.append(result)


results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
