import itertools
import logging
import multiprocessing as mp
import random
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.predicates.base import PredicateEvaluatorConfig, PredicateMprConfig

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

input_scenarios = Path(__file__).parents[3] / "scenarios-for-semantic-aware-stl" / "highD"
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results.csv"


num_vehicles_per_scenarios = 4
output_type = OutputType.OUTPUT_ROBUSTNESS
model_path = Path(__file__).parent.parent / "output" / "models"
snapshot_frequency = 4

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


def process_scenario_with_rule(
    scenario: Scenario, ego_vehicle_id: int, rule: str, use_mpr: bool
) -> dict:
    try:
        world = World.create_from_scenario(scenario)
        # Create a rule evaluator
        # Provide the vehicle to evaluate traffic rules for as ego vehicle
        predicate_evaluator_config = PredicateEvaluatorConfig(
            mpr=PredicateMprConfig(enabled=use_mpr, model_path=model_path),
            scale_rob=True,
        )

        # Create a rule evaluator
        # Provide the vehicle to evaluate traffic rules for as ego vehicle
        ego_vehicle = world.vehicle_by_id(ego_vehicle_id)
        assert ego_vehicle is not None
        _LOGGER.info(
            f"Evaluating rule {rule} with {'MPR' if use_mpr else 'MFR'} for vehicle {ego_vehicle.id} in scenario {scenario.scenario_id} from time step {ego_vehicle.start_time} to {ego_vehicle.end_time}"
        )
        rule_evaluator = OfflineRuleEvaluator.create_for_rule(
            world,
            ego_vehicle.id,
            rule,
            output_type=output_type,
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
    except Exception as e:
        _LOGGER.warning(e)
        return {
            "scenario_id": str(scenario.scenario_id),
            "rule": rule,
            "mpr": use_mpr,
            "vehicle_id": np.nan,
            "start_time_step": np.nan,
            "end_time_step": np.nan,
            "robustness": ", ".join(map(str, [np.nan, np.nan])),
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

scenarios_paths = np.random.choice(list(input_scenarios.glob("*.xml")), 100, replace=False)
scenarios = list(
    map(
        lambda scenario_path: CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)[0],
        scenarios_paths,
    )
)

ego_vehicles_per_scenario = {}
for scenario in scenarios:
    vehicle_ids = list(map(lambda v: v.obstacle_id, scenario.dynamic_obstacles))
    if len(vehicle_ids) < num_vehicles_per_scenarios:
        continue
    ego_vehicles_per_scenario[scenario.scenario_id] = random.sample(
        vehicle_ids, k=num_vehicles_per_scenarios
    )


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    results_since_last_snapshot = 0
    results = []
    tasks = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        for scenario, rule, use_mpr in itertools.product(scenarios, rules, (True, False)):
            for ego_vehicle_id in ego_vehicles_per_scenario[scenario.scenario_id]:
                task = executor.submit(
                    process_scenario_with_rule, scenario, ego_vehicle_id, rule, use_mpr
                )
                tasks[task] = (scenario.scenario_id, ego_vehicle_id, rule, use_mpr)

        for finished_future in futures.as_completed(tasks.keys()):
            exec = finished_future.exception()
            if exec is not None:
                task_arguments = tasks[finished_future]
                _LOGGER.warning(f"Exception {exec} occurred while processing {task_arguments}")
                continue

            result = finished_future.result()
            results.append(finished_future.result())
            results_since_last_snapshot += 1
            del tasks[finished_future]

            if results_since_last_snapshot >= snapshot_frequency:
                _LOGGER.info(f"Writing snapshot to {output_file}")
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_file, index=False)
                results_since_last_snapshot = 0

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
